#!/usr/bin/env python3
"""Table VIII: what the benchmark actually contains, per source.

Every column is recomputed from the case files and the per-case records, never inferred from
the sample counts. Two kinds of number are mixed deliberately:

  structural columns   from the case annotations themselves -- `n_records`, `words`,
                       `entry_spans`, `answer_region_index`, `entry_index`, `entity_offset`.
                       These are the benchmark's own labels, not our parsing.
  token column         from the `full` arm's per-case records, so it is the reader's tokenizer
                       rather than an estimate.

The script also cross-checks the block membership our `structure_probe.py` derives by parsing
against the annotated `answer_region_index` / `entry_index`. Disagreement would mean one of the
two is wrong, so it is reported rather than smoothed over.

Usage
    python src/analysis_quad4/make_dataset_table.py \
        --cases data/longctx_max.jsonl.gz \
        --percase 'results/percase/quadfull_s*_percase.gz' \
        --structure results/diag/structure.json \
        --out results/diag/dataset_stats.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import io
import json
import statistics


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def block_of_offset(spans, off):
    for i, span in enumerate(spans):
        if span[0] <= off < span[1]:
            return i
    return None


def canonical_block_membership(case):
    """Return (anchor_block, answer_block) from the benchmark's own annotations."""
    spans = case.get("entry_spans") or []
    ans_bi = case.get("answer_region_index")
    if case.get("case_kind") == "multi_entity":
        off = case.get("entity_offset")
        if off is None:
            off = case.get("target_region_start")
        a_bi = block_of_offset(spans, off) if off is not None else None
    else:
        a_bi = case.get("entry_index")
    return a_bi, ans_bi


def load_tokens(pattern, arm="full"):
    """Median context tokens per case under the reader's tokenizer, from the given arm."""
    out = {}
    for p in sorted(glob.glob(pattern)):
        with open_maybe_gz(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("arm") == arm and r.get("ctx_tokens"):
                    out[r["case_key"]] = r["ctx_tokens"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="data/longctx_max.jsonl.gz")
    ap.add_argument("--percase", default="results/percase/quadfull_s*_percase.gz")
    ap.add_argument("--structure", default="results/diag/structure.json")
    ap.add_argument("--out", default="results/diag/dataset_stats.json")
    args = ap.parse_args(argv)

    cases = []
    with open_maybe_gz(args.cases) as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    tokens = load_tokens(args.percase)
    print("cases: %d   token rows: %d" % (len(cases), len(tokens)))

    # cross-check against the parser
    parser_blocks = {}
    try:
        st = json.load(io.open(args.structure, encoding="utf-8"))
        parser_blocks = {r["key"]: r for r in st["cases"] if r["status"] == "ok"}
    except FileNotFoundError:
        print("  (structure.json missing: skipping the parser cross-check)")

    agree = disagree = 0
    disagree_keys = []

    per_source = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in cases:
        src = str(c.get("source"))
        d = per_source[src]
        d["words"].append(c.get("words"))
        if c.get("key") in tokens:
            d["tokens"].append(tokens[c["key"]])
        d["records"].append(c.get("n_records"))
        d["canaries"].append(len(c.get("canaries") or []))
        spans = c.get("entry_spans") or []
        if spans:
            d["block_chars"].append(statistics.median(s[1] - s[0] for s in spans))
        if c.get("case_kind") == "multi_entity":
            d["entities"].append(c.get("n_entities"))
            d["n_multi"] += [1]
        else:
            d["n_needle"] += [1]

        a_bi, q_bi = canonical_block_membership(c)
        if a_bi is not None and q_bi is not None:
            same = (a_bi == q_bi)
            d["same_block"].append(1 if same else 0)
            d["co_located"] += [1]
            # compare with the parser's answer
            pr = parser_blocks.get(c.get("key"))
            if pr is not None and pr.get("answer_block") is not None:
                if (pr["answer_block"] == q_bi) and (pr["anchor_block"] == a_bi):
                    agree += 1
                else:
                    disagree += 1
                    if len(disagree_keys) < 5:
                        disagree_keys.append(c.get("key"))

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 0) if xs else None

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.mean(xs), 2) if xs else None

    table = {}
    for src in sorted(per_source):
        d = per_source[src]
        n = len(d["words"])
        table[src] = {
            "cases": n,
            "median_words": med(d["words"]),
            "median_tokens": med(d["tokens"]),
            "records_per_case": mean(d["records"]),
            "median_block_chars": med(d["block_chars"]),
            "canaries_per_case": mean(d["canaries"]),
            "mean_entities_multi": mean(d["entities"]),
            "multi_cases": len(d["n_multi"]),
            "needle_cases": len(d["n_needle"]),
            "same_block_pct": (round(100.0 * sum(d["same_block"]) / len(d["same_block"]), 2)
                               if d["same_block"] else None),
        }

    payload = {
        "sources": table,
        "totals": {
            "cases": len(cases),
            "median_tokens": med([t for d in per_source.values() for t in d["tokens"]]),
            "multi_cases": sum(len(d["n_multi"]) for d in per_source.values()),
            "needle_cases": sum(len(d["n_needle"]) for d in per_source.values()),
        },
        "parser_vs_canonical": {"agree": agree, "disagree": disagree,
                                "examples": disagree_keys},
    }
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    hdr = ("source", "cases", "words", "tokens", "rec/case", "blk chars",
           "canaries", "ents(multi)", "same-block%")
    print()
    print("%-18s %5s %6s %7s %8s %9s %8s %11s %12s" % hdr)
    for src, s in table.items():
        print("%-18s %5d %6s %7s %8s %9s %8s %11s %12s"
              % (src, s["cases"], s["median_words"], s["median_tokens"],
                 s["records_per_case"], s["median_block_chars"], s["canaries_per_case"],
                 s["mean_entities_multi"], s["same_block_pct"]))
    t = payload["totals"]
    print("\ntotal %d cases (multi %d / needle %d), median tokens %s"
          % (t["cases"], t["multi_cases"], t["needle_cases"], t["median_tokens"]))
    print("parser vs canonical block membership: agree %d, disagree %d"
          % (agree, disagree))
    if disagree_keys:
        print("  first disagreements: %s" % disagree_keys)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
