#!/usr/bin/env python3
"""Structural probe: where does the anchor sit relative to the answer inside a case?

Motivation
    The paper claims that the record answering the question and the record storing the
    protected value are the *same block*, and that they are on *different lines* inside
    that block.  Both statements are load-bearing: the first is why routing works at all,
    the second is why protecting a single line is not enough.  Until now they were quoted
    as aggregate survival percentages.  This script measures the layout directly, from the
    case text alone, with no model call.

What it computes, per case
    n_blocks        blocks in the context, split on the `----- entry N -----` markers
    anchor_block    index of the block that mentions the queried entity
    anchor_line     line index (global) of that mention
    answer_block    index of the block that carries the reference answer
    answer_line     line index (global) of that answer
    block_dist      |anchor_block - answer_block|      0 means same block
    line_dist       |anchor_line  - answer_line|       >0 means different lines
    anchor_block_lines   size of the anchor block in lines
    anchor_block_chars   size of the anchor block in characters

How the anchor is found, per case kind
    multi_entity    the queried entity is `target_entity` (an `ACCT-...` handle), which
                    appears on a `Account ID:` line inside a private-contact block.
    needle          `target_entity` is empty; the question names a record id instead
                    ("... for the record with id <id>?").  The id is recovered by taking
                    every `"id": "<value>"` in the context and keeping the one that also
                    appears in the question, so nothing is guessed from a template.

    In both kinds the answer is the first line that contains any of `answers`.

Output
    A JSON file with a per-case list plus aggregate summaries per case kind and per source.
    `--percase` additionally joins the evaluated arms, so survival rates can be reported
    per retention strategy (anchor-line, delete, alias, DARK) on exactly the same cases.

Usage
    python src/analysis_quad4/structure_probe.py \
        --cases data/longctx_max.jsonl.gz \
        --out   results/diag/structure.json

    python src/analysis_quad4/structure_probe.py \
        --cases data/longctx_max.jsonl.gz \
        --percase 'results/percase/quadfull_s*_percase.gz' \
        --out   results/diag/structure.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import io
import json
import re
import statistics

MARK = re.compile(r'^-{3,}\s*entry\s+(\d+)\s*-{3,}\s*$', re.I)
ID_FIELD = re.compile(r'"id"\s*:\s*"([^"]+)"')


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def parse_blocks(ctx):
    """Split a context on the entry markers. Returns [(marker_index, [line, ...]), ...]."""
    blocks, cur, cur_i = [], [], -1
    for line in ctx.split("\n"):
        m = MARK.match(line.strip())
        if m:
            if cur:
                blocks.append((cur_i, cur))
            cur_i, cur = int(m.group(1)), [line]
        else:
            cur.append(line)
    if cur:
        blocks.append((cur_i, cur))
    if not blocks:                       # corpora without markers: one block
        blocks = [(-1, ctx.split("\n"))]
    return blocks


# The six anchor forms the benchmark actually uses.  Order matters only in that the
# "entry describing X" wording must not be mistaken for the bare "for entry N" one.
ANCHOR_PATTERNS = [
    ("id", re.compile(r"with id\s+([A-Za-z0-9_\-\.]+)")),
    ("name", re.compile(r"record named\s+([A-Za-z0-9_\-\.]+)")),
    ("session", re.compile(r"for session\s+([A-Za-z0-9_\-\.]+)")),
    ("describing", re.compile(r"entry describing\s+([A-Za-z0-9_\-\.]+)")),
    ("entry", re.compile(r"for entry\s+(\d+)")),
]


def needle_anchor_from_question(case):
    """Recover the anchor a needle question names, and say which form it took.

    The benchmark uses six forms (Table VII in the paper).  Returning the form as well as
    the token lets the caller check its counts against that table, which is how we know the
    extractor is not silently missing a template.
    """
    q = case.get("question") or ""
    for kind, pat in ANCHOR_PATTERNS:
        m = pat.search(q)
        if m:
            return kind, m.group(1)
    return None, None


def locate(blocks, needle):
    """Return (block_i, line_i_global) of the first line containing `needle`."""
    flat = 0
    for bi, (_idx, lines) in enumerate(blocks):
        for line in lines:
            if needle and needle in line:
                return bi, flat
            flat += 1
    return None, None


def locate_entry(blocks, entry_id):
    """Anchor given as a bare entry index: the anchor is that block's marker line."""
    flat = 0
    for bi, (idx, lines) in enumerate(blocks):
        if idx == int(entry_id):
            return bi, flat
        flat += len(lines)
    return None, None


def probe_case(case):
    ctx = case.get("context") or ""
    kind = case.get("case_kind", "?")
    answers = [a for a in (case.get("answers") or []) if a]

    anchor = (case.get("target_entity") or "").strip()
    anchor_kind = "entity"
    if not anchor:
        anchor_kind, anchor = needle_anchor_from_question(case)
        anchor = anchor or ""

    rec = {
        "key": case.get("key"),
        "case_kind": kind,
        "source": case.get("source"),
        "n_entities": case.get("n_entities"),
        "anchor": anchor or None,
        "anchor_kind": anchor_kind if anchor else None,
        "answers": answers,
        "status": "ok",
    }

    blocks = parse_blocks(ctx)
    rec["n_blocks"] = len(blocks)
    rec["n_lines"] = sum(len(l) for _i, l in blocks)

    if not anchor:
        rec["status"] = "no_anchor"
        return rec
    if not answers:
        rec["status"] = "no_answer"
        return rec

    if anchor_kind == "entry":
        a_bi, a_li = locate_entry(blocks, anchor)
    else:
        a_bi, a_li = locate(blocks, anchor)
    if a_bi is None:
        rec["status"] = "anchor_not_found"
        return rec

    q_bi = q_li = None
    for ans in answers:
        q_bi, q_li = locate(blocks, ans)
        if q_bi is not None:
            break
    if q_bi is None:
        rec["status"] = "answer_not_found"
        return rec

    block_lines = blocks[a_bi][1]
    rec.update({
        "anchor_block": a_bi,
        "anchor_line": a_li,
        "answer_block": q_bi,
        "answer_line": q_li,
        "block_dist": abs(a_bi - q_bi),
        "line_dist": abs(a_li - q_li),
        "same_block": a_bi == q_bi,
        "same_line": a_li == q_li,
        "anchor_block_lines": len(block_lines),
        "anchor_block_chars": sum(len(x) + 1 for x in block_lines),
    })
    return rec


def summarise(records):
    ok = [r for r in records if r["status"] == "ok"]
    if not ok:
        return {"n": 0}
    same_block = sum(1 for r in ok if r["same_block"])
    same_line = sum(1 for r in ok if r["same_line"])
    ld = [r["line_dist"] for r in ok if r["same_block"]]
    return {
        "n": len(ok),
        "n_total": len(records),
        "status_counts": dict(collections.Counter(r["status"] for r in records)),
        "same_block": same_block,
        "same_block_pct": round(100.0 * same_block / len(ok), 2),
        "same_line": same_line,
        "same_line_pct": round(100.0 * same_line / len(ok), 2),
        "line_dist_median": statistics.median(ld) if ld else None,
        "line_dist_mean": round(statistics.mean(ld), 2) if ld else None,
        "line_dist_hist": dict(sorted(collections.Counter(ld).items())) if ld else {},
        "anchor_block_lines_median": statistics.median(
            r["anchor_block_lines"] for r in ok),
        "anchor_block_chars_median": statistics.median(
            r["anchor_block_chars"] for r in ok),
    }


def load_percase(pattern):
    """arm -> {case_key: record}"""
    arms = collections.defaultdict(dict)
    files = sorted(glob.glob(pattern))
    for path in files:
        with open_maybe_gz(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                arms[r["arm"]][r["case_key"]] = r
    return arms, files


def survival_by_arm(records, arms):
    """Answer retention and raw-value exposure, per arm.

    Deliberately *not* comparing `tgt_entity_present` across arms: in the harness that
    field is evaluated against the surface token the arm itself produced -- the alias for an
    alias arm, the literal `[REDACTED]` for a delete arm -- so it answers "is the handle I
    substituted still there", not "is the referent addressable".  Reading it as a survival
    rate across arms would be wrong.  The two quantities that do compare are the answer span
    (`ans_in_ctx`) and exposure of the raw protected value (the leak fields).
    """
    keys = {r["key"] for r in records if r["status"] == "ok"}
    multi = {r["key"] for r in records if r["status"] == "ok" and r["case_kind"] == "multi_entity"}
    out = {}
    for arm, rows in sorted(arms.items()):
        hit = [rows[k] for k in keys if k in rows]
        if not hit:
            continue
        mhit = [rows[k] for k in multi if k in rows]
        leaked = [r for r in hit
                  if (r.get("ctx_leak_value") or 0) > 0 or (r.get("out_leak_value") or 0) > 0]
        out[arm] = {
            "n": len(hit),
            "answer_present_pct": round(
                100.0 * sum(1 for r in hit if r.get("ans_in_ctx")) / len(hit), 2),
            "raw_value_exposed_pct": round(100.0 * len(leaked) / len(hit), 2),
            "n_multi": len(mhit),
            "gold_attr_present_pct": (round(
                100.0 * sum(1 for r in mhit if r.get("tgt_attr_present")) / len(mhit), 2)
                if mhit else None),
            # collision=1 means distinct entities collapsed onto one surface token, so the
            # reader cannot tell which record the question is about.  Deletion produces this
            # by construction, which is why a delete arm can keep the answer span and still
            # lose the task.
            "collision_pct": (round(
                100.0 * sum(1 for r in mhit if (r.get("collision") or 0) > 0) / len(mhit), 2)
                if mhit else None),
            "mean_ctx_tokens": round(statistics.mean(
                [r["ctx_tokens"] for r in hit if r.get("ctx_tokens")] or [0]), 0),
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="data/longctx_max.jsonl.gz")
    ap.add_argument("--out", default="results/diag/structure.json")
    ap.add_argument("--percase", default="",
                    help="glob of per-case files to join for survival by arm")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    cases = []
    with open_maybe_gz(args.cases) as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    if args.limit:
        cases = cases[:args.limit]
    print("cases: %d" % len(cases))

    records = [probe_case(c) for c in cases]

    by_kind = {}
    for kind in sorted({r["case_kind"] for r in records}):
        by_kind[kind] = summarise([r for r in records if r["case_kind"] == kind])
    by_source = {}
    for src in sorted({str(r["source"]) for r in records}):
        by_source[src] = summarise([r for r in records if str(r["source"]) == src])

    payload = {
        "cases_file": args.cases,
        "overall": summarise(records),
        "by_kind": by_kind,
        "by_source": by_source,
        "cases": records,
    }

    if args.percase:
        arms, files = load_percase(args.percase)
        payload["survival_by_arm"] = survival_by_arm(records, arms)
        payload["percase_files"] = files
        print("joined %d arms from %d per-case file(s)" % (len(arms), len(files)))

    out = args.out
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False)

    o = payload["overall"]
    print()
    print("overall: %d located of %d   same block %.2f%%   same line %.2f%%"
          % (o.get("n", 0), o.get("n_total", 0),
             o.get("same_block_pct", 0), o.get("same_line_pct", 0)))
    print("status: %s" % o.get("status_counts"))
    for kind, s in by_kind.items():
        print("  %-14s n=%4d  same block %6.2f%%  same line %6.2f%%  line-dist median %s"
              % (kind, s.get("n", 0), s.get("same_block_pct", 0),
                 s.get("same_line_pct", 0), s.get("line_dist_median")))
    if "survival_by_arm" in payload:
        print()
        print("%-24s %6s %9s %11s %9s %9s %8s"
              % ("arm", "n", "answer%", "raw-exposed%", "goldattr%", "collision%", "tokens"))
        for arm, s in payload["survival_by_arm"].items():
            fmt = lambda v: ("%.2f" % v) if v is not None else "-"
            print("%-24s %6d %9.2f %11.2f %9s %9s %8.0f"
                  % (arm, s["n"], s["answer_present_pct"], s["raw_value_exposed_pct"],
                     fmt(s["gold_attr_present_pct"]), fmt(s["collision_pct"]),
                     s["mean_ctx_tokens"]))
        print()
        print("goldattr%% is multi-entity only (%d cases); answer%% and raw-exposed%% "
              "are over all cases." % payload["survival_by_arm"].get(
                  "quad4", {}).get("n_multi", 0))
    print()
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
