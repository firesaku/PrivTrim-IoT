#!/usr/bin/env python3
"""Simulate an imperfect sensitive-value detector and emit paired case files.

The paper's zero leakage is measured against a known set V of injected values. A reviewer will ask
what happens when V has to come from a detector instead. The honest answer is a curve, not a single
number: here each protected value survives detection independently with probability rho, so rho = 1
is the paper's setting and rho < 1 is a detector that misses some values.

Two things make the curve readable rather than trivially saturated:

  * Routing sees only the detected subset (`canaries`), but scoring sees the full injected set
    (`canaries_eval`). Dropping a value without this split would make the missed value both
    unprotected and uncounted, which is a false zero rather than a measured leak.

  * Every rho uses the SAME 500 cases, sampled once and stratified by (source, case_kind), so the
    points are paired and the only thing that changes between them is which values were detected.

The rho grid is fine near 1.0 on purpose. A case carries 12.8 protected values on average, so even
a 1% miss rate leaves roughly one case in eight with an undetected value; a grid of 1.0/0.75/0.5
would be saturated at the first non-trivial point.

Usage
    python src/longctx/imperfect_detector.py --outdir data/recall
    python src/longctx/imperfect_detector.py --outdir data/recall --ratios 1.0,0.98
"""
from __future__ import annotations

import argparse
import collections
import gzip
import io
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RATIOS = "1.0,0.995,0.99,0.98,0.95,0.90"
SUBSET = 500
SUBSET_SEED = 20261108


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def load_cases(path):
    out = []
    with open_maybe_gz(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def stratified_subset(cases, n):
    """Deterministic, proportional to (source, case_kind). Same set for every rho."""
    buckets = collections.defaultdict(list)
    for c in cases:
        buckets[(c.get("source"), c.get("case_kind"))].append(c)
    rng = random.Random(SUBSET_SEED)
    picked = []
    total = len(cases)
    for key in sorted(buckets, key=lambda k: (str(k[0]), str(k[1]))):
        group = buckets[key]
        group.sort(key=lambda c: c["key"])
        want = max(1, int(round(n * len(group) / total)))
        want = min(want, len(group))
        picked.extend(rng.sample(group, want))
    picked.sort(key=lambda c: c["key"])
    return picked[:n] if len(picked) > n else picked


def rho_tag(rho):
    return ("%.3f" % rho).rstrip("0").rstrip(".").replace(".", "p")


def discount(case, rho):
    """Keep each protected value independently with probability rho, seeded per (case, rho)."""
    cans = case.get("canaries") or []
    if rho >= 1.0:
        out = dict(case)
        out.pop("canaries_eval", None)
        return out, len(cans), 0
    rng = random.Random("%s|%.6f" % (case["key"], rho))
    kept, dropped = [], 0
    for c in cans:
        if rng.random() < rho:
            kept.append(c)
        else:
            dropped += 1
    out = dict(case)
    out["canaries"] = kept
    out["canaries_eval"] = cans
    return out, len(kept), dropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(ROOT / "data" / "longctx_max.jsonl.gz"))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ratios", default=DEFAULT_RATIOS)
    ap.add_argument("--subset", type=int, default=SUBSET)
    ap.add_argument("--dry-run", action="store_true", help="只报统计，不写文件")
    args = ap.parse_args(argv)

    cases = load_cases(args.cases)
    if not cases:
        raise SystemExit("no cases loaded from %s" % args.cases)
    picked = stratified_subset(cases, args.subset) if args.subset else cases
    outdir = Path(args.outdir)
    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    manifest = {"source": str(args.cases), "subset": len(picked),
                "subset_seed": SUBSET_SEED, "runs": []}

    print("cases=%d  subset=%d  rho in %s" % (len(cases), len(picked), ratios))
    print("%-8s %8s %10s %12s %14s" % ("rho", "kept", "dropped", "cases w/ drop", "cases leak-free"))
    for rho in ratios:
        rows, kept_n, drop_n, with_drop = [], 0, 0, 0
        for c in picked:
            out, k, d = discount(c, rho)
            rows.append(out)
            kept_n += k
            drop_n += d
            if d:
                with_drop += 1
        n = len(rows)
        # P(no undetected value in a case) under independence of the miss events
        exp_clean = sum(1.0 for _ in range(0))  # placeholder, computed below
        tot = sum(len(c.get("canaries") or []) for c in picked)
        # expected fraction of cases with every value detected: product over the case
        exp_clean = 0.0
        for c in picked:
            m = len(c.get("canaries") or [])
            exp_clean += rho ** m
        exp_clean = 100.0 * exp_clean / n
        print("%-8s %8d %10d %11.1f%% %13.1f%%"
              % (rho, kept_n, drop_n, 100.0 * with_drop / n, exp_clean))
        tag = rho_tag(rho)
        entry = {"rho": rho, "tag": tag, "n": n, "kept": kept_n, "dropped": drop_n,
                 "cases_with_drop": with_drop,
                 "expected_clean_pct": exp_clean,
                 "total_values": tot}
        if not args.dry_run:
            dest = outdir / ("recall_%s.jsonl" % tag)
            with io.open(dest, "w", encoding="utf-8", newline="\n") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            entry["file"] = str(dest)
            print("         wrote %s (%d cases)" % (dest, n))
        manifest["runs"].append(entry)

    if not args.dry_run:
        mpath = outdir / "recall_manifest.json"
        with io.open(mpath, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(manifest, fh, indent=1)
        print("wrote", mpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
