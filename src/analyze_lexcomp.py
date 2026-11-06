#!/usr/bin/env python3
"""Score the 2026 lexical-rule compressor against the arms already in results/percase/.

Reads the per-case records produced by a `slurm/eval_lexcomp.slurm` run plus the existing
DARK baseline (`results/percase/quadfull_s*_percase.gz`), pairs every arm on the same case
keys, and reports F1, prompt leakage, gold-in-context and realised context length, with
paired bootstrap intervals on every comparison.

The two control arms are included in the eval job on purpose: `full` must reproduce
0.6078 and `quad4` must reproduce 0.4937, or the batch is not comparable to the rest of
the paper.

Usage
    python src/analyze_lexcomp.py
    python src/analyze_lexcomp.py --results 'out/lexcomp/shard*_percase.jsonl' \
                                  --baseline 'results/percase/quadfull_s*_percase.gz'
"""
from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import random
import statistics


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def load_many(pattern):
    rows = {}
    files = sorted(glob.glob(pattern))
    for path in files:
        with open_maybe_gz(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows[(r["case_key"], r["arm"])] = r
    return rows, files


def leaked(row):
    return ((row.get("ctx_leak_value") or 0) > 0) or ((row.get("out_leak_value") or 0) > 0)


def summarise(rows):
    toks = [r["ctx_tokens"] for r in rows if r.get("ctx_tokens")]
    return {
        "n": len(rows),
        "f1": statistics.mean(r["f1"] for r in rows),
        "leak": 100.0 * sum(1 for r in rows if leaked(r)) / len(rows),
        "gic": 100.0 * sum(1 for r in rows if r.get("ans_in_ctx")) / len(rows),
        "tokens": statistics.mean(toks) if toks else float("nan"),
    }


def paired_ci(diffs, reps=4000, seed=7):
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(reps))
    return boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="out/lexcomp/shard*_percase.jsonl")
    ap.add_argument("--baseline", default="results/percase/quadfull_s*_percase.gz")
    args = ap.parse_args(argv)

    new, new_files = load_many(args.results)
    base, base_files = load_many(args.baseline)
    if not new:
        raise SystemExit("no per-case rows matched %r" % args.results)
    print("results  : %d rows from %d file(s)" % (len(new), len(new_files)))
    print("baseline : %d rows from %d file(s)" % (len(base), len(base_files)))

    arms = sorted({a for (_, a) in new})
    print()
    # NOTE: `full` and `quad4` here are the control arms re-run inside the lexcomp batch, not
    # the main run. They can differ from the main numbers in the fourth decimal because batched
    # fp16 inference is not bit-exact. The paired block below correctly uses `base` (the main run)
    # for the reference arm, so those comparisons are against the published numbers.
    print("%-16s %6s %8s %8s %8s %8s" % ("arm", "n", "F1", "leak%", "GIC%", "tokens"))
    for arm in arms:
        rows = [r for (_, a), r in new.items() if a == arm]
        s = summarise(rows)
        print("%-16s %6d %8.4f %8.1f %8.1f %8.0f"
              % (arm, s["n"], s["f1"], s["leak"], s["gic"], s["tokens"]))

    print()
    print("paired against the same cases in the baseline run")
    for arm in arms:
        for ref in ("full", "quad4"):
            keys = sorted(k for (k, a) in new if a == arm and (k, ref) in base)
            if not keys:
                continue
            diffs = [new[(k, arm)]["f1"] - base[(k, ref)]["f1"] for k in keys]
            lo, hi = paired_ci(diffs)
            flag = "significant" if (lo > 0 or hi < 0) else "n.s."
            print("  %-14s - %-6s n=%4d  %+.4f  [%+.4f,%+.4f]  %s"
                  % (arm, ref, len(keys), statistics.mean(diffs), lo, hi, flag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
