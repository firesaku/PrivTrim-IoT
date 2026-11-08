"""Cross-model generalisation: does DARK hold its position on readers it was not tuned on?

Reads the per-case records from slurm/xmodel.slurm (three readers from different families and
scales, arms `full` and `quad4`) plus the Qwen2.5-7B baseline, and reports for each model:

  F1 and leakage of each arm
  the paired difference DARK - full on the same cases, with a bootstrap interval
  the realised context length

The `full` arm is the control: on Qwen2.5-7B it must reproduce 0.6078. A model whose control
does not reproduce is reported as such rather than silently included.

Usage
    python src/analyze_xmodel.py --results 'results/percase/granite_s*_percase.gz' \
                                 --baseline 'results/percase/quadfull_s*_percase.gz'
"""
from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import random
import re
import statistics


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def load(pattern):
    """model_tag -> {(case_key, arm): row}"""
    out = {}
    for path in sorted(glob.glob(pattern)):
        # The tag is whatever precedes the shard index: `xmodel_qwen14b_s0` and `granite_s0` both
        # resolve, so the script works on the naming the jobs actually use.
        m = re.search(r"([A-Za-z0-9]+)_s\d+_percase", path)
        tag = m.group(1) if m else "unknown"
        rows = out.setdefault(tag, {})
        with open_maybe_gz(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows[(r["case_key"], r["arm"])] = r
    return out


def leaked(r):
    return ((r.get("ctx_leak_value") or 0) > 0) or ((r.get("out_leak_value") or 0) > 0)


def ci(diffs, reps=4000, seed=7):
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(reps))
    return boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--baseline", default="results/percase/quadfull_s*_percase.gz")
    ap.add_argument("--control-f1", type=float, default=None,
                    help="expected F1 of the full arm, for a regression check. Leave unset for a "
                         "different reader: the uncompressed baseline is model-specific.")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    models = load(args.results)
    if not models:
        raise SystemExit("no per-case rows matched %r" % args.results)

    base = {}
    for path in sorted(glob.glob(args.baseline)):
        with open_maybe_gz(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                base[(r["case_key"], r["arm"])] = r

    out = {}
    print("%-12s %-6s %6s %8s %8s %8s %9s"
          % ("model", "arm", "n", "F1", "leak%", "tokens", "vs full"))
    for tag in sorted(models):
        rows = models[tag]
        arms = sorted({a for (_, a) in rows})
        per = {}
        for arm in arms:
            recs = [r for (k, a), r in rows.items() if a == arm]
            toks = [r["ctx_tokens"] for r in recs if r.get("ctx_tokens")]
            per[arm] = {
                "n": len(recs),
                "f1": statistics.mean(r["f1"] for r in recs),
                "leak": 100.0 * sum(1 for r in recs if leaked(r)) / len(recs),
                "tokens": statistics.mean(toks) if toks else None,
            }
        # paired DARK - full
        diff = None
        if "full" in per and "quad4" in per:
            keys = sorted(k for (k, a) in rows if a == "quad4" and (k, "full") in rows)
            diffs = [rows[(k, "quad4")]["f1"] - rows[(k, "full")]["f1"] for k in keys]
            lo, hi = ci(diffs)
            diff = {"mean": statistics.mean(diffs), "lo": lo, "hi": hi, "n": len(diffs)}
        out[tag] = {"arms": per, "dark_minus_full": diff}

        ctrl = per.get("full", {}).get("f1")
        flag = ""
        if args.control_f1 is not None and ctrl is not None:
            if abs(ctrl - args.control_f1) > 0.02:
                flag = "  <-- control off by %.3f" % (ctrl - args.control_f1)
        for arm in arms:
            p = per[arm]
            print("%-12s %-6s %6d %8.4f %8.2f %8s %9s"
                  % (tag, arm, p["n"], p["f1"], p["leak"],
                     "%.0f" % p["tokens"] if p["tokens"] else "-",
                     ("%+.4f" % diff["mean"]) if (diff and arm == "quad4") else ""))
        if diff:
            print("%-12s DARK-full %+.4f  [%+.4f,%+.4f]  n=%d%s"
                  % ("", diff["mean"], diff["lo"], diff["hi"], diff["n"], flag))
        print()

    if args.json_out:
        with io.open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        print("wrote", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
