#!/usr/bin/env python3
"""Build results/diag/factorial.json, the 2x2 behind Table VIII and Figure 14.

The two factors are protection granularity (a window around the anchor, or the whole record) and
the operation applied inside it (aliasing, or deletion). They are crossed on the same 2,728 cases,
so every cell is a mean over the same case population and every effect is a paired difference.

Inputs
    results/percase/window2_s*_percase.gz   four arms from one batch (job 1557997)
    results/percase/quadfull_s*_percase.gz  the block+delete arm, from the main run

Both batches re-run `full` and `quad4` as controls; this script does not check them, because
`validate_fallback.py` and the caption of Table VIII already report that they reproduce.

Bootstrap settings match make_fig_frontier.py (4,000 resamples, seed 7) so that an interval quoted
in the text, drawn in a figure and written here is the same interval.

Usage
    python src/analysis_quad4/make_factorial.py
    python src/analysis_quad4/make_factorial.py --check   # compare against the committed file
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import io
import json
import random
import statistics
import sys


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def load(patterns):
    """arm -> {case_key: row}, first occurrence per case wins."""
    acc = collections.defaultdict(dict)
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open_maybe_gz(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    acc[r["arm"]].setdefault(r["case_key"], r)
    return acc


def ci(values, reps=4000, seed=7):
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(reps))
    return boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]


def paired(acc, a, b):
    keys = sorted(k for k in acc[a] if k in acc[b])
    if not keys:
        raise SystemExit("no shared cases between %s and %s" % (a, b))
    d = [acc[a][k]["f1"] - acc[b][k]["f1"] for k in keys]
    lo, hi = ci(d)
    return {"mean": statistics.mean(d), "lo": lo, "hi": hi, "n": len(keys)}


def build():
    acc = load(["results/percase/window2_s*_percase.gz",
                "results/percase/window2_s*_percase.jsonl",
                "results/percase/quadfull_s*_percase.gz"])

    need = ("quad4", "quad4_win", "quad4_del", "quad4_win_del")
    missing = [a for a in need if a not in acc]
    if missing:
        raise SystemExit("missing arms in the per-case records: %s" % missing)

    def mean(arm):
        return statistics.mean(r["f1"] for r in acc[arm].values())

    out = {
        "cells": {
            "window": {"alias": mean("quad4_win"), "delete": mean("quad4_win_del")},
            "block": {"alias": mean("quad4"), "delete": mean("quad4_del")},
        },
        "main_effects": {
            "alias_at_window": paired(acc, "quad4_win", "quad4_win_del"),
            "alias_at_block": paired(acc, "quad4", "quad4_del"),
            "block_at_alias": paired(acc, "quad4", "quad4_win"),
            "block_at_delete": paired(acc, "quad4_del", "quad4_win_del"),
        },
        "arm_totals": {a: {"n": len(v), "f1": statistics.mean(r["f1"] for r in v.values())}
                       for a, v in sorted(acc.items())},
    }

    # Interaction: the alias effect at block minus the alias effect at window, paired per case.
    keys = [k for k in acc["quad4"]
            if all(k in acc[a] for a in ("quad4", "quad4_del", "quad4_win", "quad4_win_del"))]
    inter = [(acc["quad4"][k]["f1"] - acc["quad4_del"][k]["f1"])
             - (acc["quad4_win"][k]["f1"] - acc["quad4_win_del"][k]["f1"]) for k in keys]
    lo, hi = ci(inter)
    out["interaction"] = {"mean": statistics.mean(inter), "lo": lo, "hi": hi, "n": len(inter)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/diag/factorial.json")
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed file instead of overwriting it")
    args = ap.parse_args(argv)

    built = build()

    if args.check:
        with io.open(args.out, encoding="utf-8") as fh:
            committed = json.load(fh)
        bad = []
        for group in ("cells", "main_effects", "interaction"):
            for k, v in built[group].items():
                ref = committed.get(group, {}).get(k)
                if ref is None:
                    bad.append((group, k, "missing"))
                    continue
                if isinstance(v, dict):
                    for f in ("mean", "lo", "hi"):
                        if f in v and abs(v[f] - ref[f]) > 1e-9:
                            bad.append((group, k + "." + f, "%r != %r" % (v[f], ref[f])))
                elif abs(v - ref) > 1e-9:
                    bad.append((group, k, "%r != %r" % (v, ref)))
        if bad:
            print("MISMATCH")
            for b in bad:
                print("  %s %s: %s" % b)
            return 1
        print("committed factorial.json reproduced exactly")
        return 0

    with io.open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(built, fh, indent=1)
    print("wrote", args.out)
    for k, v in built["main_effects"].items():
        print("  %-16s %+.4f [%+.4f,%+.4f]  n=%d" % (k, v["mean"], v["lo"], v["hi"], v["n"]))
    it = built["interaction"]
    print("  %-16s %+.4f [%+.4f,%+.4f]  n=%d"
          % ("interaction", it["mean"], it["lo"], it["hi"], it["n"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
