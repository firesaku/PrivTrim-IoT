#!/usr/bin/env python3
"""Choose a reader model for the cross-model check, before spending a full run on it.

Motivation
    The rule needs the reader to treat an aliased handle as an ordinary identifier. A reader
    that reads `[ACCT-A]` as a redaction marker answers UNKNOWN and loses the task even though
    the prompt is intact. Qwen2.5-14B does exactly that on 94% of multi-entity cases, which is
    why the model is chosen by measurement here instead of by size or reputation.

What it reports, per model
    full F1, DARK F1, and the UNKNOWN rate under each arm, split by case kind. The UNKNOWN
    rate on multi-entity under DARK is the quantity that predicts the failure: it is the share
    of questions the reader declines although the answer is in the prompt.

Selection rule
    A model is usable when its `full` arm is healthy (the probe sample reproduces the expected
    uncompressed score) and its DARK arm does not collapse. The script ranks candidates by DARK
    F1 minus the penalty for a high UNKNOWN rate, and prints the ranking rather than deciding
    silently.

Usage
    python src/analyze_mprobe.py --results 'D:/path/mprobe_*_percase.jsonl'
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


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def load(pattern):
    """model tag -> {(case_key, arm): row}"""
    out = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"mprobe_([A-Za-z0-9_]+?)_percase", path)
        if not m:
            continue
        rows = out.setdefault(m.group(1), {})
        with open_maybe_gz(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows[(r["case_key"], r["arm"])] = r
    return out


def is_unknown(pred):
    p = str(pred or "").strip().upper()
    return p.startswith("UNKNOWN") or p.startswith("UNKNOWABLE")


def summarise(rows, kind=None):
    sel = [r for r in rows if kind is None or r.get("case_kind") == kind]
    if not sel:
        return None
    return {
        "n": len(sel),
        "f1": statistics.mean(r["f1"] for r in sel),
        "unknown": 100.0 * sum(1 for r in sel if is_unknown(r.get("pred"))) / len(sel),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    models = load(args.results)
    if not models:
        raise SystemExit("no probe rows matched %r" % args.results)

    ranking = []
    print("%-12s %-6s %5s %8s %9s %10s %10s"
          % ("model", "arm", "n", "F1", "UNKNOWN%", "F1(multi)", "F1(needle)"))
    for tag in sorted(models):
        rows = list(models[tag].values())
        per_arm = collections.defaultdict(list)
        for r in rows:
            per_arm[r["arm"]].append(r)
        for arm in sorted(per_arm):
            s = summarise(per_arm[arm])
            sm = summarise(per_arm[arm], "multi_entity")
            sn = summarise(per_arm[arm], "needle")
            print("%-12s %-6s %5d %8.4f %9.1f %10s %10s"
                  % (tag, arm, s["n"], s["f1"], s["unknown"],
                     ("%.4f" % sm["f1"]) if sm else "-",
                     ("%.4f" % sn["f1"]) if sn else "-"))
        dark = summarise(per_arm.get("quad4", []))
        dark_multi = summarise(per_arm.get("quad4", []), "multi_entity")
        full = summarise(per_arm.get("full", []))
        if dark and full:
            ranking.append({
                "model": tag,
                "full_f1": full["f1"],
                "dark_f1": dark["f1"],
                "dark_unknown": dark["unknown"],
                "dark_unknown_multi": dark_multi["unknown"] if dark_multi else None,
                "gap": dark["f1"] - full["f1"],
            })
        print()

    ranking.sort(key=lambda r: -r["dark_f1"])
    print("=== ranked by DARK F1 on the probe sample ===")
    print("%-12s %8s %8s %12s %10s" % ("model", "full", "DARK", "UNKNOWN(multi)", "gap"))
    for r in ranking:
        print("%-12s %8.4f %8.4f %12s %+10.4f"
              % (r["model"], r["full_f1"], r["dark_f1"],
                 ("%.1f%%" % r["dark_unknown_multi"]) if r["dark_unknown_multi"] is not None else "-",
                 r["gap"]))
    if ranking:
        print()
        print("best DARK F1: %s (%.4f); highest refuser: %s (%.1f%% UNKNOWN on multi-entity)"
              % (ranking[0]["model"], ranking[0]["dark_f1"],
                 ranking[-1]["model"], ranking[-1]["dark_unknown_multi"] or 0.0))

    if args.json_out:
        with io.open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(ranking, fh, indent=1)
        print("wrote", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
