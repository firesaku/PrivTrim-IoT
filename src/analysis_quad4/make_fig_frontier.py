#!/usr/bin/env python3
"""Leakage-utility frontier with intervals, and a forest plot of the paired differences.

Two panels, because the two questions are different:

  (a) where does each arm sit, and how much of that position is noise? Both axes carry an
      interval. F1 gets a percentile bootstrap over cases. Leakage is a proportion and several
      arms sit at exactly 0% or 100%, where the bootstrap is degenerate, so it gets a Wilson
      score interval instead; an arm that leaks nothing over 2,728 cases then has an upper bound
      near 0.1%, which is the honest way to draw "zero".
  (b) which differences are actually resolved? Each row is DARK minus one comparator on the same
      cases, with a paired bootstrap interval. This separates "we are ahead" from "this
      experiment did not distinguish us".

Only arms evaluated on the full 2,728-case extension are used.

Usage
    python src/analysis_quad4/make_fig_frontier.py --out paper/figures/fig_frontier.pdf
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import io
import json
import math
import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, RED = "#141414", "#8E2A22"
G_D, G_M, G_L, PALE = "#555555", "#9A9A9A", "#D4D4D2", "#EFEFEF"
W, H = 7.10, 3.20

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "text.color": "#000000", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})

LABELS = {
    "full": ("uncompressed", "reference"),
    "quad4": ("DARK (ours)", "ours"),
    "ext_quito_x": ("Quito-X", "published"),
    "ext_cpc": ("CPC", "published"),
    "ext_provence": ("Provence", "published"),
    "ext_longllmlingua": ("LongLLMLingua", "published"),
    "ext_llmlingua2": ("LLMLingua-2", "published"),
    "ext_selective_context": ("Selective Context", "published"),
    "ext_mask_llmlingua2": ("mask + LLMLingua-2", "sanitised"),
    "alias_budget_link": ("global alias", "ours-ablation"),
    "budget_cluster_link": ("anchor line", "ours-ablation"),
    "quad4_del": ("block + delete", "ours-ablation"),
}
COLOUR = {"reference": G_M, "published": G_D, "sanitised": G_M,
          "ours-ablation": RED, "ours": INK}
MARKER = {"reference": "s", "published": "o", "sanitised": "^",
          "ours-ablation": "D", "ours": "*"}

OFFSET = {
    "uncompressed": (-2.6, 0.006, "right"),
    "Quito-X": (2.6, 0.010, "left"),
    "CPC": (-2.6, 0.018, "right"),
    "LongLLMLingua": (2.0, 0.008, "left"),
    "Provence": (-2.6, 0.006, "right"),
    "LLMLingua-2": (2.0, 0.012, "left"),
    "Selective Context": (2.0, -0.034, "left"),
    "anchor line": (1.6, 0.000, "left"),
    "DARK (ours)": (1.6, 0.012, "left"),
    "global alias": (1.6, 0.014, "left"),
    "block + delete": (1.6, -0.030, "left"),
    "mask + LLMLingua-2": (1.6, 0.000, "left"),
}


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def collect(patterns, target_n):
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
    return {a: v for a, v in acc.items() if len(v) == target_n}


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion; well behaved at 0 and 1."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h))


def boot_ci(vals, reps=4000, seed=7):
    rng = random.Random(seed)
    bs = sorted(statistics.mean(rng.choices(vals, k=len(vals))) for _ in range(reps))
    return bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--percase", nargs="+",
                    default=["results/percase/quadfull_s*_percase.gz",
                             "results/percase/extevfull_s*_percase.gz",
                             "results/percase/extnew2_s*_percase.gz"])
    ap.add_argument("--target-n", type=int, default=2728)
    ap.add_argument("--out", default="paper/figures/fig_frontier.pdf")
    ap.add_argument("--json-out", default="results/diag/frontier.json")
    args = ap.parse_args(argv)

    data = collect(args.percase, args.target_n)
    rows = []
    for arm, by_key in data.items():
        if arm not in LABELS:
            continue
        recs = list(by_key.values())
        k = sum(1 for r in recs
                if (r.get("ctx_leak_value") or 0) > 0 or (r.get("out_leak_value") or 0) > 0)
        f1 = [r["f1"] for r in recs]
        lo, hi = boot_ci(f1)
        llo, lhi = wilson(k, len(recs))
        rows.append({"arm": arm, "name": LABELS[arm][0], "group": LABELS[arm][1],
                     "n": len(recs), "f1": statistics.mean(f1), "f1_lo": lo, "f1_hi": hi,
                     "leak": 100.0 * k / len(recs), "leak_lo": llo, "leak_hi": lhi})
    rows.sort(key=lambda r: (r["leak"], -r["f1"]))

    diffs = []
    dark = data.get("quad4")
    if dark:
        for arm, by_key in data.items():
            if arm == "quad4" or arm not in LABELS:
                continue
            keys = sorted(k for k in dark if k in by_key)
            if not keys:
                continue
            d = [dark[k]["f1"] - by_key[k]["f1"] for k in keys]
            lo, hi = boot_ci(d)
            diffs.append({"arm": arm, "name": LABELS[arm][0], "group": LABELS[arm][1],
                          "n": len(keys), "mean": statistics.mean(d), "lo": lo, "hi": hi})
        diffs.sort(key=lambda d: d["mean"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, H),
                                   gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.30})

    ax1.axvspan(-2, 5, color=PALE, zorder=0)
    ax1.text(1.0, 0.02, "usable", fontsize=10, color=G_D, ha="center", va="bottom")
    for r in rows:
        ours = r["group"] == "ours"
        ax1.errorbar(r["leak"], r["f1"],
                     xerr=[[max(0.0, r["leak"] - r["leak_lo"])],
                           [max(0.0, r["leak_hi"] - r["leak"])]],
                     yerr=[[max(0.0, r["f1"] - r["f1_lo"])],
                           [max(0.0, r["f1_hi"] - r["f1"])]],
                     fmt="none", ecolor=COLOUR[r["group"]], elinewidth=0.8,
                     capsize=1.8, zorder=4)
        ax1.scatter(r["leak"], r["f1"], s=190 if ours else 70, marker=MARKER[r["group"]],
                    facecolor=COLOUR[r["group"]], edgecolor="white", lw=0.8, zorder=5)
        dx, dy, ha = OFFSET.get(r["name"], (1.6, 0.008, "left"))
        ax1.text(r["leak"] + dx, r["f1"] + dy, r["name"], fontsize=10, ha=ha, va="bottom",
                 color=COLOUR[r["group"]], zorder=6,
                 fontweight="bold" if ours else "normal")
    ax1.set_xlabel("protected values recoverable (%)  DOWN")
    ax1.set_ylabel("F1  UP")
    ax1.set_xlim(-3, 133)
    ax1.set_ylim(0, 0.68)
    ax1.grid(color=G_L, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    ax1.set_title("(a) the frontier, 95% intervals", fontsize=10, loc="left",
                  fontweight="bold")

    ax2.axvline(0, color=G_D, lw=0.9, ls="--", zorder=1)
    ys = list(range(len(diffs)))
    for y, d in zip(ys, diffs):
        col = COLOUR[d["group"]]
        sig = (d["lo"] > 0) or (d["hi"] < 0)
        ax2.plot([d["lo"], d["hi"]], [y, y], color=col, lw=1.2, zorder=4)
        ax2.scatter(d["mean"], y, s=70 if sig else 46, marker="o",
                    facecolor=col if sig else "white", edgecolor=col, lw=1.0, zorder=5)
        ax2.text(-0.58, y, "%+.3f" % d["mean"], fontsize=10, va="center", ha="left",
                 color=col if sig else G_D,
                 fontweight="bold" if sig else "normal")
    ax2.set_yticks(ys)
    ax2.set_yticklabels([d["name"] for d in diffs])
    ax2.yaxis.tick_right()
    ax2.set_xlabel("F1, DARK minus the arm")
    ax2.set_xlim(-0.62, 0.95)
    ax2.grid(axis="x", color=G_L, lw=0.5, zorder=0)
    for s in ("top", "left"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(axis="y", length=0, pad=3)
    ax2.set_title("(b) what is resolved", fontsize=10, loc="left", fontweight="bold")


    fig.subplots_adjust(left=0.075, right=0.855, top=0.88, bottom=0.16)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)

    if args.json_out:
        with io.open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"arms": rows, "paired_vs_dark": diffs}, fh, indent=1)
        print("wrote", args.json_out)

    print()
    print("%-22s %6s %8s %16s %9s %16s" % ("arm", "n", "F1", "F1 95%", "leak%", "leak 95%"))
    for r in rows:
        print("%-22s %6d %8.4f  [%.3f,%.3f] %8.2f  [%.2f,%.2f]"
              % (r["name"], r["n"], r["f1"], r["f1_lo"], r["f1_hi"], r["leak"],
                 r["leak_lo"], r["leak_hi"]))
    print()
    print("paired, DARK minus arm:")
    for d in diffs:
        sig = "sig" if (d["lo"] > 0 or d["hi"] < 0) else "n.s."
        print("  %-22s %+.4f  [%+.4f,%+.4f]  %s"
              % (d["name"], d["mean"], d["lo"], d["hi"], sig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())