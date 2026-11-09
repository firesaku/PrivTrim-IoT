#!/usr/bin/env python3
"""The price of referential integrity, measured against budget.

The paper's claim is that the operation, not the region, is what makes the task possible. A budget
sweep is the one place where that claim can be priced, because it holds the corpus and the reader
fixed and varies only how much of the prompt survives.

Two arms bracket the answer. `budget_cluster` prunes by redundancy and protects nothing: it is
leak-free by accident, because it drops the record that carries the value. `budget_cluster_link`
adds exactly one thing -- it keeps the anchor line verbatim -- and the difference between those two
arms at every budget is what the anchor is worth when you are allowed to forward it.

So the figure has two panels on purpose. F1 alone would show `budget_cluster_link` crossing above
DARK below about 5% and invite the reading that protection is optional at tight budgets. The leak
panel is what that crossing is paid with: the same arm forwards a protected value in 21% of cases at
5%, 36% at 10% and 39% at 15%. Put together, the pair states the finding the sweep actually
produced: protection is cheap when there is budget to spend and expensive when there is not, and no
arm gets both properties at every budget except this one.

Usage
    python src/analysis_quad4/make_fig_budget_all.py --out paper/figures/fig_budget_all.pdf
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import io
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INK = "#4A1412"
RED = "#8E2A22"
G_D = "#4A4A4A"
G_M = "#9A9A9A"
G_L = "#D4D4D2"
W, H = 7.05, 2.70

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.5, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "text.color": "#000000", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})

# label -> (percase glob, arm, ratio, colour, linestyle, marker)
SERIES = [
    ("DARK",                    ["budgetdark_s*_percase.jsonl", "quadfull_s*_percase.gz"],
     "quad4",                INK, "-",  "o"),
    ("redundancy, anchor kept", ["xbud_s*_percase.gz", "quadfull_s*_percase.gz"],
     "budget_cluster_link",  RED, "--", "s"),
    ("redundancy only",         ["xbud_s*_percase.gz", "quadfull_s*_percase.gz"],
     "budget_cluster",       G_M, ":",  "^"),
    # published compressors at the low budgets. Loaded from the budgetext job when it exists, and
    # from the archived 0.5 run so the curve closes: quito_x and cpc are the two strongest
    # published baselines, and whether they also cross DARK is the question that job exists to
    # answer. Absent files are skipped rather than fatal, so this figure builds before and after.
    ("Quito-X",                 ["budgetext_*_percase.jsonl", "extevfull_s*_percase.gz",
                                 "extnew2_s*_percase.gz"],
     "ext_quito_x",          "#B5645C", "-.", "D"),
    ("CPC",                     ["budgetext_*_percase.jsonl", "extevfull_s*_percase.gz",
                                 "extnew2_s*_percase.gz"],
     "ext_cpc",              "#C98F86", "-.", "v"),
]
RATIOS = [0.05, 0.10, 0.15, 0.50]


def opener(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def load_index(patterns):
    """(arm, ratio) -> {case_key: record}; first pattern wins so the 0.5 rows come from quadfull."""
    idx = collections.defaultdict(dict)
    for pattern in patterns:
        for path in sorted(glob.glob(str(ROOT / "results" / "percase" / pattern))):
            with opener(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    idx[(r["arm"], r.get("ratio"))].setdefault(r["case_key"], r)
    return idx


def leaks(r):
    return (r.get("ctx_leak_value") or 0) > 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="paper/figures/fig_budget_all.pdf")
    ap.add_argument("--json-out", default="results/diag/budget_all.json")
    args = ap.parse_args(argv)

    table = {}
    for label, patterns, arm, *_ in SERIES:
        idx = load_index(patterns)
        pts = []
        for ratio in RATIOS:
            rows = idx.get((arm, ratio))
            if not rows:
                print("  missing %s @ r=%.2f" % (label, ratio))
                continue
            vals = list(rows.values())
            pts.append({"ratio": ratio, "n": len(vals),
                        "f1": statistics.mean(r["f1"] for r in vals),
                        "leak": 100.0 * sum(1 for r in vals if leaks(r)) / len(vals),
                        "tokens": statistics.mean(r.get("ctx_tokens") or 0 for r in vals)})
        table[label] = pts

    ctrl = table.get("DARK", [])
    if not ctrl:
        print("no DARK rows found; nothing to plot")
        return 1

    # the full-context reference, for scale
    full_idx = load_index(["quadfull_s*_percase.gz"])
    full_key = ("full", None)
    full_f1 = (statistics.mean(r["f1"] for r in full_idx[full_key].values())
               if full_key in full_idx else None)

    if args.json_out:
        payload = {"series": table, "full_f1": full_f1, "full_tokens": 8482}
        with io.open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=1)
        print("wrote", args.json_out)

    fig, axes = plt.subplots(1, 2, figsize=(W, H))

    for label, patterns, arm, colour, ls, mk in SERIES:
        pts = table.get(label, [])
        if not pts:
            continue
        xs = [p["ratio"] for p in pts]
        axes[0].plot(xs, [p["f1"] for p in pts], color=colour, linestyle=ls, marker=mk,
                     markersize=4.5, linewidth=1.7 if arm == "quad4" else 1.3,
                     label=label, zorder=3 if arm == "quad4" else 2)
        axes[1].plot(xs, [p["leak"] for p in pts], color=colour, linestyle=ls, marker=mk,
                     markersize=4.5, linewidth=1.7 if arm == "quad4" else 1.3,
                     label=label, zorder=3 if arm == "quad4" else 2)

    if full_f1 is not None:
        axes[0].axhline(full_f1, color=G_L, linewidth=1.2, linestyle="-", zorder=1)
        axes[0].annotate("uncompressed  %.3f" % full_f1, xy=(0.05, full_f1),
                         xytext=(0, 3), textcoords="offset points",
                         fontsize=8, color=G_D)

    for ax, title, ylab in ((axes[0], "(a) task", "F1 $\\uparrow$"),
                            (axes[1], "(b) what it costs", "cases leaking (%) $\\downarrow$")):
        ax.set_xlabel("budget ratio $r$")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", pad=6)
        ax.set_xscale("log")
        ax.set_xticks(RATIOS)
        ax.set_xticklabels(["%.2f" % r for r in RATIOS])
        ax.minorticks_off()
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(G_L)
        ax.spines["bottom"].set_color(G_L)
        ax.grid(axis="y", color="#EFEFEF", linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].set_ylim(-0.03, max(0.72, (full_f1 or 0.6) + 0.06))
    axes[1].set_ylim(-4, 104)
    axes[0].legend(frameon=False, loc="lower right")

    fig.tight_layout(pad=0.3, w_pad=1.4)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)

    print()
    print("%-24s %6s %8s %8s %8s" % ("arm", "r", "n", "F1", "leak%"))
    for label, pts in table.items():
        for p in pts:
            print("%-24s %6.2f %8d %8.4f %8.1f" % (label, p["ratio"], p["n"], p["f1"], p["leak"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
