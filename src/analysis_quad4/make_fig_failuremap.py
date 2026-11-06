#!/usr/bin/env python3
"""Structural failure map: two factors, three regions, measured on real cases.

Two structural properties degrade DARK independently, so the figure has two panels rather than a
single surface:

  (a) anchor-answer distance, split by case kind. Distance is what the referential claim depends
      on, and it is the factor that produces a hard cliff rather than a slope.
  (b) anchor block size at a fixed distance. Keeping a large block costs budget, so the same
      distance performs worse inside a larger block.

A first version drew one 2-D grid over both factors at once. It was rejected: the rows were
confounded with case kind, half the cells were empty, and one cell of 19 needle cases sat above
both of its neighbours. Splitting by kind removes the confound and the anomaly disappears.

Regions are a reporting choice, stated in the figure: usable (F1 >= 0.45, about the arm's overall
0.494), degrading (0.15-0.45), unusable (< 0.15). Cells with fewer than 10 cases are marked sparse
and are not interpreted.

Usage
    python src/analysis_quad4/make_fig_failuremap.py --out paper/figures/fig_failure.pdf
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import io
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INK = "#141414"
GREEN, AMBER, RED = "#007A5E", "#D69A35", "#8E2A22"
G_D, G_L = "#555555", "#D4D4D2"
W, H = 7.10, 2.55

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "text.color": "#000000", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})

BINS = ["0", "1-4", "5-10", "11-30", "31+"]
ROWS = [("needle", "single-entity (needle)"), ("multi_entity", "multi-entity")]


def db(d):
    if d == 0:
        return "0"
    if d <= 4:
        return "1-4"
    if d <= 10:
        return "5-10"
    if d <= 30:
        return "11-30"
    return "31+"


def load(pattern):
    out = {}
    for path in sorted(glob.glob(pattern)):
        op = gzip.open if path.endswith(".gz") else io.open
        with op(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r["arm"] == "quad4":
                    out[r["case_key"]] = r
    return out


def region(m):
    if m >= 0.45:
        return GREEN
    if m >= 0.15:
        return AMBER
    return RED


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", default="results/diag/structure.json")
    ap.add_argument("--percase", default="results/percase/quadfull_s*_percase.gz")
    ap.add_argument("--out", default="paper/figures/fig_failure.pdf")
    ap.add_argument("--json-out", default="results/diag/failuremap.json")
    args = ap.parse_args(argv)

    sj = json.load(io.open(args.structure, encoding="utf-8"))
    cases = {c["key"]: c for c in sj["cases"]}
    f1 = load(args.percase)

    cells = collections.defaultdict(list)
    for k, c in cases.items():
        if k not in f1:
            continue
        cells[(c["case_kind"], db(c.get("line_dist") or 0))].append(f1[k]["f1"])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, H),
                                 gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.34})

    for i, (kind, _) in enumerate(ROWS):
        for j, b in enumerate(BINS):
            v = cells.get((kind, b))
            x, y = j, i
            if not v:
                ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor="white",
                                       edgecolor=G_L, lw=0.7, ls=":"))
                ax.text(x, y, "not in\nbenchmark", fontsize=9, color=G_L,
                        ha="center", va="center", linespacing=1.1)
                continue
            n, m = len(v), statistics.mean(v)
            if n < 10:
                ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor="#F2F3F5",
                                       edgecolor=G_L, lw=0.7, hatch="///"))
                ax.text(x, y, "%.3f\nn=%d sparse" % (m, n), fontsize=10, color=G_D,
                        ha="center", va="center", linespacing=1.15)
                continue
            ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=region(m),
                                   edgecolor="white", lw=1.5))
            ax.text(x, y, "%.3f\nn=%d" % (m, n), fontsize=10, color="#FFFFFF",
                    ha="center", va="center", linespacing=1.15, fontweight="bold")

    ax.set_xlim(-0.5, len(BINS) - 0.5)
    ax.set_ylim(len(ROWS) - 0.5, -0.5)
    ax.set_xticks(range(len(BINS)))
    ax.set_xticklabels(BINS)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[1] for r in ROWS])
    ax.set_xlabel("anchor$-$answer line distance (lines)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a) distance, by case kind", fontsize=10, loc="left", fontweight="bold")

    # (b) block size at a fixed distance
    gb = collections.defaultdict(list)
    for k, c in cases.items():
        if k not in f1:
            continue
        d = c.get("line_dist") or 0
        if not (1 <= d <= 4):
            continue
        bl = c.get("anchor_block_lines") or 0
        b = "6-10" if 6 <= bl <= 10 else ("11-20" if 11 <= bl <= 20 else
                                          ("21-40" if 21 <= bl <= 40 else "41+"))
        gb[b].append(f1[k]["f1"])
    labels, means, ns = [], [], []
    for b in ("6-10", "11-20", "41+"):
        v = gb.get(b)
        if v:
            labels.append(b)
            means.append(statistics.mean(v))
            ns.append(len(v))
    ys = list(range(len(labels)))[::-1]
    for y, m, n in zip(ys, means, ns):
        col = region(m) if n >= 10 else "#F2F3F5"
        bx.barh(y, m, height=0.55, color=col, edgecolor="white", lw=1.0)
        bx.text(m + 0.015, y, "%.3f  (n=%d)" % (m, n), fontsize=10, va="center",
                color=INK, fontweight="bold" if n >= 10 else "normal")
    bx.set_yticks(ys)
    bx.set_yticklabels(labels)
    bx.set_xlim(0, 0.95)
    bx.set_xlabel("F1 at distance 1$-$4")
    bx.tick_params(length=0)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    bx.set_title("(b) block size, distance held fixed", fontsize=10, loc="left",
                 fontweight="bold")
    bx.grid(axis="x", color=G_L, lw=0.5, zorder=0)

    handles = [Rectangle((0, 0), 1, 1, facecolor=GREEN, edgecolor="white"),
               Rectangle((0, 0), 1, 1, facecolor=AMBER, edgecolor="white"),
               Rectangle((0, 0), 1, 1, facecolor=RED, edgecolor="white")]
    ax.legend(handles, ["usable (F1 $\\geq$ 0.45)", "degrading (0.15$-$0.45)",
                        "unusable (F1 $<$ 0.15)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, frameon=False,
              handlelength=1.4, columnspacing=1.5)

    fig.subplots_adjust(left=0.155, right=0.995, top=0.90, bottom=0.30)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)

    table = []
    for (kind, b), v in sorted(cells.items()):
        table.append({"kind": kind, "dist": b, "n": len(v), "f1": statistics.mean(v)})
    for b, m, n in zip(labels, means, ns):
        table.append({"kind": "fixed-dist-1-4", "block": b, "n": n, "f1": m})
    if args.json_out:
        with io.open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(table, fh, indent=1)
        print("wrote", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())