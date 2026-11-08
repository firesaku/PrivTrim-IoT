#!/usr/bin/env python3
"""The two design choices as an interaction plot, not as a bar chart.

A grouped bar chart of the 2x2 would restate Table VIII in a second medium. An interaction plot
does work the table cannot: the vertical gap between the two lines is the operation effect, the
slope of each line is the granularity effect, and the two lines being nearly parallel is itself
the evidence that the two choices are close to additive. A reader sees the two-to-three ratio
before reading a single number.

Reads results/diag/factorial.json, which is produced from the two runs by the same code that fills
Table VIII, so the figure and the table cannot drift apart.

Usage
    python src/analysis_quad4/make_fig_interaction.py --out paper/figures/fig_interaction.pdf
"""
from __future__ import annotations

import argparse
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#4A1412"       # alias (ours)
RED = "#8E2A22"       # delete
G_D, G_M, G_L = "#555555", "#9A9A9A", "#D4D4D2"
W, H = 3.45, 2.55

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "text.color": "#000000", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--factorial", default="results/diag/factorial.json")
    ap.add_argument("--out", default="paper/figures/fig_interaction.pdf")
    args = ap.parse_args(argv)

    with io.open(args.factorial, encoding="utf-8") as fh:
        d = json.load(fh)

    cells = d["cells"]
    eff = d["main_effects"]
    inter = d["interaction"]

    xs = [0, 1]
    labels = ["window\n(anchor $\\pm$ 2 lines)", "block\n(whole record)"]
    alias = [cells["window"]["alias"], cells["block"]["alias"]]
    dele = [cells["window"]["delete"], cells["block"]["delete"]]

    fig, ax = plt.subplots(figsize=(W, H))

    ax.plot(xs, alias, color=INK, marker="o", ms=7, lw=2.0, label="alias (ours)",
            markeredgecolor="white", markeredgewidth=0.8, zorder=5)
    ax.plot(xs, dele, color=RED, marker="s", ms=6, lw=2.0, label="delete",
            markeredgecolor="white", markeredgewidth=0.8, zorder=5)

    # The operation effect: the vertical gap between the lines at each granularity.
    for x, a, b, e in ((0, alias[0], dele[0], eff["alias_at_window"]),
                       (1, alias[1], dele[1], eff["alias_at_block"])):
        ax.annotate("", xy=(x, a - 0.008), xytext=(x, b + 0.008),
                    arrowprops=dict(arrowstyle="<->", lw=1.0, color=G_M,
                                    shrinkA=0, shrinkB=0))
        ax.text(x + 0.055, (a + b) / 2, "%+.3f" % e["mean"], fontsize=10, color=G_D,
                va="center", ha="left")

    # The granularity effect: the rise of each line, labelled at its midpoint rather than drawn
    # as a sweeping arrow, which reads as a third series.
    mid_a = (alias[0] + alias[1]) / 2
    mid_d = (dele[0] + dele[1]) / 2
    ax.text(0.5, mid_a + 0.020, "%+.3f" % eff["block_at_alias"]["mean"], fontsize=10,
            color=G_D, ha="center", va="bottom")
    ax.text(0.5, mid_d - 0.020, "%+.3f" % eff["block_at_delete"]["mean"], fontsize=10,
            color=G_D, ha="center", va="top")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.30, 1.42)
    ax.set_ylim(0.20, 0.56)
    ax.set_ylabel("F1")
    ax.grid(axis="y", color=G_L, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.3)

    ax.text(1.40, 0.212, "interaction $%+.4f$ $[%+.4f,%+.4f]$"
            % (inter["mean"], inter["lo"], inter["hi"]),
            fontsize=10, color=G_D, ha="right", va="bottom")

    fig.tight_layout(pad=0.25)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)
    for g in ("window", "block"):
        print("  %-7s alias %.4f  delete %.4f" % (g, cells[g]["alias"], cells[g]["delete"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
