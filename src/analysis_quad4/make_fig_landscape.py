#!/usr/bin/env python3
"""Structural sensitivity landscape: which factor moves each arm, and by how much.

The failure map (Fig. 12) showed that structure decides DARK's score. This figure asks the next
question: does structure decide it *for the other arms too*, and in the same way? Each panel holds
the case population fixed and varies one factor, so the four lines are directly comparable.

  (a) single-entity cases, F1 against anchor-answer distance
  (b) multi-entity cases, same axis
  (c) block size with distance held at 1-4 lines

A leakage panel was proposed alongside this one and is not drawn: leakage in these arms is a
property of the operation, not of the case. Across all 2,728 cases it is 0.00% for DARK, 0.00% for
global alias, 0.00% for block+delete and 55.39% for the anchor-line arm, with no variation by
distance or block size, so a leakage heatmap would be four constant rows. The rates are in the
legend instead.

Colour follows the paper: dark maroon for DARK, red for the arm that leaks, greys for the
leak-free ablations.

Usage
    python src/analysis_quad4/make_fig_landscape.py --out paper/figures/fig_landscape.pdf
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

INK = "#4A1412"
RED = "#8E2A22"
G_D, G_M, G_L = "#555555", "#9A9A9A", "#D4D4D2"
W, H = 7.10, 2.45

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

BINS = ["0", "1-4", "5-10", "11-30", "31+"]
BLK = ["6-10", "11-20", "41+"]
ARMS = [
    ("quad4", "DARK", INK, "o", "-", 2.0, 190),
    ("budget_cluster_link", "anchor line (leaks 55.4%)", RED, "s", "-", 1.3, 60),
    ("alias_budget_link", "global alias", G_M, "^", "-", 1.3, 60),
    ("quad4_del", "block + delete", G_D, "D", "-", 1.3, 60),
]


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


def bb(n):
    if 6 <= n <= 10:
        return "6-10"
    if 11 <= n <= 20:
        return "11-20"
    if n >= 41:
        return "41+"
    return None


def load(pattern):
    acc = collections.defaultdict(dict)
    for path in sorted(glob.glob(pattern)):
        op = gzip.open if path.endswith(".gz") else io.open
        with op(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                acc[r["arm"]].setdefault(r["case_key"], r)
    return acc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", default="results/diag/structure.json")
    ap.add_argument("--percase", default="results/percase/quadfull_s*_percase.gz")
    ap.add_argument("--out", default="paper/figures/fig_landscape.pdf")
    ap.add_argument("--json-out", default="results/diag/landscape.json")
    args = ap.parse_args(argv)

    sj = json.load(io.open(args.structure, encoding="utf-8"))
    cases = {c["key"]: c for c in sj["cases"]}
    arms = load(args.percase)

    fig, axes = plt.subplots(1, 3, figsize=(W, H),
                             gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.30})
    out = {}

    for ax, kind, title in ((axes[0], "needle", "(a) single-entity"),
                            (axes[1], "multi_entity", "(b) multi-entity")):
        for arm, label, col, mk, ls, lw, ms in ARMS:
            d = arms.get(arm, {})
            cells = collections.defaultdict(list)
            for k, c in cases.items():
                if c["case_kind"] != kind or k not in d:
                    continue
                cells[db(c.get("line_dist") or 0)].append(d[k]["f1"])
            xs, ys = [], []
            for i, b in enumerate(BINS):
                v = cells.get(b)
                if not v:
                    continue
                xs.append(i)
                ys.append(statistics.mean(v))
                out.setdefault("distance", []).append(
                    {"arm": arm, "kind": kind, "bin": b, "n": len(v),
                     "f1": statistics.mean(v)})
            ax.plot(xs, ys, color=col, lw=lw, ls=ls, label=label,
                    zorder=5 if arm == "quad4" else 4)
            for xi, yi, b in zip(xs, ys, [BINS[i] for i in xs]):
                n_i = len(cells[b])
                ax.plot([xi], [yi], marker=mk, ms=6 if arm == "quad4" else 5,
                        color=col, zorder=6,
                        markerfacecolor="white" if n_i < 20 else col,
                        markeredgecolor=col, markeredgewidth=1.0)
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels(BINS)
        ax.set_xlabel("anchor$-$answer distance (lines)")
        ax.set_xlim(-0.35, len(BINS) - 0.65)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color=G_L, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(title, fontsize=10, loc="left", fontweight="bold")

    axes[0].set_ylabel("F1")

    # (c) block size at fixed distance
    ax = axes[2]
    for arm, label, col, mk, ls, lw, ms in ARMS:
        d = arms.get(arm, {})
        cells = collections.defaultdict(list)
        for k, c in cases.items():
            if k not in d:
                continue
            dd = c.get("line_dist") or 0
            if not (1 <= dd <= 4):
                continue
            b = bb(c.get("anchor_block_lines") or 0)
            if b:
                cells[b].append(d[k]["f1"])
        xs, ys = [], []
        for i, b in enumerate(BLK):
            v = cells.get(b)
            if not v or len(v) < 10:
                continue
            xs.append(i)
            ys.append(statistics.mean(v))
            out.setdefault("block", []).append(
                {"arm": arm, "bin": b, "n": len(v), "f1": statistics.mean(v)})
        ax.plot(xs, ys, color=col, lw=lw, ls=ls, label=label,
                zorder=5 if arm == "quad4" else 4)
        for xi, yi, b in zip(xs, ys, [BLK[i] for i in xs]):
            n_i = len(cells[b])
            ax.plot([xi], [yi], marker=mk, ms=6 if arm == "quad4" else 5,
                    color=col, zorder=6,
                    markerfacecolor="white" if n_i < 20 else col,
                    markeredgecolor=col, markeredgewidth=1.0)
    ax.set_xticks(range(len(BLK)))
    ax.set_xticklabels(BLK)
    ax.set_xlabel("anchor block size (lines), distance 1$-$4")
    ax.set_xlim(-0.35, len(BLK) - 0.65)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(color=G_L, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("(c) block size", fontsize=10, loc="left", fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=4, frameon=False, handlelength=1.8, columnspacing=1.6)
    fig.text(0.5, -0.10, "open marker: fewer than 20 cases", ha="center", fontsize=10,
             color=G_D, style="italic")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.30)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)

    if args.json_out:
        with io.open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
