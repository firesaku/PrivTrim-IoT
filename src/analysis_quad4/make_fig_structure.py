#!/usr/bin/env python3
"""Figure: what the record-block structure actually looks like, and why it matters.

Four panels, all from `results/diag/structure.json` (built by structure_probe.py) plus the
per-case records of the DARK run.  No model calls.

  (a) anatomy of one block: the anchor sits on the private-contact block, the answer on the
      record line below it -- same block, several lines apart
  (b) how far apart: histogram of the anchor-answer line distance per case kind
  (c) five retention strategies scored on the three things that can break -- answer span,
      exposure of the raw protected value, and surface-token collision
  (d) block size against DARK's F1, i.e. the size at which keeping the block stops working

Usage
    python src/analysis_quad4/make_fig_structure.py \
        --structure results/diag/structure.json \
        --percase 'results/percase/quadfull_s*_percase.gz' \
        --out paper/figures/fig_structure.pdf
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

INK, RED = "#4A1412", "#8E2A22"
G_D, G_M, G_L, PALE = "#555555", "#9A9A9A", "#D4D4D2", "#EFEFEF"
W, H = 7.10, 4.20

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.monospace": ["Consolas", "DejaVu Sans Mono"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "text.color": "#000000", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
})


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def load_percase(pattern, arm):
    rows = {}
    for p in sorted(glob.glob(pattern)):
        with open_maybe_gz(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("arm") == arm:
                    rows[r["case_key"]] = r
    return rows


def panel_anatomy(ax):
    """Show one block: contact lines carry the anchor, the record line carries the answer."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("(a) block anatomy", fontsize=10, loc="left", fontweight="bold")

    def line(y, text, colour, weight="normal"):
        ax.text(4, y, text, fontsize=10, color=colour, family="monospace",
                va="center", fontweight=weight)

    ax.add_patch(Rectangle((2, 6), 96, 84, facecolor="none", edgecolor=INK, lw=0.9))
    ax.text(4, 83, "one record block", fontsize=10, color=G_D, style="italic", va="center")

    line(72, "Private contact record:", G_D)
    line(62, "Account ID: ACCT-4cd0", RED, "bold")
    line(46, '{"device": "water_heater"', INK)
    line(38, ' "persona_id": "multi_0226"', RED, "bold")

    ax.text(97, 62, "anchor", fontsize=10, color=RED, va="center", ha="right",
            fontweight="bold")
    ax.text(97, 38, "answer", fontsize=10, color=RED, va="center", ha="right",
            fontweight="bold")
    ax.text(4, 18, "same block; 5 lines apart", fontsize=10, color=G_D, va="center")


def panel_distance(ax, cases):
    """Line distance between the anchor mention and the answer, per case kind."""
    kinds = ["multi_entity", "needle"]
    colours = {"multi_entity": RED, "needle": G_M}
    bins = list(range(0, 13))
    width = 0.42
    for i, kind in enumerate(kinds):
        hist = collections.Counter(r["line_dist"] for r in cases
                                   if r["status"] == "ok" and r["case_kind"] == kind)
        vals = [hist.get(b, 0) for b in bins]
        ax.bar([b + (i - 0.5) * width for b in bins], vals, width=width,
               facecolor=colours[kind], edgecolor="white", lw=0.5,
               label=kind.replace("_", "-"))
    ax.set_xlabel("lines between anchor mention and answer")
    ax.set_ylabel("cases")
    ax.set_xticks(bins)
    ax.set_title("(b) anchor--answer distance", fontsize=10, loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, handlelength=1.2, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_strategies(ax, surv):
    """The three things that can break, for five retention strategies."""
    order = [
        ("full", "full"),
        ("budget_cluster_link", "line"),
        ("alias_budget_link", "line\n+alias"),
        ("quad4_del", "block\n+delete"),
        ("quad4", "block\n+alias"),
    ]
    order = [(a, lab) for a, lab in order if a in surv]
    metrics = [
        ("answer_present_pct", "answer survives"),
        ("raw_value_exposed_pct", "raw value exposed"),
        ("collision_pct", "entities collide"),
    ]
    colours = [INK, RED, G_M]
    n = len(order)
    width = 0.26
    xs = list(range(n))
    for j, (field, label) in enumerate(metrics):
        vals = [surv[a].get(field) or 0 for a, _ in order]
        ax.bar([x + (j - 1) * width for x in xs], vals, width=width,
               facecolor=colours[j], edgecolor="white", lw=0.5, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in order])
    ax.set_ylabel("cases (%)")
    ax.set_ylim(0, 118)
    ax.set_title("(c) what breaks, by strategy", fontsize=10, loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, handlelength=1.2, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.30), fontsize=9, columnspacing=1.2,
              handletextpad=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_blocksize(ax, cases, percase):
    """DARK F1 against the size of the block it has to keep."""
    joined = [(r["anchor_block_lines"], percase[r["key"]]["f1"], r["case_kind"])
              for r in cases
              if r["status"] == "ok" and r["key"] in percase]
    edges = [(6, 10, "6-10"), (11, 20, "11-20"), (21, 10 ** 6, ">20")]
    kinds = ["multi_entity", "needle"]
    colours = {"multi_entity": RED, "needle": G_M}
    width = 0.36
    for i, kind in enumerate(kinds):
        vals, ns = [], []
        for lo, hi, _lab in edges:
            sub = [f for l, f, k in joined if k == kind and lo <= l <= hi]
            vals.append(statistics.mean(sub) if sub else 0)
            ns.append(len(sub))
        xs = [j + (i - 0.5) * width for j in range(len(edges))]
        ax.bar(xs, vals, width=width, facecolor=colours[kind], edgecolor="white",
               lw=0.5, label=kind.replace("_", "-"))
        for x, v, m in zip(xs, vals, ns):
            ax.text(x, v + 0.012, "%d" % m, fontsize=9, color=G_D, ha="center")
    ax.set_xticks(range(len(edges)))
    ax.set_xticklabels([lab for _lo, _hi, lab in edges])
    ax.set_xlabel("lines in the anchor block")
    ax.set_ylabel("F1 under DARK")
    ax.set_ylim(0, 1.18)
    ax.set_title("(d) block size vs F1", fontsize=10, loc="left", fontweight="bold")
    ax.legend(frameon=False, handlelength=1.2, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", default="results/diag/structure.json")
    ap.add_argument("--percase", default="results/percase/quadfull_s*_percase.gz")
    ap.add_argument("--out", default="paper/figures/fig_structure.pdf")
    args = ap.parse_args(argv)

    payload = json.load(io.open(args.structure, encoding="utf-8"))
    cases = payload["cases"]
    surv = payload.get("survival_by_arm") or {}
    if not surv:
        raise SystemExit("run structure_probe.py with --percase first")

    percase = load_percase(args.percase, "quad4")

    fig, axes = plt.subplots(2, 2, figsize=(W, H))
    panel_anatomy(axes[0][0])
    panel_distance(axes[0][1], cases)
    panel_strategies(axes[1][0], surv)
    panel_blocksize(axes[1][1], cases, percase)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.22,
                        hspace=0.62, wspace=0.30)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
