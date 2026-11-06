#!/usr/bin/env python3
"""Conceptual figure: the link that breaks, where the boundary is, and where we sit.

Three panels, each carrying one of the paper's claims:

  (a) the referent is a *link*, not a string -- deletion keeps the answer text and destroys the
      thing that says whose it is; aliasing keeps both. This is the difference the usual
      compression metric does not see.
  (b) the trust boundary: what crosses to the provider, what stays with the operator, and what
      each kind of adversary can do. Nothing here is measured; it is the threat model drawn.
  (c) the same trade-off as Figure 4, but on conceptual axes, with the measured points placed
      in it so the abstract curve is anchored to numbers.

White background, Times, nothing below 10 pt at 7.10 in wide.

Usage
    python src/analysis_quad4/make_fig_concept.py --out paper/figures/fig_concept.pdf
"""
from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch

INK, RED = "#4A1412", "#8E2A22"
G_D, G_M, G_L, PALE = "#555555", "#9A9A9A", "#D4D4D2", "#EFEFEF"
W, H = 7.10, 3.55

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "text.color": "#000000", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})


def panel_link(ax):
    """(a) the answer string survives deletion; the thing that owns it does not."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("(a) the referent is a link, not a string", fontsize=10, loc="left",
                 fontweight="bold")

    rows = [
        (80, "kept", "ACCT-4cd0", "owner known", INK, "-"),
        (54, "delete", "[REDACTED]", "owner lost", RED, "--"),
        (28, "alias", "[ACCT-A]", "owner known", INK, "-"),
    ]
    for y, tag, tok, verdict, colour, ls in rows:
        ax.text(2, y + 10, tag, fontsize=10, color=colour, fontweight="bold")
        ax.add_patch(Rectangle((2, y - 5), 54, 10, facecolor="white", edgecolor=colour,
                               lw=0.9, ls=ls))
        ax.text(5, y, tok, fontsize=10, family="monospace", color=colour, va="center")
        ax.text(60, y, verdict, fontsize=10, color=colour, va="center")

    ax.text(2, 10, "the answer text is the same", fontsize=10, color=G_D, style="italic")
    ax.text(2, 3, "in all three rows", fontsize=10, color=G_D, style="italic")


def panel_boundary(ax):
    """(b) what crosses to the provider, and what each adversary can reach."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.set_title("(b) where the boundary is", fontsize=10, loc="left", fontweight="bold")

    ax.add_patch(FancyBboxPatch((3, 60), 94, 32, boxstyle="round,pad=0,rounding_size=2",
                                facecolor="none", edgecolor=G_D, lw=0.9, ls="--"))
    ax.text(5, 88, "operator", fontsize=10, color=G_D, style="italic")
    ax.text(5, 80, "raw log  ->  routing rule", fontsize=10)
    ax.text(5, 71, "keyed alias map  (stays here)", fontsize=10, color=INK)

    ax.add_patch(FancyArrowPatch((50, 58), (50, 48), arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color=INK))
    ax.text(52, 53, "prompt only", fontsize=10, color=INK, fontweight="bold", va="center")

    ax.add_patch(FancyBboxPatch((3, 20), 94, 26, boxstyle="round,pad=0,rounding_size=2",
                                facecolor="white", edgecolor=RED, lw=1.0))
    ax.text(5, 41, "model provider", fontsize=10, color=RED, fontweight="bold")
    ax.text(5, 33, "reads the prompt,", fontsize=10)
    ax.text(5, 25, "may retain it", fontsize=10)

    ax.text(3, 12, "who can do what", fontsize=10, color=G_D, style="italic")
    ax.text(3, 6, "provider: aliases only", fontsize=10, color=INK)
    ax.text(3, 0.5, "key holder: everything", fontsize=10, color=RED)


def panel_map(ax, pts):
    """(c) the plane the paper argues in. Conceptual only: the measured points are Fig. 4."""
    ax.set_xlim(-6, 108)
    ax.set_ylim(-0.06, 0.82)
    ax.set_xlabel("privacy", fontsize=10)
    ax.set_ylabel("utility", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(c) the plane we argue in", fontsize=10, loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.add_patch(Rectangle((66, 0.44), 40, 0.26, facecolor=PALE, edgecolor="none", zorder=0))
    ax.text(86, 0.795, "the corner that stays empty", fontsize=10, ha="center", va="top",
            color=G_D)

    xs = list(range(0, 101))
    ys = [0.62 * (1 - (1 - x / 100.0) ** 0.55) for x in xs]
    ax.plot(xs, ys, color=INK, lw=1.2, zorder=3)

    ax.annotate("", xy=(98, 0.63), xytext=(34, 0.17),
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color=RED,
                                connectionstyle="arc3,rad=0.16"), zorder=4)


def measured():
    """(privacy, F1) for the arms we can read straight from the per-case records."""
    wanted = {
        "quad4": ("DARK (ours)", True),
        "full": ("uncompressed", False),
        "ext_quito_x": ("Quito-X", False),
        "alias_budget_link": ("global alias", False),
        "ext_mask_llmlingua2": ("masked", False),
    }
    acc = {}
    for fam in ("quadfull", "extevfull", "extnew2"):
        for p in sorted(glob.glob("results/percase/%s_s*_percase.gz" % fam)):
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r["arm"] in wanted:
                        acc.setdefault(r["arm"], {}).setdefault(r["case_key"], r)
    out = []
    for arm, rows in acc.items():
        if len(rows) != 2728:
            continue
        leak = sum(1 for r in rows.values()
                   if (r.get("ctx_leak_value") or 0) > 0 or (r.get("out_leak_value") or 0) > 0)
        out.append((wanted[arm][0], 100.0 * (1 - leak / len(rows)),
                    statistics.mean(r["f1"] for r in rows.values()), wanted[arm][1]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="paper/figures/fig_concept.pdf")
    args = ap.parse_args(argv)

    fig, axes = plt.subplots(1, 3, figsize=(W, H),
                             gridspec_kw={"width_ratios": [1.12, 1.0, 1.0], "wspace": 0.30})
    panel_link(axes[0])
    panel_boundary(axes[1])
    panel_map(axes[2], measured())
    fig.subplots_adjust(left=0.04, right=0.99, top=0.89, bottom=0.16)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)
    for name, privacy, f1, ours in measured():
        print("  %-14s privacy %5.1f  F1 %.4f" % (name, privacy, f1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
