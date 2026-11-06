#!/usr/bin/env python3
"""Leakage-utility frontier over every arm evaluated at the same scale.

The paper's argument is a two-dimensional one: an arm is only usable if it answers the
question *and* forwards nothing. Tables report the two coordinates separately; this figure
puts every arm on both axes at once, so the trade-off is visible rather than assembled by the
reader.

Only arms evaluated on the full 2,728-case extension are plotted, so no point comes from a
different sample. The numbers are recomputed from the per-case records rather than copied
from the table.

Usage
    python src/analysis_quad4/make_fig_frontier.py \
        --percase 'results/percase/*_percase.gz' \
        --out paper/figures/fig_frontier.pdf
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

INK, RED = "#141414", "#8E2A22"
G_D, G_M, G_L, PALE = "#555555", "#9A9A9A", "#D4D4D2", "#EFEFEF"
W, H = 7.10, 3.40

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

# display name, group. Group decides colour and marker.
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



# label offset per arm: (dx, dy, horizontal alignment). The right-hand cluster holds five
# points within 0.08 F1 of each other, so those labels are stacked to the right of the marks
# and the canvas is widened to make room.
OFFSET = {
    "uncompressed": (2.0, 0.008, "left"),
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
    """arm -> (n, f1, leak%), keeping only arms with exactly `target_n` distinct cases.

    Several arms appear in more than one run, so a row is keyed by (arm, case_key) and the
    first occurrence wins. Pass the canonical run for each arm family; a glob that pulls in a
    different sample size will make that arm drop out rather than silently mix sets.
    """
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
    out = {}
    for arm, by_key in acc.items():
        if len(by_key) != target_n:
            continue
        rows = list(by_key.values())
        leak = sum(1 for r in rows
                   if (r.get("ctx_leak_value") or 0) > 0 or (r.get("out_leak_value") or 0) > 0)
        out[arm] = (len(rows), statistics.mean(r["f1"] for r in rows),
                    100.0 * leak / len(rows))
    return out


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
    rows = [(LABELS[a][0], LABELS[a][1], v[1], v[2])
            for a, v in data.items() if a in LABELS]
    if not rows:
        raise SystemExit("no arms matched; check --percase glob and --target-n")
    rows.sort(key=lambda r: (r[3], -r[2]))

    fig, ax = plt.subplots(figsize=(W, H))
    # the region an operator would accept: few leaks, high F1
    ax.axvspan(-2, 5, color=PALE, zorder=0)
    ax.text(1.0, 0.02, "usable region", fontsize=10, color=G_D, ha="center", va="bottom")

    for name, group, f1, leak in rows:
        ax.scatter(leak, f1, s=180 if group == "ours" else 70,
                   marker=MARKER[group], facecolor=COLOUR[group],
                   edgecolor="white", linewidth=0.8, zorder=5)
        dx, dy, ha = OFFSET.get(name, (1.6, 0.008, "left"))
        ax.text(leak + dx, f1 + dy, name, fontsize=10, ha=ha, va="bottom",
                color=COLOUR[group], zorder=6,
                fontweight="bold" if group == "ours" else "normal")

    ax.set_xlabel("protected values recoverable from the prompt (%)  $\\downarrow$")
    ax.set_ylabel("F1 $\\uparrow$")
    ax.set_xlim(-3, 133)
    ax.set_ylim(0, 0.68)
    ax.grid(axis="both", color=G_L, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)

    if args.json_out:
        with io.open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump([{"arm": k, "n": v[0], "f1": v[1], "leak": v[2]}
                       for k, v in sorted(data.items())], fh, indent=1)
        print("wrote", args.json_out)

    print()
    print("%-22s %6s %8s %8s" % ("arm", "n", "F1", "leak%"))
    for name, group, f1, leak in rows:
        print("%-22s %6d %8.4f %8.1f   [%s]" % (name, args.target_n, f1, leak, group))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
