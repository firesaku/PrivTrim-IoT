#!/usr/bin/env python3
"""Where the aggregate comparison hides the advantage: two regimes, opposite directions.

The paper's headline comparison is a tie (0.4937 against 0.4831). That tie is an average over two
populations that behave in opposite ways, and splitting them is what shows the contribution plainly:

  question names a sensitive account  -> every compressor that beats us on F1 leaks on 84--100% of
      the same cases, and among the arms that do not leak we are the only one that answers at all
      (0.3273 against 0.0007 for the next leak-free arm).

  question names a non-sensitive record id -> there is nothing to protect by removing, and we beat
      every compressor outright by +0.12 to +0.48 while matching the uncompressed prompt at half
      the tokens.

The mechanism is the same in both halves: the F1 cost is the price of making the referent
unreadable, so it appears exactly where the referent is the secret and nowhere else.

Usage
    python src/analysis_quad4/make_fig_regimes.py --out paper/figures/fig_regimes.pdf
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK = "#4A1412"
RED = "#8E2A22"
G_D = "#4A4A4A"
G_M = "#9A9A9A"
G_L = "#D4D4D2"
PALE = "#EFEFEF"
W, H = 3.45, 4.30

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9.5,
    "text.color": "#000000", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})

ARMS = [
    ("full",                   "uncompressed"),
    ("quad4",                  "DARK"),
    ("ext_quito_x",            "Quito-X"),
    ("ext_longllmlingua",      "LongLLM"),
    ("ext_cpc",                "CPC"),
    ("ext_provence",           "Provence"),
    ("ext_llmlingua2",         "LLMLingua-2"),
    ("ext_mask_llmlingua2",    "mask+L2"),
    ("budget_cluster",         "redund."),
]
KINDS = [("multi_entity", "(a) question names a sensitive account"),
         ("needle", "(b) question names a non-sensitive record id")]


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def load(patterns):
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


def leaks(r):
    return (r.get("ctx_leak_value") or 0) > 0 or (r.get("out_leak_value") or 0) > 0


def paired_ci(a, b, kind, acc, reps=4000, seed=7):
    keys = [k for k in acc[a] if k in acc[b] and acc[a][k].get("case_kind") == kind]
    if not keys:
        return None
    diffs = [acc[a][k]["f1"] - acc[b][k]["f1"] for k in keys]
    rng = random.Random(seed)
    bs = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(reps))
    return {"mean": statistics.mean(diffs),
            "lo": bs[int(0.025 * reps)], "hi": bs[int(0.975 * reps)], "n": len(diffs)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--percase", nargs="+",
                    default=["results/percase/quadfull_s*_percase.gz",
                             "results/percase/extevfull_s*_percase.gz",
                             "results/percase/extnew2_s*_percase.gz"])
    ap.add_argument("--out", default="paper/figures/fig_regimes.pdf")
    ap.add_argument("--json-out", default="results/diag/regimes.json")
    args = ap.parse_args(argv)

    acc = load(args.percase)
    stats = {}
    for kind, _ in KINDS:
        stats[kind] = {}
        for arm, label in ARMS:
            rows = [r for r in acc[arm].values() if r.get("case_kind") == kind]
            stats[kind][label] = {
                "n": len(rows),
                "f1": statistics.mean(r["f1"] for r in rows),
                "leak": 100.0 * sum(1 for r in rows if leaks(r)) / len(rows),
                "gic": 100.0 * sum(1 for r in rows if r.get("ans_in_ctx")) / len(rows),
            }

    # order by the multi-entity column so the eye can follow one ordering across both panels
    order = sorted([l for _, l in ARMS], key=lambda l: stats["multi_entity"][l]["f1"])

    fig, axes = plt.subplots(2, 1, figsize=(W, H), sharey=True)
    ys = range(len(order))

    for ax, (kind, title) in zip(axes, KINDS):
        for y, label in zip(ys, order):
            s = stats[kind][label]
            leaky = s["leak"] > 50.0
            ours = label == "DARK"
            color = INK if ours else (RED if leaky else G_M)
            ax.barh(y, s["f1"], height=0.62, color=color,
                    edgecolor=INK if ours else "none", linewidth=0.8 if ours else 0)
            ax.text(s["f1"] + 0.012, y, "%.2f\u00b7%.0f%%" % (s["f1"], s["leak"]),
                    va="center", ha="left", fontsize=7.2,
                    color=INK if ours else G_D,
                    fontweight="bold" if ours else "normal")
        ax.set_yticks(list(ys))
        ax.set_yticklabels(order, fontsize=8)
        for tick, label in zip(ax.get_yticklabels(), order):
            if label == "DARK":
                tick.set_fontweight("bold")
                tick.set_color(INK)
        ax.set_xlim(0, 0.86)
        ax.set_xlabel("F1", fontsize=8.5, labelpad=1)
        ax.set_title(title, fontsize=9, pad=4, loc="left")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(G_L)
        ax.spines["bottom"].set_color(G_L)
        ax.grid(axis="x", color=PALE, linewidth=0.8)
        ax.set_axisbelow(True)

    for ax in axes:
        ax.text(0.60, len(order) - 0.30, "F1  ·  leak", ha="left", va="center",
                fontsize=8, color=G_M, style="italic")

    handles = [Patch(facecolor=INK, label="DARK"),
               Patch(facecolor=RED, label="leaks on most cases"),
               Patch(facecolor=G_M, label="leak-free, weaker")]
    # panel (a)'s lower right is empty: the two leak-free weak arms have zero-length bars there
    axes[0].legend(handles=handles, loc="lower right", frameon=False, fontsize=7,
                   handlelength=1.0, borderpad=0.1, labelspacing=0.25)

    fig.tight_layout(pad=0.3, h_pad=1.0)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)

    payload = {"by_kind": stats}
    for kind, _ in KINDS:
        for arm, label in [("ext_quito_x", "Quito-X"), ("ext_cpc", "CPC"),
                           ("full", "uncompressed"), ("ext_mask_llmlingua2", "mask + L2"),
                           ("budget_cluster", "redundancy only")]:
            c = paired_ci("quad4", arm, kind, acc)
            if c:
                payload["%s_DARK_minus_%s" % (kind, label.replace(" ", "_"))] = c
    if args.json_out:
        with io.open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=1)
        print("wrote", args.json_out)

    print()
    for kind, _ in KINDS:
        print("%-14s %6s %8s %8s" % (kind, "n", "F1", "leak%"))
        for label in order:
            s = stats[kind][label]
            print("  %-18s %5d %7.4f %7.1f" % (label, s["n"], s["f1"], s["leak"]))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
