#!/usr/bin/env python3
"""Gold-in-context by arm and source: one flat row against a patchwork.

Table IV reports GIC as a single number per arm, which hides the fact that it is the arm's most
source-dependent property. Every published compressor retains the answer-bearing record in some
sources and loses it in others --- Quito-X ranges from 16.8% on privacylens to 100% on simuhome,
Selective Context from 3.2% on simuhome to 99.8% on smartbench_tdcs. The arms that keep the anchor
block are flat at 100% on all six.

That flatness is the mechanism, and the figure is careful about what it does and does not show:
both block-keeping arms are flat, including the delete variant, so GIC separates *region* from
*operation* and does not by itself justify the operation. That is why the paper treats GIC as a
necessary condition and reports referential integrity separately (Table I).

Usage
    python src/analysis_quad4/make_fig_gic.py --out paper/figures/fig_gic.pdf
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
from matplotlib.colors import LinearSegmentedColormap

INK = "#4A1412"
G_L = "#D4D4D2"
W, H = 6.30, 3.05

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "text.color": "#000000", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})

SOURCES = ["pec_home", "privacylens", "simuhome", "smartbench_tdcs", "toolmem", "toolmem_raw"]
SRC_DISPLAY = ["pec\nhome", "privacy\nlens", "simu\nhome", "smartbench\ntdcs", "tool\nmem",
               "toolmem\nraw"]
ROWS = [
    ("quad4",              "DARK",                True),
    ("quad4_del",          "delete variant",      False),
    ("ext_provence",       "Provence",            False),
    ("ext_cpc",            "CPC",                 False),
    ("ext_longllmlingua",  "LongLLMLingua",       False),
    ("ext_quito_x",        "Quito-X",             False),
    ("ext_mask_llmlingua2", "mask + LLMLingua-2", False),
    ("ext_llmlingua2",     "LLMLingua-2",         False),
    ("alias_budget_link",  "global alias",        False),
    ("ext_selective_context", "Selective Context", False),
]
CMAP = LinearSegmentedColormap.from_list(
    "gic", ["#FFFFFF", "#E3CFCB", "#B08A85", "#7A4A44", INK])


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--percase", nargs="+",
                    default=["results/percase/quadfull_s*_percase.gz",
                             "results/percase/extevfull_s*_percase.gz",
                             "results/percase/extnew2_s*_percase.gz"])
    ap.add_argument("--out", default="paper/figures/fig_gic.pdf")
    ap.add_argument("--json-out", default="results/diag/gic.json")
    args = ap.parse_args(argv)

    acc = load(args.percase)
    matrix, overall, table = [], [], []
    for arm, label, ours in ROWS:
        rows = acc[arm]
        vals = []
        for s in SOURCES:
            sub = [r for r in rows.values() if r.get("source") == s]
            v = 100.0 * sum(1 for r in sub if r.get("ans_in_ctx")) / len(sub) if sub else float("nan")
            vals.append(v)
            table.append({"arm": arm, "source": s, "gic": v, "n": len(sub)})
        matrix.append(vals)
        o = 100.0 * sum(1 for r in rows.values() if r.get("ans_in_ctx")) / len(rows)
        overall.append(o)
        table.append({"arm": arm, "source": "overall", "gic": o, "n": len(rows)})

    fig, ax = plt.subplots(figsize=(W, H))
    im = ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=100, aspect="auto")

    for i in range(len(ROWS)):
        for j in range(len(SOURCES)):
            v = matrix[i][j]
            ax.text(j, i, "%.0f" % v, ha="center", va="center", fontsize=10,
                    color="#FFFFFF" if v > 62 else "#000000")

    ax.set_xticks(range(len(SOURCES)))
    ax.set_xticklabels(SRC_DISPLAY, fontsize=10)
    ax.set_yticks(range(len(ROWS)))
    labels = [lbl for _, lbl, _ in ROWS]
    ax.set_yticklabels(labels, fontsize=10)
    for tick, (_, _, ours) in zip(ax.get_yticklabels(), ROWS):
        if ours:
            tick.set_fontweight("bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(SOURCES))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(ROWS))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("answer record survives (%)", fontsize=10)
    cb.ax.tick_params(labelsize=10, length=0)
    cb.outline.set_visible(False)

    fig.tight_layout(pad=0.3)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)

    if args.json_out:
        with io.open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(table, fh, indent=1)
        print("wrote", args.json_out)

    print()
    print("%-20s %s  overall" % ("arm", " ".join("%6s" % s[:6] for s in SOURCES)))
    for (arm, label, _), vals, o in zip(ROWS, matrix, overall):
        print("%-20s %s  %6.1f" % (label, " ".join("%6.1f" % v for v in vals), o))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
