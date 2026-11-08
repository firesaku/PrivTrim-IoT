#!/usr/bin/env python3
"""Detection-recall curve: what zero leakage costs when the value set is not known for free.

The paper measures leakage against a known set V of injected values. A reviewer will ask what
happens when V has to come from a detector. Each run here keeps every protected value with
probability rho, so the curve answers that question directly, and rho = 1.0 is the control that
must reproduce the paper's zero.

Two lines are plotted because they answer different things:

  measured    what the harness reports: the share of cases with at least one value recoverable
  expected    rho^m averaged over cases, where m is that case's protected-value count -- the leak
              rate a per-value detector model predicts if each miss is independent

Where the two agree, the mechanism is simply "every undetected value stays in the prompt"; where
they diverge, compression is dropping the units that carry the missed values, which is the more
interesting case and the one worth stating in the text.

Usage
    python src/analysis_quad4/make_fig_recall.py --out paper/figures/fig_recall.pdf
"""
from __future__ import annotations

import argparse
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
W, H = 3.45, 2.65

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "text.color": "#000000", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})

ORDER = ["1", "0p995", "0p99", "0p98", "0p95", "0p9"]
RHO = {"1": 1.0, "0p995": 0.995, "0p99": 0.99, "0p98": 0.98, "0p95": 0.95, "0p9": 0.90}


def open_maybe_gz(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def load(tag):
    rows = []
    for path in sorted(glob.glob(str(ROOT / "results" / "percase" / ("recall_%s*" % tag)))):
        with open_maybe_gz(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def summarise(rows, arm):
    sub = [r for r in rows if r.get("arm") == arm]
    if not sub:
        return None
    n = len(sub)
    leaked_cases = sum(1 for r in sub if (r.get("ctx_leak_value") or 0) > 0)
    leaked_vals = sum((r.get("ctx_leak_value") or 0) for r in sub)
    total_vals = sum((r.get("n_canaries") or 0) for r in sub)
    return {
        "arm": arm, "n": n,
        "leak_case_pct": 100.0 * leaked_cases / n,
        "leak_value_pct": 100.0 * leaked_vals / max(1, total_vals),
        "f1": statistics.mean(r["f1"] for r in sub),
        "gic_pct": 100.0 * sum(1 for r in sub if r.get("ans_in_ctx")) / n,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="paper/figures/fig_recall.pdf")
    ap.add_argument("--json-out", default="results/diag/recall.json")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    table, expected = [], []
    for tag in ORDER:
        rows = load(tag)
        s = summarise(rows, "quad4")
        if s is None:
            print("missing per-case data for rho tag %s -- skipping" % tag)
            continue
        s["rho"] = RHO[tag]
        table.append(s)
        # expected leak rate under an independent per-value detector
        q = [r for r in rows if r.get("arm") == "quad4"]
        exp = sum((1.0 - RHO[tag]) ** 0 * (1.0 - (RHO[tag] ** (r.get("n_canaries") or 0)))
                  for r in q) / max(1, len(q))
        expected.append({"rho": RHO[tag], "expected_leak_case_pct": 100.0 * exp})

    if not table:
        print("no recall results yet under results/percase/")
        return 1

    ctrl = [t for t in table if t["rho"] >= 1.0]
    if ctrl and ctrl[0]["leak_case_pct"] > 0.0:
        print("CONTROL FAILED: rho=1.0 leaks %.2f%% of cases; expected 0%%"
              % ctrl[0]["leak_case_pct"])
        return 2

    payload = {"runs": table, "expected": expected}
    if args.json_out:
        with io.open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=1)
        print("wrote", args.json_out)
    if args.check:
        print("check mode: not writing figure")
        return 0

    fig, ax = plt.subplots(figsize=(W, H))
    xs = [100.0 * t["rho"] for t in table]
    ys = [t["leak_case_pct"] for t in table]
    ex = [100.0 * e["expected_leak_case_pct"] for e in expected]

    ax.plot(xs, ex, linestyle="--", color=G_M, linewidth=1.2,
            label="expected if every miss leaks")
    ax.plot(xs, ys, marker="o", color=INK, linewidth=1.6, markersize=4.5,
            label="measured")
    for x, y, t in zip(xs, ys, table):
        ax.annotate("%.0f" % y, (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7, color=INK)
    ax.set_xlabel("share of protected values the detector finds (%)")
    ax.set_ylabel("cases with a value recoverable (%)")
    ax.set_xlim(min(xs) - 1.0, 100.6)
    ax.set_ylim(-2, 102)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(G_L)
    ax.spines["bottom"].set_color(G_L)
    ax.grid(color="#EFEFEF", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", args.out)

    print()
    print("%-8s %8s %10s %10s %8s %8s" % ("rho", "n", "leak case%", "leak val%", "F1", "GIC%"))
    for t, e in zip(table, expected):
        print("%-8.3f %8d %10.1f %10.1f %8.3f %8.1f   expected %.1f"
              % (t["rho"], t["n"], t["leak_case_pct"], t["leak_value_pct"], t["f1"],
                 t["gic_pct"], e["expected_leak_case_pct"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
