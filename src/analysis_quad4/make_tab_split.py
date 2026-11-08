#!/usr/bin/env python3
"""Generate tables/tab_split.tex: every arm, split by whether the question names a protected value.

The aggregate table answers "is DARK competitive on average", which is the wrong question for a
setting whose whole point is that one half of the data is different from the other. This table
reports the two halves separately, with the paired difference against DARK recomputed inside each
half, so a reader can see that the aggregate tie is not a uniform result.

Sign convention: delta is DARK minus the row's method, so positive means DARK is ahead.

Usage
    python src/analysis_quad4/make_tab_split.py            # write the table
    python src/analysis_quad4/make_tab_split.py --check    # compare with the committed file
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
import sys

# arm -> (display, is_ours, from_lexcomp_batch)
ROWS = [
    ("full",                   r"\arm{full}",                       False, False),
    ("quad4",                  r"\textbf{\textsc{DARK}}",           True,  False),
    ("ext_quito_x",            "Quito-X",                           False, False),
    ("ext_cpc",                "CPC",                               False, False),
    ("ext_longllmlingua",      "LongLLMLingua",                     False, False),
    ("ext_provence",           "Provence",                          False, False),
    ("ext_lexcomp",            r"LexComp \arm{\_preprint}",         False, True),
    ("quad4_del",              r"\arm{quad4\_del}",                 False, False),
    ("ext_llmlingua2",         "LLMLingua-2",                       False, False),
    ("ext_mask_llmlingua2",    r"\arm{mask\_llmlingua2}",           False, False),
    ("alias_budget",           r"\arm{alias\_budget}",              False, False),
    ("budget_cluster",         r"\arm{budget\_cluster}",            False, False),
]

CAPTION = r"""\caption{The same arms as Table~\ref{tab:external}, split by whether the question names a
protected value. A \emph{multi-entity} question names an account that is itself a protected value;
a \emph{needle} question names a record identifier that is not. $\Delta$ is the paired difference
\textsc{DARK} minus that row's method on the same cases, with a $95\%$ percentile bootstrap interval
over $4{,}000$ resamples (seed 7); positive means \textsc{DARK} is ahead, and a blank cell is the
\textsc{DARK} row itself. Leakage is recomputed inside each half. The aggregate in
Table~\ref{tab:external} averages these two columns, and they move in opposite directions.}"""


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


def pooled(rows, kind):
    return [r for r in rows.values() if r.get("case_kind") == kind]


def leaks(r):
    return (r.get("ctx_leak_value") or 0) > 0 or (r.get("out_leak_value") or 0) > 0


def paired(dark, other, kind, reps=4000, seed=7):
    keys = [k for k in dark if k in other and dark[k].get("case_kind") == kind]
    diffs = [dark[k]["f1"] - other[k]["f1"] for k in keys]
    if not diffs:
        return None
    rng = random.Random(seed)
    bs = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(reps))
    return statistics.mean(diffs), bs[int(0.025 * reps)], bs[int(0.975 * reps)], len(diffs)


def fmt_delta(d):
    if d is None:
        return "---"
    m, lo, hi, _ = d
    if lo <= 0 <= hi:
        return r"$%+.3f$ {\scriptsize$[%+.3f,%+.3f]$}" % (m, lo, hi)
    return r"$\mathbf{%+.3f}$ {\scriptsize$[%+.3f,%+.3f]$}" % (m, lo, hi)


def build():
    main = load(["results/percase/quadfull_s*_percase.gz",
                 "results/percase/extevfull_s*_percase.gz",
                 "results/percase/extnew2_s*_percase.gz"])
    lex = load(["results/percase/lexcomp_eval_s*_percase.gz"])
    if "ext_lexcomp" in lex:
        main["ext_lexcomp"] = lex["ext_lexcomp"]

    missing = [a for a, *_ in ROWS if a not in main]
    if missing:
        raise SystemExit("no per-case records for: %s" % missing)

    out = [r"\begin{table*}[t]", CAPTION, r"\label{tab:split}",
           r"\centering\footnotesize", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lrrrrrr}", r"\toprule",
           r" & \multicolumn{2}{c}{multi-entity ($n=1{,}500$)}"
           r" & \multicolumn{2}{c}{needle ($n=1{,}228$)}"
           r" & \multicolumn{2}{c}{$\Delta$ vs \textsc{DARK}} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
           r"Method & F1 $\uparrow$ & leak $\downarrow$ & F1 $\uparrow$ & leak $\downarrow$"
           r" & multi & needle \\", r"\midrule"]

    dark = main["quad4"]
    for arm, display, ours, _ in ROWS:
        cells = [display]
        for kind in ("multi_entity", "needle"):
            rows = pooled(main[arm], kind)
            n = len(rows)
            f1 = statistics.mean(r["f1"] for r in rows)
            lk = 100.0 * sum(1 for r in rows if leaks(r)) / n
            cells.append("%.4f" % f1)
            cells.append("%.1f\\%%" % lk)
        dm = paired(dark, main[arm], "multi_entity") if not ours else None
        dn = paired(dark, main[arm], "needle") if not ours else None
        cells.append(fmt_delta(dm))
        cells.append(fmt_delta(dn))
        out.append(" & ".join(cells) + r" \\")

    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(out)


def main_cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="paper/tables/tab_split.tex")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    built = build()
    if args.check:
        committed = io.open(args.out, encoding="utf-8").read()
        ok = built.strip() == committed.strip()
        print("committed tab_split.tex reproduced exactly" if ok else "MISMATCH in " + args.out)
        return 0 if ok else 1
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(built)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())
