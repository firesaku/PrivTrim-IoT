#!/usr/bin/env python3
"""Generate tables/tab_external.tex: every arm, every metric, on the same 2,728 cases.

The table used to carry F1 and leakage only, and its numbers were transcribed by hand. This script
reads them from the archived per-case records instead, so the table cannot drift from the data,
and adds the three metrics that were reported only in prose: exact match, gold-in-context (the
share of cases in which the record carrying the answer survives compression) and realised context
length.

Ordering follows the manuscript: ours first, then published compressors by F1, then our ablations
interleaved where their scores place them.

Usage
    python src/analysis_quad4/make_tab_external.py            # write the table
    python src/analysis_quad4/make_tab_external.py --check    # compare with the committed table

One caveat is encoded rather than hidden: the LexComp row comes from its own evaluation batch, in
which the control arms score 0.6078 and 0.4934 rather than 0.6078 and 0.4937 because batched fp16
inference is not bit-exact. The table reports the main-run values for DARK and the LexComp batch's
value for LexComp; the difference is in the fourth decimal and is stated in the caption.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import io
import json
import statistics
import sys

# arm -> (display, year, family, bold)
ARMS = [
    ("full",                      r"\arm{full}",                          "---",  "uncompressed",             False),
    ("ext_full",                  r"\arm{ext\_full}",                     "---",  "uncompressed (rerun)",     False),
    ("quad4",                     r"\textbf{\textsc{DARK}}",              "---",  "ours (routes)",            True),
    ("ext_quito_x",               "Quito-X",                               "2025", "query-aware",              False),
    ("ext_quito_x_native",        r"Quito-X \arm{\_native}",              "2025", "query-aware",              False),
    ("ext_cpc",                   "CPC",                                   "2025", "query-aware",              False),
    ("ext_cpc_native",            r"CPC \arm{\_native}",                   "2025", "query-aware",              False),
    ("ext_longllmlingua",         "LongLLMLingua",                         "2024", "task-agnostic",            False),
    ("ext_provence",              "Provence",                              "2025", "query-aware",              False),
    ("ext_provence_native",       r"Provence \arm{\_native}",              "2025", "query-aware",              False),
    ("ext_lexcomp",               r"LexComp \arm{\_preprint}",             "2026", "lexical rules",            False),
    ("alias_budget_link",         r"\arm{alias\_budget\_link}",            "---",  "ours, no key",             False),
    ("budget_cluster_link",       r"\arm{budget\_cluster\_link}",          "---",  "ours, anchor kept",        False),
    ("quad4_del",                 r"\arm{quad4\_del}",                     "---",  "ours, delete",             False),
    ("ext_llmlingua2",            "LLMLingua-2",                           "2024", "task-agnostic",            False),
    ("ext_selective_context",     "Selective Context",                     "2023", "task-agnostic",            False),
    ("ext_llmlingua2_native",     r"LLMLingua-2 \arm{\_native}",           "2024", "task-agnostic",            False),
    ("ext_mask_llmlingua2",       r"\arm{mask\_llmlingua2}",               "---",  "sanitize + compress",      False),
    ("ext_mask_llmlingua2_native", r"\arm{mask\_llmlingua2\_native}",      "---",  "sanitize + compress",      False),
    ("alias_budget",              r"\arm{alias\_budget}",                  "---",  "ours, global alias",       False),
    ("budget_cluster",            r"\arm{budget\_cluster}",                "---",  "ours, redundancy only",    False),
]

CAPTION = r"""\caption{Every arm on the same $2{,}728$ cases (Qwen2.5-7B-Instruct, $r=0.50$), with every
endpoint we measure. Published compressors are re-run through their reference implementations on
these cases; arms suffixed \arm{\_native} use each method's own default length. \emph{F1} and
\emph{EM} are the task metrics. \emph{GIC} is gold-in-context: the share of cases in which the
record carrying the reference answer survives compression at all, and it is the column that
separates the two families of method---every arm that keeps the anchor block holds it at $100\%$,
while the published compressors drop that record in $11$ to $54\%$ of cases. \emph{leak} is the
fraction of cases in which a protected value is recoverable from the prompt; \emph{tok} is the
mean realised context length. \textsc{DARK} is the only arm that reaches $100\%$ GIC and $0\%$
leakage together. Against the strongest leak-free published baseline, the masking wrapper, it
exceeds by $+0.3226$ $[+0.3050,+0.3413]$; against the strongest compressor, Quito-X, the
difference is $+0.0107$ $[-0.0091,+0.0304]$, i.e.\ not significant, at $0\%$ versus $98.1\%$
leakage. The \arm{lexcomp} row is from its own batch, where the control arms score $0.4934$ rather
than $0.4937$.}"""


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


def metrics(rows):
    recs = list(rows.values())
    n = len(recs)
    leak = sum(1 for r in recs
               if (r.get("ctx_leak_value") or 0) > 0 or (r.get("out_leak_value") or 0) > 0)
    gic = sum(1 for r in recs if r.get("ans_in_ctx"))
    toks = [r.get("ctx_tokens") for r in recs if r.get("ctx_tokens")]
    return {
        "n": n,
        "f1": statistics.mean(r["f1"] for r in recs),
        "em": statistics.mean(r.get("em") or 0 for r in recs),
        "gic": 100.0 * gic / n,
        "leak": 100.0 * leak / n,
        "tok": statistics.mean(toks) if toks else float("nan"),
    }


def build():
    main = load(["results/percase/quadfull_s*_percase.gz",
                 "results/percase/extevfull_s*_percase.gz",
                 "results/percase/extnew2_s*_percase.gz"])
    lex = load(["results/percase/lexcomp_eval_s*_percase.gz"])
    if "ext_lexcomp" in lex:
        main["ext_lexcomp"] = lex["ext_lexcomp"]

    missing = [a for a, *_ in ARMS if a not in main]
    if missing:
        raise SystemExit("no per-case records for: %s" % missing)

    lines = [r"\begin{table*}[t]", CAPTION, r"\label{tab:external}",
             r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lllrrrrr}", r"\toprule",
             r"Method & Year & Family & F1 $\uparrow$ & EM $\uparrow$ & GIC $\uparrow$"
             r" & leak $\downarrow$ & tok \\", r"\midrule"]

    for arm, display, year, family, bold in ARMS:
        m = metrics(main[arm])
        f1 = "%.4f" % m["f1"]
        em = "%.4f" % m["em"]
        gic = "%.1f\\%%" % m["gic"]
        leak = "%.1f\\%%" % m["leak"]
        tok = "%d" % round(m["tok"])
        if bold:
            f1, em, gic, leak = ["\\textbf{%s}" % x for x in (f1, em, gic, leak)]
        lines.append("%-30s & %-5s & %-20s & %s & %s & %s & %s & %s \\\\"
                     % (display, year, family, f1, em, gic, leak, tok))

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)



SOURCES = ["pec_home", "privacylens", "simuhome", "smartbench_tdcs", "toolmem", "toolmem_raw"]
SRC_LABEL = {
    "pec_home": r"\arm{pec\_home}", "privacylens": r"\arm{privacylens}",
    "simuhome": r"\arm{simuhome}", "smartbench_tdcs": r"\arm{smartbench\_tdcs}",
    "toolmem": r"\arm{toolmem}", "toolmem_raw": r"\arm{toolmem\_raw}",
}


def build_persource():
    """Every arm against every source, F1 only, so the table stays readable."""
    main = load(["results/percase/quadfull_s*_percase.gz",
                 "results/percase/extevfull_s*_percase.gz",
                 "results/percase/extnew2_s*_percase.gz"])
    lex = load(["results/percase/lexcomp_eval_s*_percase.gz"])
    if "ext_lexcomp" in lex:
        main["ext_lexcomp"] = lex["ext_lexcomp"]

    head = [r"\begin{table*}[t]",
            r"\caption{Per-source F1 for every arm on the extension set ($r=0.50$). The aggregate in "
            r"Table~\ref{tab:external} hides the spread: \textsc{DARK} ranges from $0.2631$ on "
            r"\arm{toolmem\_raw}, where every arm is weak, to $0.6756$ on \arm{pec\_home}. No source "
            r"reverses the ordering between \textsc{DARK} and the strongest leak-free published "
            r"baseline.}",
            r"\label{tab:persource_all}",
            r"\centering\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{l" + "r" * (len(SOURCES) + 1) + "}",
            r"\toprule",
            "Method & " + " & ".join(SRC_LABEL[s] for s in SOURCES) + r" & overall \\",
            r"\midrule"]
    body = []
    for arm, display, year, family, bold in ARMS:
        rows = main[arm]
        cells = []
        for s in SOURCES:
            vals = [r["f1"] for k, r in rows.items() if r.get("source") == s]
            cells.append("%.4f" % statistics.mean(vals) if vals else "---")
        overall = "%.4f" % statistics.mean(r["f1"] for r in rows.values())
        if bold:
            cells = ["\\textbf{%s}" % c for c in cells]
            overall = "\\textbf{%s}" % overall
        body.append("%-30s & %s & %s \\\\" % (display, " & ".join(cells), overall))
    tail = [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(head + body + tail)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="paper/tables/tab_external.tex")
    ap.add_argument("--out-persource", default="paper/tables/tab_persource_all.tex")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    built = build()
    if args.check:
        committed = io.open(args.out, encoding="utf-8").read()
        if built.strip() == committed.strip():
            print("committed tab_external.tex reproduced exactly")
            return 0
        print("MISMATCH in", args.out)
        return 1

    with io.open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(built)
    print("wrote", args.out)
    with io.open(args.out_persource, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(build_persource())
    print("wrote", args.out_persource)
    return 0


if __name__ == "__main__":
    sys.exit(main())
