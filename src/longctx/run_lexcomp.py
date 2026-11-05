#!/usr/bin/env python3
"""Compress a benchmark file with the 2026 lexical-rule compressor (external baseline).

Method
    Chokshi, "Lexical Prompt Compression for Large Language Models: A Training-Free,
    Deterministic Pipeline with Empirical Pareto Analysis Across Eleven Task Categories",
    arXiv:2609.13154 (2026).  The peer-reviewed companion result is Ma et al., EMNLP 2026.

Reference implementation (external, not vendored here)
    https://github.com/shaminchokshi/Lexical-Prompt-Compressor---Research-Experiment-Code
    Also packaged as `pip install less-tokens`.

What it is
    A training-free, deterministic, CPU-only compressor: a fixed sequence of lexical rules
    (filler-phrase removal, abbreviation and contraction substitution, filler-word and
    stop-word removal, function-word pruning, POS-based pruning, lemmatisation, synonym
    shortening) applied to the text.  No model is called at compression time.

What we use
    The `maximum` preset, i.e. all eleven rules.  On this benchmark it is the only preset
    that shortens anything: `light` actually grows the text by about 8% because the rules
    re-space punctuation, while `medium` and `aggressive` retain roughly 96-98%.

Output
    One JSON object per line, in the schema the DARK harness consumes through
    `e2e_quad.py --ext-arms`:

        {"key": <case key>, "method": "lexcomp", "text": <compressed context>,
         "original_chars": <int>, "compressed_chars": <int>}

    Only the context is compressed.  The question is left untouched, which is what every
    other arm in the comparison sees.

Usage
    python src/longctx/run_lexcomp.py \
        --cases   data/longctx_max.jsonl.gz \
        --out     out/ext_lexcomp.jsonl \
        --lexcomp /path/to/Lexical-Prompt-Compressor---Research-Experiment-Code

    # score it with the normal harness (see slurm/eval_lexcomp.slurm):
    #   e2e_quad.py --arms full,quad4 --ext-arms out/ext_lexcomp.jsonl --ext-ratio 0.5 ...
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import statistics
import sys
import time


PRESETS = ("light", "medium", "aggressive", "maximum")


def build_toggles(preset: str, technique_order):
    """Map a preset name onto the toggle dict the reference implementation expects."""
    def on(*names):
        return {k: (k in names) for k in technique_order}

    return {
        "light": on("remove_filler_phrases", "apply_contractions", "remove_filler_words",
                    "preserve_named_entities", "normalize_whitespace_punct"),
        "medium": on("remove_filler_phrases", "apply_abbreviations", "apply_contractions",
                     "remove_filler_words", "remove_stopwords",
                     "preserve_named_entities", "normalize_whitespace_punct"),
        "aggressive": on("remove_filler_phrases", "apply_abbreviations", "apply_contractions",
                         "remove_filler_words", "remove_stopwords", "remove_function_words",
                         "lemmatize", "preserve_named_entities", "normalize_whitespace_punct"),
        "maximum": on(*technique_order),
    }[preset]


def open_maybe_gz(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True,
                    help="benchmark file, .jsonl or .jsonl.gz, one case per line")
    ap.add_argument("--out", required=True, help="ext-arm JSONL to write")
    ap.add_argument("--lexcomp", default=os.environ.get("LEXCOMP_REPO"),
                    help="path to the Lexical-Prompt-Compressor checkout "
                         "(or set LEXCOMP_REPO)")
    ap.add_argument("--preset", default="maximum", choices=PRESETS,
                    help="toggle configuration (default: maximum)")
    ap.add_argument("--method", default="lexcomp",
                    help="name recorded in the `method` field (default: lexcomp)")
    ap.add_argument("--context-field", default="context",
                    help="which field of the case to compress (default: context)")
    ap.add_argument("--key-field", default="key",
                    help="which field identifies the case (default: key)")
    ap.add_argument("--limit", type=int, default=0, help="compress only the first N cases")
    args = ap.parse_args(argv)

    if not args.lexcomp:
        ap.error("--lexcomp is required (path to the reference implementation checkout)")
    # The reference implementation exposes `backend/compressor.py` at the checkout root.
    sys.path.insert(0, args.lexcomp)
    try:
        from backend.compressor import compress, TECHNIQUE_ORDER  # noqa: E402
    except Exception as exc:  # pragma: no cover - depends on the external checkout
        sys.stderr.write(
            "could not import backend.compressor from %r: %s\n"
            "clone https://github.com/shaminchokshi/"
            "Lexical-Prompt-Compressor---Research-Experiment-Code and pass --lexcomp\n"
            % (args.lexcomp, exc))
        return 2

    toggles = build_toggles(args.preset, TECHNIQUE_ORDER)

    cases = []
    with open_maybe_gz(args.cases) as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    if args.limit:
        cases = cases[:args.limit]
    print("cases: %d   preset: %s" % (len(cases), args.preset), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    t0 = time.time()
    written = 0
    char_ratio = []
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as out:
        for i, case in enumerate(cases):
            src = case.get(args.context_field) or ""
            try:
                text = compress(src, toggles)["compressed"]
            except Exception as exc:  # a single bad case must not kill the run
                sys.stderr.write("  case %s failed: %s\n" % (case.get(args.key_field), exc))
                continue
            if not text:
                continue
            written += 1
            char_ratio.append(len(text) / max(1, len(src)))
            out.write(json.dumps({
                "key": case[args.key_field],
                "method": args.method,
                "text": text,
                "original_chars": len(src),
                "compressed_chars": len(text),
            }, ensure_ascii=False) + "\n")
            if (i + 1) % 500 == 0:
                print("  %d/%d   %.1f min" % (i + 1, len(cases), (time.time() - t0) / 60),
                      flush=True)

    elapsed = time.time() - t0
    print("wrote %d rows to %s in %.1f min" % (written, args.out, elapsed / 60))
    if char_ratio:
        print("character retention: median %.1f%%  mean %.1f%%"
              % (100 * statistics.median(char_ratio), 100 * statistics.mean(char_ratio)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
