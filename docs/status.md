# Status: what is in this repository

This is the canonical PrivTrim repository: the method, every benchmark, the per-case evidence behind
every reported number, the manuscript, and the lab-notebook documentation the project was run on. This
page is the inventory, plus the things that are *not* in the tree and why.

Short version: **nothing is missing.** Verifying the numbers, rebuilding the manuscript and re-running
the reader are all possible from a fresh clone; only model weights are fetched at run time.

## Authorship

The project was written by two people, and eleven of Yukun Wen's commits were originally recorded under
the wrong identity: ten carried a placeholder e-mail (`yukun.wen@example.com`, so GitHub did not
attribute them) and one carried the repository owner's handle instead of a display name — that commit
(`fe2d10e`) is the one that added the LaTeX sources and compiled PDF. Checking the whole commit list
against dates, committers and file provenance gives a clean split:

| Window | Commits | Signed as |
| --- | ---: | --- |
| 2026-09-24 … 2026-10-17 | 68 | `Shijie Zeng` |
| 2026-10-25 … 2026-11-03 | 29 | `Yukun Wen` (27 under the correct identity, 2 under the bare handle) |

The reorganisation commit that produced the current layout is authored `Yukun Wen`.

## Access

`firesaku` (GitHub id 54803898, display name *Shijie Zeng*) owns this repository. `Yukun-Wen` is a
collaborator with **push** permission. If another collaborator is ever added, remember that write
permission allows pushing commits and branches and editing files, but not changing repository settings,
deleting the repository, or inviting others.

## ✅ Inventory

| Area | Content |
| --- | --- |
| Method | `src/longctx/` — harness, predicate library, quadrant router; `src/analysis_quad4/` — reproducibility checks |
| Benchmarks | `data/` — 20 files pinned by `MANIFEST.sha256`: the 112-case main set, the 2,728-case extension, three 300-case public variants, both paraphrase sets, the frozen V1–V3 sample sets and the external-baseline inputs |
| Evidence | `results/percase/` (755 files + 54 quad4 shards, ~997k rows), `results/cells/` (189), `results/reports/` (21), `results/logs/` (34 — the original job logs, failures included) |
| Manuscript | `paper/` — `main.pdf` (18 pages) and the complete LaTeX source; builds clean with 0 undefined references, 0 undefined citations, 0 overfull hboxes |
| V1–V3 stage | `src/e2e*.py`, `src/v2c_*`, `src/v2d_*`, `src/v3_batch.py`, `src/longctx/e2e_longctx*.py`, `src/mechanism/`, `src/paraphrase_probe/`, `src/verify/` |
| Tooling | official SQuAD v2 scorer, baseline stubs, `slurm/` job template + shard splitter, `tests/` (34 CPU-only unit tests) |
| Demos | `demo/` — three self-contained interactive HTML pages |
| Docs | `docs/*.md` — ten English guides; `docs/README_zh.md`, `docs/00_…`–`28_…`, `docs/analysis/` — the original Chinese lab notebook |
| Archive | `archive/manuscript-v0/` — the superseded earlier manuscript, kept rather than deleted |

## Verification

Everything below runs on a CPU with no third-party package:

```bash
python src/analysis_quad4/reproduce.py          # all PASS, exit 0
python src/analysis_quad4/validate_fallback.py  # routing / budget / leakage, four datasets
python -m unittest discover -s tests -v         # 34 tests
python src/rank_attack.py                       # 100.0% -> 19.1%, matches the paper
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`rank_attack.py`, and anything else that reads `data/longctx_max.jsonl`, needs `gunzip -k` first.

## Not versioned

| Excluded | Why |
| --- | --- |
| model weights (`Qwen2.5-7B-Instruct`, `Mistral-7B-Instruct-v0.3`, the XLM-R encoder) | 13–29 GB; fetched from the model hub |
| the project's private result database (≈ 6 GB) | holds the published baselines' raw payloads, which are theirs to distribute |
| the six source corpora in raw form | third-party licences; only the derived canary-bearing excerpts ship here |

Details: [Data reference](data.md#not-versioned).

## What the reorganisation changed

The layout was moved to a conventional research-repository shape, and the whole tree — including files
that previously existed only in the older working copy — was de-identified in the same pass. **No
numerical result, prompt, predicate or data value was modified**: `reproduce.py`, `rank_attack.py` and
`validate_fallback.py` reproduce the manuscript's values from this tree, and [Results](results.md)
restates the manuscript's tables value for value.

| Change | Reason |
| --- | --- |
| `code/` → `src/`, `Latex/V1/` → `paper/`, `zeal/` → `demo/` | one conventional layout; the earlier `paper/` tree moved to `archive/manuscript-v0/` instead of being deleted |
| `README.md` replaced with an English README; the Chinese one preserved as `docs/README_zh.md` | the release is English-first, but the lab notebook is the project's primary record |
| added `docs/*.md` (10), `assets/`, `tests/`, `slurm/`, `requirements.txt`, `CITATION.cff` | navigation, figures, checks, and a cluster-neutral job template |
| cluster account paths removed from 27 files | they identify a specific institution's cluster and a specific account |
| per-node host names replaced in 9 files | same |
| GPU model names replaced with `GPU` / `32 GB GPU` | the hardware class narrows down the cluster |
| student-ID e-mail addresses removed from the manuscript byline | personal data with no scientific content |
| absolute Windows development paths replaced with `/path/to/…` in 52 files | machine-specific, no scientific content |
| earlier user placeholders normalised to `/path/to/…` | the previous half-done sanitisation left them in 41 files |
| credential-shaped ignore rules kept, cluster-named rules dropped | the former are needed by every contributor, the latter identify a system |
| `data/MANIFEST.sha256` regenerated over all 20 files | integrity is checkable without the upstream manifest |
| module docstrings and printed report labels of the load-bearing files translated to English | this is the English release; older analysis scripts keep their inline comments |

## Before making a publication claim

The manuscript, the benchmark construction and the V1–V3 mechanism study are joint work. Any decision
about submitting or publicising is one the co-author should agree to.
