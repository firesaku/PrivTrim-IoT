# Code reference

Every path is relative to the repository root. The **Runs here** column says whether the script can be
executed from a fresh checkout:

- ✅ — needs only the files in this repository (plus `numpy`/`torch`/`transformers` where noted);
- 🔌 — needs an input that is *not* in this repository (a model snapshot, an upstream dataset, an
  external baseline's own checkout);
- ⚙️ — runs against the archived run outputs (`out/`), so it needs a prior run rather than a model.

Everything below is present in the tree; nothing is marked pending.

## `src/longctx/` — the method

| File | Purpose | Runs here |
| --- | --- | --- |
| `e2e_quad.py` | the DARK harness: builds every arm, enforces the budget, scores, records per case | 🔌 needs a model snapshot |
| `e2e_longctx_r5full.py` | predecessor harness; `e2e_quad.py` is generated from it | 🔌 same |
| `privcheck.py` | the predicate library — `norm`, `redact`, `EMAIL/PHONE/ACCOUNT`, `exposure_*`. **Single source of truth** | ✅ |
| `v4_router.py` | the (sensitivity × relevance) router, `Aliaser`, `apply_replacements` | ✅ |
| `ext_compress.py` | drives a published compressor over a benchmark, writes per-case contexts | 🔌 needs the baseline's checkout |
| `patch_ext_arms.py` | adds `--ext-arms` to the harness (historical patch, kept for provenance) | ✅ |
| `para_robustness.py` | checks that the arm ordering survives question rewrites | 🔌 needs results |
| `test_budget_guard_real_tok.py` | regression + property tests for the budget guard with the real tokenizer | 🔌 needs the tokenizer; ⏳ for the predecessor harness it imports |
| `README_VERSIONS.md` | which harness version is authoritative, and the two budget bugs found | — |

## `src/analysis_quad4/` — reproducibility

| File | Purpose | Runs here |
| --- | --- | --- |
| `reproduce.py` | re-derives every reported number from `results/percase/` and diffs it | ✅ **start here** |
| `validate_fallback.py` | model-free check of routing, forced-set sizing, budget compliance and leakage on four datasets | ✅ |
| `make_e2e_quad.py` | regenerates `e2e_quad.py` from `e2e_longctx_r5full.py`, byte-exactly | ✅ |

## `src/` — benchmark construction

| File | Purpose | Runs here |
| --- | --- | --- |
| `make_public_bench.py` | LongBench → project case format, with synthetic PII canaries injected | 🔌 needs LongBench raw |
| `make_typeb.py` | derives the Type B (single-hop) variant from `public_lb.jsonl` | ✅ after `gunzip -k` |
| `make_typec.py` | builds the Type C (two-hop) variant: answer six sentences from the anchor | 🔌 needs LongBench raw |
| `make_pc_input.py` | converts LongBench cases to the Perception Compressor's input format | 🔌 |
| `gen_guiding.py` | generates the guiding questions (`pc_query_dict.json`) for that baseline | 🔌 |
| `build_ext_arms.py` | collects external-baseline outputs into the `--ext-arms` file the harness consumes | 🔌 after a baseline run |
| `build_ext_arms_pub.py` | same, for `public_lb` | 🔌 |
| `make_dac_plan.py`, `make_ext2_plan.py`, `make_extc_plan.py` | static shard plans, computed before submission to remove run-time races | ✅ |
| `make_intact_sync.py` | applies INTACT's substitution map to the question as well | 🔌 |
| `apply_intact.py` | writes INTACT's replacements back into the per-case result format | 🔌 |
| `merge_max32.py` | merges 32 shards into one result set plus aggregate tables | 🔌 after a run |
| `budget_key.py` | keyed vs unkeyed budget-arm comparison | 🔌 after runs |
| `data_build/dataset_probe2.py` | structural probe over the source corpora — one line per source | 🔌 needs the corpora |

## `src/` — analysis and audit

| File | Purpose | Runs here |
| --- | --- | --- |
| `rank_attack.py` | the rank-inversion attack on the alias map (zero GPU); reproduces 100.0% → 19.1% | ✅ after `gunzip -k` |
| `aliasdbg.py` | prints the alias map under an unkeyed and a keyed assignment | ✅ after `gunzip -k` |
| `sigtest.py` | paired bootstrap significance tests (needs `numpy`) | 🔌 needs results |
| `analyze_ext.py` | external baseline vs. internal arms, final comparison table | 🔌 |
| `analyze_pc.py` | Perception Compressor against the other methods on the public benchmark | 🔌 |
| `analyze_pub.py` | public-benchmark utility + leakage, with paired tests | 🔌 |
| `analyze_typec.py` | does the link mechanism hold on two-hop public contexts | 🔌 |
| `analyze_intact.py` | INTACT vs. DARK under equal privacy (both leak-free) | 🔌 |
| `analyze_keyfull.py` | keyed vs unkeyed full-arm comparison | 🔌 |
| `analyze2.py`, `analyze_123.py`, `analyze_ab2.py` | earlier analysis passes over the mechanism experiments | 🔌 |
| `status_ext.py` | external-baseline progress and resume guide (zero GPU) | 🔌 |
| `numchk.py`, `tgtchk.py` | numeric / target-presence spot checks | 🔌 |
| `run_pc.py` | runs the Perception Compressor on LongBench | 🔌 needs its checkout + `pc_stubs` on the path |
| `peek_payload.py`, `join_baseline.py` | inspect and join the published baselines' result database | 🔌 |
| `diag_e2e.py` | prints the aggregate view of an archived e2e cells file | 🔌 |
| `tools/dump*.py` | export helpers over the archived artifacts | 🔌 |
| `verify/audit_driver_diag_e2e.py` | runs `diag_e2e.py` verbatim against a local mirror and cross-checks its arithmetic | 🔌 |
| `v1_checker.py`, `v1b_diag.py` | the V1 paraphrase-aware leak checker and its diagnostic | ✅ needs `src/privcheck.py`, which is present |
| `mechanism/mech.py` | mechanism-level probe (V2 era) | ⚙️ reads the archived `final/*.json` |

## `src/` — the V1–V3 stage

The earlier mechanism study lives beside the long-context stage, in the layout it was written for:

| Path | Purpose |
| --- | --- |
| `e2e.py`, `e2e2.py`, `e2e3.py`, `e2e4.py`, `e2e5.py` | the earlier end-to-end harnesses (V1 → V5) |
| `v2c_feasibility.py`, `v2d_gpu.py`, `v3_batch.py` | feasibility and constraint-ablation drivers |
| `privcheck.py`, `v4_router.py` | this stage's copies of the predicate and router modules |
| `privcheck_selftest.py` | asserts the two containment claims stated in `privcheck.py` |
| `build_conflict_cases.py` | builds the conflict-resolution cases |
| `longctx/e2e_longctx.py`, `e2e_longctx_r2.py`, `r3`, `r4` | harness revisions r1–r4; read `longctx/README_VERSIONS.md` before trusting r1's bytes |
| `longctx/` (analysis helpers) | per-source tables, binding diagnostics, bootstrap joins, question-leak scans |
| `paraphrase_probe/` | the motivation study behind the paraphrase-coverage claim |
| `verify/` (audit scripts) | the earlier independent audit passes |

Note that `src/privcheck.py` is byte-identical to `src/longctx/privcheck.py`; the long-context stage
keeps its own copy so that a pipeline cannot silently pick up a different predicate version.
`src/v4_router.py` does differ from `src/longctx/v4_router.py` — the V4-era scripts expect the former.

## `src/external/` and `src/pc_stubs/`

| File | Purpose | Runs here |
| --- | --- | --- |
| `external/squad_v11/evaluate-v2.0.py` | official SQuAD v2 scorer, vendored so scoring never depends on a pip install | ✅ |
| `pc_stubs/icecream.py`, `pynvml.py`, `seaborn.py` | import-only stubs for the Perception Compressor checkout | ✅ |

## Top-level

| Path | Purpose |
| --- | --- |
| `paper/` | LaTeX sources and compiled PDF of the manuscript |
| `data/` | every benchmark + `MANIFEST.sha256` |
| `results/percase/` | per-case evidence; the input of `reproduce.py` |
| `results/cells/`, `results/reports/`, `results/logs/` | per-cell aggregates, generated reports and the original job logs |
| `demo/` | three self-contained HTML demos — no server, no dependencies |
| `slurm/` | cluster-neutral job template and the shard splitter |
| `tests/` | CPU-only smoke tests over the predicates and the router |
| `assets/` | figures used by the README |
| `archive/manuscript-v0/` | the superseded earlier manuscript, kept for the record |
| `docs/README_zh.md`, `docs/00_…`–`28_…`, `docs/analysis/` | the Chinese lab notebook this release was built from |

## Conventions worth knowing

**One predicate implementation.** `privcheck.py` defines the three value patterns and the leak
verdict. Nothing else carries a copy, because earlier copies drifted and produced a false "12.35% miss"
figure that took a full audit to retract.

**Estimated vs. measured tokens.** `--dry-run` estimates tokens as `words × 1.33`; a real run measures
with the tokenizer. Any budget claim must come from the measured path — a word-count proxy cannot see
the cost of the separator between units, which is exactly where the one-token accounting gap lived.

**Arms are not comparable across blocks.** Length-preserving arms (`full`, `delete`, `alias`) are not
budget-matched to the compressed arms. The reports keep them in separate blocks on purpose.
