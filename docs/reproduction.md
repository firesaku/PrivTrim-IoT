# Reproduction

Three levels. Levels 1 and 2 need **no GPU, no model weights and no network**; both run in this
repository as shipped.

| Level | Question it answers | Command | Cost |
| --- | --- | --- | --- |
| **1. Re-derive the numbers** | Do the archived per-case results still produce exactly the reported tables? | `python src/analysis_quad4/reproduce.py` | ~10 s, CPU |
| **2. Rebuild the arms** | Is prompt construction — routing, aliasing, budget accounting, leakage — deterministic and correct without a model? | `python src/analysis_quad4/validate_fallback.py` + `python -m unittest discover -s tests` | ~5 s, CPU |
| **3. Re-run the reader** | Do the F1 values themselves come back? | `src/longctx/e2e_quad.py` on a GPU | ~30 min per arm set |

---

## Level 1 — re-derive every reported number

```bash
python src/analysis_quad4/reproduce.py
```

This reads **only** `results/percase/` and, for each of the ~90 numbers stated in
[Results](results.md), recomputes it from the raw per-case rows and diffs it against the expected
value with a tolerance of 0.0015 (fp16 non-determinism level).

```
==============================================================================
1) longctx_max extension, 2,728 cases (forward fill, no answer key)
==============================================================================
  PASS  full                                     got 0.6081  expected 0.6081
  PASS  quad4                                    got 0.4937  expected 0.4937
  ...
  ---- quad4 leakage must be 0; measured 0.000
  ...
  ---- full arm identical case-by-case with the paper reference batch: 112 / 112
  ---- quad4 mean tokens as a share of the full context: 48.6% (expected ~48.6%)
  ...
RESULT: all PASS - the archived results in this repository agree with every number stated in the docs
```

Exit code is **0 only when every item passes**; the script is usable as a gate. Two properties are
worth calling out because they are the ones that catch silent corruption:

- the **identity baseline** — the uncompressed arm must reproduce the reference batch *case by case*,
  not just in the mean. A mismatch means the run differs in model, precision or data;
- **zero leakage** is asserted as an exact equality, not rounded.

If an item fails, the script names it and exits 1.

## Level 2 — rebuild the arms without a model

```bash
# routing, forced-set sizing, budget compliance and leakage, on four datasets
python src/analysis_quad4/validate_fallback.py

# predicate and router unit tests
python -m unittest discover -s tests -v      # 34 tests, no third-party dependency
```

`validate_fallback.py` walks the construction path of `e2e_quad.py` on 150 cases from each of the four
shipped benchmarks and reports:

```
=== longctx_max (2,728 cases, record-block structure) ===
  mode: {'entry': 150}
  forced/budget: mean=0.398 max=0.493  cases over budget=0/150
  emitted prompt over budget: 0 / 150
  gold answer present in prompt: 150 / 150 (100.0%)
  cases with a leaked canary: 0 / 150
```

The four rows are the four invariants: the mode selector picks `entry` when record markers exist and
`window` when they do not; forced retention never exceeds the budget; the emitted prompt never exceeds
the budget; and no canary survives. `gold answer present in prompt` is a model-independent upper bound
on F1 — on Type A (topical) it drops to 16.0%, which is the honest ceiling for a rule that has no
entity to route on.

You can also build prompts for any benchmark without loading a model:

```bash
gunzip -k data/longctx_max.jsonl.gz
python src/longctx/e2e_quad.py --dry-run --cases data/longctx_max.jsonl \
    --out out/dry.txt --json out/dry.json --percase out/dry_percase.jsonl
```

`--dry-run` runs "read cases → build arms → reconcile budget → report" and prints the measured token
count of every arm in every cell (mean, max absolute deviation, max percentage deviation), so a budget
mismatch cannot hide. Token counts in dry-run are estimates (words × 1.33); a real run measures with
the tokenizer.

## Level 3 — re-run the reader

Needs a 7B reader on one GPU; every case file it reads ships in `data/`. See
[Evaluation](evaluation.md) for the exact invocation.

---

## What is verified and what is not

| Claim | Status |
| --- | --- |
| Every F1 / leakage / token-share number in [Results](results.md) | ✅ re-derived from `results/percase/` on every run |
| Prompt construction, budget accounting, zero leakage | ✅ re-executed on the shipped benchmarks |
| Predicates, alias map, quadrant routing | ✅ unit-tested |
| `e2e_quad.py` is byte-identical to what `make_e2e_quad.py` regenerates | ✅ `python src/analysis_quad4/make_e2e_quad.py`, compare hashes |
| The manuscript builds from source | ✅ `pdflatex` ×2 + `bibtex`, 25 pages, no undefined references |
| The F1 values themselves | 🔌 requires a GPU run; all inputs are present |
| The V1–V3 mechanism results | 🔌 re-running needs the same GPU setup; the archives are in `results/` |

## Runs archived after the first release

Four result families were produced after the initial archive and are now committed, because the
manuscript quotes numbers that cannot be re-derived without them. All are compressed per-case
records, the same format as the older families.

| Family in `results/percase/` | Job | What it backs |
| --- | --- | --- |
| `window2_s*_percase.gz` | 1557997 | Table VIII and Fig. 14 — the 2×2 over granularity × operation |
| `granite_s*_percase.gz` | 1557517 | Table VI — the second reader |
| `lexcomp_eval_s*_percase.gz` | LexComp eval batch | the 2026 rule-based compressor comparison |
| `mprobe_*_percase.gz` | 1557495 | the reader-selection probe quoted in Section IX |

Each derived summary can be rebuilt and checked against what the manuscript printed:

```bash
python src/analysis_quad4/make_factorial.py --check     # Table VIII / Fig. 14
python src/analyze_xmodel.py \
    --results 'results/percase/granite_s*_percase.gz' \
    --baseline 'results/percase/quadfull_s*_percase.gz' # Table VI, granite row
python src/analyze_lexcomp.py \
    --results 'results/percase/lexcomp_eval_s*_percase.gz' \
    --baseline 'results/percase/quadfull_s*_percase.gz' # the LexComp row
```

`make_factorial.py --check` exits non-zero on any mismatch at 1e-9, which is how the figure, the
table and the text are kept from drifting apart. All bootstrap intervals use 4,000 resamples with
seed 7 for the same reason.

## Runs added for the second round of experiments

Five more families were produced for the experiments added after the first submission. They extend
the same per-case format, so every analysis script reads them without changes.

| Family in `results/percase/` | Job | What it backs |
| --- | --- | --- |
| `recall_{1,0p995,0p99,0p98,0p95,0p9}_percase.jsonl` | 1561909 | the detection-recall curve in Section IX |
| `budgetdark_s*_percase.jsonl` | 1561910 | `quad4` at r = 0.05/0.10/0.15, the missing half of the budget sweep |
| `budgetext_{0p05,0p15}_s0_percase.jsonl` | 1562592 | published compressors at the two tight budgets |
| `frontend_s*_percase.jsonl` | 1562593 | DARK as a front end, Section VII |
| `reid_s*_percase.jsonl` | 1561924 | the extraction attack, Section IX |

Two of these need a note on how the code they call was prepared, because the published compressors
are not shipped here:

**The ratio patch.** `batch1` hard-codes `budget = n // 2`. For the low-budget sweep, a copy of the
published code sits at `code/batch1_ratio/` whose only difference from the original is that one
expression, which now reads `PT_EXT_RATIO` and defaults to `0.5`. A smoke job (1561925) confirmed
both halves: at `PT_EXT_RATIO=0.5` Quito-X reproduces `n // 2` exactly (2254 -> 1127) and at `0.1`
it yields 225. The original code is untouched.

**The detector sweep.** `data/recall/` is generated, not stored: `src/longctx/imperfect_detector.py`
writes six case files from `data/longctx_max.jsonl.gz`, keeping each protected value with
probability rho. The subset of 500 cases and the per-case RNG seed are both fixed, so the files are
reproducible byte for byte.

Because the sweep varies what the router is *told* to protect, two small hooks were added to
`e2e_quad.py`: `canaries_eval` (scoring counts the full injected set while routing sees only the
detected subset, otherwise a missed value would be neither protected nor counted) and a carve-out in
`check_question_alias` for an undetected target entity (the question then goes out unaliased, which
is what a real system would do). Both are no-ops when the field is absent, which the committed
`--check` outputs confirm byte for byte.

Rebuild and check:

```bash
python src/analysis_quad4/make_fig_recall.py --check          # the recall curve
python src/analysis_quad4/make_fig_budget_all.py --json-out results/diag/budget_all.json
python src/analysis_quad4/make_tab_split.py --check           # Table V, the by-kind main table
```

All of these read only `results/percase/`, so a reader without a GPU can re-derive every number the
manuscript prints from these experiments.

## Regenerating `e2e_quad.py`

`e2e_quad.py` is generated from `e2e_longctx_r5full.py` by a text transform, and the transform is
byte-exact:

```bash
cp src/longctx/e2e_quad.py /tmp/before.py
python src/analysis_quad4/make_e2e_quad.py
diff /tmp/before.py src/longctx/e2e_quad.py     # no output = identical
```

This is why both files carry the same module docstring: editing either one alone breaks the identity
check, which is the point — a claim of "this is the code that produced the numbers" is only meaningful
if it can be re-derived.

## Reproducibility controls baked into the harness

The harness enforces two controls in every batch, described in the manuscript's experimental setup:

1. the **uncompressed arm is recomputed** in every run and must reproduce its reference value;
2. every **budget arm asserts non-overshoot at the case level**, and the harness raises rather than
   truncating.

Runs that fail either control are discarded rather than reported. `reproduce.py` re-checks both from
the archived evidence.
