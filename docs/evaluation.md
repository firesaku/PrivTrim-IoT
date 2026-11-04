# Evaluation

How to run the reader end to end, and how to read what it prints.

> Nothing in this repository executes a language model. The harness is invoked on a machine that has
> the weights; the CPU-only parts are described in [Reproduction](reproduction.md).

## What you need

| Component | Version used for the paper |
| --- | --- |
| Python | 3.10+ |
| PyTorch | 2.6 |
| Transformers | 4.57 |
| Reader (main) | `Qwen/Qwen2.5-7B-Instruct`, fp16, greedy |
| Reader (cross-model) | `mistralai/Mistral-7B-Instruct-v0.3`, fp16, greedy |
| Embedding model (optional, `--sel emb`) | `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` |
| Scorer | the official SQuAD v2 script, vendored at `src/external/squad_v11/evaluate-v2.0.py` |

Decoding is greedy with `max_new_tokens=64` and `maxlen=24576` under `--maxlen-policy raise`: a
prompt that exceeds the limit **aborts** rather than being truncated, because truncation silently
removes the answer (see [Benchmark](benchmark.md#known-caveats)). The seed is fixed at `20261015`.

## The command

```bash
gunzip -k data/longctx_max.jsonl.gz

python src/longctx/e2e_quad.py \
    --cases data/longctx_max.jsonl \
    --out out/report.txt --json out/cells.json --percase out/percase.jsonl \
    --lm  /models/Qwen2.5-7B-Instruct \
    --scorer src/external/squad_v11/evaluate-v2.0.py \
    --arms quad4,quad4_del,alias,delete,full,budget_cluster_link,alias_budget_link \
    --budgets 0.5 --dtype fp16 --batch 8 --max-new 64 --maxlen 24576 \
    --alias-question --no-oracle --seed 20261015
```

`--no-oracle` is not optional for any number you intend to report: it turns off the gold-answer oracle
used for diagnostics and makes relevance come from the question alone. `--alias-question` applies the
alias map to the question as well as the context, which is the *(q′, P)* form the method is defined on.

## Options that matter

| Flag | Effect |
| --- | --- |
| `--arms` | comma-separated; `all` expands to the full arm matrix |
| `--budgets` | e.g. `0.10,0.15,0.25,0.5,0.75`; used for the budget curve |
| `--sel auto\|lex\|emb` | redundancy measure: sentence embeddings when `--emb` is given, else lexical Jaccard |
| `--dry-run` | no model; build arms and reconcile the budget only |
| `--stub-preds FILE` | feed external predictions through the full scoring/leakage/report chain |
| `--dump-arms FILE` | write the built arm texts out for manual inspection |
| `--ext-arms FILE` | inject pre-compressed contexts from an external baseline (see below) |
| `--limit N`, `--n-per-source N` | subsample for a smoke run |
| `--maxlen-policy raise\|truncate` | `raise` by default; `truncate` exists only for reproducing the old artefact |
| `--seed` | case-order and sampling seed |

## Arms

| Arm | Compresses? | Sanitises? | Protects the anchor? |
| --- | --- | --- | --- |
| `full` | no | no | — |
| `delete` | no | redact | — |
| `alias` | no | alias | — |
| `budget_cluster` | yes | no | no |
| `budget_cluster_link` | yes | no | line |
| `budget_random` | yes | no | random control |
| `delete_budget` | yes | redact | line |
| `alias_budget` | yes | alias | no |
| `alias_budget_link` | yes | alias | line |
| **`quad4`** | yes | alias | **block** |
| `quad4_del` | yes | delete | block |

`quad4_del` is the ablation that replaces the alias operation by deletion with everything else held
fixed; it is the arm that isolates the value of aliasing rather than deleting.

## What the report prints

The run report keeps two blocks strictly apart, because mixing them is how budget comparisons go wrong:

1. **length-preserving block** — `full`, `delete`, `alias`. These do not compress, so they are an
   upper bound and a privacy reference, not budget-matched rivals.
2. **equal-budget block** — every arm whose per-case budget is `B = floor(r * n_C)`. For these the
   report prints, per arm and per ratio, the *measured* token count with its mean, maximum absolute
   deviation from `B`, and maximum percentage deviation. A budget mismatch cannot hide in an average.

Per case the harness records both leak channels separately: `ctx_leak_*` (recoverable from the emitted
prompt) and `out_leak_*` (emitted by the model). Two controls run in every batch:

- the uncompressed arm is recomputed and must match its reference value, case by case;
- every budget arm asserts non-overshoot, and raises instead of truncating.

Runs failing either control are discarded, not reported.

## Scoring

Scoring uses the official SQuAD v2 script (`compute_exact`, `compute_f1`), a fixed, model-independent
procedure. `data/pc_qa.jsonl.gz` gives the same 300 Type A cases in the format the Perception
Compressor expects, so an external baseline can be run and joined by `case_key`.

## External baselines

`src/longctx/ext_compress.py` drives an external compressor over a benchmark and writes per-case
contexts; `src/build_ext_arms.py` collects the results into the `--ext-arms` file the harness consumes.
The run then evaluates baseline arms through the identical scoring, leakage and reporting chain, which
is what makes the "same cases, same budget, same scorer" claim checkable.

The external compressor's own checkout and its raw batch input are **not** part of this repository
(third-party code and corpora); see [Data reference](data.md#not-versioned). Publishing a competing
method's numbers is only defensible if they were produced by that method's reference implementation at
its own 50% setting, with the realised token count reported next to them — which is what the results
tables do.

## Cost

| Configuration | Wall clock (one GPU-class GPU) |
| --- | --- |
| 112 cases × 11 arms × 1 ratio | ~5 min |
| 112 cases × 11 arms × 5 ratios | ~25 min |
| 2,728 cases × 7 arms × 1 ratio | ~4 h |

See [Compute](compute.md) for a batch-system template and the sharding pattern actually used.
