# Compute

Everything in §§1–2 of [Reproduction](reproduction.md) runs on a laptop. Only the reader
(§3) needs an accelerator. This page describes how the GPU runs were organised, in
cluster-neutral form, so they can be re-issued on any Slurm system.

## The rule that matters most

**Never load a model on a login node.** Login nodes are shared by everyone using the cluster; a single
7B fp16 load plus generation is enough to make the machine unusable for others and, on managed
clusters, enough to get an account suspended. The submission templates below exist precisely so that
the only thing executed on a login node is `sbatch`, `squeue`, `sacct` and `scancel`.

The CPU-only checks — `reproduce.py`, `validate_fallback.py`, `unittest`, `--dry-run` — cost
milliseconds and are fine anywhere, but they do not need the cluster at all.

## Resource shape

| Job | Nodes | GPUs | Wall clock | Notes |
| --- | ---: | ---: | --- | --- |
| 112-case main table | 1 | 1 | ~5 min | 11 arms × 1 ratio |
| budget sweep | 1 | 1 | ~25 min | 11 arms × 5 ratios |
| cross-model / paraphrase | 1 | 1 | ~10 min each | one reader per job |
| 2,728-case extension | 1 | 1 per shard | ~4 h total across 32 shards | `--array=0-31` |

The whole project fits on **one** GPU at a time. Nothing here needs multi-node parallelism; sharding is
used for throughput and for resumability, not for memory.

## Template

`slurm/eval_quad4.slurm` in this repository is a working template, and
`slurm/shard_cases.py` is the shard splitter it calls. Edit the four `PRIVTRIM_*` paths at the top (or
export them) and submit:

```bash
mkdir -p logs
export PRIVTRIM_LM=/path/to/Qwen2.5-7B-Instruct
sbatch slurm/eval_quad4.slurm
```

The shape of it:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --array=0-31%8            # 32 shards, 8 at a time
#SBATCH --output=logs/%x-%A_%a.out

SHARD="${SLURM_ARRAY_TASK_ID:-0}"; NSHARD="${SLURM_ARRAY_TASK_COUNT:-1}"
OUT="$REPO/out/shard${SHARD}"
[ -f "$OUT/report.txt" ] && exit 0          # idempotent resubmission
mkdir -p "$OUT"

python "$REPO/slurm/shard_cases.py" "$CASES" "$OUT/cases.jsonl" "$SHARD" "$NSHARD"
python "$REPO/src/longctx/e2e_quad.py" --cases "$OUT/cases.jsonl" \
    --out "$OUT/report.txt" --json "$OUT/cells.json" --percase "$OUT/percase.jsonl" \
    --lm "$LM" --scorer "$REPO/src/external/squad_v11/evaluate-v2.0.py" \
    --arms "$ARMS" --budgets "$BUDGETS" --dtype fp16 --maxlen 24576 \
    --alias-question --no-oracle --seed 20261015
```

The harness has no built-in sharding flag — a run is sharded by splitting the case list
(`shard_cases.py` takes every `count`-th line, which keeps shards disjoint and stable) and merging the
per-case files afterwards. Each shard writes a plain `.jsonl` file, so the merge is a concatenation,
and `reproduce.py` reads them the same way it reads `results/percase/`.

## Practices that were worth the trouble

**Shard, and make each shard idempotent.** Write to `<out>/shard<N>/` and skip a shard whose report
already exists. A four-hour job that times out at hour three should cost one shard, not four hours.

**Keep the wall clock short.** Short limits backfill faster. A 40-minute limit for a 30-minute job is
better than a 6-hour limit for the same work.

**Keep logs.** `results/reports/`, `results/cells/` and `results/percase/` in this repository are the
literal outputs of these jobs, including the failed and superseded ones. The failures are part of the
evidence: a one-token budget-accounting gap and a non-monotone answer survival were both found by
reading them.

**One GPU, not a node.** Every run reported here used a single device. Nothing was measured on a
whole-node allocation, and no number depends on exclusive access.

**Dry-run before you spend.** `--dry-run` builds every arm and reconciles every budget without loading
a model, in seconds. It catches the class of error that costs a four-hour slot: wrong case file, wrong
arm list, a budget that cannot be met, a prompt that exceeds `maxlen`. It does **not** catch token
counting errors from the estimate (`words × 1.33`), which is why the harness re-measures with the real
tokenizer on a real run.

## Environment

The paper's runs used PyTorch 2.6 with Transformers 4.57 in a virtual environment against a
pre-downloaded model snapshot; no run needed network access at job time. Pinning the interpreter and
pointing the harness at a local snapshot path (`--lm /path/to/snapshot`) is what makes a job
reproducible months later.

Optional: an embedding model is only needed for `--sel emb`
(`microsoft/llmlingua-2-xlm-roberta-large-meetingbank` in the paper's runs). The default `--sel auto`
falls back to lexical Jaccard when `--emb` is not given, which is the setting used for the reported
numbers.
