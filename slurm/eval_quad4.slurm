#!/bin/bash
# Evaluate the DARK harness on one shard of a benchmark.
#
# This is the cluster-neutral form of the jobs that produced results/percase/.
# It runs on ONE GPU; the array exists for throughput and resumability, not for
# memory.  Nothing here loads a model on the login node.
#
#   1. edit the four paths below (or export the variables before sbatch)
#   2. mkdir -p logs
#   3. sbatch slurm/eval_quad4.slurm
#
# Re-submitting after a timeout re-runs only the shards whose report is missing.
#
#SBATCH --job-name=privtrim
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --array=0-31%8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
export PYTHONIOENCODING=utf-8

# ---------------------------------------------------------------- configuration
REPO="${PRIVTRIM_ROOT:-$PWD}"                                  # repository checkout
PY="${PRIVTRIM_PY:-python}"                                    # interpreter
LM="${PRIVTRIM_LM:?export PRIVTRIM_LM=/path/to/Qwen2.5-7B-Instruct}"
CASES="${PRIVTRIM_CASES:-$REPO/data/longctx_max.jsonl}"        # gunzip -k first
ARMS="${PRIVTRIM_ARMS:-quad4,quad4_del,alias,delete,full,budget_cluster_link,alias_budget_link}"
BUDGETS="${PRIVTRIM_BUDGETS:-0.5}"
MAXLEN="${PRIVTRIM_MAXLEN:-24576}"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARD="${SLURM_ARRAY_TASK_COUNT:-1}"
OUT="$REPO/out/shard${SHARD}"

# ---------------------------------------------------------------- resumability
if [ -f "$OUT/report.txt" ]; then
    echo "[shard $SHARD/$NSHARD] report already exists, nothing to do"
    exit 0
fi
mkdir -p "$OUT"

# ---------------------------------------------------------------- one shard
"$PY" "$REPO/slurm/shard_cases.py" "$CASES" "$OUT/cases.jsonl" "$SHARD" "$NSHARD"

"$PY" "$REPO/src/longctx/e2e_quad.py" \
    --cases   "$OUT/cases.jsonl" \
    --out     "$OUT/report.txt" \
    --json    "$OUT/cells.json" \
    --percase "$OUT/percase.jsonl" \
    --lm      "$LM" \
    --scorer  "$REPO/src/external/squad_v11/evaluate-v2.0.py" \
    --arms    "$ARMS" \
    --budgets "$BUDGETS" \
    --dtype fp16 --batch 8 --max-new 64 \
    --maxlen "$MAXLEN" --maxlen-policy raise \
    --alias-question --no-oracle --seed 20261015

echo "[shard $SHARD/$NSHARD] done"
