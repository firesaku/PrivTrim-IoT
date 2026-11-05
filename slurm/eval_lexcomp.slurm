#!/bin/bash
# Score the 2026 lexical-rule compressor on one shard of the benchmark.
#
# The compressor itself runs on CPU and is not called here: it is run once, ahead of time,
# by src/longctx/run_lexcomp.py, which writes an ext-arm JSONL.  This job only puts that
# pre-compressed text through the same reader, scorer, leak detector and diagnostics as
# every other arm, so the only variable is where the context came from.
#
# `full` and `quad4` are included deliberately.  They are the batch's own control: in the
# paper's run `full` reproduces 0.6078 and `quad4` reproduces 0.4937 on 2,728 cases.  If
# either drifts, the batch is not comparable and should be discarded rather than reported.
#
#   1. produce the ext-arm file first (CPU, no GPU needed):
#        python src/longctx/run_lexcomp.py --cases data/longctx_max.jsonl.gz \
#            --out out/ext_lexcomp.jsonl --lexcomp /path/to/Lexical-Prompt-Compressor
#   2. export the paths below (or edit the defaults)
#   3. mkdir -p logs out
#   4. sbatch slurm/eval_lexcomp.slurm
#
#SBATCH --job-name=lexcomp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
export PYTHONIOENCODING=utf-8

# ---------------------------------------------------------------- configuration
REPO="${PRIVTRIM_ROOT:-$PWD}"                                  # repository checkout
PY="${PRIVTRIM_PY:-python}"                                    # interpreter
LM="${PRIVTRIM_LM:?export PRIVTRIM_LM=/path/to/Qwen2.5-7B-Instruct}"
CASES="${PRIVTRIM_CASES:-$REPO/data/longctx_max.jsonl}"        # gunzip -k first
EXT_ARMS="${PRIVTRIM_EXT_ARMS:-$REPO/out/ext_lexcomp.jsonl}"   # from run_lexcomp.py
ARMS="${PRIVTRIM_ARMS:-full,quad4}"
BUDGETS="${PRIVTRIM_BUDGETS:-0.5}"
MAXLEN="${PRIVTRIM_MAXLEN:-24576}"
EXT_RATIO="${PRIVTRIM_EXT_RATIO:-0.5}"                         # label only; text is pre-compressed

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARD="${SLURM_ARRAY_TASK_COUNT:-1}"
OUT="$REPO/out/lexcomp/shard${SHARD}"

# ---------------------------------------------------------------- resumability
if [ -f "$OUT/report.txt" ]; then
    echo "[shard $SHARD/$NSHARD] report already exists, nothing to do"
    exit 0
fi
mkdir -p "$OUT"

# ---------------------------------------------------------------- one shard
"$PY" "$REPO/slurm/shard_cases.py" "$CASES" "$OUT/cases.jsonl" "$SHARD" "$NSHARD"

"$PY" "$REPO/src/longctx/e2e_quad.py" \
    --cases    "$OUT/cases.jsonl" \
    --out      "$OUT/report.txt" \
    --json     "$OUT/cells.json" \
    --percase  "$OUT/percase.jsonl" \
    --lm       "$LM" \
    --scorer   "$REPO/src/external/squad_v11/evaluate-v2.0.py" \
    --arms     "$ARMS" \
    --ext-arms "$EXT_ARMS" \
    --ext-ratio "$EXT_RATIO" \
    --budgets  "$BUDGETS" \
    --dtype fp16 --batch 8 --max-new 64 \
    --maxlen "$MAXLEN" --maxlen-policy raise \
    --alias-question --seed 20261015

echo "[shard $SHARD/$NSHARD] done"
