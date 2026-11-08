#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-recall
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --array=0-5
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/recall-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/recall-%A_%a.err
#
# Imperfect-detector degradation: one task per detection rate rho.
#
# Task id -> rho tag. Every task runs the SAME 500 cases (paired by construction: the
# generator picks them once and only changes which protected values the router is told about).
#
#   rho = 1.0    the paper's setting, and the control: quad4 must report 0% leakage here,
#                which is what makes any leakage at lower rho attributable to the detector.
#
# Data are produced locally by src/longctx/imperfect_detector.py and uploaded to
# $RD/data/recall/. Nothing is generated on the login node.
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true

RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PT_ALIAS_KEY='rankfix-2026'
export PYTHONIOENCODING=utf-8
PY=$RD/env/bin/python

TAGS=(1 0p995 0p99 0p98 0p95 0p9)
T=${SLURM_ARRAY_TASK_ID:-0}
TAG=${TAGS[$T]}

CASES=$RD/data/recall/recall_${TAG}.jsonl
OUT=$RD/out/recall_${TAG}

if [ ! -f "$CASES" ]; then
    echo "FATAL: missing $CASES (run imperfect_detector.py locally and upload data/recall/)"
    exit 2
fi
if [ -f "$OUT/report.txt" ]; then
    echo "[recall $TAG] report exists, nothing to do"
    exit 0
fi
mkdir -p "$OUT"

MODEL_DIR=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then echo "FATAL: no Qwen2.5-7B snapshot"; exit 2; fi

cd $RD/code/longctx || exit 1
echo "recall rho=$TAG task=$T host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "model dir: $MODEL_DIR"

START=$(date +%s)
$PY -u e2e_quad.py \
  --cases   "$CASES" \
  --out     "$OUT/report.txt" \
  --json    "$OUT/cells.json" \
  --percase "$OUT/percase.jsonl" \
  --lm      "$MODEL_DIR" \
  --scorer  $RD/external/squad_v11/evaluate-v2.0.py \
  --arms    full,quad4 \
  --budgets 0.5 --alias-question --no-oracle \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 8 --max-new 64 --seed 20261015
RC=$?
echo "recall_${TAG}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC
