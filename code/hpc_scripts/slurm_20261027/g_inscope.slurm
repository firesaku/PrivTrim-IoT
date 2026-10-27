#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-inscope
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:08:00
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/inscope-%j.out
#SBATCH --error=/path/to/project/logs/inscope-%j.err
#
# in-scope F1 æ‰“åˆ†ï¼š19 ä¾‹ x 9 è‡‚ x 3 é¢„ç®— + å¤–éƒ¨åŸºçº¿è‡‚ï¼ˆinscope_ext_armsï¼‰
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
PY=$RD/env/bin/python
cd $RD/code/longctx || exit 1

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)

for V in target_last target_first multi_cases; do
  TAG="inscope_${V}"
  [ -s "$RD/out/${TAG}_cells.json" ] && { echo "SKIP $TAG"; continue; }
  EXT=""
  [ "$V" = "multi_cases" ] && EXT="--ext-arms $RD/data/inscope_ext_arms.jsonl --ext-ratio 0.5"
  echo "=== $TAG START=$(date -Is) ==="
  $PY -u e2e_longctx_r5full.py \
    --cases $RD/data/inscope_${V}.jsonl \
    --out $RD/out/${TAG}_report.txt \
    --json $RD/out/${TAG}_cells.json \
    --percase $RD/out/${TAG}_percase.jsonl \
    --lm "$QWEN" \
    --scorer $RD/external/squad_v11/evaluate-v2.0.py \
    --arms all --budgets 0.25,0.5,0.75 --alias-question \
    $EXT \
    --maxlen 24576 --maxlen-policy raise --sel lex \
    --dtype fp16 --batch 8 --max-new 64 --seed 20261015
  echo "${TAG}_RC=$?"
done
echo INSCOPE_ALL_DONE
