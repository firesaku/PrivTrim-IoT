#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-xmodel
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-23%3
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/xmodel-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/xmodel-%A_%a.err
#
# Cross-model generalisation. Three readers from different families/scales, two arms each
# (full and DARK), same 2,728 cases and the same deterministic sharding as every other run,
# so arms pair per case across batches.
#
# Task id -> (model, shard): model = T / 8, shard = T % 8.
#
# The full arm is the batch's own control: it must reproduce 0.6078. If a model cannot be
# loaded (remote code, architecture), the shard fails loudly rather than reporting partial
# numbers.
#
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
PY=$RD/env/bin/python

NSHARD=8
T=$SLURM_ARRAY_TASK_ID
MIDX=$((T / NSHARD))
S=$((T % NSHARD))

MODELS=(
  "models--Qwen--Qwen2.5-14B-Instruct:qwen14b"
  "models--ibm-granite--granite-3.1-8b-instruct:granite8b"
  "models--microsoft--phi-4:phi4"
)
ENTRY=${MODELS[$MIDX]}
REPO_DIR=${ENTRY%%:*}
TAG=${ENTRY##*:}

cd $RD/code/longctx || exit 1
echo "xmodel tag=$TAG shard=$S/$NSHARD host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_DIR=$(ls -d $RD/hf_cache/$REPO_DIR/snapshots/* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then echo "FATAL: no snapshot for $REPO_DIR"; exit 2; fi
echo "model dir: $MODEL_DIR"
du -sh "$MODEL_DIR" 2>/dev/null

# deterministic shard, identical to quad_full / abl_* / lexcomp so arms pair per case
$PY - <<PY
import json, io
RD = '$RD'
NL = chr(10)
allc = [json.loads(l) for l in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if l.strip()]
allc.sort(key=lambda c: c['key'])
part = allc[int('$S')::int('$NSHARD')]
with io.open(RD + '/data/xmodel_s$S.jsonl', 'w', encoding='utf-8', newline=NL) as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + NL)
print('  shard=$S total=%d slice=%d' % (len(allc), len(part)))
PY

START=$(date +%s)
$PY -u e2e_quad.py \
  --cases $RD/data/xmodel_s${S}.jsonl \
  --out $RD/out/xmodel_${TAG}_s${S}_report.txt \
  --json $RD/out/xmodel_${TAG}_s${S}_cells.json \
  --percase $RD/out/xmodel_${TAG}_s${S}_percase.jsonl \
  --lm "$MODEL_DIR" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,quad4 \
  --budgets 0.5 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "xmodel_${TAG}_s${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC