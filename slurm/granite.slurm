#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-granite
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%3
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/granite-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/granite-%A_%a.err
#
# Cross-family check on IBM Granite 3.1 8B.
#
# The reader was chosen by measurement, not by size: on a 300-case probe this model answered
# UNKNOWN on 1.5% of multi-entity cases under the aliased prompt, against 16.5% for phi-4 and
# 96.5% for Qwen2.5-14B. A reader that refuses the aliased handle cannot show the rule at all,
# and the probe is cheap (3 minutes per model) compared with a full run.
#
# Arms: full (control, must reproduce the probe's healthy uncompressed score) and quad4.
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
S=$SLURM_ARRAY_TASK_ID
NSHARD=8

cd $RD/code/longctx || exit 1
echo "granite shard=$S/$NSHARD host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

MODEL_DIR=$(ls -d $RD/hf_cache/models--ibm-granite--granite-3.1-8b-instruct/snapshots/* 2>/dev/null | head -1)
if [ -z "$MODEL_DIR" ]; then echo "FATAL: no granite snapshot"; exit 2; fi
echo "model dir: $MODEL_DIR"

# deterministic shard, identical to every other batch so arms pair per case
$PY - <<PY
import json, io
RD = '$RD'
NL = chr(10)
allc = [json.loads(l) for l in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if l.strip()]
allc.sort(key=lambda c: c['key'])
part = allc[int('$S')::int('$NSHARD')]
with io.open(RD + '/data/granite_s$S.jsonl', 'w', encoding='utf-8', newline=NL) as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + NL)
print('  shard=$S total=%d slice=%d' % (len(allc), len(part)))
PY

START=$(date +%s)
$PY -u e2e_quad.py \
  --cases $RD/data/granite_s${S}.jsonl \
  --out $RD/out/granite_s${S}_report.txt \
  --json $RD/out/granite_s${S}_cells.json \
  --percase $RD/out/granite_s${S}_percase.jsonl \
  --lm "$MODEL_DIR" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,quad4 \
  --budgets 0.5 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "granite_s${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC