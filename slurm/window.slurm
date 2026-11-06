#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-window
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-7%3
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/window-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/window-%A_%a.err
#
# Protection granularity: line / window / block, at fixed operation.
#
# quad4      keeps the whole record block that holds the anchor (previous result)
# quad4_win  keeps only the anchor line +/- 2 units, ignoring record markers
# quad4_win_del  the same window with the alias replaced by deletion
#
# full and quad4 are re-run in the same batch as controls: they must reproduce 0.6078 and
# 0.4937. If they do not, the harness upload changed behaviour and the window arms are void.
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
OUT=$RD/out/window_s${S}

cd $RD/code/longctx || exit 1
# resume guard: a shard that already finished is skipped
if [ -s "${OUT}_percase.jsonl" ]; then echo "shard $S already done, skip"; exit 0; fi

echo "window shard=$S/$NSHARD host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader
MODEL_DIR=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* 2>/dev/null | head -1)
[ -z "$MODEL_DIR" ] && { echo "FATAL: no qwen snapshot"; exit 2; }

$PY - <<PY
import json, io
RD = '$RD'
NL = chr(10)
allc = [json.loads(l) for l in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if l.strip()]
allc.sort(key=lambda c: c['key'])
part = allc[int('$S')::int('$NSHARD')]
with io.open(RD + '/data/window_s$S.jsonl', 'w', encoding='utf-8', newline=NL) as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + NL)
print('  shard=$S total=%d slice=%d' % (len(allc), len(part)))
PY

START=$(date +%s)
$PY -u e2e_quad.py \
  --cases $RD/data/window_s${S}.jsonl \
  --out ${OUT}_report.txt \
  --json ${OUT}_cells.json \
  --percase ${OUT}_percase.jsonl \
  --lm "$MODEL_DIR" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,quad4,quad4_win,quad4_win_del \
  --budgets 0.5 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "window_s${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC