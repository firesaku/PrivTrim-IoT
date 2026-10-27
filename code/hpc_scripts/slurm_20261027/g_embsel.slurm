#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-embsel
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:35:00
#SBATCH --array=0-31%16
#SBATCH --requeue
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/embsel-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/embsel-%A_%a.err
#
# å¥å‘é‡å†—ä½™åº¦ vs è¯é¢å†—ä½™åº¦ï¼š2728 ä¾‹ x 9 è‡‚ x 3 é¢„ç®—ï¼Œ--sel emb + XLM-R
# å¤šä¸€ä¸ª XLM-R å‰å‘ï¼Œæ—¶é—´æ¯” lex ç•¥é•¿ -> 25min
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=2
PY=$RD/env/bin/python
cd $RD/code/longctx || exit 1

S=$SLURM_ARRAY_TASK_ID
NSHARD=32
TAG="embsel_s${S}"

if [ -s "$RD/out/${TAG}_cells.json" ]; then
  echo "SKIP $TAG"; exit 0
fi
echo "HOST=$(hostname) $TAG START=$(date -Is)"

$PY -c "
import io
L=[x for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
n=$NSHARD; s=$S; part=L[s::n]
with io.open('$RD/data/${TAG}.jsonl','w',encoding='utf-8',newline='\n') as f:
    for x in part: f.write(x if x.endswith('\n') else x+'\n')
print('shard',s,'cases',len(part))"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
XLM=$(ls -d $RD/hf_cache/models--FacebookAI--xlm-roberta-large/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/${TAG}.jsonl \
  --out $RD/out/${TAG}_report.txt \
  --json $RD/out/${TAG}_cells.json \
  --percase $RD/out/${TAG}_percase.jsonl \
  --lm "$QWEN" --emb "$XLM" --sel emb \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms all --budgets 0.25,0.5,0.75 --alias-question \
  --maxlen 24576 --maxlen-policy raise \
  --dtype fp16 --batch 8 --max-new 64 --seed 20261015
RC=$?
echo "${TAG}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC
