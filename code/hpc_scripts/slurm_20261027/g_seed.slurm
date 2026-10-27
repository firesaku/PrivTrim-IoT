#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-seed
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:35:00
#SBATCH --array=0-63%24
#SBATCH --requeue
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/seed-%A_%a.out
#SBATCH --error=/path/to/project/logs/seed-%A_%a.err
#
# å¤š seed ç¨³å¥æ€§ï¼š2728 ä¾‹ x 9 è‡‚ x 3 é¢„ç®—ï¼Œseed={7001,7002}
# max_opt å®žæµ‹ 43ä¾‹/ç‰‡ ~8minï¼›æ­¤å¤„ seed2 ä¹Ÿ 64 ç‰‡ -> 43 ä¾‹/ç‰‡ï¼Œ--time=20min å®½è£•
# å¹‚ç­‰ï¼šå·²æœ‰ cells.json çš„åˆ†ç‰‡è‡ªåŠ¨è·³è¿‡
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=2
PY=$RD/env/bin/python
cd $RD/code/longctx || exit 1

S=$SLURM_ARRAY_TASK_ID
NSHARD=64
SEED=$(( 7001 + S / 32 ))     # 0-31 -> 7001 ; 32-63 -> 7002
SHARD=$(( S % 32 ))
TAG="seed${SEED}_s${SHARD}"

if [ -s "$RD/out/${TAG}_cells.json" ]; then
  echo "SKIP $TAG"; exit 0
fi
echo "HOST=$(hostname) $TAG SEED=$SEED START=$(date -Is)"

$PY -c "
import io
L=[x for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
n=$NSHARD//2; s=$SHARD; part=L[s::n]
with io.open('$RD/data/${TAG}.jsonl','w',encoding='utf-8',newline='\n') as f:
    for x in part: f.write(x if x.endswith('\n') else x+'\n')
print('shard',s,'cases',len(part))"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/${TAG}.jsonl \
  --out $RD/out/${TAG}_report.txt \
  --json $RD/out/${TAG}_cells.json \
  --percase $RD/out/${TAG}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms all --budgets 0.25,0.5,0.75 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 8 --max-new 64 --seed $SEED
RC=$?
echo "${TAG}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC
