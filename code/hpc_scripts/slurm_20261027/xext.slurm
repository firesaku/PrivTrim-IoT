#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-xext
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:18:00
#SBATCH --array=0-2%3
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/xext-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/xext-%A_%a.err
#
# A: 8 个外部基线在 Mistral 上的评测（同 300 例）。
#
# 关键：**不需要重新压缩**。5 个方法的压缩产物已覆盖全部 2728 例
# （out/ext_full/<method>/），这里只是换生成模型（Mistral 而非 Qwen）跑评测。
#
# 规模: 300 例 / 3 片 = 100 例/片 x (5 方法 x 2 变体 + ext_full + full) ~12 臂
#       = 1200 次生成 x 0.43s ≈ 516s ≈ 9min -> --time=18min
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
PY=$RD/env/bin/python
S=$SLURM_ARRAY_TASK_ID
N=3

cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) XEXT shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

MISTRAL=$(ls -d $RD/hf_cache/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/* | head -1)
[ -z "$MISTRAL" ] && { echo MISSING; exit 1; }

$PY $RD/build_ext_arms.py $RD/data/ext_arms_xext_s${S}.jsonl \
    quito_x,provence,cpc,llmlingua2,mask_llmlingua2 2>&1 | tail -3

$PY -c "
import json,io
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()][:300]
part=cs[int('$S')::int('$N')]
with io.open('$RD/data/xext_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('shard=$S cases=%d'%len(part))
"

START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/xext_s${S}.jsonl \
  --out $RD/out/xext_s${S}_report.txt \
  --json $RD/out/xext_s${S}_cells.json \
  --percase $RD/out/xext_s${S}_percase.jsonl \
  --lm "$MISTRAL" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 \
  --ext-arms $RD/data/ext_arms_xext_s${S}.jsonl \
  --ext-ratio 0.5 \
  --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "XEXT_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0
