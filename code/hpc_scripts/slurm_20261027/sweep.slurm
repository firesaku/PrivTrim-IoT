#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-sweep
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --array=0-31%32
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/sweep-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/sweep-%A_%a.err
#
# 预算下探: 0.05 / 0.10 / 0.15 / 0.20 —— 填补 max32 未测的区间。
#
# 动机(来自 max32 已完成的 57,288 条记录):
#   alias_budget_link 的 F1 随预算单调下降:
#     0.25 -> 0.6799 | 0.50 -> 0.6372 | 0.75 -> 0.5766
#   最好成绩出现在**测试区间的最低端点**。审稿人必问:
#     "0.25 是不是挑出来的甜点? 继续压到 0.1 会不会更好,还是会崩?"
#   这是当前唯一真正需要 GPU 才能回答的问题。
#
# 成本: 只跑 4 个臂 x 4 个预算 = 10 prompts/例(原 21),约 48% 的 max32 成本。
#       86 例 x 10 = 860 prompts, 100 + 860*0.43 ≈ 470s ≈ 8min -> --time=15min。
#
# 与 max32 保持一致以便逐例配对: 同 32 分片切法、batch 2、同 seed、同 scorer。
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
SHARD=$SLURM_ARRAY_TASK_ID
NSHARD=32

cd $RD/code/longctx || exit 1

if [ -s "$RD/out/sweep_s${SHARD}_cells.json" ]; then
  echo "SKIP shard $SHARD"
  exit 0
fi

echo "HOST=$(hostname) SHARD=$SHARD/$NSHARD START=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY -c "
import io
L=[x for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
part=L[int('$SHARD')::32]
with io.open('$RD/data/sweep_shard%d.jsonl'%int('$SHARD'),'w',encoding='utf-8',newline='\n') as f:
    for x in part: f.write(x if x.endswith('\n') else x+'\n')
print('shard','$SHARD','cases',len(part))"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/sweep_shard${SHARD}.jsonl \
  --out $RD/out/sweep_s${SHARD}_report.txt \
  --json $RD/out/sweep_s${SHARD}_cells.json \
  --percase $RD/out/sweep_s${SHARD}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,alias,alias_budget_link,budget_cluster_link \
  --budgets 0.05,0.10,0.15,0.20 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "SWEEP_S${SHARD}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC
