#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-typec
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --array=0-3%4
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/typec-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/typec-%A_%a.err
#
# Type C：公开上下文 + 「锚点单元 ≠ 答案承载单元」的两跳任务。
#
# 这是**唯一**能真正检验 link 机制的公开实验：
#   Type A（原问题）      : 问题不引用 PII，link 无事可做
#   Type B（同行记录）    : 账号与设备同一行 -> 锚点单元已含答案，
#                           protect_of 无增量 -> 两臂必然同值（已验证 300/300）
#   Type C（两跳，分离）  : 账号在 directory 行、设备在 registry 行，
#                           link 保护锚点行才有意义
#
# 只跑我们自己的 4 个臂，不需要外部方法压缩。
# 规模: 300 例 / 4 片 = 75 例/片 x 4 臂 = 300 次生成 ≈ 2-3min
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
N=4

cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) TYPEC shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY -c "
import json,io
cs=[json.loads(x) for x in io.open('$RD/data/public_lb_typec.jsonl',encoding='utf-8') if x.strip()]
part=cs[int('$S')::int('$N')]
with io.open('$RD/data/typec_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('shard=$S cases=%d'%len(part))
"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/typec_s${S}.jsonl \
  --out $RD/out/typec_s${S}_report.txt \
  --json $RD/out/typec_s${S}_cells.json \
  --percase $RD/out/typec_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,alias,alias_budget,alias_budget_link --budgets 0.5 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "TYPEC_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0
