#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-xmistral
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-2%3
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/xmistral-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/xmistral-%A_%a.err
#
# ③ 跨模型泛化：把主实验的关键臂换到 Mistral-7B-Instruct-v0.3 上重跑。
#
# 为什么只需要这几个臂：要验证的是**机制结论**是否能跨模型成立——
#   alias vs full            -> 纯脱敏是否仍然有害（Qwen 上是 -0.0822）
#   alias_budget vs alias    -> 压缩本身贡献多少
#   alias_budget_link vs alias_budget -> link 机制的增益（Qwen 上是 +0.3009）
# 所以取 full / alias / alias_budget / alias_budget_link 四臂，预算固定 0.5。
#
# 规模: 300 例（与 A2/A3 同批，便于横向对比）x 4 臂 = 1200 次生成
#       按 0.43s/次 ≈ 516s 单卡 -> 3 片 x 100 例 ≈ 3 分钟/片
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
echo "HOST=$(hostname) XMISTRAL shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

MISTRAL=$(ls -d $RD/hf_cache/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/* 2>/dev/null | head -1)
echo "MISTRAL=$MISTRAL"
[ -z "$MISTRAL" ] && { echo "MISSING MISTRAL"; exit 1; }

$PY -c "
import json,io
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()][:300]
part=cs[int('$S')::int('$N')]
with io.open('$RD/data/xm_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('shard=$S cases=%d'%len(part))
"

START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/xm_s${S}.jsonl \
  --out $RD/out/xm_s${S}_report.txt \
  --json $RD/out/xm_s${S}_cells.json \
  --percase $RD/out/xm_s${S}_percase.jsonl \
  --lm "$MISTRAL" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,alias,alias_budget,alias_budget_link --budgets 0.5 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "XMISTRAL_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0
