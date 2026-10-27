#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-typeb
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --array=0-3%4
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/typeb-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/typeb-%A_%a.err
#
# Type B 条件：公开上下文 + **引用注入 PII** 的合成问题。
#
# 与 pubeval 的唯一差别是 --cases 换成 data/public_lb_typeb.jsonl
# （同 key、同上下文，只换了 question/answers）。
# 因此外部方法的压缩产物**完全复用**，不需要重压。
#
# 目的：检验第 11 节的解释——自建集优势来自"问题点名敏感标识符"这一设计。
#   link 增益恢复为正 -> 解释成立
#   仍为 0          -> 解释不成立，机制在公开上下文上另有问题
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
echo "HOST=$(hostname) TYPEB shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY $RD/build_ext_arms_pub.py "$RD/data/public_lb_typeb.jsonl" \
    "$RD/data/ext_arms_typeb_s${S}.jsonl" \
    quito_x,provence,cpc,llmlingua2,mask_llmlingua2 2>&1 | tail -8

$PY -c "
import json,io
cs=[json.loads(x) for x in io.open('$RD/data/public_lb_typeb.jsonl',encoding='utf-8') if x.strip()]
part=cs[int('$S')::int('$N')]
with io.open('$RD/data/typeb_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('shard=$S cases=%d'%len(part))
"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/typeb_s${S}.jsonl \
  --out $RD/out/typeb_s${S}_report.txt \
  --json $RD/out/typeb_s${S}_cells.json \
  --percase $RD/out/typeb_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,alias,alias_budget,alias_budget_link --budgets 0.5 --alias-question \
  --ext-arms $RD/data/ext_arms_typeb_s${S}.jsonl \
  --ext-ratio 0.5 \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "TYPEB_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0
