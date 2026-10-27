#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-evint
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --array=0-1%2
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/evint-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/evint-%A_%a.err
#
# INTACT 匿名化文本的 QA 评测 —— 唯一目的是拿到 F1，
# 好和「零泄漏」阵营里的我们自己（F1 0.6888 / 长度 20%）做等隐私对比。
#
# 规模: 100 例 / 2 片 = 50 例/片 x (ext_intact + ext_full + full) 3 臂
#       = 150 次生成 x 0.43s ≈ 65s -> --time=15min 极宽松
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
N=2

cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) EVINT shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

# 只收 intact 一个方法，避免把其他臂重算
$PY $RD/build_ext_arms.py $RD/data/ext_arms_intact_s${S}.jsonl intact 2>&1 | tail -4

$PY -c "
import json,io,os
ks=[f[:-5] for f in os.listdir('$RD/out/ext_full/intact') if f.endswith('.json')]
ks=sorted(ks)[int('$S')::int('$N')]
cset=set(ks)
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
part=[c for c in cs if c['key'] in cset]
with io.open('$RD/data/evint_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('shard=$S cases=%d'%len(part))
"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/evint_s${S}.jsonl \
  --out $RD/out/evint_s${S}_report.txt \
  --json $RD/out/evint_s${S}_cells.json \
  --percase $RD/out/evint_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 \
  --ext-arms $RD/data/ext_arms_intact_s${S}.jsonl \
  --ext-ratio 0.5 \
  --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "EVINT_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0
