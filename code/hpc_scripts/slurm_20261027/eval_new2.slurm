#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-ev2new
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:22:00
#SBATCH --array=0-15%16
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/ev2new-%A_%a.out
#SBATCH --error=/path/to/project/logs/ev2new-%A_%a.err
#
# 增量评测：只评新增的两个外部臂（longllmlingua / selective_context）。
#
# 为什么单独跑而不是重跑全量：已有的 12 个臂在 2728 例上都评过了，
# 重跑会把 38k 次生成再算一遍（约 4.2 GPU-h）。用 build_ext_arms.py 的
# ONLY 过滤只保留新增臂，成本降到 2728 x 2 ≈ 5.5k 次生成 ≈ 0.65 GPU-h。
#
# 规模: 16 片 = 171 例/片 x 4 臂 = 684 次生成；实测 1364 次要 14min，故 684 次约 7min -> --time=22min 留 3 倍余量
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
PY=$RD/env/bin/python
S=$SLURM_ARRAY_TASK_ID
N=16
NEWM="longllmlingua,selective_context"

cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) EV2NEW shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

# 每个 task 写自己的清单，且只含新增臂
$PY $RD/build_ext_arms.py $RD/data/ext_arms_new_s${S}.jsonl "$NEWM" 2>&1 | tail -4

$PY -c "
import json,io
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
part=cs[int('$S')::int('$N')]
with io.open('$RD/data/ev2new_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('shard=$S cases=%d'%len(part))
"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/ev2new_s${S}.jsonl \
  --out $RD/out/ev2new_s${S}_report.txt \
  --json $RD/out/ev2new_s${S}_cells.json \
  --percase $RD/out/ev2new_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 \
  --ext-arms $RD/data/ext_arms_new_s${S}.jsonl \
  --ext-ratio 0.5 \
  --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "EV2NEW_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0
