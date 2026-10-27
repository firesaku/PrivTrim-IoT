#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-b8chk
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/b8chk-%j.out
#
# batch=8 对照实验: 只跑分片 0,输出写 b8_s0_*,不覆盖 max32_s0_*。
# 目的: 与已有 max32_s0(batch=2)逐例比对,回答两件事
#   1) 提速倍数(generation 占单片 89%)
#   2) 贪婪解码在 fp16 下是否**批不变** —— 若输出有差异,
#      跨 batch 的 run 就不可比,提速方案要作废。
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
cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) BATCH8 START=$(date -Is)"
$PY -c "
import io
L=[x for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
part=L[0::32]
with io.open('$RD/data/b8_shard0.jsonl','w',encoding='utf-8',newline='\n') as f:
    for x in part: f.write(x if x.endswith('\n') else x+'\n')
print('shard 0 cases',len(part))"
QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/b8_shard0.jsonl \
  --out $RD/out/b8_s0_report.txt \
  --json $RD/out/b8_s0_cells.json \
  --percase $RD/out/b8_s0_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms all --budgets 0.25,0.5,0.75 --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 8 --max-new 64 --seed 20261015
RC=$?
echo "B8_S0_RC=$RC ELAPSED=$(( $(date +%s) - START ))s  (batch2 baseline = 867s)"
exit $RC
