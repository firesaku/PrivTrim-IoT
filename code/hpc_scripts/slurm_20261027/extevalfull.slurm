#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-extevfull
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --array=0-23%24
#SBATCH --dependency=afterany:1484279
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/extevfull-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/extevfull-%A_%a.err
#
# 全量外部评测 —— 排在压缩作业 1483708 之后自动启动（afterok 依赖）。
#
# 这样整条链在 20:47 的释放窗口里一次跑完，不需要人工盯着再提交：
#   1483708 (压缩 5 方法, 30x20min)  ->  afterok  ->  本作业 (评测, 24x20min)
#
# 每个 task 启动时先重建 data/ext_arms_all.jsonl（幂等、零成本），
# 所以拿到的一定是压缩刚跑完的最新覆盖；压缩哪部分没成，这里就自动少评那部分。
#
# 规模: 2728 例 x (12 个 ext 臂 + 1 个 full 参照) = 35,464 次生成
#       按 0.43s/次 ≈ 4.2 GPU-小时；24 片 x 114 例 x 13 次 x 0.43s ≈ 10.6 min/片
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
N=24

cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) EXTEVFULL shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

# 重建清单：每个 task 写自己的副本，避免 24 个进程同时覆盖同一文件
$PY $RD/build_ext_arms.py $RD/data/ext_arms_s${S}.jsonl 2>&1 | tail -4

$PY -c "
import json,io
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
part=cs[int('$S')::int('$N')]
with io.open('$RD/data/extevfull_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('shard=$S cases=%d'%len(part))
"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/extevfull_s${S}.jsonl \
  --out $RD/out/extevfull_s${S}_report.txt \
  --json $RD/out/extevfull_s${S}_cells.json \
  --percase $RD/out/extevfull_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 \
  --ext-arms $RD/data/ext_arms_s${S}.jsonl \
  --ext-ratio 0.5 \
  --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "EXTEVFULL_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0
