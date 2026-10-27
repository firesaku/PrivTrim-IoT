#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-extc2
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --array=0-29%30
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/extc2-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/extc2-%A_%a.err
#
# 外部基线全量压缩（5 个方法，暂缓 dac）。
#
# 与上一版(pt-extc)的唯一区别 —— 修掉一个会导致**漏跑**的竞态：
#   旧版每个 task 自己算 todo=[未压缩案例] 再 todo[S::N]；
#   各 task 启动时刻不同 -> todo 长度不同 -> 切到的是不同列表，案例会被跳过。
#   新版读 data/extc_plan.json（提交前静态生成并自检 PASS），与启动顺序无关。
#
# 分片大小 233–583 例；按实测速率（quito_x 0.42s/例、provence 0.52、cpc 2.34、
# llmlingua2 0.95、mask 0.94），最慢的 cpc 分片 p95 约 11 分钟 -> --time=15min。
#
# ext_compress.py 自带 fp.exists() 跳过，所以重跑安全、可断点续跑。
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=$RD/env-ext/bin/python
B1=$RD/research/token_compression_batch1_v1
i=$SLURM_ARRAY_TASK_ID

cd $RD/code/longctx || exit 1
echo "=== EXTC2 task=$i host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

M=$($RD/env/bin/python -c "
import json; p=json.load(open('$RD/data/extc_plan.json'))['$i']; print(p['method'])")
echo "method=$M"

$RD/env/bin/python -c "
import json,io
p=json.load(open('$RD/data/extc_plan.json'))['$i']
keys=set(p['keys'])
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
part=[c for c in cs if c['key'] in keys]
with io.open('$RD/data/extc2_t$i.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('task=$i assigned=%d materialized=%d'%(len(keys),len(part)))
"

T0=$(date +%s)
timeout -k 30 800 $PY -u ext_compress.py \
  --batch1 $B1 \
  --cases $RD/data/extc2_t$i.jsonl \
  --outdir $RD/out/ext_full \
  --methods $M
RC=$?
echo "=== EXTC2 task=$i method=$M rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
echo "=== count now: $(ls $RD/out/ext_full/$M 2>/dev/null | wc -l)"
exit 0
