#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-edg2
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:08:00
#SBATCH --array=0-5
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/edg2-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/edg2-%A_%a.err
#
# 外部基线诊断 v2 —— 相对 v1 只改一件事: 把墙钟从 20min 砍到 8min。
#
# 理由: 上一轮 pt-sweep(2cpu/24G/15min) 在同样的拥堵里 23:45 就开跑了,
#       而 v1(8cpu/64G/20min) 拿到的是 Reason=Priority、预计等 11 小时。
#       回填只认"能不能塞进空洞",8min 的空洞比 20min 多得多。
#       1 个案例 + 模型加载实际只需 2-4 分钟,8min 足够;真卡死也能从日志看出来。
#
# 保留 8cpu/64G: 10-19 那次成功的 396 例用的就是 8cpu/64G,
# 而失败的 extfull 用的是 2cpu/24G —— 资源不足是嫌疑之一,先不动它。
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

METHODS=(quito_x provence cpc llmlingua2 mask_llmlingua2 dac)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "=== EDG2 method=$M host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$RD/env/bin/python -c "
import json,io
L=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
c=L[0]
with io.open('$RD/data/edg2_1.jsonl','w',encoding='utf-8',newline='\n') as f:
    f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('diag case:', c['key'], 'ctx_chars=', len(c['context']))
"

T0=$(date +%s)
timeout -k 15 400 $PY -u ext_compress.py \
  --batch1 $B1 \
  --cases $RD/data/edg2_1.jsonl \
  --outdir $RD/out/ext_diag \
  --methods $M \
  --limit 1
RC=$?
echo "=== EDG2 method=$M rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
ls -l $RD/out/ext_diag/$M 2>/dev/null
exit 0
