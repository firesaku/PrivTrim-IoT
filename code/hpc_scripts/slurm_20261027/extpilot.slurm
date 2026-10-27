#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-extpilot
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:08:00
#SBATCH --array=0-5
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/extpilot-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/extpilot-%A_%a.err
#
# 外部基线全量化的第一步：在**从未压缩过的新案例**上验证流程并测速。
#
# 背景（已核实）：
#   out/ext_full 现有 396/2728 例（dac 217），且 ext_manifest.json 不存在
#   —— 说明任何一次 ext_compress 都没有跑到最后的汇总步骤。
#   10-20 那次 extfull 用的是**已全部完成的 396 例** + `| tail -8`，
#   所以既没有产出也没有日志，不能据此断定"卡死"。
#
# 本作业：取 longctx_max 中**不在 out/ext_full/quito_x 里**的前 2 个案例，
#   每个 task 跑 1 个方法，8 分钟墙钟（回填友好），无管道，逐条 flush 到日志。
#   产出直接写 out/ext_full（与最终全量同一目录，可续跑）。
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

echo "=== PILOT method=$M host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 选 2 个尚未压缩过的案例
$RD/env/bin/python -c "
import json,io,os
done=set(f[:-5] for f in os.listdir('$RD/out/ext_full/quito_x') if f.endswith('.json'))
out=[]
for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8'):
    if not x.strip(): continue
    c=json.loads(x)
    if c['key'] in done: continue
    out.append(c)
    if len(out)>=2: break
with io.open('$RD/data/extpilot.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in out: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('pilot cases:',[c['key'] for c in out])
"

T0=$(date +%s)
timeout -k 15 420 $PY -u ext_compress.py \
  --batch1 $B1 \
  --cases $RD/data/extpilot.jsonl \
  --outdir $RD/out/ext_full \
  --methods $M
RC=$?
echo "=== PILOT method=$M rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
ls -l $RD/out/ext_full/$M 2>/dev/null | tail -3
exit 0
