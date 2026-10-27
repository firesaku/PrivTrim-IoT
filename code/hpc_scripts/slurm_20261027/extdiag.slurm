#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-extdiag
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --array=0-5
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/extdiag-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/extdiag-%A_%a.err
#
# 诊断 ext_compress.py 为什么在 40 分钟内零产出。
# 上一轮 extfull(1479745) 的致命错误: 命令行是
#     python -u ext_compress.py ... 2>&1 | tail -8
# tail 要等 stdin 关闭才输出,进程被 TIMEOUT 杀掉时它从未 flush,
# 所以 40 分钟的日志里只有 HOST/shard/nvidia-smi 三行。
# 本作业: 一个方法一个 task,不用管道,加 timeout 让卡住的方法自己退出(rc=124)。
#
# 产物写 out/ext_diag,与 out/ext_full 隔离,保证不会被 fp.exists() 跳过。
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

echo "=== DIAG method=$M host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== free -g"; free -g | head -2

# 取 1 个案例;用 longctx_max 的第 0 条(属于 396 子集,便于和已有结果对照)
$RD/env/bin/python -c "
import json,io
L=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
c=L[0]
with io.open('$RD/data/extdiag_1.jsonl','w',encoding='utf-8',newline='\n') as f:
    f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('diag case:', c['key'], 'ctx_chars=', len(c['context']))
"

T0=$(date +%s)
timeout -k 30 900 $PY -u ext_compress.py \
  --batch1 $B1 \
  --cases $RD/data/extdiag_1.jsonl \
  --outdir $RD/out/ext_diag \
  --methods $M \
  --limit 1
RC=$?
echo "=== DIAG method=$M rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
echo "=== produced:"; ls -l $RD/out/ext_diag/$M 2>/dev/null
exit 0
