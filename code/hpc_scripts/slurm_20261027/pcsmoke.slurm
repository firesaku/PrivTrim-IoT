#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-pcsmoke
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/pcsmoke-%j.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/pcsmoke-%j.err
#
# 冒烟测试：Perception Compressor 跑 2 例，确认
#   1) Llama-2-7b / sbert / Llama-3 tokenizer 都能加载
#   2) 三个作者函数能串起来
#   3) 输出文本确实比原文短
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/pc_stubs:$RD/repos/PerceptionCompressor:$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd $RD || exit 1
echo "PCSMOKE start=$(date -Is) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

T0=$(date +%s)
$RD/env-ext/bin/python -u run_pc.py \
  --rate 2.0 --shard 0 --nshards 1 --limit 2 \
  --outdir $RD/out/ext_full/perception_smoke
echo "PCSMOKE rc=$? elapsed=$(( $(date +%s) - T0 ))s"

echo "### 产物"
ls -la $RD/out/ext_full/perception_smoke 2>/dev/null
$RD/env-ext/bin/python - <<'PY'
import json, io, glob
for p in sorted(glob.glob('/dpc/kuex0095/100069961_ku/privtrim/out/ext_full/perception_smoke/*.json')):
    d = json.load(io.open(p, encoding='utf-8'))
    if d.get('status') != 'ok':
        print('  %-40s ERROR: %s' % (p.split('/')[-1], d.get('error', '')[:120]))
        continue
    t = d['matched']['text']
    print('  %-40s ctx_tokens=%-6d 前 160 字:' % (p.split('/')[-1], d.get('ctx_tokens', 0)))
    print('     ', t[:160].replace('\n', ' | '))
PY
echo "PCSMOKE end=$(date -Is)"
