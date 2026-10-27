#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-extc
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --array=0-29%30
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/extc-%A_%a.out
#SBATCH --error=/path/to/project/logs/extc-%A_%a.err
#
# 外部基线全量压缩（5 个方法，**不含 dac**）。
#
# 成本依据（从 out/ext_full 已有 396 例的 compression_seconds 实测，见 extcost.py）:
#   quito_x 0.35s/例  provence 0.45s/例  cpc 1.52s/例  llmlingua2 0.86s/例  mask 0.84s/例
#   剩下 2332 例合计仅 ~3.0 GPU-小时。
#   dac 是 35.94s/例（p95 81.8s），全量要 27 GPU-小时 —— 单独处理，见报告。
#
# 分片: 每个 task 只跑 1 个方法，按 stride 切「尚未压缩」的案例，
#       与已有产物无重叠；ext_compress.py 自身也有 fp.exists() 跳过，双重幂等。
#       5 个方法的已完成集合不同（dac 只有 217），所以每个方法各自过滤。
#
# task -> (method, shard, nshards):
#   0-3   quito_x          n=4
#   4-7   provence         n=4
#   8-17  cpc              n=10
#   18-23 llmlingua2       n=6
#   24-29 mask_llmlingua2  n=6
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
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
if   [ $i -le 3 ];  then M=quito_x;         S=$((i-0));  N=4
elif [ $i -le 7 ];  then M=provence;        S=$((i-4));  N=4
elif [ $i -le 17 ]; then M=cpc;             S=$((i-8));  N=10
elif [ $i -le 23 ]; then M=llmlingua2;      S=$((i-18)); N=6
else                     M=mask_llmlingua2; S=$((i-24)); N=6
fi

echo "=== EXTC method=$M shard=$S/$N task=$i host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

# 该方法尚未压缩的案例，按 stride 切
$PY -c "
import json,io,os
done=set(f[:-5] for f in os.listdir('$RD/out/ext_full/$M') if f.endswith('.json'))
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
todo=[c for c in cs if c['key'] not in done]
part=todo[int('$S')::int('$N')]
with io.open('$RD/data/extc_${M}_${S}.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('method=$M todo_total=%d shard_cases=%d'%(len(todo),len(part)))
"

T0=$(date +%s)
timeout -k 30 1100 $PY -u ext_compress.py \
  --batch1 $B1 \
  --cases $RD/data/extc_${M}_${S}.jsonl \
  --outdir $RD/out/ext_full \
  --methods $M
RC=$?
echo "=== EXTC method=$M shard=$S rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
echo "=== count now: $(ls $RD/out/ext_full/$M 2>/dev/null | wc -l)"
exit 0
