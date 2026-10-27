#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-dac
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --array=0-35%36
#SBATCH --dependency=afterany:1484279
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/dac-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/dac-%A_%a.err
#
# DAC 补跑：把 396 例对照集里缺的 179 例补齐，让「6 个方法在同一批案例上」成立。
#
# 排队策略（基于实测，不是拍脑袋）:
#   1) 队列里运行中的 GPU 作业有 8 个在 15 分钟内结束 -> 空洞是碎片化的
#   2) dac 加载只要 1.6-3.0 秒，切碎几乎零代价（对比 cpc 要 15.9 秒，就不该碎切）
#   3) dac 每例均值 35.9s、p95 81.8s、最大 103s -> 5 例/片 ≈ 3-9 分钟，p95 约 7 分钟
#      => 切成 36 个 5 例的小任务，任何一个 10 分钟的空洞都塞得进去
#   4) 依赖 afterany:1484279，等 5 个方法的压缩跑完再上，不与主链抢卡
#
# 缺的这 179 例比已完成的 217 例**更长**（已完成的均值 4049 token，缺的估计更高），
# 所以实际每片可能慢于 35.9s/例的均值；--time=20min 留足余量，超时也只丢个别案例，
# 幂等重跑可续。
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

cd $RD/code/longctx || exit 1      # ext_compress.py 在这里；上一轮就是漏了这行

echo "=== DAC task=$i host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

$RD/env/bin/python -c "
import json,io,os
plan=json.load(open('$RD/data/dac_plan.json'))
if '$i' not in plan:
    print('task $i not in plan'); raise SystemExit(0)
keys=set(plan['$i']['keys'])
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
part=[c for c in cs if c['key'] in keys]
with io.open('$RD/data/dac_t$i.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('task=$i assigned=%d materialized=%d'%(len(keys),len(part)))
"

T0=$(date +%s)
timeout -k 30 1050 $PY -u ext_compress.py \
  --batch1 $B1 \
  --cases $RD/data/dac_t$i.jsonl \
  --outdir $RD/out/ext_full \
  --methods dac
RC=$?
echo "=== DAC task=$i rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
echo "=== dac count now: $(ls $RD/out/ext_full/dac 2>/dev/null | wc -l)"
exit 0
