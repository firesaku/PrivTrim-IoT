#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-probe2
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --array=0-1
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/probe2-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/probe2-%A_%a.err
#
# 新增两个外部基线的可行性探针 —— DAC 的教训：没量过显存就直接铺 30 个分片是要出事的。
# 每个 task 跑 1 个方法，取最短 / 中位 / 最长 3 个案例，报告 显存 + 耗时 + 预算是否达标。
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
METHODS=(longllmlingua selective_context)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}

echo "=== PROBE2 method=$M host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 取最短/中位/最长 3 个案例
$RD/env/bin/python -c "
import json,io
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
cs.sort(key=lambda c: len(c['context']))
sel=[cs[0], cs[len(cs)//2], cs[-1]]
with io.open('$RD/data/probe2_3.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in sel: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('probe cases (chars):', [len(c['context']) for c in sel])
"

T0=$(date +%s)
timeout -k 30 900 $PY -u $RD/ext_compress2.py \
  --batch1 $B1 \
  --cases $RD/data/probe2_3.jsonl \
  --outdir $RD/out/ext_probe2 \
  --methods $M
RC=$?
echo "=== PROBE2 method=$M rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
echo "=== RESULTS"
for f in $RD/out/ext_probe2/$M/*.json; do
  [ -f "$f" ] && $RD/env/bin/python -c "
import json,sys
o=json.load(open(sys.argv[1]))
m=o.get('matched') or {}
print('  %-42s status=%-6s orig=%-6s target=%-6s got=%-6s sec=%-7s trim=%s'
 % (sys.argv[1].split('/')[-1][:42], o.get('status'), o.get('original_tokens'),
    o.get('target_tokens'), m.get('tokens'), o.get('seconds'), o.get('adapter_trimmed_tokens')))
" "$f"
done
echo "=== ERR FILE"
tail -5 $RD/logs/probe2-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err 2>/dev/null
exit 0
