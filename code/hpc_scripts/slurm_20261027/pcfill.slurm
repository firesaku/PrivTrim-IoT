#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-pcfill
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/pcfill-%j.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/pcfill-%j.err
#
# 补齐 perception 评测：上一轮 4 分片启动时压缩还没跑完，只覆盖了 274/300 例。
# 这里把缺的案例补评一遍（压缩已 300/300 完成）。
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PT_ALIAS_KEY='rankfix-2026'
PY=$RD/env/bin/python

cd $RD/code/longctx || exit 1
echo "PCFILL start=$(date -Is)"

$PY $RD/build_ext_arms_pub.py "$RD/data/public_lb.jsonl" \
    "$RD/data/ext_arms_pc.jsonl" perception 2>&1 | tail -3

# 找出已评过 perception 的 key，只跑缺的
$PY - <<PY
import json, io, os, glob
RD = '$RD'
done = set()
for p in glob.glob(RD + '/out/pcev2_s*_percase.jsonl'):
    for x in io.open(p, encoding='utf-8'):
        if x.strip():
            r = json.loads(x)
            if r['arm'] == 'ext_perception':
                done.add(r['case_key'])
cases = [json.loads(x) for x in io.open(RD + '/data/public_lb.jsonl', encoding='utf-8') if x.strip()]
have = set(f[:-5] for f in os.listdir(RD + '/out/ext_full/perception') if f.endswith('.json'))
miss = [c for c in cases if c['key'] in have and c['key'] not in done]
with io.open(RD + '/data/pcfill_cases.jsonl', 'w', encoding='utf-8', newline='\n') as f:
    for c in miss:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')
print('  已评=%d  有压缩产物=%d  本次补=%d' % (len(done), len(have), len(miss)))
PY

START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/pcfill_cases.jsonl \
  --out $RD/out/pcfill_report.txt \
  --json $RD/out/pcfill_cells.json \
  --percase $RD/out/pcfill_percase.jsonl \
  --lm $(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1) \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 --alias-question \
  --ext-arms $RD/data/ext_arms_pc.jsonl \
  --ext-ratio 0.5 \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
echo "PCFILL rc=$? elapsed=$(( $(date +%s) - START ))s rows=$(wc -l < $RD/out/pcfill_percase.jsonl 2>/dev/null)"
