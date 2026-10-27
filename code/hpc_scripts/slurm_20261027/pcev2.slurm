#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-pcev2
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --array=0-3%4
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/pcev2-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/pcev2-%A_%a.err
#
# Perception Compressor 评测（必须走 GPU：评测脚本 e2e_longctx_r5full.py
# 在第 594 行无条件调用 torch.cuda.get_device_name(0)，CPU 分区会直接崩）。
#
# 只评 perception + full 两臂；其余臂在 out/pubeval_* 已有 300 例结果，key 相同可合并。
# 案例集取"当前已有压缩产物"的那部分（随压缩作业推进而增减）。
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
S=$SLURM_ARRAY_TASK_ID
N=4

cd $RD/code/longctx || exit 1
echo "PCEV2 shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY $RD/build_ext_arms_pub.py "$RD/data/public_lb.jsonl" \
    "$RD/data/ext_arms_pc.jsonl" perception 2>&1 | tail -3

$PY - <<PY
import json, io, os
RD = '$RD'
cases = [json.loads(x) for x in io.open(RD + '/data/public_lb.jsonl', encoding='utf-8') if x.strip()]
have = set(f[:-5] for f in os.listdir(RD + '/out/ext_full/perception') if f.endswith('.json'))
sub = [c for c in cases if c['key'] in have]
part = sub[$S::$N]
with io.open(RD + '/data/pcev2_s$S.jsonl', 'w', encoding='utf-8', newline='\n') as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')
print('  可用=%d 本片=%d' % (len(sub), len(part)))
PY

START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/pcev2_s${S}.jsonl \
  --out $RD/out/pcev2_s${S}_report.txt \
  --json $RD/out/pcev2_s${S}_cells.json \
  --percase $RD/out/pcev2_s${S}_percase.jsonl \
  --lm $(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1) \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 --alias-question \
  --ext-arms $RD/data/ext_arms_pc.jsonl \
  --ext-ratio 0.5 \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "PCEV2_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
