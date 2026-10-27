#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-pceval
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%16
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/pceval-%A_%a.out
#SBATCH --error=/path/to/project/logs/pceval-%A_%a.err
#
# 评测 Perception Compressor 的压缩产物（CPU 分区，绕开 GPU 队列）。
#
# 只评 perception 一个外部臂 + full 参照臂：其余臂（alias/alias_budget/
# alias_budget_link/quito_x/cpc/...）在 out/pubeval_* 里已经评过 300 例，
# 且案例 key 相同，可直接按 key 合并比较。
#
# 只覆盖当前已产出压缩文件的案例（driver 会跳过没有 ext 产物的 key）。
set -uo pipefail
RD=/path/to/project
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PT_ALIAS_KEY='rankfix-2026'
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=$RD/env/bin/python
S=$SLURM_ARRAY_TASK_ID
N=16

cd $RD/code/longctx || exit 1
echo "PCEVAL shard=$S/$N start=$(date -Is)"

$PY $RD/build_ext_arms_pub.py "$RD/data/public_lb.jsonl" \
    "$RD/data/ext_arms_pc.jsonl" perception 2>&1 | tail -4

# 只取有 perception 产物的案例
$PY - <<PY
import json, io, os
RD = '$RD'
cases = [json.loads(x) for x in io.open(RD + '/data/public_lb.jsonl', encoding='utf-8') if x.strip()]
have = set(f[:-5] for f in os.listdir(RD + '/out/ext_full/perception') if f.endswith('.json'))
sub = [c for c in cases if c['key'] in have]
part = sub[$S::$N]
with io.open(RD + '/data/pceval_s$S.jsonl', 'w', encoding='utf-8', newline='\n') as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')
print('  shard=$S 可用案例=%d 本片=%d' % (len(sub), len(part)))
PY

START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/pceval_s${S}.jsonl \
  --out $RD/out/pceval_s${S}_report.txt \
  --json $RD/out/pceval_s${S}_cells.json \
  --percase $RD/out/pceval_s${S}_percase.jsonl \
  --lm $(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1) \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 --alias-question \
  --ext-arms $RD/data/ext_arms_pc.jsonl \
  --ext-ratio 0.5 \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype bf16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "PCEVAL_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
