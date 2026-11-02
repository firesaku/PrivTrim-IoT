#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-pubquad
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --array=0-11%12
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/pubquad-%A_%a.out
#SBATCH --error=/path/to/project/logs/pubquad-%A_%a.err
#
# 公开基准（LongBench 三套）用新的 quad4（含锚点句窗回退）重跑。
#   任务 S -> 数据集 S/4，分片 S%4
#     0 = public_lb        (Type A, 300 例，问题不含实体)
#     1 = public_lb_typeb  (Type B, 300 例，锚点与答案同句)
#     2 = public_lb_typec  (Type C, 300 例，锚点与答案相距 6 句)
#
# 臂（5 个，全部 --no-oracle）：full, alias, alias_budget_link, quad4, quad4_del
# 规模：3 数据集 x 300 例 x 5 臂 = 4500 次生成，每任务 75 例 x 5 臂 = 375 次
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PT_ALIAS_KEY='rankfix-2026'
PY=$RD/env/bin/python
S=$SLURM_ARRAY_TASK_ID
DS=$((S / 4))
SH=$((S % 4))
N=4

case $DS in
  0) NAME=public_lb;        FILE=$RD/data/public_lb.jsonl ;;
  1) NAME=public_lb_typeb;  FILE=$RD/data/public_lb_typeb.jsonl ;;
  2) NAME=public_lb_typec;  FILE=$RD/data/public_lb_typec.jsonl ;;
  *) echo "bad ds"; exit 1 ;;
esac

cd $RD/code/longctx || exit 1
echo "PUBQUAD S=$S ds=$NAME shard=$SH host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

$PY - <<PY
import gzip, json, io, collections
NAME = '$NAME'
FILE = '$FILE'
SH = int('$SH')
N = 4
OUT = '$RD/data/pubquad_%s_s%d.jsonl' % (NAME, SH)
cases = []
with io.open(FILE, encoding='utf-8') as fh:
    for line in fh:
        if line.strip():
            cases.append(json.loads(line))
cases.sort(key=lambda c: c['key'])
part = cases[SH::N]
with io.open(OUT, 'w', encoding='utf-8', newline='\n') as f:
    for c in part:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')
print('  %s shard=%d total=%d slice=%d' % (NAME, SH, len(cases), len(part)))
PY

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_quad.py \
  --cases $RD/data/pubquad_${NAME}_s${SH}.jsonl \
  --out $RD/out/pubquad_${NAME}_s${SH}_report.txt \
  --json $RD/out/pubquad_${NAME}_s${SH}_cells.json \
  --percase $RD/out/pubquad_${NAME}_s${SH}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full,alias,alias_budget_link,quad4,quad4_del \
  --budgets 0.5 --alias-question --no-oracle \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "PUBQUAD_${NAME}_s${SH}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
