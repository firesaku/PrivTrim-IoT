#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-extev
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --array=0-7%8
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/extev-%A_%a.out
#SBATCH --error=/path/to/project/logs/extev-%A_%a.err
#
# 外部基线评测：把已有 396 例压缩上下文真正送进 Qwen 打分。
#
# 现状: out/ext_full 有 396 例的压缩文本，但只有 112 例被评测过（extjoin_percase）。
#       data/ext_arms_all.jsonl 由 build_ext_arms.py 从这些文件重建（7122 行，12 个外部臂）。
#
# 规模: 396 例 / 8 片 = 50 例/片；每例 1 个 full 参照臂 + 12 个 ext_* 臂 = 13 次生成
#       650 次生成 x ~0.43s ≈ 280s + 启动 ~60s ≈ 6min -> --time=10min 留余量
#       形如 max32 的 (2 cpu / 24G)，实测能回填。
#
# 产物: out/extev_s<N>_percase.jsonl，臂名 ext_<method> / ext_<method>_native / full
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
PY=$RD/env/bin/python
S=$SLURM_ARRAY_TASK_ID
N=8

cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) EXTEV shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

# 只取"至少有一个外部臂"的案例，再按 shard 切
$PY -c "
import json,io,collections
have=set()
for x in io.open('$RD/data/ext_arms_all.jsonl',encoding='utf-8'):
    if x.strip():
        o=json.loads(x)
        if o['method']!='full': have.add(o['key'])
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
cs=[c for c in cs if c['key'] in have]
part=cs[int('$S')::$N]
with io.open('$RD/data/extev_shard%d.jsonl'%int('$S'),'w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('total ext cases=%d  shard=%d cases=%d'%(len(cs),int('$S'),len(part)))
"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/extev_shard${S}.jsonl \
  --out $RD/out/extev_s${S}_report.txt \
  --json $RD/out/extev_s${S}_cells.json \
  --percase $RD/out/extev_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 \
  --ext-arms $RD/data/ext_arms_all.jsonl \
  --ext-ratio 0.5 \
  --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "EXTEV_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit $RC
