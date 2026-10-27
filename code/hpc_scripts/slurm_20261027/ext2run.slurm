#!/bin/bash
# 用法: sbatch --export=ALL,EXT2_METHOD=longllmlingua,EXT2_NSHARD=16 ext2run.slurm
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-ext2run
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:12:00
#SBATCH --array=0-15%16
#SBATCH --chdir=/path/to/project
#SBATCH --output=/path/to/project/logs/ext2run-%A_%a.out
#SBATCH --error=/path/to/project/logs/ext2run-%A_%a.err
#
# 新增外部基线的全量压缩（LongLLMLingua / Selective Context）。
#
# 分片依据（实测探针）:
#   LongLLMLingua: 0.23-0.45 s/例，加载 12.3 s
#     -> 2728 例 / 16 片 = 171 例/片 ≈ 77 s 计算 + 12 s 加载 ≈ 1.5 min
#     -> 12 分钟墙钟给 8 倍余量；切这么碎是为了塞进碎片化空洞
#
# 输出写到 out/ext_full/<method>/，schema 与 ext_compress.py 一致，
# build_ext_arms.py 与 e2e 评测器无需改动即可消费。
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
M=${EXT2_METHOD:?need EXT2_METHOD}

cd $RD/code/longctx || exit 1
echo "=== EXT2 method=$M task=$i host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

$RD/env/bin/python -c "
import json,io
p=json.load(open('$RD/data/ext2_plan_$M.json'))['$i']
keys=set(p['keys'])
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
part=[c for c in cs if c['key'] in keys]
with io.open('$RD/data/ext2_${M}_t$i.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('task=$i assigned=%d materialized=%d'%(len(keys),len(part)))
"

T0=$(date +%s)
timeout -k 30 620 $PY -u $RD/ext_compress2.py \
  --batch1 $B1 \
  --cases $RD/data/ext2_${M}_t$i.jsonl \
  --outdir $RD/out/ext_full \
  --methods $M
RC=$?
echo "=== EXT2 method=$M task=$i rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
echo "=== count now: $(ls $RD/out/ext_full/$M 2>/dev/null | wc -l)"
exit 0
