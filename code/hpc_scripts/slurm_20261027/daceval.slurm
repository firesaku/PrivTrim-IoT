#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-dacev
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --array=0-3%4
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/dacev-%A_%a.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/dacev-%A_%a.err
#
# 最后一块拼图: 把 DAC 新补的 179 例送进评测。
#
# 背景: 全量评测(1484378)跑的时候 DAC 只有 217 例压缩产物，
#       所以 ext_dac 那行只有 217 例，与其他臂的 2728 例不可比。
#       现在 DAC 已补齐 396 例（对照集），只差这 179 例的评测。
#
# 规模: 179 例 x 13 臂 x 0.43s ≈ 1000s 单卡 -> 4 片 x 45 例 ≈ 4-5 min/片
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/dpc/kuex0095/100069961_ku/privtrim
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export NLTK_DATA=$RD/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
PY=$RD/env/bin/python
S=$SLURM_ARRAY_TASK_ID
N=4

cd $RD/code/longctx || exit 1
echo "HOST=$(hostname) DACEV shard=$S/$N start=$(date -Is)"
nvidia-smi --query-gpu=name --format=csv,noheader

# 重建清单（现在 dac 已补到 396，清单会覆盖全部可用臂）
$PY $RD/build_ext_arms.py $RD/data/ext_arms_dacev.jsonl 2>&1 | tail -3

# 找出「有 dac 压缩产物但还没有 ext_dac 评测行」的案例
$PY -c "
import json,io,glob,os
have_eval=set()
for p in glob.glob('$RD/out/extev*_s*_percase.jsonl'):
    for x in io.open(p,encoding='utf-8'):
        if x.strip():
            r=json.loads(x)
            if r['arm']=='ext_dac': have_eval.add(r['case_key'])
dac=set(f[:-5] for f in os.listdir('$RD/out/ext_full/dac') if f.endswith('.json'))
need=sorted(dac-have_eval)
cs=[json.loads(x) for x in io.open('$RD/data/longctx_max.jsonl',encoding='utf-8') if x.strip()]
sel=[c for c in cs if c['key'] in set(need)]
part=sel[int('$S')::int('$N')]
with io.open('$RD/data/dacev_s$S.jsonl','w',encoding='utf-8',newline='\n') as f:
    for c in part: f.write(json.dumps(c,ensure_ascii=False)+'\n')
print('dac_done=%d have_eval=%d need=%d shard=$S cases=%d'%(len(dac),len(have_eval),len(need),len(part)))
"

QWEN=$(ls -d $RD/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/* | head -1)
START=$(date +%s)
$PY -u e2e_longctx_r5full.py \
  --cases $RD/data/dacev_s${S}.jsonl \
  --out $RD/out/dacev_s${S}_report.txt \
  --json $RD/out/dacev_s${S}_cells.json \
  --percase $RD/out/dacev_s${S}_percase.jsonl \
  --lm "$QWEN" \
  --scorer $RD/external/squad_v11/evaluate-v2.0.py \
  --arms full --budgets 0.5 \
  --ext-arms $RD/data/ext_arms_dacev.jsonl \
  --ext-ratio 0.5 \
  --alias-question \
  --maxlen 24576 --maxlen-policy raise --sel lex \
  --dtype fp16 --batch 2 --max-new 64 --seed 20261015
RC=$?
echo "DACEV_S${S}_RC=$RC ELAPSED=$(( $(date +%s) - START ))s"
exit 0
