#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-extsmoke
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:35:00
#SBATCH --chdir=/dpc/kuex0095/100069961_ku/privtrim
#SBATCH --output=/dpc/kuex0095/100069961_ku/privtrim/logs/extsmoke-%j.out
#SBATCH --error=/dpc/kuex0095/100069961_ku/privtrim/logs/extsmoke-%j.err
#
# Smoke test for the ratio-parameterised copy of the published compressors.
#
# Two things must be true before any sweep is worth submitting:
#   1. env-ext can load and run the published methods on this node at all;
#   2. the one-line budget patch actually takes effect, i.e. PT_EXT_RATIO=0.1 yields ~10% of the
#      tokens rather than 50%.
#
# Both are checked here on four cases with two methods. The archived 0.5 behaviour is the control:
# if the 0.5 pass does not also produce the expected token counts, the patch broke something and the
# sweep must not run.
#
# The published code is NOT modified. $RD/code/batch1_ratio is a copy whose only diff from
# $RD/research/token_compression_batch1_v1 is the budget expression, and the patch reads an
# environment variable that defaults to 0.5.
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true

RD=/dpc/kuex0095/100069961_ku/privtrim
B1=$RD/code/batch1_ratio
export PYTHONNOUSERSITE=1
export PYTHONPATH=$RD/code:$RD/code/longctx
export HF_HOME=$RD/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PYX=$RD/env-ext/bin/python

if [ ! -f "$B1/compress.py" ]; then echo "FATAL: no patched batch1 at $B1"; exit 2; fi
if [ ! -x "$PYX" ]; then echo "FATAL: no env-ext python at $PYX"; exit 2; fi

echo "extsmoke host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "--- env-ext ---"
$PYX -c "import torch,sys; print('python',sys.version.split()[0]); print('torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | head -5

# four cases only: fastest thing that can still catch a broken loader
$PYX - <<'PY'
import json, io
RD='/dpc/kuex0095/100069961_ku/privtrim'
allc=[json.loads(l) for l in io.open(RD+'/data/longctx_max.jsonl',encoding='utf-8') if l.strip()]
allc.sort(key=lambda c:c['key'])
pick=allc[::700][:4]
with io.open(RD+'/data/extsmoke.jsonl','w',encoding='utf-8',newline=chr(10)) as f:
    for c in pick:
        f.write(json.dumps(c,ensure_ascii=False)+chr(10))
print('extsmoke cases: %d  keys=%s' % (len(pick),[c['key'] for c in pick]))
PY

cd $RD/code/longctx || exit 1
for RATIO in 0.5 0.1; do
    TAG=$(echo $RATIO | tr '.' 'p')
    echo "=== ratio=$RATIO (tag=$TAG) ==="
    PT_EXT_RATIO=$RATIO $PYX -u ext_compress.py \
        --batch1 "$B1" \
        --cases  $RD/data/extsmoke.jsonl \
        --outdir $RD/out/extsmoke_$TAG \
        --methods quito_x,cpc 2>&1 | tail -25
    echo "--- ratio=$RATIO token accounting ---"
    $PYX - <<PY
import json,glob,os
for m in ('quito_x','cpc'):
    fs=sorted(glob.glob('$RD/out/extsmoke_$TAG/'+m+'/*.json'))
    for f in fs[:2]:
        r=json.load(open(f,encoding='utf-8'))
        print('  %-8s %-30s orig=%5d target=%5d matched=%5d native=%5d'
              % (m, r.get('id','')[-24:], r.get('original_tokens',-1),
                 r.get('target_tokens',-1), (r.get('matched') or {}).get('tokens',-1),
                 (r.get('native') or {}).get('tokens',-1)))
PY
done
echo "extsmoke done"
