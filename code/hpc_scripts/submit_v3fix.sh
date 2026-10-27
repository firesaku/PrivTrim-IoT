#!/bin/bash
# Almesbar: corrected v3 runs -- early layer (PACT口径) + size-matched random control,
# plus a reproduction check that --layer last --rand-legacy still returns the published table.
set -uo pipefail
B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
RD=/path/to/100065732/privtrim_v2c_20261015
PY="$B/.venv-main/bin/python"
EMB="$B/models/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/ebaba9b0e874dadd3003ffcff828e4397e568089"
mkdir -p "$RD/logs" "$RD/v3out2"

echo "===== verify patched v3 ====="
"$PY" -c "import ast;src=open('$RD/v3_batch.py',encoding='utf-8').read();ast.parse(src);print('AST_OK rand_min_gap', ('--rand-min-gap' in src), 'layer', ('--layer' in src))" 2>&1

cat > "$RD/v3fix.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-v3fix
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v3fix-%A_%a.out
#SBATCH --error=$RD/logs/v3fix-%A_%a.err
set -uo pipefail
CFGS=(scatter guard gran bound bound bound)
EXTRA=("" "" "" "--rand-min-gap 2" "--rand-min-gap 0" "--layer last --rand-legacy")
TAG=(scatter_early guard_early gran_early bound_early_gap2 bound_early_gap0 bound_LAST_legacy)
i=\$SLURM_ARRAY_TASK_ID
CFG=\${CFGS[\$i]}; EX=\${EXTRA[\$i]}; TG=\${TAG[\$i]}
echo "HOST=\$(hostname) TASK=\$i CFG=\$CFG EXTRA=[\$EX] TAG=\$TG DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
cd "$RD"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RD/v3out2/\$TG"
PYTHONUNBUFFERED=1 "$PY" -u v3_batch.py \\
  --config "\$CFG" --datadir "$RD" --outdir "$RD/v3out2/\$TG" --model "$EMB" \\
  --cap 200 --batch 64 --maxlen 256 --seed 20261015 \$EX
rc=\$?
echo "V3FIX_RC=\$rc TAG=\$TG"
exit \$rc
EOF
chmod +x "$RD/v3fix.slurm"

cd "$RD"
J=$(sbatch --parsable --array=0-5 "$RD/v3fix.slurm" 2>&1); echo "ALM_V3FIX_JOB=$J"
sleep 5
squeue -u "$(id -un)" -o '%.12i %.9P %.12j %.9T %.11M %.8b %R' 2>&1
echo "V3FIX_SUBMIT_DONE"
