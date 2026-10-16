#!/bin/bash
# Almesbar (V100) parallel path for the pair-budget fix: v2d real embeddings, both modes.
# /dpc is shared with Madar, so RD and the uploaded scripts are already visible here.
set -uo pipefail
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
MD=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/models
PY=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python
mkdir -p "$RD/logs"

echo "===== host / gpu ====="
hostname; date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -4

echo "===== resolve encoder snapshot ====="
MODEL=""
for d in "$MD"/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/*/ ; do
  if [ -f "$d/config.json" ]; then MODEL="${d%/}"; fi
done
echo "MODEL=$MODEL"
[ -z "$MODEL" ] && { echo NO_MODEL_FOUND; exit 4; }

echo "===== write v2d slurm (1xV100, fp16) ====="
cat > "$RD/v2d_fix_alm.slurm" <<EOF
#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-v2dalm
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v2dalm-%j.out
#SBATCH --error=$RD/logs/v2dalm-%j.err

set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1 || true
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

for M in legacy disjoint; do
  echo "########## MODE=\$M ##########"
  PYTHONUNBUFFERED=1 "$PY" -u v2d_gpu.py \\
    --samples "$RD/frozen_samples.jsonl" \\
    --out "$RD/v2d_alm_report_\$M.txt" \\
    --json "$RD/v2d_alm_cells_\$M.json" \\
    --model "$MODEL" \\
    --ratios 0.125,0.25,0.5 \\
    --cap 200 --batch 64 --maxlen 256 --seed 20261015 --dtype fp16 \\
    --pair-mode \$M
  echo "V2DALM_\${M}_RC=\$?"
done
echo "V2DALM_DONE"
EOF
chmod +x "$RD/v2d_fix_alm.slurm"

echo "===== submit ====="
cd "$RD"
J=$(sbatch --parsable "$RD/v2d_fix_alm.slurm" 2>&1); echo "ALM_GPU_JOB=$J"
sleep 4
squeue -u "$(id -un)" -o '%.12i %.10P %.14j %.9T %.11M %.8b %R' 2>&1
echo "ALM_SUBMIT_DONE"
