#!/bin/bash
# 补跑 gran 配置（单卡），并收集已有 V3 报告
set -uo pipefail
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
MD=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/models
PY=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python
MODEL=""
for d in "$MD"/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/*/ ; do
  [ -f "$d/config.json" ] && MODEL="${d%/}"
done
echo "MODEL=$MODEL"

cat > "$RD/v3gran.slurm" <<EOF
#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-v3gran
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v3gran-%j.out
#SBATCH --error=$RD/logs/v3gran-%j.err
set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
cd "$RD"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1 "$PY" -u v3_batch.py --config gran --datadir "$RD" --outdir "$RD/v3out" \\
  --model "$MODEL" --cap 200 --batch 64 --maxlen 256 --seed 20261015
echo "V3GRAN_RC=\$?"
EOF
chmod +x "$RD/v3gran.slurm"
cd "$RD"
JID=$(sbatch --parsable "$RD/v3gran.slurm" 2>&1 | tr -d ' \r')
echo "V3GRAN_JOB=$JID"
sleep 5
squeue -u "$(id -un)" -o '%.14i %.9P %.12j %.9T %.11M %.8b %.18R' 2>&1
echo "===== 已有 V3 报告 ====="
for c in scatter guard bound; do
  f="$RD/v3out/v3_${c}_report.txt"
  [ -f "$f" ] && { echo "########## $c"; cat "$f"; echo; }
done
echo "SETUP_GRAN_DONE"
