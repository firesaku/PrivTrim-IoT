#!/bin/bash
# 提交 V1b 诊断并等结果（prod CPU，很快）
set -uo pipefail
RD=/path/to/100065732/privtrim_v2c_20261015
PY=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python

cat > "$RD/v1b.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-v1b
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v1b-%j.out
#SBATCH --error=$RD/logs/v1b-%j.err
set -uo pipefail
cd "$RD"
PYTHONUNBUFFERED=1 "$PY" -u v1b_diag.py --out "$RD/v1b_report.txt" --nsample 30
echo "V1B_RC=\$?"
EOF
chmod +x "$RD/v1b.slurm"
cd "$RD"
JID=$(sbatch --parsable "$RD/v1b.slurm" 2>&1 | tr -d ' \r')
echo "V1B_JOB=$JID"
for i in $(seq 1 40); do
  st=$(squeue -j "$JID" -h -o '%T' 2>/dev/null | tr -d ' \r')
  [ -z "$st" ] && break
  sleep 5
done
echo "===== V1B 结果 ====="
cat "$RD/v1b_report.txt" 2>&1 | head -90
echo
echo "===== V3 数组状态 ====="
squeue -u "$(id -un)" -o '%.14i %.9P %.10j %.9T %.11M %.8b %.18R' 2>&1
echo "V1B_SETUP_DONE"
