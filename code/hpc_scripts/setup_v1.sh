#!/bin/bash
# 提交 V1 paraphrase 检查器（prod CPU），并顺带查 V3 数组状态
set -uo pipefail
RD=/path/to/100065732/privtrim_v2c_20261015
PY=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python
mkdir -p "$RD/logs"

echo "===== V3 数组状态 ====="
squeue -u "$(id -un)" -o '%.14i %.10P %.10j %.9T %.11M %.8b %.18R' 2>&1
echo "--- V3 已完成任务的报告 ---"
ls -l "$RD/v3out" 2>&1
for f in "$RD"/v3out/v3_*_report.txt; do [ -f "$f" ] && { echo "### $f"; cat "$f"; }; done

echo "===== 写 V1 slurm ====="
cat > "$RD/v1.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-v1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v1-%j.out
#SBATCH --error=$RD/logs/v1-%j.err

set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
cd "$RD"
PYTHONUNBUFFERED=1 "$PY" -u v1_checker.py \\
  --out "$RD/v1_report.txt" \\
  --json "$RD/v1_cells.json"
rc=\$?
echo "V1_RC=\$rc"
cat "$RD/v1_report.txt" 2>&1
echo "V1_DONE"
exit \$rc
EOF
chmod +x "$RD/v1.slurm"

echo "===== 提交 V1 ====="
cd "$RD"
sbatch "$RD/v1.slurm" 2>&1
sleep 4
squeue -u "$(id -un)" -o '%.14i %.10P %.10j %.9T %.11M %.8b %.18R' 2>&1
echo "SETUP1_DONE"
