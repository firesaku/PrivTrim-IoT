#!/bin/bash
# 改投 interactive 分区（vis-91，空闲 L40S 48GB），并取消要等 6 小时的 gpu 作业
set -uo pipefail
B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
PY="$B/.venv-main/bin/python"
LM=""; EMB=""
for d in "$B"/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ ; do [ -f "$d/config.json" ] && LM="${d%/}"; done
for d in "$B"/models/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/*/ ; do [ -f "$d/config.json" ] && EMB="${d%/}"; done
SC="$B/project/external/squad_v11/evaluate-v2.0.py"
CASES="$B/run/qwen25_7b/cases.jsonl.gz"

echo "===== 1. 取消排到 23:08 的 gpu 作业 ====="
scancel 1451107 2>&1 && echo "cancelled 1451107" || echo "(already gone)"
sleep 2

echo "===== 2. 写 interactive slurm（L40S 单卡）====="
cat > "$RD/e2eint.slurm" <<EOF
#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=interactive
#SBATCH --job-name=pt-e2ei
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/e2ei-%j.out
#SBATCH --error=$RD/logs/e2ei-%j.err
set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$RD"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONUNBUFFERED=1 "$PY" -u e2e.py \\
  --cases "$CASES" \\
  --out "$RD/e2eout/e2e_report.txt" \\
  --json "$RD/e2eout/e2e_cells.json" \\
  --lm "$LM" --emb "$EMB" --scorer "$SC" \\
  --n-per-dataset 500 --budget-frac 0.5 --max-new 64 --batch 8 --seed 20261015
rc=\$?
echo "E2E_RC=\$rc"
echo "===== REPORT ====="
cat "$RD/e2eout/e2e_report.txt" 2>&1
echo "E2EI_DONE"
exit \$rc
EOF
chmod +x "$RD/e2eint.slurm"

echo "===== 3. 提交 ====="
cd "$RD"
JID=$(sbatch --parsable "$RD/e2eint.slurm" 2>&1 | tr -d ' \r')
echo "E2EINT_JOB=$JID"
echo "$JID" > "$RD/.e2eint_jid"
sleep 6
squeue -j "$JID" -o '%.14i %.12P %.10j %.9T %.11M %.8b %.18R' 2>&1
echo "SETUP_E2EINT_DONE"
