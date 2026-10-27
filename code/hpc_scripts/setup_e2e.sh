#!/bin/bash
# 提交端到端 F1 对比（单卡 H200）
set -uo pipefail
B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
RD=/path/to/100065732/privtrim_v2c_20261015
PY="$B/.venv-main/bin/python"
mkdir -p "$RD/logs" "$RD/e2eout"

LM=""; EMB=""; SC=""
for d in "$B"/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ ; do [ -f "$d/config.json" ] && LM="${d%/}"; done
for d in "$B"/models/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/*/ ; do [ -f "$d/config.json" ] && EMB="${d%/}"; done
SC="$B/project/external/squad_v11/evaluate-v2.0.py"
CASES="$B/run/qwen25_7b/cases.jsonl.gz"

echo "LM=$LM"; echo "EMB=$EMB"; echo "SC=$SC"; echo "CASES=$CASES"
ls -l "$CASES" 2>&1
[ -f "$SC" ] || { echo NO_SCORER; exit 4; }
[ -f "$CASES" ] || { echo NO_CASES; exit 5; }

cat > "$RD/e2e.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-e2e
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/e2e-%j.out
#SBATCH --error=$RD/logs/e2e-%j.err
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
  --n-per-dataset 500 --budget-frac 0.5 --max-new 64 --batch 16 --seed 20261015
rc=\$?
echo "E2E_RC=\$rc"
cat "$RD/e2eout/e2e_report.txt" 2>&1
echo "E2E_DONE"
exit \$rc
EOF
chmod +x "$RD/e2e.slurm"
cd "$RD"
JID=$(sbatch --parsable "$RD/e2e.slurm" 2>&1 | tr -d ' \r')
echo "E2E_JOB=$JID"
sleep 5
squeue -u "$(id -un)" -o '%.14i %.9P %.12j %.9T %.11M %.8b %.18R' 2>&1
echo "SETUP_E2E_DONE"
