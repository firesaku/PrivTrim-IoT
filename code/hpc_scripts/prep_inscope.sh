RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
mkdir -p "$RD/inscope_out"
echo "=== inputs present? ==="
wc -l "$RD/inscope_target_last.jsonl" "$RD/inscope_ext_arms.jsonl" 2>&1
cat > "$RD/inscope.slurm" <<'EOF'
#!/bin/bash
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-inscope
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/dpc/kuex0095/100065732/privtrim_v2c_20261015
#SBATCH --output=/dpc/kuex0095/100065732/privtrim_v2c_20261015/logs/inscope-%j.out
#SBATCH --error=/dpc/kuex0095/100065732/privtrim_v2c_20261015/logs/inscope-%j.err
set -uo pipefail
echo "HOST=$(hostname) DATE=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1 "/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python" -u e2e_longctx_r5.py \
--cases "/dpc/kuex0095/100065732/privtrim_v2c_20261015/inscope_target_last.jsonl" \
--out "/dpc/kuex0095/100065732/privtrim_v2c_20261015/inscope_out/report.txt" \
--percase "/dpc/kuex0095/100065732/privtrim_v2c_20261015/inscope_out/percase.jsonl" \
--json "/dpc/kuex0095/100065732/privtrim_v2c_20261015/inscope_out/cells.json" \
--lm "/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28" \
--scorer "/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/project/external/squad_v11/evaluate-v2.0.py" \
--arms all --budgets 0.5 \
--maxlen 24576 --maxlen-policy raise --sel lex \
--dtype fp16 --batch 2 --max-new 64 --seed 20261015 \
--ext-arms "/dpc/kuex0095/100065732/privtrim_v2c_20261015/inscope_ext_arms.jsonl" --ext-ratio 0.5
rc=$?
echo "INSCOPE_RC=$rc"
echo "===== REPORT ====="
cat "/dpc/kuex0095/100065732/privtrim_v2c_20261015/inscope_out/report.txt" 2>&1
echo "INSCOPE_DONE"
exit $rc
EOF
head -n 1 "$RD/inscope.slurm"
echo "SLURM_BYTES=$(wc -c < "$RD/inscope.slurm")"
