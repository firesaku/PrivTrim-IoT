RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
PY=$B/.venv-main/bin/python
mkdir -p "$RD/aliaslink_mistral_out" "$RD/logs"

cat > "$RD/mistral.slurm" <<'EOF'
#SBATCH --account=kuex0095
#SBATCH --partition=gpu
#SBATCH --job-name=pt-mistral
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:30:00
#SBATCH --chdir=/dpc/kuex0095/100065732/privtrim_v2c_20261015
#SBATCH --output=/dpc/kuex0095/100065732/privtrim_v2c_20261015/logs/mistral-%j.out
#SBATCH --error=/dpc/kuex0095/100065732/privtrim_v2c_20261015/logs/mistral-%j.err
set -uo pipefail
echo "HOST=$(hostname) DATE=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || true
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1 "/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python" -u e2e_longctx_r4.py \
--cases "/dpc/kuex0095/100065732/privtrim_v2c_20261015/longctx_cases.jsonl" \
--out "/dpc/kuex0095/100065732/privtrim_v2c_20261015/aliaslink_mistral_out/report.txt" \
--percase "/dpc/kuex0095/100065732/privtrim_v2c_20261015/aliaslink_mistral_out/percase.jsonl" \
--json "/dpc/kuex0095/100065732/privtrim_v2c_20261015/aliaslink_mistral_out/cells.json" \
--lm "/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/models/mistral-7b-instruct-v03" \
--scorer "/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/project/external/squad_v11/evaluate-v2.0.py" \
--arms all --budgets 0.25,0.5,0.75 \
--maxlen 24576 --maxlen-policy raise --sel lex \
--dtype fp16 --batch 2 --max-new 64 --seed 20261015
rc=$?
echo "MISTRAL_RC=$rc"
echo "===== REPORT ====="
cat "/dpc/kuex0095/100065732/privtrim_v2c_20261015/aliaslink_mistral_out/report.txt" 2>&1
echo "MISTRAL_DONE"
exit $rc
EOF
echo "=== submitting mistral ==="
sbatch "$RD/mistral.slurm" 2>&1
echo "=== queue ==="
squeue --me -o "%.10i %.16j %.9T %.8M %.10R" 2>&1 | head

echo
echo "=== DRY-RUN extreme budgets 0.10,0.15 (tokenizer only, no GPU) ==="
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd "$RD"
$PY -u e2e_longctx_r4.py \
  --cases "$RD/longctx_cases.jsonl" \
  --out /tmp/dry_x.txt --json /tmp/dry_x.json --percase /tmp/dry_x.jsonl \
  --lm "$B/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28" \
  --scorer "$B/project/external/squad_v11/evaluate-v2.0.py" \
  --arms all --budgets 0.10,0.15,0.25 \
  --maxlen 24576 --maxlen-policy raise --sel lex --dry-run 2>&1 | tail -n 30
echo "DRY_RC=$?"
