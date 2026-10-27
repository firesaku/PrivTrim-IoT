#!/bin/bash
# Submit the pair-budget fix verification on <CLUSTER>.
#   GPU job (1xH200, single card -- not the whole node): v2d real embeddings, both pair modes
#   CPU job (prod, 8 cores):                             v2c char-3gram, both pair modes
set -uo pipefail
RD=/path/to/100065732/privtrim_v2c_20261015
MD=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/models
PY=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python
mkdir -p "$RD/logs"

echo "===== 0. sanity ====="
"$PY" -c "import numpy;print('numpy',numpy.__version__)" 2>&1
"$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | head -3
ls -l "$RD"/frozen_samples.jsonl "$RD"/v2d_gpu.py "$RD"/v2c_feasibility.py 2>&1
"$PY" -c "
import ast,sys
for f in ['v2d_gpu.py','v2c_feasibility.py']:
    src=open('$RD/'+f,encoding='utf-8').read()
    ast.parse(src)
    assert '--pair-mode' in src, f+' missing --pair-mode'
    print('AST_OK', f, 'pair_mode_present')
" 2>&1

echo "===== 1. resolve encoder snapshot ====="
MODEL=""
for d in "$MD"/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/*/ ; do
  if [ -f "$d/config.json" ]; then MODEL="${d%/}"; fi
done
echo "MODEL=$MODEL"
if [ -z "$MODEL" ]; then echo "NO_MODEL_FOUND"; exit 4; fi

echo "===== 2. write GPU slurm (v2d, both pair modes, 1xH200) ====="
cat > "$RD/v2d_fix.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-v2dfix
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v2dfix-%j.out
#SBATCH --error=$RD/logs/v2dfix-%j.err

set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1 || true
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

echo "########## MODE=legacy (reproduce published) ##########"
PYTHONUNBUFFERED=1 "$PY" -u v2d_gpu.py \\
  --samples "$RD/frozen_samples.jsonl" \\
  --out "$RD/v2d_report_legacy.txt" \\
  --json "$RD/v2d_cells_legacy.json" \\
  --model "$MODEL" \\
  --ratios 0.125,0.25,0.5 \\
  --cap 200 --batch 64 --maxlen 256 --seed 20261015 \\
  --pair-mode legacy
echo "LEGACY_RC=\$?"

echo "########## MODE=disjoint (FIXED) ##########"
PYTHONUNBUFFERED=1 "$PY" -u v2d_gpu.py \\
  --samples "$RD/frozen_samples.jsonl" \\
  --out "$RD/v2d_report_fixed.txt" \\
  --json "$RD/v2d_cells_fixed.json" \\
  --model "$MODEL" \\
  --ratios 0.125,0.25,0.5 \\
  --cap 200 --batch 64 --maxlen 256 --seed 20261015 \\
  --pair-mode disjoint
echo "FIXED_RC=\$?"
echo "V2DFIX_DONE"
EOF
chmod +x "$RD/v2d_fix.slurm"

echo "===== 3. write CPU slurm (v2c, both pair modes, prod/8cpu) ====="
cat > "$RD/v2c_fix.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=prod
#SBATCH --job-name=pt-v2cfix
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v2cfix-%j.out
#SBATCH --error=$RD/logs/v2cfix-%j.err

set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
for M in legacy disjoint; do
  echo "########## MODE=\$M ##########"
  PYTHONUNBUFFERED=1 "$PY" -u v2c_feasibility.py \\
    --samples "$RD/frozen_samples.jsonl" \\
    --out "$RD/v2c_report_\$M.txt" \\
    --json "$RD/v2c_cells_\$M.json" \\
    --ratios 0.125,0.25,0.5 --cap 200 --seed 20261015 \\
    --pair-mode \$M
  echo "V2C_\${M}_RC=\$?"
done
echo "V2CFIX_DONE"
EOF
chmod +x "$RD/v2c_fix.slurm"

echo "===== 4. submit both ====="
cd "$RD"
J1=$(sbatch --parsable "$RD/v2d_fix.slurm" 2>&1); echo "GPU_JOB=$J1"
J2=$(sbatch --parsable "$RD/v2c_fix.slurm" 2>&1); echo "CPU_JOB=$J2"
sleep 4
echo "===== 5. queue ====="
squeue -u "$(id -un)" -o '%.12i %.10P %.14j %.9T %.11M %.8b %R' 2>&1
echo "SUBMIT_DONE"
