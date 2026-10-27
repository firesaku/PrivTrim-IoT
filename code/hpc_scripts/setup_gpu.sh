#!/bin/bash
# 建 GPU 单卡作业并提交（不占整节点）
set -uo pipefail
RD=/path/to/100065732/privtrim_v2c_20261015
MD=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/models
PY=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python
mkdir -p "$RD/logs"

echo "===== 1. 解析编码器 snapshot 路径 ====="
MODEL=""
for d in "$MD"/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/*/ ; do
  if [ -f "$d/config.json" ]; then MODEL="${d%/}"; fi
done
echo "MODEL=$MODEL"
if [ -z "$MODEL" ]; then echo "NO_MODEL_FOUND"; ls -R "$MD/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank" 2>&1 | head -20; exit 4; fi
ls -l "$MODEL" 2>&1 | head -10

echo "===== 2. 写单卡 slurm ====="
cat > "$RD/v2d.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-v2d
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v2d-%j.out
#SBATCH --error=$RD/logs/v2d-%j.err

set -uo pipefail
echo "HOST=\$(hostname) DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1 || true
cd "$RD"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1 "$PY" -u v2d_gpu.py \\
  --samples "$RD/frozen_samples.jsonl" \\
  --out "$RD/v2d_report.txt" \\
  --json "$RD/v2d_cells.json" \\
  --model "$MODEL" \\
  --ratios 0.125,0.25,0.5 \\
  --cap 200 --batch 64 --maxlen 256 --seed 20261015
rc=\$?
echo "V2D_RC=\$rc"
echo "V2D_DONE"
exit \$rc
EOF
chmod +x "$RD/v2d.slurm"
head -9 "$RD/v2d.slurm"

echo "===== 3. 提交单卡作业 ====="
cd "$RD"
sbatch "$RD/v2d.slurm" 2>&1
sleep 3
squeue -u "$(id -un)" -o '%.12i %.10P %.14j %.9T %.11M %.8b %R' 2>&1
echo "SETUP2_DONE"
