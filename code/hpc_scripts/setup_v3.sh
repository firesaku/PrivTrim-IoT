#!/bin/bash
# V3 作业数组：4 个配置，每任务 1 张卡，最多 2 并发（不占整节点）
set -uo pipefail
RD=/path/to/100065732/privtrim_v2c_20261015
MD=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/models
PY=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python
mkdir -p "$RD/logs" "$RD/v3out"

echo "===== 1. 解析 encoder ====="
MODEL=""
for d in "$MD"/models--microsoft--llmlingua-2-xlm-roberta-large-meetingbank/snapshots/*/ ; do
  [ -f "$d/config.json" ] && MODEL="${d%/}"
done
echo "MODEL=$MODEL"
[ -z "$MODEL" ] && { echo NO_MODEL; exit 4; }

echo "===== 2. 数据文件 ====="
ls -l "$RD"/frozen_samples*.jsonl 2>&1

echo "===== 3. 写数组 slurm ====="
cat > "$RD/v3.slurm" <<EOF
#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-v3
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --chdir=$RD
#SBATCH --output=$RD/logs/v3-%A_%a.out
#SBATCH --error=$RD/logs/v3-%A_%a.err

set -uo pipefail
CFGS=(scatter guard gran bound)
CFG=\${CFGS[\$SLURM_ARRAY_TASK_ID]}
echo "HOST=\$(hostname) TASK=\$SLURM_ARRAY_TASK_ID CFG=\$CFG DATE=\$(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>&1 || true
cd "$RD"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
PYTHONUNBUFFERED=1 "$PY" -u v3_batch.py \\
  --config "\$CFG" \\
  --datadir "$RD" \\
  --outdir "$RD/v3out" \\
  --model "$MODEL" \\
  --cap 200 --batch 64 --maxlen 256 --seed 20261015
rc=\$?
echo "V3_RC=\$rc CFG=\$CFG"
exit \$rc
EOF
chmod +x "$RD/v3.slurm"
head -10 "$RD/v3.slurm"

echo "===== 4. 提交数组 ====="
cd "$RD"
sbatch "$RD/v3.slurm" 2>&1
sleep 4
squeue -u "$(id -un)" -o '%.12i %.10P %.14j %.9T %.11M %.8b %.20R' 2>&1
echo "SETUP3_DONE"
