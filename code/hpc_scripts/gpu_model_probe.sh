B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
echo "=== partitions ==="
sinfo -o "%.14P %.6a %.10l %.6D %.6t %.20G %.20N" 2>&1 | head -n 12
echo "=== idle gpus ==="
sinfo -N -o "%.18N %.12P %.6t %.20G" 2>&1 | grep -i "idle\|mix" | head -n 20
echo "=== queue depth ==="
squeue -o "%.10i %.10u %.9T %.20G" 2>&1 | awk 'NR==1{print} NR>1{c[$3]++} END{for(k in c) print k, c[k]}'
echo "=== local model dirs ==="
for d in "$B/models" "$B/../models" "$B/project/models" /path/to/100065732/models; do
  [ -d "$d" ] && { echo "--- $d"; ls -1 "$d" 2>&1 | head -n 20; }
done
echo "=== hf cache ==="
ls -1 ~/.cache/huggingface/hub 2>&1 | head -n 20
echo "=== where is qwen25_7b ==="
ls -ld $B/models/* 2>&1 | head -n 10
