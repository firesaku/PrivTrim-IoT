B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
PY=$B/.venv-main/bin/python
echo "=== net test ==="
timeout 20 curl -sS -o /dev/null -w "hf_status=%{http_code} time=%{time_total}\n" https://huggingface.co/api/models/Qwen/Qwen2.5-3B-Instruct 2>&1
timeout 20 curl -sS -o /dev/null -w "hfmirror=%{http_code} time=%{time_total}\n" https://hf-mirror.com/api/models/Qwen/Qwen2.5-3B-Instruct 2>&1
echo "=== hf_hub version ==="
$PY -c "import huggingface_hub as h; print('hf_hub', h.__version__)" 2>&1 | tail -n 2
echo "=== caches on disk ==="
ls -1 /dpc/kuex0095/100065732/privtrim_iot_hpc_20260921/models/ 2>&1 | head
echo "=== sizes ==="
du -sh /dpc/kuex0095/100065732/privtrim_iot_hpc_20260921/models/* 2>/dev/null | head
echo "=== quota ==="
df -h /dpc/kuex0095/100065732 2>&1 | tail -n 2
echo "=== which model does r4 default to ==="
grep -n "Qwen\|mkdir\|-m \|--model" /dpc/kuex0095/100065732/privtrim_v2c_20261015/submit_aliaslink.sh 2>&1 | head -n 10
[ -f /dpc/kuex0095/100065732/privtrim_v2c_20261015/../../privtrim_v2c_20261015/submit_aliaslink.sh ] && echo found
find /dpc/kuex0095/100065732 -maxdepth 3 -name "submit_aliaslink.sh" 2>/dev/null
