B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
PY=$B/.venv-main/bin/python
DST="$B/models/mistral-7b-instruct-v03"
LOG=/dpc/kuex0095/100065732/privtrim_v2c_20261015/logs/dl_mistral.log
echo "=== running downloaders ==="
pgrep -f "mistral-7b-instruct-v03" 2>&1 | tr '\n' ' '; echo
echo "=== killing all ==="
pkill -f "mistral-7b-instruct-v03" 2>&1; sleep 4
pgrep -f "mistral-7b-instruct-v03" 2>&1 | tr '\n' ' '; echo "(should be empty)"
echo "=== partial size ==="
du -sb "$DST" 2>&1
ls -1 "$DST" 2>&1 | head -n 20
echo "=== relaunch single ==="
setsid nohup env PYTHONUNBUFFERED=1 $PY -c "
from huggingface_hub import snapshot_download
p = snapshot_download('mistralai/Mistral-7B-Instruct-v0.3', local_dir='$DST',
    allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','*.txt'], max_workers=6)
print('DONE', p)
" > "$LOG" 2>&1 < /dev/null &
disown 2>/dev/null
echo "PID=$!"
sleep 15
echo "=== log ==="
tr '\r' '\n' < "$LOG" 2>&1 | tail -n 4
echo "=== size ==="
du -sb "$DST" 2>&1
