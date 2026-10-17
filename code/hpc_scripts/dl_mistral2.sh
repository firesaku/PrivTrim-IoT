B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
PY=$B/.venv-main/bin/python
DST="$B/models/mistral-7b-instruct-v03"
LOG=/dpc/kuex0095/100065732/privtrim_v2c_20261015/logs/dl_mistral.log
mkdir -p "$DST"
echo "=== already downloading? ==="
pgrep -af "snapshot_download|Mistral-7B" 2>&1 | head -n 5
echo "=== log ==="
tail -n 6 "$LOG" 2>&1
echo "=== size ==="
du -sh "$DST" 2>&1
echo "=== starting if needed ==="
if pgrep -f "Mistral-7B-Instruct-v0.3" >/dev/null 2>&1; then
  echo "ALREADY_RUNNING"
elif [ -f "$DST/model-00001-of-00006.safetensors" ] && [ -f "$DST/config.json" ]; then
  echo "ALREADY_PRESENT"
else
  setsid nohup env PYTHONUNBUFFERED=1 $PY -c "
from huggingface_hub import snapshot_download
p = snapshot_download('mistralai/Mistral-7B-Instruct-v0.3', local_dir='$DST',
    allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','*.txt'], max_workers=4)
print('DONE', p)
" > "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null
  echo "LAUNCHED"
fi
sleep 10
echo "=== after launch ==="
tail -n 4 "$LOG" 2>&1
du -sh "$DST" 2>&1
