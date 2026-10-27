B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
PY=$B/.venv-main/bin/python
DST="$B/models/mistral-7b-instruct-v03"
LOG=/path/to/100065732/privtrim_v2c_20261015/logs/dl_mistral.log
mkdir -p "$DST"
echo "=== existing? ==="
ls -1 "$DST" 2>&1 | head -n 5
du -sh "$DST" 2>&1
echo "=== launching background download ==="
if [ -f "$DST/config.json" ] && [ -f "$DST/model.safetensors.index.json" ]; then
  echo "ALREADY_PRESENT"
else
  cd "$B" && nohup env HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1 \
    $PY -c "
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    local_dir='$DST',
    local_dir_use_symlinks=False,
    allow_patterns=['*.json','*.safetensors','*.model','tokenizer*','*.txt'],
    max_workers=4,
)
print('DONE', p)
" > "$LOG" 2>&1 &
  echo "launched pid=$!"
fi
sleep 8
echo "=== log head ==="
tail -n 5 "$LOG" 2>&1
echo "=== size now ==="
du -sh "$DST" 2>&1
