RD=/path/to/100065732/privtrim_v2c_20261015
B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
echo "=== how r4 finds the model ==="
grep -n "model" "$RD/e2e_longctx_r4.py" 2>&1 | grep -i "default\|argparse\|add_argument\|qwen\|llama\|mistral" | head -n 20
echo "=== all hf-style caches under RD ==="
find "$RD" -maxdepth 3 -type d -name "models--*" 2>/dev/null | head -n 30
echo "=== big dirs under RD ==="
du -sh "$RD"/* 2>/dev/null | sort -hr | head -n 12
echo "=== search snapshots ==="
find /path/to/100065732 -maxdepth 4 -type d -name "models--Qwen*" -o -maxdepth 4 -type d -name "models--meta-llama*" -o -maxdepth 4 -type d -name "models--mistralai*" 2>/dev/null | head -n 20
echo "=== hf_cache envs / offline dirs ==="
grep -rn "HF_HOME\|TRANSFORMERS_CACHE\|model_path\|MODEL=" "$B"/run/*.sh "$RD"/*.sh 2>/dev/null | head -n 15
