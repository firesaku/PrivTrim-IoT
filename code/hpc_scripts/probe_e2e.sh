#!/bin/bash
B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
echo "===== 1. models 目录全列 ====="
ls -1 "$B/models" 2>&1
echo
echo "===== 2. 找 Qwen ====="
ls -d "$B"/models/*[Qq]wen* 2>&1
for d in "$B"/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ ; do
  [ -d "$d" ] && { echo "SNAP=$d"; ls -l "$d" 2>&1 | head -12; }
done
echo
echo "===== 3. HF 缓存 ====="
ls -d ~/.cache/huggingface/hub/*/ 2>&1 | head -20
echo "HF_HOME=$HF_HOME"
echo
echo "===== 4. 官方评分器 ====="
find "$B" -maxdepth 4 -name 'evaluate-v2.0.py' 2>/dev/null | head -5
find "$B" -maxdepth 4 -iname '*squad*' -maxdepth 4 2>/dev/null | head -10
echo
echo "===== 5. 数据集 ====="
ls -l "$B/external/squad_v11/" 2>&1 | head
ls -l "$B/external/hotpotqa/" 2>&1 | head
echo
echo "===== 6. 现有 run 目录（可复用输入/金标准） ====="
ls -l "$B/run/qwen25_7b/" 2>&1 | head -12
echo
echo "===== 7. 是否有 sentence-transformers / sklearn ====="
PY="$B/.venv-main/bin/python"
"$PY" -c "import sklearn; print('sklearn', sklearn.__version__)" 2>&1 | tail -1
"$PY" -c "import sentence_transformers as s; print('st', s.__version__)" 2>&1 | tail -1
echo "PROBE_E2E_DONE"
