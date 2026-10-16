#!/bin/bash
echo "HOST=$(hostname)"
date -Is
echo "--- PARTITIONS ---"
sinfo -o "%22P %6a %6D %14t %12C %30G" 2>&1
echo "--- MY QUEUE ---"
squeue -u "$(id -un)" -o '%.12i %.11P %.14j %.9T %.11M %.8b %.16R' 2>&1 | head -20
echo "--- SHARED PROJECT DIR ---"
ls -d /dpc/kuex0095/100065732/privtrim_v2c_20261015 2>&1
ls -l /dpc/kuex0095/100065732/privtrim_v2c_20261015/e2e.py 2>&1
echo "--- MODEL VISIBLE? ---"
ls -d /dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/ 2>&1
echo "--- VENV ---"
P=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python
"$P" -c "import torch,sys;print('PY',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | tail -2
echo "ALMESBAR_PROBE_DONE"
