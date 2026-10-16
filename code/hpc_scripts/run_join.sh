#!/bin/bash
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
PY=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python
echo "--- 逐例文件 ---"
ls -l "$RD/e2eout/e2e2_percase.jsonl" 2>&1
wc -l "$RD/e2eout/e2e2_percase.jsonl" 2>&1
echo "--- 跑 join ---"
"$PY" "$RD/join_baseline.py" 2>&1 | tail -60
echo "JOIN_RUN_DONE"
