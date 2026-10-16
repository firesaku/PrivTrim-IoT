#!/bin/bash
PY=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python
"$PY" /dpc/kuex0095/100065732/privtrim_v2c_20261015/diag_e2e.py 2>&1 | head -60
