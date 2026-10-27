#!/bin/bash
# 快速探测：只用秒级命令
echo "===== GPU 分区 ====="
timeout 20 sinfo -p gpu -o "%22P %6a %6D %14t %12C %28G" 2>&1
echo
echo "===== 全部节点状态（简短）====="
timeout 20 sinfo -N -p gpu -o "%N %14t %12C %28G" 2>&1
echo
echo "===== 我的队列 ====="
timeout 20 squeue -u "$(id -un)" -o '%.12i %.10P %.16j %.9T %.11M %.6D %.20b %R' 2>&1
echo
echo "===== GPU 分区空闲卡估算 ====="
timeout 20 sinfo -p gpu -o "%N %t %G %C" 2>&1
echo
echo "===== 模型目录（只 ls，不 find）====="
MD=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/models
timeout 20 ls -d "$MD"/*/ 2>&1 | head -15
echo
echo "===== venv python ====="
P=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python
timeout 60 "$P" -c "import sys,torch,transformers;print('PY_OK',sys.version.split()[0],torch.__version__,transformers.__version__,torch.cuda.is_available())" 2>&1 | tail -3
echo
echo "===== 磁盘余量 ====="
timeout 20 df -h /path/to 2>&1 | tail -2
echo "FASTPROBE_DONE"
