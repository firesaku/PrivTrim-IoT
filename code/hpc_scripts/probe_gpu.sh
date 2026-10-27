#!/bin/bash
# 找空闲 GPU + 盘点可用模型/环境
echo "===== 1. GPU 节点与空闲卡 ====="
sinfo -o "%22P %6a %6D %14t %20C %30G" 2>&1
echo
echo "--- 每节点 GRES 分配详情 ---"
scontrol show nodes 2>/dev/null | awk '
/NodeName=/ {n=""; g=""; a=""; s=""; c=""}
/NodeName=/ {for(i=1;i<=NF;i++) if($i ~ /^NodeName=/) n=substr($i,10)}
/ Gres=/ {for(i=1;i<=NF;i++) if($i ~ /^Gres=/) g=substr($i,6)}
/State=/ {for(i=1;i<=NF;i++) if($i ~ /^State=/) s=substr($i,7)}
/CfgTRES=/ {for(i=1;i<=NF;i++) if($i ~ /^CfgTRES=/) c=substr($i,9)}
/AllocTRES=/ {for(i=1;i<=NF;i++) if($i ~ /^AllocTRES=/) a=substr($i,11)}
/^$/ {if(n!="") print n" state="s" gres="g" cfg="c" alloc="a; n=""}
END {if(n!="") print n" state="s" gres="g" cfg="c" alloc="a}
' 2>&1 | head -20
echo
echo "===== 2. 我的队列 ====="
squeue -u "$(id -un)" -o '%.12i %.10P %.16j %.9T %.11M %.6D %b %R' 2>&1
echo
echo "===== 3. 可用 python 环境 ====="
for p in \
  /path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-main/bin/python \
  /path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/.venv-compression/bin/python ; do
  if [ -x "$p" ]; then
    echo "--- $p"
    "$p" -c "import sys,torch,transformers;print('py',sys.version.split()[0],'torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available())" 2>&1 | head -3
    "$p" -c "import sentence_transformers as s;print('sentence_transformers',s.__version__)" 2>&1 | head -2
  fi
done
echo
echo "===== 4. 本地模型目录 ====="
MD=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005/models
ls -d $MD/*/ 2>&1 | head -20
echo "--- HF 缓存 ---"
ls -d ~/.cache/huggingface/hub/*/ 2>&1 | head -20
echo
echo "===== 5. 小嵌入模型是否存在 ====="
find /path/to/100065732 -maxdepth 6 -iname '*minilm*' -o -maxdepth 6 -iname '*sentence*' -o -maxdepth 6 -iname '*bge*' -o -maxdepth 6 -iname '*e5-*' 2>/dev/null | head -10
echo "PROBE2_DONE"
