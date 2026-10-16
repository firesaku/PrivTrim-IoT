#!/bin/bash
# PrivTrim-IoT 资源盘点（只读）
echo "===== 1. CLUSTER ====="
hostname
date -Is
echo
echo "===== 2. PARTITIONS ====="
sinfo -h -o '%P|avail=%a|tl=%l|nodes=%D|state=%t|cpu/node=%c|mem=%m|gres=%G' 2>&1
echo
echo "===== 3. MY ASSOC ====="
sacctmgr -nP show assoc where user=$(id -un) format=Account,Partition,QOS 2>&1
echo
echo "===== 4. MY QUEUE ====="
squeue -u $(id -un) -o '%.12i %.12P %.18j %.9T %.11M %.5D %R' 2>&1
echo
echo "===== 5. FREE GPUs ====="
sinfo -o "%22P %6a %6D %14t %20C %34G" 2>&1
echo
echo "===== 6. GPU NODES DETAIL ====="
scontrol show nodes -o 2>/dev/null | grep -oE 'NodeName=[^ ]*|Gres=[^ ]*|AllocTRES=[^ ]*|State=[^ ]*' | paste - - - - 2>/dev/null | head -40
echo
echo "===== 7. MY RECENT JOBS (7d) ====="
sacct -u $(id -un) --starttime=now-7days -o JobID,JobName%22,Partition,Account,State,Elapsed,AllocCPUS,ReqMem,AllocTRES%32 -X 2>&1 | head -30
echo
echo "===== 8. PROJECT DIR ====="
ls -la /dpc/kuex0095/ 2>&1 | head -20
echo "--- my subdir ---"
ls -la /dpc/kuex0095/100065732/ 2>&1 | head -30
echo
echo "===== 9. QUOTA ====="
df -h /dpc/kuex0095 2>&1
echo
echo "===== 10. MODULES ====="
module avail 2>&1 | head -20
echo "PROBE_DONE"
