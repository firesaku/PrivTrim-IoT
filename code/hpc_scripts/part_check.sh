#!/bin/bash
echo "===== interactive 分区 ====="
sinfo -p interactive -o "%N %t %C %G %m" 2>&1
scontrol show partition interactive 2>&1 | head -20
echo
echo "===== 我的作业预计启动时间 ====="
squeue -j 1451107 --start 2>&1
squeue -u "$(id -un)" -o '%.14i %.9P %.12j %.9T %.11M %.20S %.18R' 2>&1
echo
echo "===== l40s 节点详情 ====="
scontrol show node vis-91 2>&1 | head -20
echo
echo "===== 我的优先级 vs 别人 ====="
sprio -j 1451107 2>&1
echo
echo "===== gpu 分区在跑什么 ====="
squeue -p gpu -o '%.14i %.10u %.14j %.9T %.11M %.6D %b' 2>&1 | head -25
echo "PART_CHECK_DONE"
