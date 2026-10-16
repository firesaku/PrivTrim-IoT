#!/bin/bash
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
JID=${1:-1444010}
echo "===== 该作业状态 ====="
squeue -j "$JID" -o '%.12i %.10P %.14j %.9T %.11M %.8b %.20R' 2>&1
echo "===== 该作业记账 ====="
sacct -j "$JID" -o JobID,JobName%12,Partition,State,ExitCode,Elapsed,AllocCPUS,AllocTRES%40 -X 2>&1
echo "===== 我的全部队列 ====="
squeue -u "$(id -un)" -o '%.12i %.10P %.14j %.9T %.11M %.8b %.20R' 2>&1
echo "===== GPU 空闲情况 ====="
sinfo -p gpu -o "%N %t %C %G" 2>&1
echo "===== 文件 ====="
ls -l "$RD" 2>&1
echo "===== GPU 输出尾部 ====="
tail -30 "$RD/logs/v2d-$JID.out" 2>&1
echo "===== GPU 错误尾部 ====="
tail -20 "$RD/logs/v2d-$JID.err" 2>&1
echo "===== 报告（若有）====="
if [ -f "$RD/v2d_report.txt" ]; then cat "$RD/v2d_report.txt"; else echo "(尚未生成)"; fi
echo "CHECK2_DONE"
