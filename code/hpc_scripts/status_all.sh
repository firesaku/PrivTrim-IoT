#!/bin/bash
RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
echo "===== QUEUE ====="
squeue -u "$(id -un)" -o '%.14i %.9P %.10j %.9T %.11M %.8b %.18R' 2>&1
echo
echo "===== GPU 空闲 ====="
sinfo -p gpu -o "%N %t %C %G" 2>&1
sinfo -p gpu -o "%N %t %G" -N 2>&1 | head
echo
echo "===== V1 记账 ====="
sacct -j 1444037 -o JobID,JobName%10,State,ExitCode,Elapsed,AllocCPUS -X 2>&1
echo "===== V3 记账 ====="
sacct -j 1444036 -o JobID,JobName%10,State,ExitCode,Elapsed,AllocCPUS -X 2>&1
echo
echo "===== V1 报告 ====="
[ -f "$RD/v1_report.txt" ] && cat "$RD/v1_report.txt" || echo "(V1 尚未生成)"
echo
echo "===== V1 out 尾部 ====="
tail -20 "$RD"/logs/v1-1444037.out 2>&1
echo "===== V1 err 尾部 ====="
tail -10 "$RD"/logs/v1-1444037.err 2>&1
echo
echo "===== V3 报告 ====="
for f in "$RD"/v3out/v3_*_report.txt; do [ -f "$f" ] && { echo "### $(basename $f)"; cat "$f"; }; done
echo "(V3 报告列出完毕)"
echo "STATUS_DONE"
