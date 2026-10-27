#!/bin/bash
echo "===== ALMESBAR: 我的队列 ====="
squeue -u "$(id -un)" -o '%.12i %.10P %.12j %.9T %.11M %.16R' 2>&1 | head -12
echo "===== ALMESBAR: 今日我的作业终态 ====="
sacct -u "$(id -un)" --starttime=2026-10-15 -o JobID,JobName%12,Partition,State,ExitCode,Elapsed -X 2>&1 | grep -E 'e2e|pt-|----' | head -12
echo
echo "===== 产出文件 ====="
ls -l /path/to/100065732/privtrim_v2c_20261015/e2eout/ 2>&1
echo
echo "===== 关键报告行（E2E v2）====="
head -6 /path/to/100065732/privtrim_v2c_20261015/e2eout/e2e2_report.txt 2>&1
echo "ALM_STATUS_DONE"
