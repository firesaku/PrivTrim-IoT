RD=/path/to/100065732/privtrim_v2c_20261015
echo "===== 我的队列 ====="
squeue -u $(id -un) -o '%.14i %.9P %.14j %.9T %.11M %.8b %.18R' 2>&1
echo "===== 我所有作业的终态 ====="
sacct -u $(id -un) --starttime=2026-10-15 -o JobID,JobName%16,Partition,State,ExitCode,Elapsed -X 2>&1 | head -30
echo "===== 产出文件 ====="
ls -l "$RD" 2>&1
echo "--- v3out ---"
ls -l "$RD/v3out" 2>&1
echo "===== 磁盘 ====="
du -sh "$RD" 2>&1
echo "STATUS_OK"
