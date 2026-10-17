B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
C=$B/project/research/token_compression_batch1_v1
echo "=== common.py FULL ==="
cat -n "$C/common.py" 2>&1
echo
echo "=== venvs ==="
ls -d $B/.venv* $B/*/.venv* /dpc/kuex0095/100065732/.venv* 2>/dev/null
echo "=== which venv has llmlingua ==="
for v in $B/.venv*; do
  [ -x "$v/bin/python" ] && echo -n "$v: " && "$v/bin/python" -c "import llmlingua;print('llmlingua OK')" 2>&1 | tail -n 1
done
echo "=== run dir ==="
ls -l "$B"/run/ 2>&1 | head -n 20
echo "=== published run scripts ==="
find "$B/run" -maxdepth 2 -name "*.sh" -o -maxdepth 2 -name "*.py" 2>/dev/null | head -n 20
