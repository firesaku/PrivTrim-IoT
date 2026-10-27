B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
C=$B/project/research/token_compression_batch1_v1
echo "=== dir ==="
ls -l "$C" 2>&1 | head -n 30
echo "=== common.py: METHODS / ALGORITHMS ==="
grep -n "^METHODS\|^ALGORITHMS\|^TRACKS\|^def key\|^def " "$C/common.py" 2>&1 | head -n 30
echo "--- ALGORITHMS content ---"
sed -n '/^ALGORITHMS/,/^\]/p' "$C/common.py" 2>&1 | head -n 20
echo "=== compress.py: defs ==="
grep -n "^def \|^class \|^    def " "$C/compress.py" 2>&1 | head -n 40
echo "=== compress.py: first 80 lines ==="
sed -n '1,80p' "$C/compress.py" 2>&1
