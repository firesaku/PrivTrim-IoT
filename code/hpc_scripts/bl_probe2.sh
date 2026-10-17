B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
C=$B/project/research/token_compression_batch1_v1
echo "=== compress.py lines 95-175 ==="
sed -n '95,175p' "$C/compress.py" 2>&1
echo "=== compress.py main() 175-260 ==="
sed -n '175,260p' "$C/compress.py" 2>&1
echo "=== case schema used by compress_case (grep c\[) ==="
grep -n "c\['" "$C/compress.py" 2>&1 | head -n 20
grep -n "c\.get(" "$C/compress.py" 2>&1 | head -n 20
echo "=== runner scripts referencing compress.py ==="
grep -rln "compress.py" "$B"/run 2>/dev/null | head -n 10
echo "=== model_manifest repos ==="
$B/.venv-main/bin/python -c "
import json;d=json.load(open('$C/model_manifest.json'));
print(type(d));
print(list(d.keys())[:20] if isinstance(d,dict) else d[:5])" 2>&1 | head -n 10
