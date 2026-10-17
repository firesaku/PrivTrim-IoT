RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
echo "=== r4 exists / size / sha ==="
ls -l "$RD/e2e_longctx_r4.py" 2>&1
sha256sum "$RD/e2e_longctx_r4.py" 2>&1
wc -l "$RD/e2e_longctx_r4.py" 2>&1
echo "=== entity_diag alias branch on REMOTE r4 (lines 258-282) ==="
sed -n '258,282p' "$RD/e2e_longctx_r4.py" 2>&1
echo "=== does remote have the pre-seed? ==="
grep -n "canaries') or \[\])" "$RD/e2e_longctx_r4.py" 2>&1 | head
grep -n "for _c in" "$RD/e2e_longctx_r4.py" 2>&1 | head
echo "=== ext-arms already present? ==="
grep -n "ext_arms\|ext-arms" "$RD/e2e_longctx_r4.py" 2>&1 | head
echo "=== BASE64 for exact copy ==="
base64 -w0 "$RD/e2e_longctx_r4.py" 2>&1 | head -c 200
echo
echo "=== job progress ==="
tail -n 2 "$RD/logs/aliaslink-10602422.out" 2>&1
tail -n 2 "$RD/logs/mistral-10602423.out" 2>&1
