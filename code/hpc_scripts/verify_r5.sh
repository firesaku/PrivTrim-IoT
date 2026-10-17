RD=/dpc/kuex0095/100065732/privtrim_v2c_20261015
echo "=== r5 remote integrity ==="
sha256sum "$RD/e2e_longctx_r5.py" 2>&1
wc -c "$RD/e2e_longctx_r5.py" 2>&1
grep -c 'alias-question' "$RD/e2e_longctx_r5.py" 2>&1
grep -c 'ext-arms' "$RD/e2e_longctx_r5.py" 2>&1
echo "=== job B progress (full tail) ==="
date +%H:%M:%S
squeue --me -o "%.10i %.14j %.9T %.9M %.10R" 2>&1 | head -n 5
tail -n 6 "$RD/logs/mistral-10602423.out" 2>&1
echo "=== job B .out age (s) ==="
now=$(date +%s); mt=$(stat -c %Y "$RD/logs/mistral-10602423.out"); echo $((now-mt))
echo "=== job B .err size ==="
wc -c "$RD/logs/mistral-10602423.err" 2>&1
echo "=== gpu partition free ==="
sinfo -p gpu -o "%.8t %.12N %.16G %.6D" 2>&1 | head -n 8
