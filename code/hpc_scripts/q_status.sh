RD=/path/to/100065732/privtrim_v2c_20261015
date +%H:%M:%S
squeue --me -o "%.10i %.16j %.9T %.8M %.10R" 2>&1 | head
echo "--- A tail ---"; tail -n 2 "$RD/logs/aliaslink-10602422.out" 2>&1
echo "--- B tail ---"; tail -n 3 "$RD/logs/mistral-10602423.out" 2>&1
echo "--- A out ---"; ls -l --time-style=+%H:%M "$RD/aliaslink_out/" 2>&1 | head -n 5
