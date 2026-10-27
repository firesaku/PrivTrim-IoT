B=/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
PY=$B/.venv-main/bin/python
RD=/path/to/100065732/privtrim_v2c_20261015
echo "=== r4 argparse ==="
grep -n "add_argument" "$RD/e2e_longctx_r4.py" 2>&1 | sed 's/^ *//' | head -n 40
echo "=== apply_chat_template usage ==="
grep -n "apply_chat_template\|chat_template\|tokenizer(" "$RD/e2e_longctx_r4.py" 2>&1 | head -n 10
echo
echo "=== tokenizer preflight ==="
$PY - <<'EOF'
import json, sys
from transformers import AutoTokenizer
RD='/path/to/100065732/privtrim_v2c_20261015'
B='/path/to/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005'
qwen = B+'/models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28'
mist = B+'/models/mistral-7b-instruct-v03'
cases=[json.loads(l) for l in open(RD+'/longctx_cases.jsonl', encoding='utf-8')]
print('cases=%d' % len(cases))
print('case keys=%s' % sorted(cases[0].keys())[:14])
for name, path in (('qwen', qwen), ('mistral', mist)):
    tk = AutoTokenizer.from_pretrained(path)
    ct = getattr(tk, 'chat_template', None)
    print('\n[%s] chat_template=%s  eos=%r  pad=%r' % (name, 'YES' if ct else 'NO', tk.eos_token, tk.pad_token))
    mx, over = 0, 0
    for c in cases:
        ctx = c.get('context') or c.get('full_context') or ''
        n = len(tk(ctx, add_special_tokens=False)['input_ids'])
        mx = max(mx, n)
        if n > 24576: over += 1
    print('[%s] max_ctx_tokens=%d  over_24576=%d/%d' % (name, mx, over, len(cases)))
    if ct:
        msgs=[{'role':'system','content':'x'},{'role':'user','content':'y'}]
        ids = tk.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        print('[%s] chat_template renders ok, n=%d' % (name, len(ids)))
EOF
echo "=== mistral config ==="
$PY -c "import json;c=json.load(open('$B/models/mistral-7b-instruct-v03/config.json'));print({k:c.get(k) for k in ('model_type','max_position_embeddings','sliding_window','torch_dtype','num_hidden_layers')})"
