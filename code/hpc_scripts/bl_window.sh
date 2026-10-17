B=/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005
C=$B/project/research/token_compression_batch1_v1
PY=$B/.venv-main/bin/python
echo "=== manifest paths + each model context window ==="
$PY - <<'EOF'
import json, os
B='/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005'
C=B+'/project/research/token_compression_batch1_v1'
man=json.load(open(C+'/model_manifest.json'))
for repo,info in man.items():
    print('\n%-58s status=%s' % (repo, info.get('status')))
    p = info.get('path') or info.get('local_path') or info.get('snapshot')
    print('   path=%s' % p)
    cfg=None
    for cand in ([p] if p else []) + [os.path.join(p,'config.json') if p else None]:
        if cand and os.path.isfile(cand): cfg=cand; break
        if cand and os.path.isfile(os.path.join(cand,'config.json')): cfg=os.path.join(cand,'config.json'); break
    if cfg:
        c=json.load(open(cfg))
        print('   model_type=%s max_pos=%s' % (c.get('model_type'), c.get('max_position_embeddings')))
EOF
echo
echo "=== does provence chunk internally? ==="
grep -rn "max_position\|512\|chunk\|split\|window" $B/models/models--naver--provence-reranker-debertav3-v1/snapshots/*/modeling_provence.py 2>/dev/null | head -n 15
echo "=== llmlingua2 compress signature / chunking ==="
$PY -c "
import inspect, llmlingua
from llmlingua import PromptCompressor
print('llmlingua', getattr(llmlingua,'__version__','?'))
print(inspect.signature(PromptCompressor.compress))
" 2>&1 | tail -n 6
echo "=== installed llmlingua compress: chunk handling ==="
$PY -c "
import inspect
from llmlingua import PromptCompressor
src=inspect.getsource(PromptCompressor.compress)
import re
for ln in src.splitlines():
    if re.search(r'chunk|max_position|512|split_text|truncat', ln): print(ln.strip()[:140])
" 2>&1 | tail -n 25
echo "=== our longctx context lengths (Qwen tokens) ==="
$PY -c "
import json
R='/dpc/kuex0095/100065732/privtrim_v2c_20261015/longctx_cases.jsonl'
cs=[json.loads(l) for l in open(R)]
print('n=%d min=%d median=%d max=%d' % (len(cs), min(len(c['context']) for c in cs)//4, sorted(len(c['context']) for c in cs)[len(cs)//2]//4, max(len(c['context']) for c in cs)//4))
" 2>&1 | tail -n 3
