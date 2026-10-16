"""跨数据集结构探测 v2：每个数据源只输出一行（抗折叠）"""
import io, json, os, gzip, glob

C = r"C:\Users\ADMIN\Documents\ChatGPT\Cybersec_IoT"
D = r"D:\CodexData\tmp\privtrim-review-20261015\dsprobe"
os.makedirs(D, exist_ok=True)
n = [0]

def emit(tag, text):
    n[0] += 1
    io.open(os.path.join(D, "%02d_%s.txt" % (n[0], tag)), "w", encoding="utf-8").write(text + "\n")

def keys_of(obj):
    return ",".join(list(obj.keys())[:14]) if isinstance(obj, dict) else type(obj).__name__

# 1 SQuAD
sq = os.path.join(C, 'external', 'squad_v11', 'dev-v1.1.json')
if os.path.exists(sq):
    d = json.loads(io.open(sq, encoding='utf-8').read())
    para = d['data'][0]['paragraphs'][0]; qa = para['qas'][0]
    emit("SQUAD", "SQUAD | articles=%d | qa_keys=%s | answers=%s | ctx_chars=%d | path=%s" % (
        len(d['data']), keys_of(qa), json.dumps(qa['answers'][0], ensure_ascii=False), len(para['context']), sq))

# 2 HotpotQA
hp = os.path.join(C, 'external', 'hotpotqa', 'dev_distractor.parquet')
if os.path.exists(hp):
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(hp)
        t = pf.read_row_group(0).slice(0, 1).to_pylist()[0]
        desc = " ; ".join("%s=%s" % (k, (json.dumps(v, ensure_ascii=False)[:90] if not isinstance(v, str) else v[:90])) for k, v in t.items())
        emit("HOTPOTQA", "HOTPOTQA | rows=%d | cols=%s | row0: %s" % (pf.metadata.num_rows, pf.schema_arrow.names, desc))
    except Exception as e:
        emit("HOTPOTQA", "HOTPOTQA | pyarrow ERR %r | path=%s" % (e, hp))

# 3 每个 external 子目录：最大数据文件路径 + 大小
for name in sorted(os.listdir(os.path.join(C, 'external'))):
    dd = os.path.join(C, 'external', name)
    if not os.path.isdir(dd):
        continue
    hits = []
    for root, dirs, files in os.walk(dd):
        if '.git' in root:
            continue
        for f in files:
            if f.endswith(('.json', '.jsonl', '.csv', '.parquet', '.tsv', '.gz')):
                p = os.path.join(root, f)
                try:
                    hits.append((os.path.getsize(p), os.path.relpath(p, C)))
                except OSError:
                    pass
    hits.sort(reverse=True)
    top = hits[:4]
    emit("EXT_%s" % name, "EXT %s | files=%d | top: %s" % (
        name, len(hits), " ; ".join("%s (%.2fMB)" % (r, s/1e6) for s, r in top)))

# 4 toolmem 首行
tm = os.path.join(C, 'research', 'mem2act_scope_v1', 'data', 'toolmem_conversation.jsonl')
if os.path.exists(tm):
    with io.open(tm, encoding='utf-8', errors='replace') as fh:
        line = fh.readline()
    try:
        o = json.loads(line)
        emit("TOOLMEM", "TOOLMEM | keys=%s | sample=%s" % (keys_of(o), line.strip()[:300]))
    except Exception as e:
        emit("TOOLMEM", "TOOLMEM | parse ERR %r | raw=%s" % (e, line[:250]))

# 5 AgentDojo
g = glob.glob(os.path.join(C, '**', '*agentdojo*'), recursive=True)
g = [p for p in g if '.git' not in p][:8]
emit("AGENTDOJO", "AGENTDOJO | hits=%s" % " ; ".join(p[len(C)+1:] for p in g))

# 6 SimuHome 任务
for tag, pat in [("SIMUHOME_TASKS", os.path.join(C, 'research', 'simuhome_generated_tasks_v1', '*', '*.json')),
                 ("SIMUHOME_BENCH", os.path.join(C, 'benchmarks', 'simuhome_prospective_capability_dev_v1', '*', '*.json'))]:
    gg = glob.glob(pat)[:4]
    emit(tag, "%s | %s" % (tag, " ; ".join(p[len(C)+1:] for p in gg)))

# 7 HPC cases 结构（已有的 17975 例，最全的 SQuAD+HotpotQA 增强版）
hpc = r"D:\CodexData\artifacts\privtrim-madar-20261013\hpc_full\run\qwen25_7b\cases.jsonl.gz"
if os.path.exists(hpc):
    with gzip.open(hpc, 'rt', encoding='utf-8') as fh:
        first = json.loads(fh.readline())
    emit("HPC_CASES", "HPC_CASES | keys=%s | canaries=%s | path=%s" % (
        keys_of(first), json.dumps(first.get('canaries'), ensure_ascii=False)[:220], hpc))

print("wrote", n[0], "files to", D)
