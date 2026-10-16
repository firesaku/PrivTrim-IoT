"""跨数据集结构探测（只读）"""
import io, json, os, gzip, glob

C = r"C:\Users\ADMIN\Documents\ChatGPT\Cybersec_IoT"
OUT = r"D:\CodexData\tmp\privtrim-review-20261015\dataset_probe.txt"
L = []
def w(s=""): L.append(str(s))

def head(path, n=2, label=None):
    w("--- %s" % (label or path))
    try:
        if path.endswith('.gz'):
            fh = gzip.open(path, 'rt', encoding='utf-8', errors='replace')
        else:
            fh = io.open(path, encoding='utf-8', errors='replace')
        for i, line in enumerate(fh):
            if i >= n: break
            w("    " + line.strip()[:400])
        fh.close()
    except Exception as e:
        w("    ERR %r" % (e,))

w("=" * 90)
w("A. external/ 目录树（2 层）")
for d in sorted(glob.glob(os.path.join(C, 'external', '*'))):
    w("  %s" % os.path.basename(d))
    if os.path.isdir(d):
        for f in sorted(os.listdir(d))[:12]:
            p = os.path.join(d, f)
            sz = os.path.getsize(p) if os.path.isfile(p) else 0
            w("      %-45s %s" % (f[:45], ("%.2f MB" % (sz/1e6)) if sz else "DIR"))

w("")
w("=" * 90)
w("B. 各候选数据源结构")

# SQuAD
sq = os.path.join(C, 'external', 'squad_v11', 'dev-v1.1.json')
if os.path.exists(sq):
    d = json.loads(io.open(sq, encoding='utf-8').read())
    art = d['data'][0]
    para = art['paragraphs'][0]
    qa = para['qas'][0]
    w("--- SQuAD 1.1")
    w("    articles=%d  paragraphs(art0)=%d" % (len(d['data']), len(art['paragraphs'])))
    w("    qa keys=%s  answers[0]=%s" % (list(qa.keys()), qa['answers'][0]))
    w("    context chars=%d" % len(para['context']))
else:
    w("--- SQuAD 未找到: %s" % sq)

# HotpotQA parquet
hp = os.path.join(C, 'external', 'hotpotqa', 'dev_distractor.parquet')
if os.path.exists(hp):
    w("--- HotpotQA parquet")
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(hp)
        w("    rows=%d cols=%s" % (pf.metadata.num_rows, pf.schema_arrow.names))
        t = pf.read_row_group(0).slice(0, 1).to_pylist()[0]
        for k, v in t.items():
            s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            w("      %-14s %s" % (k, s[:180]))
    except Exception as e:
        w("    pyarrow 不可用: %r" % (e,))
else:
    w("--- HotpotQA 未找到")

# PrivacyLens / PEC-Home / PISanitizer / SAGE
for name in ['PrivacyLens', 'PEC-Home', 'PISanitizer', 'SAGE']:
    d = os.path.join(C, 'external', name)
    if not os.path.isdir(d):
        w("--- %s 不存在" % name); continue
    w("--- %s" % name)
    hits = []
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.endswith(('.json', '.jsonl', '.csv', '.parquet', '.tsv')):
                p = os.path.join(root, f)
                hits.append((os.path.getsize(p), os.path.relpath(p, d)))
    hits.sort(reverse=True)
    for sz, rel in hits[:6]:
        w("      %-60s %.2f MB" % (rel[:60], sz/1e6))
    if hits:
        biggest = os.path.join(d, hits[0][1])
        head(biggest, 2, "  head: " + hits[0][1])

# toolmem
tm = os.path.join(C, 'research', 'mem2act_scope_v1', 'data', 'toolmem_conversation.jsonl')
if os.path.exists(tm):
    w("--- toolmem_conversation.jsonl")
    head(tm, 1)

# AgentDojo
w("--- AgentDojo 位置搜索")
for pat in [os.path.join(C, '**', '*agentdojo*', '*.json'),
            os.path.join(C, 'external', '*', '*banking*')]:
    for p in glob.glob(pat, recursive=True)[:5]:
        w("      %s" % p[len(C)+1:])

# SimuHome tasks
w("--- SimuHome 任务样例位置")
for pat in [os.path.join(C, 'research', 'simuhome_generated_tasks_v1', '*', '*.json'),
            os.path.join(C, 'benchmarks', 'simuhome_prospective_capability_dev_v1', 'tasks', '*.json')]:
    g = glob.glob(pat)[:3]
    for p in g:
        w("      %s" % p[len(C)+1:])

io.open(OUT, 'w', encoding='utf-8').write("\n".join(L))
print("wrote", OUT, len(L), "lines")
