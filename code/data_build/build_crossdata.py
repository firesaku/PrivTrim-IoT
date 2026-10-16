"""
跨数据集抽样 + 敏感项注入 + 冻结（CPU only, 0 model calls）
输出：frozen_samples.jsonl + MANIFEST.json + sha256

设计：
  - 每个来源抽 N 例，统一 schema: {sid, source, context, question, answers, canaries, variants}
  - canaries: 合成敏感标识符（canonical，沿用项目既有 email/phone/account 三型）
  - variants: 每型 3 个 paraphrase 形式（用于测"正则够不到"的覆盖率）
  - 冻结后写 sha256，登记不再改
"""
import io, json, os, re, gzip, glob, hashlib, random, sys

C = r"C:\Users\ADMIN\Documents\ChatGPT\Cybersec_IoT"
OUTDIR = r"D:\CodexData\tmp\privtrim-review-20261015\crossdata"
os.makedirs(OUTDIR, exist_ok=True)
N_PER_SOURCE = 96
random.seed(20261015)

def norm_ws(t):
    return re.sub(r'\s+', ' ', t).strip()

def sha(p):
    return hashlib.sha256(io.open(p, 'rb').read()).hexdigest()

# ---------------------------------------------------------------- 抽取器
def x_squad():
    p = os.path.join(C, 'external', 'squad_v11', 'dev-v1.1.json')
    d = json.loads(io.open(p, encoding='utf-8').read())
    out = []
    for art in d['data']:
        for para in art['paragraphs']:
            if not (128 <= len(para['context']) <= 4000):
                continue
            for qa in para['qas']:
                if not qa['answers']:
                    continue
                a = qa['answers'][0]['text']
                if not a.strip() or len(a) > 40:
                    continue
                out.append(dict(question=norm_ws(qa['question']),
                                context=norm_ws(para['context']), answers=[a]))
    return out

def x_hotpotqa():
    p = os.path.join(C, 'external', 'hotpotqa', 'dev_distractor.parquet')
    import pyarrow.parquet as pq
    t = pq.ParquetFile(p).read().to_pylist()
    out = []
    for r in t:
        titles = r['context']['title']; sents = r['context']['sentences']
        ctx = " ".join("%s: %s" % (ti, " ".join(ss)) for ti, ss in zip(titles, sents))
        if not (200 <= len(ctx) <= 12000):
            continue
        out.append(dict(question=norm_ws(r['question']), context=norm_ws(ctx),
                        answers=[str(r['answer'])], meta=dict(type=r.get('type'), level=r.get('level'))))
    return out

def x_simuhome():
    out = []
    for p in sorted(glob.glob(os.path.join(C, 'research', 'simuhome_generated_tasks_v1', 'tasks', '*.json'))):
        try:
            o = json.loads(io.open(p, encoding='utf-8').read())
        except Exception:
            continue
        txt = json.dumps(o, ensure_ascii=False)
        ctx = txt[:6000]
        q = ""
        for k in ('instruction', 'goal', 'task', 'question', 'prompt', 'description'):
            if isinstance(o.get(k), str):
                q = o[k]; break
        if not q:
            q = o.get('name') or os.path.basename(p)
        out.append(dict(question=norm_ws(str(q)), context=norm_ws(ctx),
                        answers=[], meta=dict(file=os.path.basename(p))))
    return out

def x_pec_home():
    out = []
    for fn in ['multi_user_preferences.jsonl', 'dynamic_user_preferences.jsonl']:
        p = os.path.join(C, 'external', 'PEC-Home', 'data', 'release', fn)
        if not os.path.exists(p):
            continue
        with io.open(p, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                ctx = json.dumps(o, ensure_ascii=False)
                if len(ctx) < 200:
                    continue
                out.append(dict(question="What do the stored user preferences say?",
                                context=norm_ws(ctx)[:6000], answers=[], meta=dict(file=fn)))
    return out

def x_toolmem():
    p = os.path.join(C, 'research', 'mem2act_scope_v1', 'data', 'toolmem_conversation.jsonl')
    out = []
    if not os.path.exists(p):
        return out
    with io.open(p, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except Exception:
                continue
            turns = o.get('turns') or []
            parts = []
            for t in turns:
                c = t.get('content')
                if isinstance(c, str):
                    parts.append("%s: %s" % (t.get('role', '?'), c))
            ctx = "\n".join(parts)
            if len(ctx) < 200:
                continue
            first_user = next((t.get('content') for t in turns if t.get('role') == 'user'
                               and isinstance(t.get('content'), str)), "Summarize this session.")
            out.append(dict(question=norm_ws(str(first_user))[:300], context=norm_ws(ctx)[:8000],
                            answers=[], meta=dict(session=o.get('session_id'))))
    return out

def x_privacylens():
    p = os.path.join(C, 'external', 'PrivacyLens', 'data', 'main_data.json')
    out = []
    if not os.path.exists(p):
        return out
    d = json.loads(io.open(p, encoding='utf-8').read())
    items = d if isinstance(d, list) else (d.get('data') or list(d.values())[0])
    if not isinstance(items, list):
        return out
    for o in items:
        if not isinstance(o, dict):
            continue
        ctx = json.dumps(o, ensure_ascii=False)
        q = ""
        for k in ('question', 'prompt', 'instruction', 'query'):
            if isinstance(o.get(k), str):
                q = o[k]; break
        out.append(dict(question=norm_ws(q or "Handle this record."), context=norm_ws(ctx)[:6000],
                        answers=[], meta=dict(keys=list(o.keys())[:8])))
    return out

def x_smartbench():
    out = []
    base = os.path.join(C, 'external', 'SmartBench', 'context-dependent', 'tdcs_rbe')
    files = glob.glob(os.path.join(base, '*', '*.json'))
    for p in sorted(files):
        try:
            o = json.loads(io.open(p, encoding='utf-8').read())
        except Exception:
            continue
        ctx = json.dumps(o, ensure_ascii=False)
        if len(ctx) < 300:
            continue
        label = 'anomaly' if os.sep + 'anomaly' + os.sep in p else 'normal'
        out.append(dict(question="Is there an anomaly in this device context, and why?",
                        context=norm_ws(ctx)[:8000], answers=[],
                        meta=dict(label=label, file=os.path.basename(p))))
    return out

SOURCES = [('smartbench_tdcs', x_smartbench), ('simuhome', x_simuhome), ('pec_home', x_pec_home),
           ('toolmem', x_toolmem), ('privacylens', x_privacylens), ('hotpotqa', x_hotpotqa),
           ('squad', x_squad)]

# ---------------------------------------------------------------- 敏感项 + 变体
def mk_canaries(sid, i):
    h = hashlib.sha256(("canary:" + sid).encode()).hexdigest()
    em = h[:12]; ac = h[12:24]
    email = "contact.%s@example.invalid" % em
    phone = "+1-202-555-%04d" % (i % 10000)
    acct = "ACCT-" + ac
    can = [dict(type='email', value=email, marker=em, fragment_length=6),
           dict(type='phone', value=phone, marker=re.sub(r'\D', '', phone), fragment_length=8),
           dict(type='account', value=acct, marker=ac, fragment_length=6)]
    return can

def variants_for(c):
    t, v, mk = c['type'], c['value'], c['marker']
    if t == 'email':
        local, _, dom = v.partition('@')
        return [dict(kind='canonical', text=v),
                dict(kind='case', text=v.upper()),
                dict(kind='at_dot_words', text="%s at %s" % (local, dom.replace('.', ' dot '))),
                dict(kind='local_only', text=local),
                dict(kind='paraphrase', text="the contact email ending in %s" % dom.split('.')[-1])]
    if t == 'phone':
        d = mk; a, e, l = d[1:4], d[4:7], d[7:11]
        return [dict(kind='canonical', text=v),
                dict(kind='spaces', text="+1 %s %s %s" % (a, e, l)),
                dict(kind='digits_only', text=d),
                dict(kind='paraphrase', text="the private phone number ending in %s" % l)]
    return [dict(kind='canonical', text=v),
            dict(kind='lower', text=v.lower()),
            dict(kind='prefix_space', text="ACCT " + mk),
            dict(kind='no_prefix', text=mk),
            dict(kind='paraphrase', text="the account id %s" % mk)]

BLOCK = ("Private contact record:\nContact email: {e}\nContact phone: {p}\nAccount ID: {a}")

# ---------------------------------------------------------------- 主流程
records = []
stats = []
for name, fn in SOURCES:
    try:
        pool = fn()
    except Exception as e:
        stats.append((name, 0, "ERR %r" % (e,)))
        continue
    random.shuffle(pool)
    take = pool[:N_PER_SOURCE]
    for i, r in enumerate(take):
        sid = "%s_%04d" % (name, i)
        can = mk_canaries(sid, i)
        block = BLOCK.format(e=can[0]['value'], p=can[1]['value'], a=can[2]['value'])
        # 位置轮换：前 / 后 / 中
        ctx = r['context']; pos = ['prefix', 'suffix', 'middle'][i % 3]
        if pos == 'prefix':
            aug = block + "\n\n" + ctx
        elif pos == 'suffix':
            aug = ctx + "\n\n" + block
        else:
            cut = len(ctx) // 2
            sp = ctx.find(' ', cut)
            sp = sp if sp > 0 else cut
            aug = ctx[:sp] + "\n\n" + block + "\n\n" + ctx[sp:]
        rec = dict(sid=sid, source=name, question=r['question'], context=aug,
                   answers=r.get('answers', []), canaries=can,
                   variants=[dict(ci=ci, **v) for ci, c in enumerate(can) for v in variants_for(c)],
                   canary_position=pos, meta=r.get('meta', {}))
        records.append(rec)
    stats.append((name, len(take), "ok"))

# ---------------------------------------------------------------- 冻结
jp = os.path.join(OUTDIR, 'frozen_samples.jsonl')
with io.open(jp, 'w', encoding='utf-8') as fh:
    for r in records:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

man = dict(
    created_utc="2026-10-15",
    purpose="cross-dataset sampling for unsupervised clustering desensitization (PrivTrim-IoT)",
    n_per_source_target=N_PER_SOURCE,
    total_records=len(records),
    per_source=[dict(source=s, n=n, note=note) for s, n, note in stats],
    canary_types=["email", "phone", "account"],
    variant_kinds=sorted({v['kind'] for r in records for v in r['variants']}),
    positions=["prefix", "middle", "suffix"],
    frozen=True, seed=20261015,
    source_paths=dict(squad=os.path.join(C, 'external', 'squad_v11', 'dev-v1.1.json'),
                      hotpotqa=os.path.join(C, 'external', 'hotpotqa', 'dev_distractor.parquet'),
                      privacylens=os.path.join(C, 'external', 'PrivacyLens', 'data', 'main_data.json'),
                      pec_home=os.path.join(C, 'external', 'PEC-Home', 'data', 'release'),
                      smartbench=os.path.join(C, 'external', 'SmartBench', 'context-dependent', 'tdcs_rbe'),
                      simuhome=os.path.join(C, 'research', 'simuhome_generated_tasks_v1', 'tasks'),
                      toolmem=os.path.join(C, 'research', 'mem2act_scope_v1', 'data', 'toolmem_conversation.jsonl')),
    file_sha256=sha(jp),
)
mp = os.path.join(OUTDIR, 'MANIFEST.json')
io.open(mp, 'w', encoding='utf-8').write(json.dumps(man, ensure_ascii=False, indent=2))

print("records:", len(records))
for s, n, note in stats:
    print("  %-18s %4d  %s" % (s, n, note))
print("sha256:", man['file_sha256'][:16])
print("out:", jp)
