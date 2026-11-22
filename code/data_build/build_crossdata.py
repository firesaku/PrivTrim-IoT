"""
跨数据集抽样 + 敏感项注入 + 冻结（CPU only, 0 model calls）

===================== v1 / v2 双版本说明（审计修复 2026-10-16）=====================
v1 产物 crossdata/frozen_samples.jsonl（581 例 / 7 来源）
    sha256 cb44ab1745a9f72f76bb799a2e649541fc6fa79fd9b69717e9abce357cfd42ed
已登记在 docs/06_数据资产.md，且是已发表结果（v2c/v3/join）的样本集。
本脚本**只读 v1、绝不改写它**（也不动它的 MANIFEST.json），保留存证。

v2 产物 crossdata/frozen_samples_v2.jsonl + MANIFEST_v2.json，取代 v1 **仅用于 variant 实验**：
  1) email paraphrase 坏模板已修：v1 是常量串 "the contact email ending in invalid"
     （域名固定 example.invalid）——不含任何 PII，实测 distinct = 1/581，用它评估 email
     改述会**虚假通过**。现与 hpc/v1_checker.py::variants() 对齐，改成
     "the mailbox <local> at the example domain" 这类真正携带 canary 材料的模板；
     构建期强制每个变体都携带对应材料（email 带 local part / phone 带尾 4 位或完整号码 /
     account 带完整 marker），不满足直接 raise。
  2) 变体真的注入：新字段 context_variants = canonical 块 + 变体块（位置独立轮换），
     并逐条给出 variant_spans 字符区间，使 variant-removal-rate 可以在冻结集上按区间测量。
     `context` 保持 canonical-only：源文本/块格式/位置轮换与 v1 完全一致，唯一差别是 phone 值按第 4 条
     改成逐记录唯一（脚本末尾按"仅 phone 不同"逐条对拍），canonical 表因此仍可与 v1 对齐——
     把变体塞进 `context` 会污染 canonical 评估，故用成对的 context_variants。
  3) simuhome 实际只有 5 个任务文件 -> note 记 SHORT，不再无条件 "ok"。
  4) phone 由 sha256(sid) 派生（与 email/account 同源），逐记录唯一，并保证 marker 的
     4 个 8 位窗口互不相同（v1 用来源内序号 i，7 来源共用同一批号码：96 个号码代表 581 条）。
  5) local_only / no_prefix 归一化后是 canonical 的子串，**不可能证明独立检出** -> 记录里
     显式标 role=subsumption_probe（并给出 norm_relation），另外补 genuinely independent
     的替代形式（email local_named / account hash_style 等）。

确定性：固定 seed + 与 PYTHONHASHSEED 无关（所有集合都 sorted 后输出）；
        两次不同 PYTHONHASHSEED 运行产出 sha256 逐字节一致（v1 已实测复现）。
输出：frozen_samples_v2.jsonl + MANIFEST_v2.json
"""
import io, json, os, re, gzip, glob, hashlib, random, sys

C = r"C:\Users\ADMIN\Documents\ChatGPT\Cybersec_IoT"
OUTDIR = r"D:\CodexData\tmp\privtrim-review-20261015\crossdata"
os.makedirs(OUTDIR, exist_ok=True)
N_PER_SOURCE = 96
SEED = 20261015
random.seed(SEED)

V1_JSONL = os.path.join(OUTDIR, 'frozen_samples.jsonl')        # v1：只读对拍，绝不可写
V1_SHA = "cb44ab1745a9f72f76bb799a2e649541fc6fa79fd9b69717e9abce357cfd42ed"
OUT_JSONL = os.path.join(OUTDIR, 'frozen_samples_v2.jsonl')
OUT_MAN = os.path.join(OUTDIR, 'MANIFEST_v2.json')
POSITIONS = ['prefix', 'suffix', 'middle']

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
PHONE_USED = set()          # 确定性去重：号码本体 + 4 个 8 位窗口

def mk_phone(sid):
    """phone 与 email/account 同源，由 sha256(sid) 派生（v1 用来源内序号 i，7 来源共用）。
    保持 +1-<区号>-555-<4位> 的虚构号段形状与项目 PHONE 正则一致；用 salt 计数器做
    确定性去重，保证号码本体与 marker 的 4 个 8 位窗口对 581 条记录互不相同。"""
    for salt in range(10000):
        hh = hashlib.sha256(("phone:%s:%d" % (sid, salt)).encode()).hexdigest()
        d = ''.join(str(int(ch, 16) % 10) for ch in hh[:10])
        npa = str(2 + int(hh[10], 16) % 8) + d[0:2]         # 区号首位 2-9（NANP 约束）
        val = "+1-%s-555-%s" % (npa, d[2:6])
        dig = re.sub(r'\D', '', val)
        win = [dig[k:k + 8] for k in range(4)]
        if dig in PHONE_USED or any(x in PHONE_USED for x in win):
            continue
        PHONE_USED.add(dig)
        PHONE_USED.update(win)
        return val
    raise RuntimeError("phone 确定性去重失败: %s" % sid)

def mk_canaries(sid, i):
    h = hashlib.sha256(("canary:" + sid).encode()).hexdigest()
    em = h[:12]; ac = h[12:24]
    email = "contact.%s@example.invalid" % em
    phone = mk_phone(sid)
    acct = "ACCT-" + ac
    can = [dict(type='email', value=email, marker=em, fragment_length=6),
           dict(type='phone', value=phone, marker=re.sub(r'\D', '', phone), fragment_length=8),
           dict(type='account', value=acct, marker=ac, fragment_length=6)]
    return can

DIGIT_WORDS = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
               '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}   # 与 hpc/v1_checker.py 一致

def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower())

def variants_for(c):
    """每型给 1 个 canonical + 若干变体；text 必须真正携带 canary 材料（构建期强校验）。
    模板与 hpc/v1_checker.py::variants() 对齐（它已有正确写法 "the mailbox %s at the example domain"）。"""
    t, v, mk = c['type'], c['value'], c['marker']
    if t == 'email':
        local, _, dom = v.partition('@')
        return [dict(kind='canonical', text=v, carries='full'),
                dict(kind='case', text=v.upper(), carries='full'),
                dict(kind='at_dot_words', text="%s at %s" % (local, dom.replace('.', ' dot ')), carries='full'),
                dict(kind='local_only', text=local, carries='local'),
                dict(kind='local_named', text="the local part %s of the address" % local, carries='local'),
                dict(kind='at_word', text="%s (at) %s" % (local, dom), carries='full'),
                # v1 的坏模板是 "the contact email ending in <域名后缀>"，无 PII；这里是正确写法
                dict(kind='paraphrase', text="the mailbox %s at the example domain" % local, carries='local')]
    if t == 'phone':
        d = mk; a, e, l = d[1:4], d[4:7], d[7:11]
        return [dict(kind='canonical', text=v, carries='full'),
                dict(kind='no_plus', text="%s-%s-%s-%s" % (d[0], a, e, l), carries='full'),
                dict(kind='dots', text="+1.%s.%s.%s" % (a, e, l), carries='full'),
                dict(kind='parens', text="+1 (%s) %s-%s" % (a, e, l), carries='full'),
                dict(kind='spaces', text="+1 %s %s %s" % (a, e, l), carries='full'),
                dict(kind='digits_only', text=d, carries='full'),
                dict(kind='spoken', text=" ".join(DIGIT_WORDS.get(ch, ch) for ch in d), carries='full'),
                dict(kind='paraphrase', text="the private phone number ending in %s" % l, carries='last4')]
    return [dict(kind='canonical', text=v, carries='full'),
            dict(kind='lower', text=v.lower(), carries='full'),
            dict(kind='prefix_space', text="ACCT " + mk, carries='marker'),
            dict(kind='no_prefix', text=mk, carries='marker'),
            dict(kind='hash_style', text="account #%s" % mk, carries='marker'),
            dict(kind='paraphrase', text="the account id %s" % mk, carries='marker')]

def annotate(c, ci):
    """给每个变体算 norm_relation / role / independent：
      equal   —— 归一化后与 canonical 完全相同（纯格式变化：大小写、分隔符）
      subset  —— 归一化后是 canonical 的子串（local_only / no_prefix）：
                 **不可能证明独立检出**，只能当 subsumption 探针
      superset—— 归一化后包含 canonical 全串（检出可从 canonical 继承）
      other   —— 真正独立的改述形式，才是 paraphrase 能力的证据
    role 直接写进记录，避免再被当成独立的 paraphrase 能力证据。"""
    nv = norm(c['value'])
    out = []
    for vi, v in enumerate(variants_for(c)):
        nt = norm(v['text'])
        rel = 'equal' if nt == nv else ('subset' if nt in nv else ('superset' if nv in nt else 'other'))
        if v['kind'] == 'canonical':
            role = 'canonical'
        elif rel == 'equal':
            role = 'format_only'
        elif rel == 'subset':
            role = 'subsumption_probe'
        else:
            role = 'paraphrase'
        out.append(dict(ci=ci, vi=vi, type=c['type'], kind=v['kind'], text=v['text'],
                        carries=v['carries'], norm_relation=rel, role=role,
                        independent=(role == 'paraphrase')))
    return out

def carries_material(c, v):
    """构建期硬校验：变体必须真的携带 canary 材料，否则 raise（v1 的 email paraphrase 就栽在这）。"""
    t, mk, val, txt = c['type'], c['marker'], c['value'], v['text']
    n = norm(txt)
    if v['kind'] == 'spoken':                       # 数字读法：逐位词形携带完整号码
        d = re.sub(r'\D', '', val)
        return all(DIGIT_WORDS[ch] in txt for ch in d)
    if t == 'email':
        return mk in n or norm(val) in n
    if t == 'phone':
        return mk[-4:] in n or mk in n
    return mk in n

BLOCK = ("Private contact record:\nContact email: {e}\nContact phone: {p}\nAccount ID: {a}")
VHEAD = "Alternate renderings of the same record (paraphrase / format variants):"

def insert_block(ctx, block, pos):
    """与 v1 相同的三点位插入（prefix / suffix / middle），保证 canonical 的 context 逐字节同源。"""
    if pos == 'prefix':
        return block + "\n\n" + ctx
    if pos == 'suffix':
        return ctx + "\n\n" + block
    cut = len(ctx) // 2
    sp = ctx.find(' ', cut)
    sp = sp if sp > 0 else cut
    return ctx[:sp] + "\n\n" + block + "\n\n" + ctx[sp:]

def vprefix(v):
    return "%s|%s: " % (v['type'], v['kind'])

def inject_variants(ctx, vs, pos):
    """把变体真的注入 context：每条一行 "<type>|<kind>: <text>"，单行前缀保证可精确定位。
    返回 (context_variants, variant_spans)；spans 是变体文本在 context_variants 里的字符区间，
    评测方按区间算 variant-removal-rate 即可（不会再出现'字段没人读'的死数据）。"""
    block = VHEAD + "\n" + "\n".join(vprefix(v) + v['text'] for v in vs)
    aug = insert_block(ctx, block, pos)
    spans, cursor = [], 0
    for v in vs:
        pre = vprefix(v)
        i = aug.find(pre + v['text'], cursor)
        if i < 0:
            raise RuntimeError("变体未注入/无法定位: %s %s" % (v['type'], v['kind']))
        s = i + len(pre); e = s + len(v['text'])
        if aug[s:e] != v['text']:
            raise RuntimeError("span 对不上: %s %s" % (v['type'], v['kind']))
        spans.append(dict(ci=v['ci'], vi=v['vi'], type=v['type'], kind=v['kind'],
                          role=v['role'], carries=v['carries'], start=s, end=e))
        cursor = e
    # 区间必须单调不重叠，否则区间级测量会互相污染
    for a, b in zip(spans, spans[1:]):
        if b['start'] < a['end']:
            raise RuntimeError("span 重叠: %s/%s" % (a['kind'], b['kind']))
    return aug, spans

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
        # canonical 位置轮换：前 / 后 / 中（与 v1 完全一致）；变体位置错开一格，两块不重叠
        ctx = r['context']; pos = POSITIONS[i % 3]; vpos = POSITIONS[(i + 1) % 3]
        aug = insert_block(ctx, block, pos)
        vs = []
        for ci, c in enumerate(can):
            for v in annotate(c, ci):
                if not carries_material(c, v):
                    raise RuntimeError("变体不含 canary 材料: %s %s %s" % (sid, v['type'], v['kind']))
                vs.append(v)
        cvar, spans = inject_variants(aug, vs, vpos)
        rec = dict(schema_version=2, sid=sid, source=name, question=r['question'],
                   context=aug, context_variants=cvar, variant_spans=spans,
                   answers=r.get('answers', []), canaries=can, variants=vs,
                   canary_position=pos, variant_position=vpos, meta=r.get('meta', {}))
        records.append(rec)
    if len(take) >= N_PER_SOURCE:
        stats.append((name, len(take), "ok"))
    else:
        stats.append((name, len(take), "SHORT %d/%d (源里只有 %d 例可用)" % (len(take), N_PER_SOURCE, len(take))))

# ---------------------------------------------------------------- 自检（构建期硬校验）
def need(cond, msg):
    """用显式 raise 而不是 assert —— python -O 下 assert 会被跳过，审计要求"响亮"。"""
    if not cond:
        raise RuntimeError("自检失败: " + msg)

need(len({r['sid'] for r in records}) == len(records), "sid 不唯一")
for r in records:                                   # 变体必须真的落在 context_variants 的 span 上
    for s, v in zip(r['variant_spans'], r['variants']):
        need(r['context_variants'][s['start']:s['end']] == v['text'], "span 复核失败 " + r['sid'])
    need(r['canary_position'] != r['variant_position'], "canonical/变体位置未错开 " + r['sid'])

allv = [v for r in records for v in r['variants']]
ind = [v for v in allv if v['role'] == 'paraphrase']
sub = [v for v in allv if v['role'] == 'subsumption_probe']
by_kind = {}
for v in allv:
    by_kind[(v['type'], v['kind'])] = by_kind.get((v['type'], v['kind']), 0) + 1
# 关键回归：email paraphrase 必须逐记录不同（v1 是常量，distinct = 1/581）
em_para = [v['text'] for v in allv if v['type'] == 'email' and v['kind'] == 'paraphrase']
need(len(set(em_para)) == len(em_para) == len(records), "email paraphrase 仍是常量串")
ph = [c['value'] for r in records for c in r['canaries'] if c['type'] == 'phone']
ph_win = [re.sub(r'\D', '', p)[k:k + 8] for p in ph for k in range(4)]
need(len(set(ph)) == len(ph), "phone 号码不唯一")
need(len(set(ph_win)) == len(ph_win), "phone 8 位窗口不唯一")
em = [c['value'] for r in records for c in r['canaries'] if c['type'] == 'email']
ac = [c['value'] for r in records for c in r['canaries'] if c['type'] == 'account']
need(len(set(em)) == len(em) and len(set(ac)) == len(ac), "email/account 不唯一")
need({v['kind'] for v in sub} == {'local_only', 'no_prefix'}, "subsumption 探针集合变了")
for r in records:                                   # 逐条复核独立性判定（12k+ 条变体）
    for v in r['variants']:
        nv = norm(r['canaries'][v['ci']]['value']); nt = norm(v['text'])
        rel = 'equal' if nt == nv else ('subset' if nt in nv else ('superset' if nv in nt else 'other'))
        need(rel == v['norm_relation'], "norm_relation 不一致 " + v['kind'])
        if v['role'] == 'paraphrase':               # 独立变体不能是 canonical 的归一化子串/超串
            need(rel == 'other' and nv not in nt and nt not in nv, "independent 变体不独立 " + v['kind'])
        if not v['independent']:
            need(rel != 'other', "非 independent 变体关系异常 " + v['kind'])

# ---------------------------------------------------------------- 与 v1 对拍（只读，验证 canonical 口径未变）
v1_note = "v1 缺失，跳过对拍"
v1_em_para_distinct = None
if os.path.exists(V1_JSONL):
    v1 = {}
    for line in io.open(V1_JSONL, encoding='utf-8'):
        if line.strip():
            o = json.loads(line); v1[o['sid']] = o
    same_ctx = sum(1 for r in records if v1.get(r['sid'], {}).get('context') == r['context'])
    same_can = sum(1 for r in records if v1.get(r['sid'], {}).get('canaries', [{}])[0].get('value') == r['canaries'][0]['value']
                   and v1[r['sid']]['canaries'][2].get('value') == r['canaries'][2]['value'])
    # phone 按审计要求换成逐记录唯一值，context 里那一行必然不同 -> 按"仅 phone 不同"复核
    same_ctx_mod = sum(1 for r in records
                       if v1[r['sid']]['context'].replace(v1[r['sid']]['canaries'][1]['value'],
                                                          r['canaries'][1]['value']) == r['context'])
    v1_para = {v['text'] for r in v1.values() for v in r['variants'] if v.get('ci') == 0 and v.get('kind') == 'paraphrase'}
    v1_em_para_distinct = (len(v1_para), len(v1))
    v1_note = ("context 逐条一致 %d/%d；把 v1 的 phone 串替换为 v2 的逐记录唯一 phone 后逐条一致 %d/%d"
               "（即除 phone 外 canonical context 与 v1 完全同源）；email/account canary 一致 %d/%d；"
               "v1 email paraphrase 只有 %d 种写法覆盖 %d 条记录"
               % (same_ctx, len(records), same_ctx_mod, len(records), same_can, len(records),
                  len(v1_para), len(v1)))
    if sha(V1_JSONL) != V1_SHA:
        v1_note += "  [WARN] v1 文件 sha256 与登记值不一致"

# ---------------------------------------------------------------- 冻结
with io.open(OUT_JSONL, 'w', encoding='utf-8') as fh:
    for r in records:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

man = dict(
    schema_version=2,
    created_utc="2026-10-16",
    purpose="cross-dataset sampling for unsupervised clustering desensitization (PrivTrim-IoT)",
    supersedes=dict(file="frozen_samples.jsonl", sha256=V1_SHA,
                    note="v1 原样保留以存证（已登记 docs/06_数据资产.md）；v2 仅用于 variant 实验"),
    n_per_source_target=N_PER_SOURCE,
    total_records=len(records),
    per_source=[dict(source=s, n=n, note=note) for s, n, note in stats],
    canary_types=["email", "phone", "account"],
    variant_kinds=sorted({v['kind'] for v in allv}),
    variant_roles=sorted({v['role'] for v in allv}),
    variant_role_counts={k: sum(1 for v in allv if v['role'] == k) for k in sorted({v['role'] for v in allv})},
    variant_per_kind={"%s|%s" % k: by_kind[k] for k in sorted(by_kind)},
    positions=POSITIONS,
    variant_injection=dict(
        field="context_variants", span_field="variant_spans",
        canonical_field="context",
        canonical_field_note="context 是 canonical-only 上下文：源文本/块格式/位置轮换与 v1 完全一致，"
                             "唯一差别是 phone 值按审计要求改为逐记录唯一（清单 v1_compare 给出逐条对拍）；"
                             "变体不入 context，避免污染 canonical 评估",
        position_rotation="canonical=i%3 与 variant=(i+1)%3 错开，两块不重叠",
        line_format="<type>|<kind>: <text>",
        n_variant_spans=sum(len(r['variant_spans']) for r in records),
        independent_kinds=sorted({v['kind'] for v in ind}),
        subsumption_probe_kinds=sorted({v['kind'] for v in sub}),
        subsumption_probe_note="归一化后是 canonical 的子串，只能作为 subsumption 探针，不能当作独立检出证据",
    ),
    self_checks=dict(
        every_variant_located_on_its_span=True,
        every_variant_carries_canary_material=True,
        email_paraphrase_distinct="%d/%d" % (len(set(em_para)), len(em_para)),
        email_paraphrase_distinct_v1=("%d/%d" % v1_em_para_distinct) if v1_em_para_distinct else None,
        phone_unique="%d/%d" % (len(set(ph)), len(ph)),
        phone_8char_windows_unique="%d/%d" % (len(set(ph_win)), len(ph_win)),
        canonical_positions_rotated=POSITIONS,
    ),
    v1_compare=v1_note,
    frozen=True, seed=SEED,
    source_paths=dict(squad=os.path.join(C, 'external', 'squad_v11', 'dev-v1.1.json'),
                      hotpotqa=os.path.join(C, 'external', 'hotpotqa', 'dev_distractor.parquet'),
                      privacylens=os.path.join(C, 'external', 'PrivacyLens', 'data', 'main_data.json'),
                      pec_home=os.path.join(C, 'external', 'PEC-Home', 'data', 'release'),
                      smartbench=os.path.join(C, 'external', 'SmartBench', 'context-dependent', 'tdcs_rbe'),
                      simuhome=os.path.join(C, 'research', 'simuhome_generated_tasks_v1', 'tasks'),
                      toolmem=os.path.join(C, 'research', 'mem2act_scope_v1', 'data', 'toolmem_conversation.jsonl')),
    file_sha256=sha(OUT_JSONL),
)
io.open(OUT_MAN, 'w', encoding='utf-8').write(json.dumps(man, ensure_ascii=False, indent=2))

print("records:", len(records))
for s, n, note in stats:
    print("  %-18s %4d  %s" % (s, n, note))
print("variants: %d 条（independent %d / subsumption_probe %d / format_only %d / canonical %d）"
      % (len(allv), len(ind), len(sub),
         sum(1 for v in allv if v['role'] == 'format_only'), sum(1 for v in allv if v['role'] == 'canonical')))
print("变体注入: context_variants + %d 个 span，全部按区间复核命中" % sum(len(r['variant_spans']) for r in records))
print("email paraphrase distinct: %d/%d（v1 实测 %s，常量串）"
      % (len(set(em_para)), len(em_para),
         ("%d/%d" % v1_em_para_distinct) if v1_em_para_distinct else "1/581"))
print("phone 唯一性: 号码 %d/%d，8 位窗口 %d/%d" % (len(set(ph)), len(ph), len(set(ph_win)), len(ph_win)))
print("v1 对拍:", v1_note)
print("v1 未改动:", V1_JSONL, V1_SHA[:16], "（本脚本只读）")
print("sha256:", man['file_sha256'])
print("out:", OUT_JSONL)
print("manifest:", OUT_MAN)
