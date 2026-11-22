# -*- coding: utf-8 -*-
"""build_longctx_cases.py —— 长上下文、确定性评分的 IoT/agent QA 基准构建器。

为什么存在（问题陈述）
---------------------
项目现有评测卡在 SQuAD / HotpotQA：上下文只有 135 / 910 词，最强公开基线
mask_quito_x 在 50% 预算下只掉 0.75 F1 —— 这两个语料上"压缩已经解决"，
问不出压缩真正重要的问题。冻结集里的长上下文 IoT/agent 源（toolmem 1201 词、
privacylens 588、smartbench_tdcs 584、simuhome 436、pec_home 187）恰恰**没有 gold
answer**，所以压缩问题无处可问。本脚本把这些真实源变成"答案按构造可导"的
长上下文 QA 案例。

核心原则：**答案必须是源自身的结构化记录里已经发射出来的字段值**，
问题由该字段确定性渲染，全程不用任何 LLM 写题/写答案。

案例类型
--------
1) needle（针 + 干扰）
   答案只存在于长日志中某一个可辨识的区域（entry），其余 entry 是同源的合理填充。
   丢掉正确区域就丢答案，丢掉填充是免费的。
   逐例给出答案所在区域的字符偏移 answer_char_offset。
   构建期硬校验：context.count(answer) == 1 且答案落在 needle 区域内。

2) multi_entity（多实体消歧，对齐 conflict 基准）
   同一类型（account）的 k 个不同实体出现在日志里，问题只问其中一个，答案要求把
   该实体链接到"在别处陈述"的属性（属性在 record 正文里，实体标识在 canary 记录块里）。
   删除式脱敏会把 k 个实体压成同一个 [REDACTED]，链接随之毁灭 —— 这正是只做删除
   可证伪的地方。逐例给出 n_entities / target_entity / target_attribute。
   构建期硬校验：实体两两不同、属性两两不同（=> 属性由目标实体唯一确定）、
   gold 属性在 context 中只出现一次、且只出现在目标 record 区域内。

数据源（全部为真实源，已逐个核对 schema，见 MANIFEST.source_schemas）
------------------------------------------------------------------
  <source>      crossdata/frozen_samples.jsonl 中 source ∈ {toolmem, privacylens,
                smartbench_tdcs, simuhome, pec_home} 的记录；
                record 文本 = 冻结 context 逐字节原文（含 canary 记录块）。
  toolmem_raw   research/mem2act_scope_v1/data/toolmem_conversation.jsonl（~31.8 MB，
                2029 个 session）的原始会话，按确定规则渲染成 role: content 日志；
                tool_calls 一并渲染，因为冻结渲染只取 content、把它丢了。

用法
----
    python build_longctx_cases.py --out ..\\longctx\\longctx_cases.jsonl \
        --manifest ..\\longctx\\MANIFEST_longctx.json --min-words 800 --seed 20261016

确定性：排序后的 glob / 排序后的 record 列表 / 无 set 迭代序依赖 / 单一 seeded RNG /
       sorted(obj.keys()) / json.dumps 固定 key 顺序 / newline="\n"。
       换 PYTHONHASHSEED 重跑，产出 sha256 必须逐字节一致。
"""
import argparse
import collections
import hashlib
import io
import json
import os
import random
import re

# ---------------------------------------------------------------- 源路径（真实文件）
CYBER = r'C:\Users\ADMIN\Documents\ChatGPT\Cybersec_IoT'
FROZEN = r'D:\CodexData\tmp\privtrim-review-20261015\crossdata\frozen_samples.jsonl'
TOOLMEM_RAW = os.path.join(CYBER, 'research', 'mem2act_scope_v1', 'data',
                           'toolmem_conversation.jsonl')

SCHEMA_VERSION = 1
FROZEN_SOURCES = ['pec_home', 'privacylens', 'simuhome', 'smartbench_tdcs', 'toolmem']
POOLS = FROZEN_SOURCES + ['toolmem_raw']

# 与 build_crossdata.py 完全一致的 canary 记录块格式与插入点位
BLOCK = "Private contact record:\nContact email: {e}\nContact phone: {p}\nAccount ID: {a}"
POSITIONS = ['prefix', 'suffix', 'middle']
SEP = '----- entry %d -----'
DEC = json.JSONDecoder()


def log(*a):
    print('[build_longctx]', *a, flush=True)


def need(cond, msg):
    """显式 raise（不用 assert）—— python -O 下 assert 会被跳过，审计要求"响亮"。"""
    if not cond:
        raise RuntimeError('自检失败: ' + msg)


def sha256_text(t):
    return hashlib.sha256(t.encode('utf-8')).hexdigest()


def nwords(t):
    return len(re.findall(r'\S+', t))


# ---------------------------------------------------------------- canary 派生
# 配方与 build_crossdata.py::mk_canaries / mk_phone 一致：email/account 由
# sha256("canary:"+sid) 派生，phone 由 sha256("phone:"+sid:salt) 派生并做确定性去重。
# 构建期用全部冻结记录对拍 email/account，防止配方漂移。
PHONE_USED = set()


def mk_phone(sid):
    for salt in range(10000):
        hh = hashlib.sha256(('phone:%s:%d' % (sid, salt)).encode()).hexdigest()
        d = ''.join(str(int(ch, 16) % 10) for ch in hh[:10])
        npa = str(2 + int(hh[10], 16) % 8) + d[0:2]
        val = '+1-%s-555-%s' % (npa, d[2:6])
        dig = re.sub(r'\D', '', val)
        win = [dig[k:k + 8] for k in range(4)]
        if dig in PHONE_USED or any(x in PHONE_USED for x in win):
            continue
        PHONE_USED.add(dig)
        PHONE_USED.update(win)
        return val
    raise RuntimeError('phone 确定性去重失败: %s' % sid)


def mk_canaries(sid):
    h = hashlib.sha256(('canary:' + sid).encode()).hexdigest()
    em, ac = h[:12], h[12:24]
    phone = mk_phone(sid)
    return [dict(type='email', value='contact.%s@example.invalid' % em, marker=em,
                 fragment_length=6),
            dict(type='phone', value=phone, marker=re.sub(r'\D', '', phone),
                 fragment_length=8),
            dict(type='account', value='ACCT-' + ac, marker=ac, fragment_length=6)]


def block_of(canaries):
    d = {c['type']: c['value'] for c in canaries}
    return BLOCK.format(e=d['email'], p=d['phone'], a=d['account'])


def strip_block(ctx, blk):
    """把 canary 记录块从 context 里摘掉，还原源文本（三种插入位置都覆盖）。"""
    if blk in ctx:
        return ctx.replace(blk, '').strip(), True
    return ctx, False


def insert_block(ctx, blk, pos):
    if pos == 'prefix':
        return blk + '\n\n' + ctx
    if pos == 'suffix':
        return ctx + '\n\n' + blk
    cut = len(ctx) // 2
    sp = ctx.find(' ', cut)
    sp = sp if sp > 0 else cut
    return ctx[:sp] + '\n\n' + blk + '\n\n' + ctx[sp:]


# ---------------------------------------------------------------- 结构化字段抽取
# 全部用"带终止符"的正则从源**原文**里取字段：值后面的闭引号/分隔符必须在场，
# 否则说明该字段被 [:6000]/[:8000] 截断切掉了 —— 一律丢弃，宁缺勿造。
def jstr(txt, key):
    m = re.search(r'"%s":\s*"([^"\n]{1,120})"' % re.escape(key), txt)
    return m.group(1) if m else None


def jnum(txt, key, maxlen=12):
    m = re.search(r'"%s":\s*(-?\d+(?:\.\d+)?)(?=[,\}\]\s])' % re.escape(key), txt)
    if not m:
        return None
    v = m.group(1)
    return v if len(v) <= maxlen else None


def jlist_str(txt, key):
    m = re.search(r'"%s":\s*\[([^\]]*)\]' % re.escape(key), txt)
    if not m:
        return None
    items = re.findall(r'"([^"]*)"', m.group(1))
    return ', '.join(items) if items else None


def usable(v):
    """只保留可判别、可唯一化的取值（够长、含字母数字、无换行、纯 1-2 位数不要）。"""
    if v is None:
        return False
    v = str(v).strip()
    if len(v) < 3 or len(v) > 80 or '\n' in v:
        return False
    if not re.search(r'[A-Za-z0-9]', v):
        return False
    if re.fullmatch(r'-?\d{1,2}', v):
        return False
    return True


def distinctive(v):
    """多实体属性额外要求"可判别"：含字母，或够长的数字串（降低子串碰撞）。"""
    v = str(v)
    return bool(re.search(r'[A-Za-z]', v)) or len(v) >= 4


def tool_json_objs(txt):
    """解出 `tool: {json}` 序列。

    冻结渲染把换行压成空格，多个 tool 结果会连成一行（"... tool: {a} tool: {b} ..."），
    所以必须按字面 'tool: ' 切分，再用 JSONDecoder.raw_decode 取合法 JSON 前缀
    （对 [:8000] 造成的截断容错）。
    """
    out = []
    for part in re.split(r'tool: (?=\{)', txt)[1:]:
        try:
            obj, _ = DEC.raw_decode(part.lstrip())
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


IDENT_KEYS = ('video', 'word', 'sentence', 'name', 'file', 'location', 'query', 'title',
              'city', 'country', 'symbol', 'id', 'booking_id', 'product', 'topic')


# ------- 每源 fact 抽取：[(fact_key, value, label, anchor_kind, anchor)] -------
def f_smartbench(text):
    ent = jstr(text, 'id')
    out = []
    v = jstr(text, 'occupancy_type')
    if usable(v):
        out.append(('occupancy_type', v, 'the occupancy type', 'id', ent))
    v = jnum(text, 'duration_days')
    if usable(v):
        out.append(('duration_days', v, 'the observation duration in days', 'id', ent))
    for key, lab, pat in (('total_events', 'the total number of events',
                           r'total_events=(\d+)'),
                          ('top_rooms', 'the top rooms',
                           r'top_rooms=([A-Za-z][A-Za-z_,]{2,60})')):
        m = re.search(pat, text)
        if m and usable(m.group(1)):
            out.append((key, m.group(1), lab, 'id', ent))
    seen = set()
    for m in sorted(re.finditer(
            r'Evidence: ([a-z_0-9]{4,40}) \(([a-z_ ]{2,30})\): (\d+) events', text),
            key=lambda x: x.start()):
        if m.group(1) in seen:
            continue
        seen.add(m.group(1))
        out.append(('evidence_events', m.group(3),
                    'the number of events for device %s' % m.group(1), 'id', ent))
    return out


def f_simuhome(text):
    ad = re.search(r'"anchor":\s*\{"room_id":\s*"[^"]+",\s*"device_id":\s*"([^"]+)"', text)
    anchor = ad.group(1) if ad else None
    out = []
    for key, lab in (('seed', 'the simulator seed'), ('num_rooms', 'the number of rooms'),
                     ('num_devices', 'the number of devices'),
                     ('query_type', 'the query type'), ('scenario', 'the scenario name'),
                     ('base_time', 'the base time')):
        v = jnum(text, key) if key in ('seed', 'num_rooms', 'num_devices') else jstr(text, key)
        if usable(v):
            out.append((key, v, lab, 'anchor_device', anchor))
    if usable(anchor):
        out.append(('anchor_device', anchor, 'the anchor device identifier',
                    'anchor_device', anchor))
    return out


def f_pec_home(text):
    ent = jstr(text, 'id')
    out = []
    for key, lab in (('device', 'the target device'), ('persona_id', 'the persona id'),
                     ('personalized_operation', 'the personalized operation'),
                     ('room', 'the target room'), ('room_id', 'the target room identifier'),
                     ('source_record_id', 'the source record identifier'),
                     ('task', 'the task name')):
        v = jstr(text, key)
        if usable(v):
            out.append((key, v, lab, 'id', ent))
    return out


def f_privacylens(text):
    ent = jstr(text, 'name')
    out = []
    for key, lab in (('data_subject', 'the data subject'), ('data_sender', 'the data sender'),
                     ('data_recipient', 'the data recipient'), ('data_type', 'the data type'),
                     ('transmission_principle', 'the transmission principle'),
                     ('user_name', 'the user name'), ('user_email', 'the user email'),
                     ('final_action', 'the final action'),
                     ('data_type_concrete', 'the concrete data type')):
        v = jstr(text, key)
        if usable(v):
            out.append((key, v, lab, 'name', ent))
    v = jlist_str(text, 'toolkits')
    if usable(v):
        out.append(('toolkits', v, 'the toolkits', 'name', ent))
    return out


def f_toolmem(text):
    out = []
    for oi, obj in enumerate(tool_json_objs(text)):
        ident = None
        for k in IDENT_KEYS:
            if isinstance(obj.get(k), str) and obj[k].strip():
                ident = obj[k]
                break
        for k in sorted(obj.keys()):            # sorted：与 PYTHONHASHSEED 无关
            v = obj[k]
            if k == 'status' or not isinstance(v, (str, int, float, bool)):
                continue
            v = str(v).strip()
            if not usable(v):
                continue
            out.append(('%s@%s' % (k, (ident[:32] if ident else 'r%d' % oi)), v,
                        'the %s field' % k, 'native' if ident else 'entry', ident))
    return out


RENDER_HDR = ('session {sid} | turns={n} | token_count={tok} | has_tool_calls={htc}\n'
              'original_conversation_ids: {ids}\n')


def render_toolmem_session(o):
    """把原始 session 渲染成 role: content 日志（tool_calls 一并渲染）。

    冻结渲染（build_crossdata.py::x_toolmem）只取 t['content']，于是**丢掉了
    tool_calls**（实体参数就在 arguments 里）。本池存在的理由就是保留它。
    """
    parts = [RENDER_HDR.format(sid=o.get('session_id'), n=o.get('turn_count'),
                               tok=o.get('token_count'), htc=o.get('has_tool_calls'),
                               ids=', '.join(o.get('original_conversation_ids') or []))]
    for t in o.get('turns') or []:
        c = t.get('content')
        if isinstance(c, str) and c.strip():
            parts.append('%s: %s' % (t.get('role') or '?', c))
        for tc in t.get('tool_calls') or []:
            fn = tc.get('function') or {}
            parts.append('assistant: [tool_call id=%s name=%s] %s'
                         % (tc.get('id'), fn.get('name'), fn.get('arguments')))
    return '\n'.join(parts)


def f_toolmem_raw(text):
    out = []
    m = re.search(r'^session (\S+) \| turns=(\d+) \| token_count=(\d+) \| has_tool_calls=(\S+)$',
                  text, re.M)
    sess = m.group(1) if m else None
    if m:
        out.append(('token_count', m.group(3), 'the token count', 'session', sess))
        out.append(('turn_count', m.group(2), 'the number of turns', 'session', sess))
    for tcm in re.finditer(r'\[tool_call id=(\S+) name=([^\]]*)\] (\{.*)$', text, re.M):
        try:
            obj, _ = DEC.raw_decode(tcm.group(3))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for k in sorted(obj.keys()):
            v = obj[k]
            if not isinstance(v, (str, int, float, bool)):
                continue
            if not usable(v):
                continue
            out.append(('%s@call' % k, str(v).strip(), 'the %s argument' % k, 'session', sess))
    for obj in tool_json_objs(text):
        for k in sorted(obj.keys()):
            v = obj[k]
            if k == 'status' or not isinstance(v, (str, int, float, bool)):
                continue
            if not usable(v):
                continue
            out.append(('%s@result' % k, str(v).strip(), 'the %s field' % k, 'session', sess))
    return out


EXTRACT = {'smartbench_tdcs': f_smartbench, 'simuhome': f_simuhome, 'pec_home': f_pec_home,
           'privacylens': f_privacylens, 'toolmem': f_toolmem, 'toolmem_raw': f_toolmem_raw}

ANCHOR_DESC = {'id': 'the record with id %s', 'name': 'the record named %s',
               'session': 'session %s', 'native': 'the entry describing %s',
               'anchor_device': 'the entry whose anchor device is %s', 'entry': 'entry %s'}


# ---------------------------------------------------------------- record 池
def load_frozen(path):
    recs = collections.defaultdict(list)
    for line in io.open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get('source') in FROZEN_SOURCES:
            recs[o['source']].append(o)
    for k in recs:
        recs[k].sort(key=lambda r: r['sid'])
    return recs


def build_pool_frozen(src, frozen):
    pool = []
    for o in frozen[src]:
        can = o['canaries']
        ctx = o['context']
        src_text, ok = strip_block(ctx, block_of(can))
        need(ok, '冻结记录里找不到 canary 记录块: %s' % o['sid'])
        pool.append(dict(rid=o['sid'], source=src, text=ctx, src_text=src_text,
                         canaries=can, canary_provenance='frozen',
                         block_pos=o.get('canary_position'),
                         frozen_sid=o['sid'],
                         facts=EXTRACT[src](src_text), meta=o.get('meta') or {},
                         pool_record_sha256=sha256_text(ctx)))
    return pool


def build_pool_toolmem_raw(path):
    pool = []
    for line in io.open(path, encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        rid = 'toolmem_raw::%s' % o.get('session_id')
        text = render_toolmem_session(o)
        can = mk_canaries(rid)
        pos = POSITIONS[len(pool) % 3]
        pool.append(dict(rid=rid, source='toolmem_raw', text=insert_block(text, block_of(can), pos),
                         src_text=text, canaries=can, canary_provenance='derived',
                         block_pos=pos, frozen_sid=None,
                         facts=EXTRACT['toolmem_raw'](text),
                         meta=dict(session=o.get('session_id'),
                                   turn_count=o.get('turn_count'),
                                   token_count=o.get('token_count')),
                         pool_record_sha256=None))
    for r in pool:
        r['pool_record_sha256'] = sha256_text(r['text'])
    pool.sort(key=lambda r: r['rid'])
    return pool


# ---------------------------------------------------------------- 装配
def render_log(recs):
    """拼成长日志；返回 (text, spans)，spans[i] = (start, end, rid)。"""
    parts, spans, cur = [], [], 0
    for i, r in enumerate(recs):
        head = SEP % i
        st = cur + len(head) + 1
        parts.append(head + '\n' + r['text'] + '\n\n')
        spans.append((st, st + len(r['text']), r['rid']))
        cur += len(head) + 1 + len(r['text']) + 2
    return ''.join(parts), spans


def dedup_canaries(recs):
    seen, out = set(), []
    for r in recs:
        for c in r['canaries']:
            k = (c['type'], c['value'])
            if k in seen:
                continue
            seen.add(k)
            out.append(dict(c))
    out.sort(key=lambda c: (c['type'], c['value']))
    return out


def canonical_records(recs):
    return [dict(rid=r['rid'], source=r['source'], pool_record_sha256=r['pool_record_sha256'],
                 words=nwords(r['text']), canary_provenance=r['canary_provenance'])
            for r in recs]


def pick_fillers(pool, needle, order, forbidden, min_words, max_words, min_distractors):
    """确定性挑填充，三轮：
      1) 尊重词数硬上限，凑够 min_words + min_distractors；
      2) 仍不够 min_words 就放宽上限（保证长度下限优先）；
      3) 仍不够 min_distractors 就补最短的还能塞进去的填充。
    """
    picked, words = [], nwords(needle['text'])

    def ok(r):
        return (r['rid'] != needle['rid']
                and not any(f and f in r['text'] for f in forbidden)
                and all(p['rid'] != r['rid'] for p in picked))

    for r in order:                                            # 第 1 轮：硬上限
        if words >= min_words and len(picked) >= min_distractors:
            break
        if ok(r) and words + nwords(r['text']) <= max_words:
            picked.append(r)
            words += nwords(r['text'])
    if words < min_words:                                      # 第 2 轮：放宽上限
        for r in order:
            if words >= min_words and len(picked) >= min_distractors:
                break
            if ok(r):
                picked.append(r)
                words += nwords(r['text'])
    if len(picked) < min_distractors:                          # 第 3 轮：补最短的
        for r in sorted([x for x in order if ok(x)], key=lambda x: (nwords(x['text']), x['rid'])):
            if len(picked) >= min_distractors or words + nwords(r['text']) > max_words:
                break
            picked.append(r)
            words += nwords(r['text'])
    return picked, words


def build_needle_case(src, needle, fact, pool, order, cfg, cid):
    fk, val, label, akind, anchor = fact
    fillers, _w = pick_fillers(pool, needle, order, [val], cfg['min_words'],
                               cfg['max_words'], cfg['min_distractors'])
    pos = cfg['rng'].randrange(len(fillers) + 1) if fillers else 0
    recs = fillers[:pos] + [needle] + fillers[pos:]
    ctx, spans = render_log(recs)
    ns = next(s for s in spans if s[2] == needle['rid'])
    entry_idx = [s[2] for s in spans].index(needle['rid'])

    need(val in ctx, 'needle gold 不在 context 里: %s %s' % (needle['rid'], fk))
    need(ctx.count(val) == 1, 'needle gold 在 context 里不唯一: %s %s' % (needle['rid'], fk))
    aoff = ctx.index(val)
    need(ns[0] <= aoff < ns[1], 'needle gold 不在 needle 区域内')

    # anchor：优先用源自身的标识；不唯一就退回 entry 编号（entry 编号用分隔行做唯一性探针）
    anchor_val, anchor_kind, probe = str(entry_idx), 'entry', SEP % entry_idx
    if akind != 'entry' and anchor and usable(anchor) and ctx.count(anchor) == 1:
        anchor_val, anchor_kind, probe = anchor, akind, anchor
    need(ctx.count(probe) == 1, 'needle anchor 不唯一: %s' % needle['rid'])

    words = nwords(ctx)
    need(words >= cfg['min_words'], 'needle 案例词数低于下限: %d' % words)
    return dict(
        schema_version=SCHEMA_VERSION, key='longctx-%s-needle-%04d' % (src, cid),
        case_kind='needle', dataset='longctx', source=src,
        question='In the log above, what is %s recorded for %s?'
                 % (label, ANCHOR_DESC[anchor_kind] % anchor_val),
        answers=[val], context=ctx, context_sha256=sha256_text(ctx),
        attribute_key=fk, target_attribute=val,
        anchor=anchor_val, anchor_kind=anchor_kind,
        needle_rid=needle['rid'], needle_frozen_sid=needle['frozen_sid'],
        answer_char_offset=ns[0], answer_value_offset=aoff,
        region_start=ns[0], region_end=ns[1], region_words=nwords(needle['text']),
        answer_region_index=entry_idx,
        entry_spans=[[s[0], s[1], s[2]] for s in spans],
        words=words, n_records=len(recs), n_distractors=len(fillers), entry_index=entry_idx,
        canaries=dedup_canaries(recs), pool_records=canonical_records(recs))


def pick_multi_attr(pool, k):
    """挑一个"均匀属性"：在尽量多的 record 里出现，且能给出 >=k 个可判别、互不相同的取值。

    只在"取值在该 record 原文里确实出现"时才计入 —— 保证属性值真的能被读到。
    返回 (attribute_key, label, [(value, rid)])；不存在返回 None。
    """
    by_key = collections.defaultdict(lambda: collections.defaultdict(list))
    labels, texts = {}, {r['rid']: r['text'] for r in pool}
    for r in pool:
        for fk, v, label, _ak, _an in r['facts']:
            if '@' in fk:                       # 带实体后缀的 fact 不是均匀属性
                continue
            if v not in texts[r['rid']]:
                continue
            by_key[fk][v].append(r['rid'])
            labels.setdefault(fk, label)
    best = None
    for fk in sorted(by_key.keys()):
        vals = by_key[fk]
        uniq = [v for v in vals if len(vals[v]) == 1 and distinctive(v)]
        n_recs = len({rid for v in vals for rid in vals[v]})
        if n_recs < k or len(uniq) < k:
            continue
        # 优先"含字母"的属性（比纯数字更像可读的多实体消歧任务），再比可取值的数量
        n_letter = len([v for v in uniq if re.search(r'[A-Za-z]', v)])
        cand = (n_recs, n_letter, len(uniq))
        if best is None or cand > best[0]:
            best = (cand, fk)
    if best is None:
        return None
    fk = best[1]
    # 取值按 (record 词数, 取值) 排序：确定性，且优先用短 record，避免多实体案例被
    # 单条超长会话撑爆（toolmem_raw 的会话最长 4716 词）
    pairs = sorted(((v, by_key[fk][v][0]) for v in sorted(by_key[fk])
                    if len(by_key[fk][v]) == 1 and distinctive(v)),
                   key=lambda p: (nwords(texts[p[1]]), p[0]))
    return fk, labels[fk], pairs


def build_multi_case(src, pool, cfg, cid, picked_attr, k):
    """一个多实体案例：k 个同源 record，各带一个 account 实体与一个互异属性值。

    唯一性口径（与需求一致）：属性取值在 k 个实体间两两互异 => 属性由目标实体唯一确定；
    并且目标属性 / 目标实体都不得出现在别的 record 区域里（否则可以绕过目标实体拿到答案）。
    """
    fk, label, pairs = picked_attr
    rmap = {r['rid']: r for r in pool}
    pairs = [p for p in pairs if nwords(rmap[p[1]]['text']) <= cfg['max_needle_words']]
    if len(pairs) < k:
        return None
    # 同源多个多实体案例之间轮换候选集合（纯 cid 索引，不用 rng）——否则 cid 只改变
    # 顺序与目标，记录集合完全重复
    rot = (cid * max(1, k)) % len(pairs)
    pairs = pairs[rot:] + pairs[:rot]
    chosen = list(pairs[:k])
    idx = k
    # 词数下限必须按"装配后的日志"算（含分隔行），所以装配-检查-加长来回迭代
    while True:
        if len({rid for _v, rid in chosen}) != len(chosen):
            return None
        recs = [rmap[rid] for _v, rid in chosen]
        attr_of = {rid: v for v, rid in chosen}
        ctx, spans = render_log(recs)
        if nwords(ctx) >= cfg['min_words'] or idx >= len(pairs):
            break
        chosen = chosen + [pairs[idx]]
        idx += 1
    order = list(range(len(recs)))
    cfg['rng'].shuffle(order)
    recs = [recs[i] for i in order]
    ctx, spans = render_log(recs)               # 顺序定了才装配：spans 必须与 recs 对齐
    n_ent = len(recs)
    if n_ent < 2 or nwords(ctx) < cfg['min_words']:
        return None
    ent_of = {r['rid']: [c['value'] for c in r['canaries'] if c['type'] == 'account'][0]
              for r in recs}
    ent_vals = [ent_of[r['rid']] for r in recs]
    attr_vals = [attr_of[r['rid']] for r in recs]
    if len(set(ent_vals)) != n_ent or len(set(attr_vals)) != n_ent:
        return None
    segs = [ctx[s[0]:s[1]] for s in spans]
    start = cfg['rng'].randrange(n_ent)
    for off in range(n_ent):                    # 依次试目标实体，全部条件通过才收
        ti = (start + off) % n_ent
        tgt_ent, tgt_attr = ent_vals[ti], attr_vals[ti]
        ts = spans[ti]
        if tgt_attr not in segs[ti] or tgt_ent not in segs[ti]:
            continue
        if any(tgt_attr in segs[j] or tgt_ent in segs[j]
               for j in range(n_ent) if j != ti):
            continue
        if not (ts[0] <= ctx.index(tgt_attr) < ts[1] and ts[0] <= ctx.index(tgt_ent) < ts[1]):
            continue
        return dict(
            schema_version=SCHEMA_VERSION, key='longctx-%s-multi-%04d' % (src, cid),
            case_kind='multi_entity', dataset='longctx', source=src,
            question='In the log above, which %s is recorded for the account %s?'
                     % (label.replace('the ', ''), tgt_ent),
            answers=[tgt_attr], context=ctx, context_sha256=sha256_text(ctx),
            n_entities=n_ent, target_entity=tgt_ent, target_attribute=tgt_attr,
            entity_type='account', entity_key='account', attribute_key=fk,
            entity_values=ent_vals, attribute_values=attr_vals,
            target_region_start=ts[0], target_region_end=ts[1],
            target_attribute_offset=ts[0] + segs[ti].index(tgt_attr),
            target_attribute_occurrences_in_region=segs[ti].count(tgt_attr),
            entity_offset=ts[0] + segs[ti].index(tgt_ent),
            answer_char_offset=ts[0],
            answer_value_offset=ts[0] + segs[ti].index(tgt_attr),
            answer_region_index=ti,
            entry_spans=[[s[0], s[1], s[2]] for s in spans],
            words=nwords(ctx), n_records=len(recs), guess_baseline=round(1.0 / n_ent, 4),
            canaries=dedup_canaries(recs), pool_records=canonical_records(recs))
    return None


# ---------------------------------------------------------------- 统计小工具
def pct(xs, p):
    if not xs:
        return 0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--min-words', type=int, default=800, help='长度下限（词），默认 800')
    ap.add_argument('--max-words', type=int, default=4800,
                    help='装配长度硬上限（词），默认 4800（对应 e2e 的 --maxlen 8192）')
    ap.add_argument('--max-needle-words', type=int, default=1400,
                    help='needle/multi 候选 record 自身的词数上限，默认 1400（挡掉超长会话）')
    ap.add_argument('--min-distractors', type=int, default=3)
    ap.add_argument('--multi-entities', type=int, default=5)
    ap.add_argument('--n-needle-per-source', type=int, default=12)
    ap.add_argument('--n-multi-per-source', type=int, default=8)
    ap.add_argument('--sources', default=','.join(POOLS))
    ap.add_argument('--seed', type=int, default=20261016)
    ap.add_argument('--frozen', default=FROZEN)
    ap.add_argument('--toolmem-raw', default=TOOLMEM_RAW)
    ap.add_argument('--force', action='store_true', help='允许覆盖已存在的产物（默认拒绝）')
    a = ap.parse_args()

    for p in (a.out, a.manifest):
        if os.path.exists(p) and not a.force:
            raise SystemExit('产物已存在，拒绝覆盖（加 --force 才覆盖）: %s' % p)

    frozen = load_frozen(a.frozen) if os.path.exists(a.frozen) else {}

    # ---- canary 派生配方对拍：防止 build_crossdata.py 的派生逻辑漂移 ----
    drift_n = drift_bad = 0
    for src in FROZEN_SOURCES:
        for o in frozen.get(src, []):
            d = {c['type']: c['value'] for c in o['canaries']}
            mine = {c['type']: c['value'] for c in mk_canaries(o['sid'])}
            drift_n += 1
            if mine['email'] != d['email'] or mine['account'] != d['account']:
                drift_bad += 1
    if frozen:
        need(drift_n > 0 and drift_bad == 0,
             'canary 派生配方与冻结集不一致 (%d/%d)' % (drift_bad, drift_n))
        log('canary 配方对拍: email+account %d/%d 与冻结集逐条一致（phone 走 v2 逐记录唯一配方）'
            % (drift_n - drift_bad, drift_n))
    else:
        log('[WARN] 冻结集缺失，跳过 canary 配方对拍: %s' % a.frozen)

    pools = {}
    for src in [s.strip() for s in a.sources.split(',') if s.strip()]:
        if src == 'toolmem_raw':
            if not os.path.exists(a.toolmem_raw):
                log('跳过 toolmem_raw：找不到 %s' % a.toolmem_raw)
                continue
            pools[src] = build_pool_toolmem_raw(a.toolmem_raw)
        elif src in frozen:
            pools[src] = build_pool_frozen(src, frozen)
        else:
            log('跳过 %s：冻结集里没有该源' % src)
    need(pools, '没有任何可用源')

    # 全局唯一性：email/account 由 sha256(sid) 派生，必须全局唯一；
    # phone 在 v1 冻结集里是"来源内序号"共享池（95 个号码覆盖 389 条，见
    # build_crossdata.py 第 4 条），重复是已登记的历史事实，只统计不判失败。
    allcan, allph = collections.Counter(), collections.Counter()
    for src in sorted(pools):
        for r in pools[src]:
            for c in r['canaries']:
                (allph if c['type'] == 'phone' else allcan)[(c['type'], c['value'])] += 1
    dup = sorted(k for k, v in allcan.items() if v > 1)
    need(not dup, 'email/account canary 不全局唯一: %s' % dup[:5])
    ph_dup = sum(v - 1 for v in allph.values() if v > 1)
    log('canary 唯一性: email/account %d 个互不相同；phone %d 个取值、%d 个重复'
        '（v1 共享号码池，已登记）' % (len(allcan), len(allph), ph_dup))

    rng = random.Random(a.seed)
    cfg = dict(min_words=a.min_words, max_words=a.max_words,
               max_needle_words=a.max_needle_words,
               min_distractors=a.min_distractors, multi_entities=a.multi_entities, rng=rng)

    cases, stats = [], []
    for src in sorted(pools):
        pool = pools[src]
        order = sorted(pool, key=lambda r: hashlib.sha256(
            ('order:%s:%s' % (src, r['rid'])).encode()).hexdigest())
        # needle 候选：record 词数不得超过 --max-needle-words（把超长会话挡在外面，
        # 否则单条 toolmem_raw 会话就能把案例撑到 8000 词）
        cand = [(r, f) for r in sorted(pool, key=lambda r: r['rid'])
                if nwords(r['text']) <= a.max_needle_words for f in r['facts']]
        rng.shuffle(cand)
        n_needle = 0
        for r, f in cand:
            if n_needle >= a.n_needle_per_source:
                break
            try:
                cases.append(build_needle_case(src, r, f, pool, order, cfg, n_needle))
            except RuntimeError:
                continue
            n_needle += 1
        # multi_entity 候选：从 --multi-entities 起逐级降 k，取最大的可行 k
        n_multi, why, pa, k_eff = 0, '', None, 0
        for k in range(a.multi_entities, 1, -1):
            pa = pick_multi_attr(pool, k)
            if pa is not None:
                k_eff = k
                break
        if pa is None:
            why = ('该源没有"均匀属性"（同一字段在 >=2 个 record 里出现、且能给出 >=2 个'
                   '可判别且互不相同的取值），确定性多实体抽取不成立 -> 该源只出 needle')
        else:
            for cid in range(a.n_multi_per_source):
                try:
                    c = build_multi_case(src, pool, cfg, cid, pa, k_eff)
                except RuntimeError:
                    break
                if c is None:
                    break
                cases.append(c)
                n_multi += 1
        stats.append(dict(source=src, pool_records=len(pool),
                          facts_total=sum(len(r['facts']) for r in pool),
                          needle=n_needle, multi_entity=n_multi,
                          multi_attribute_key=(pa[0] if pa else None),
                          multi_entities_effective=(k_eff if pa else 0),
                          multi_skipped_reason=why,
                          canary_provenance=(pool[0]['canary_provenance'] if pool else None),
                          mean_pool_words=round(sum(nwords(r['text']) for r in pool)
                                                / max(1, len(pool)), 1)))

    need(cases, '没有产出任何案例')
    keys = [c['key'] for c in cases]
    need(len(set(keys)) == len(keys), 'case key 不唯一')

    # ---- 逐案例独立复核（不复用构建分支的结论）----
    for c in cases:
        ctx = c['context']
        need(all(ans in ctx for ans in c['answers']), 'gold 不在 context: %s' % c['key'])
        need(c['context_sha256'] == sha256_text(ctx), 'context sha256 对不上: %s' % c['key'])
        need(nwords(ctx) >= a.min_words, '低于词数下限: %s' % c['key'])
        for ct in ('email', 'phone', 'account'):                 # 案例内实体值必须互异
            vs = [x['value'] for x in c['canaries'] if x['type'] == ct]
            need(len(set(vs)) == len(vs), '案例内 %s canary 不互异: %s' % (ct, c['key']))
        if c['case_kind'] == 'multi_entity':
            need(len(set(c['entity_values'])) == c['n_entities'], 'multi 实体不互异: %s' % c['key'])
            need(len(set(c['attribute_values'])) == c['n_entities'],
                 'multi 属性不互异（属性不再由实体唯一确定）: %s' % c['key'])
            need(c['target_entity'] in c['entity_values'], 'multi 目标实体不在集合: %s' % c['key'])
            need(c['target_attribute'] in c['attribute_values'], 'multi 目标属性不在集合')
            need(c['target_region_start'] <= c['target_attribute_offset'] < c['target_region_end'],
                 'multi 属性偏移越界: %s' % c['key'])
            need(c['target_attribute_offset'] == c['answer_value_offset'],
                 'multi answer_value_offset 与属性偏移不一致: %s' % c['key'])
            segs = [ctx[s[0]:s[1]] for s in c['entry_spans']]
            ti = c['answer_region_index']
            need(segs[ti].count(c['target_attribute']) ==
                 c['target_attribute_occurrences_in_region'], 'multi 属性出现次数对不上')
            for v in c['attribute_values']:
                need(any(v in s for s in segs), 'multi 属性在某 record 里读不到: %s' % v)
            for j, s in enumerate(segs):
                if j == ti:
                    continue
                need(c['target_attribute'] not in s, 'multi gold 属性出现在别的 record 区域')
                need(c['target_entity'] not in s, 'multi 目标实体出现在别的 record 区域')
        else:
            need(ctx.count(c['answers'][0]) == 1, 'needle gold 不唯一: %s' % c['key'])
            need(c['region_start'] <= c['answer_value_offset'] < c['region_end'],
                 'needle 偏移越界: %s' % c['key'])

    # ---- 逐 record 溯源复核：pool record 文本必须逐字节出现在 context 里 ----
    for c in cases:
        for rec in c['pool_records']:
            hit = [s for s in pools[rec['source']] if s['rid'] == rec['rid']]
            need(len(hit) == 1, 'pool record 找不到: %s' % rec['rid'])
            need(hit[0]['text'] in c['context'],
                 'pool record 未逐字节出现在 context 里: %s' % rec['rid'])
            need(sha256_text(hit[0]['text']) == rec['pool_record_sha256'],
                 'pool record sha256 对不上: %s' % rec['rid'])

    cases.sort(key=lambda c: (c['source'], c['case_kind'], c['key']))
    body = ''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    io.open(a.out, 'w', encoding='utf-8', newline='\n').write(body)
    file_sha = sha256_text(body)

    words = [c['words'] for c in cases]
    buckets = collections.Counter()
    for w in words:
        buckets[800 if w < 1000 else (1000 if w < 1500 else (1500 if w < 2000 else
                                                             (2000 if w < 3000 else 3000)))] += 1
    per_src = collections.Counter(c['source'] for c in cases)
    per_kind = collections.Counter(c['case_kind'] for c in cases)
    per_sk = collections.Counter((c['source'], c['case_kind']) for c in cases)

    man = dict(
        schema_version=SCHEMA_VERSION,
        artifact=os.path.basename(a.out), artifact_path=os.path.abspath(a.out),
        file_sha256=file_sha,
        bytes=len(body.encode('utf-8')),
        purpose='long-context deterministically-scored QA built from the real IoT/agent '
                'sources (PrivTrim-IoT); gold answers are fields the source simulator/log '
                'itself emitted, never written by a model',
        build_command='python build_longctx_cases.py --out %s --manifest %s --min-words %d '
                      '--max-words %d --max-needle-words %d --min-distractors %d '
                      '--multi-entities %d --n-needle-per-source %d --n-multi-per-source %d '
                      '--sources %s --seed %d'
                      % (a.out, a.manifest, a.min_words, a.max_words, a.max_needle_words,
                         a.min_distractors, a.multi_entities, a.n_needle_per_source,
                         a.n_multi_per_source, a.sources, a.seed),
        seed=a.seed, min_words=a.min_words, max_words=a.max_words,
        min_distractors=a.min_distractors, multi_entities=a.multi_entities,
        n_cases=len(cases),
        per_source={k: per_src[k] for k in sorted(per_src)},
        per_case_kind={k: per_kind[k] for k in sorted(per_kind)},
        per_source_case_kind={'%s|%s' % k: per_sk[k] for k in sorted(per_sk)},
        per_source_detail=stats,
        length_distribution=dict(
            min=min(words), p10=pct(words, 10), p25=pct(words, 25), median=pct(words, 50),
            p75=pct(words, 75), p90=pct(words, 90), max=max(words),
            mean=round(sum(words) / len(words), 1),
            buckets_words={'800-999': buckets[800], '1000-1499': buckets[1000],
                           '1500-1999': buckets[1500], '2000-2999': buckets[2000],
                           '3000+': buckets[3000]},
            est_ctx_tokens_mean=round(sum(words) / len(words) * 1.33, 1),
            note='词数 = 空白切分。est_ctx_tokens 只是 1.33 词/token 的粗估；真实 token 数'
                 '由 e2e_longctx.py 用 tokenizer 实测并逐臂上报'),
        source_paths=dict(frozen=os.path.abspath(a.frozen),
                          toolmem_raw=os.path.abspath(a.toolmem_raw)),
        source_schemas=SOURCE_SCHEMAS,
        extraction=EXTRACTION_NOTES,
        self_checks=dict(
            gold_answer_in_context=True,
            needle_gold_unique_in_context=True,
            needle_answer_offset_inside_region=True,
            multi_entities_pairwise_distinct=True,
            multi_attributes_pairwise_distinct=True,
            multi_gold_attribute_unique_in_context=True,
            multi_target_entity_only_in_its_region=True,
            every_pool_record_verbatim_in_context=True,
            canary_recipe_matches_frozen_set='%d/%d email+account' % (drift_n - drift_bad, drift_n),
            canary_values_globally_unique_distinct=len(allcan),
            case_keys_unique=True,
            length_floor_words=a.min_words,
        ),
        determinism=dict(
            mechanism='sorted globs; sorted record lists; no set-iteration-order dependence; '
                      'single seeded RNG; sorted(obj.keys()) for tool-JSON facts; json.dumps '
                      'with fixed key insertion order; newline="\\n"',
            verify='用两个不同 PYTHONHASHSEED 重跑同一命令，sha256 必须逐字节一致',
        ),
        not_run_here=['7B 模型推理（本地无 GPU）—— 见 e2e_longctx.py',
                      'hotpotqa / squad（已有 gold，无需重建）'],
    )
    io.open(a.manifest, 'w', encoding='utf-8', newline='\n').write(
        json.dumps(man, ensure_ascii=False, indent=2) + '\n')

    log('cases=%d  sha256=%s' % (len(cases), file_sha))
    for s in stats:
        log('  %-18s pool=%4d facts=%5d needle=%3d multi=%3d %s'
            % (s['source'], s['pool_records'], s['facts_total'], s['needle'], s['multi_entity'],
               ('multi_attr=%s' % s['multi_attribute_key']) if s['multi_attribute_key']
               else ('SKIP: %s' % s['multi_skipped_reason'])))
    log('长度分布: min=%d p10=%d median=%d p90=%d max=%d mean=%.1f'
        % (min(words), pct(words, 10), pct(words, 50), pct(words, 90), max(words),
           sum(words) / len(words)))
    log('out=%s' % a.out)
    log('manifest=%s' % a.manifest)


# ---------------------------------------------------------------- schema / 抽取说明（进 MANIFEST）
SOURCE_SCHEMAS = {
    'crossdata/frozen_samples.jsonl': {
        'path': r'D:\CodexData\tmp\privtrim-review-20261015\crossdata\frozen_samples.jsonl',
        'format': 'jsonl, 581 行, 3860859 bytes, '
                  'sha256 cb44ab1745a9f72f76bb799a2e649541fc6fa79fd9b69717e9abce357cfd42ed',
        'fields': ['sid', 'source', 'question', 'context', 'answers', 'canaries', 'variants',
                   'canary_position', 'meta'],
        'per_source': {'smartbench_tdcs': 96, 'simuhome': 5, 'pec_home': 96, 'toolmem': 96,
                       'privacylens': 96, 'hotpotqa': 96, 'squad': 96},
        'canaries': "[{type: email|phone|account, value, marker, fragment_length}]",
        'context_construction': 'norm_ws(源文本)[:6000 或 8000] 与 canary 记录块按 '
                                'prefix/suffix/middle 三点位轮换拼接（build_crossdata.py）',
        'note': '5 个 IoT/agent 源的 answers 全为空数组 —— 这正是本构建器要解决的问题；'
                'hotpotqa/squad 有 gold，不在本构建器范围内',
    },
    'toolmem_conversation.jsonl（原始语料）': {
        'path': r'C:\Users\ADMIN\Documents\ChatGPT\Cybersec_IoT\research\mem2act_scope_v1'
                r'\data\toolmem_conversation.jsonl',
        'format': 'jsonl, 2029 行, 33361325 bytes（约 31.8 MB）',
        'fields': ['session_id', 'original_conversation_ids', 'turns', 'turn_count',
                   'has_tool_calls', 'token_count'],
        'turns_fields': ['role', 'content', 'source_id', 'tool_calls', 'tool_call_id', 'name'],
        'roles': {'user': 25701, 'assistant': 30910, 'tool': 19506},
        'tool_calls_shape': "[{id, type:'function', function:{name, arguments(JSON 字符串)}}]，"
                            "共 19591 个",
        'tool_result_shape': "role=='tool' 的 content 可 json.loads 成 dict（19346 条）；"
                             "顶层键高频：status 17282 / message 3561 / results 704 / data 619 / "
                             "location 415 / date 364 / summary 308 / cwd 299 / file_path 285 / "
                             "confidence 162 / price 158 / temperature / condition / city",
        'note': '冻结集里的 toolmem 渲染（build_crossdata.py::x_toolmem）只取 t["content"]，'
                '因此丢掉了 tool_calls —— 实体参数就在 arguments 里。本构建器的 toolmem_raw 池'
                '额外渲染 tool_calls，并渲染 session_id/turn_count/token_count 头行',
    },
    'raw_siblings_verified（已核对 schema，本构建器默认不取）': {
        'smartbench_tdcs': r'external\SmartBench\context-dependent\tdcs_rbe\{normal,anomaly}'
                           r'\*.json —— 600 个文件（normal 300 / anomaly 300），top keys: '
                           r'id, label, description, reasoning, context{occupancy_type,'
                           r'duration_days}, events[{timestamp, device, status}], '
                           r'anomaly_details, anomaly_types, evidence_devices, metadata, '
                           r'pair_id, paired_id',
        'pec_home': r'external\PEC-Home\data\release\multi_user_preferences.jsonl (?) + '
                    r'dynamic_user_preferences.jsonl (712 行)，keys: device, environment, id, '
                    r'persona, persona_id, personalized_operation, room, room_id, '
                    r'source_record_id, target_actions, task, turns',
        'privacylens': r'external\PrivacyLens\data\main_data.json —— list，493 项，'
                       r'keys: name, seed, vignette, trajectory',
        'simuhome': r'research\simuhome_generated_tasks_v1\tasks\*.json —— 5 个文件，'
                    r'keys: meta{seed,query_type,case,num_rooms,num_devices,scenario}, query, '
                    r'user_location, eval{goals,required_actions}, '
                    r'initial_home_config{base_time,rooms,devices,attributes,tick_interval}',
        'note': '这些真实源都在本地存在（路径已逐个 Test-Path 通过），schema 已核对。本构建器'
                '默认只从冻结节 + toolmem 原始语料取 record，以与项目其他实验保持同源、可审计；'
                '需要时可把它们接成新池',
    },
}

EXTRACTION_NOTES = {
    'principle': '答案一律是源自身结构化记录里已经发射出来的字段值；问题由该字段确定性渲染，'
                 '构建过程不调用任何模型/LLM',
    'truncation_guard': 'frozen context 被 [:6000]/[:8000] 截断，所以字段抽取全部用"带终止符"'
                        '正则（要求闭引号在场）；被切断的字段直接丢弃，不猜、不补',
    'per_source': {
        'smartbench_tdcs': 'occupancy_type / duration_days / total_events / top_rooms'
                           '（后者在 reasoning 文本里）/ evidence_events:<device>；anchor = 记录 id',
        'simuhome': 'meta 的 seed / num_rooms / num_devices / query_type / scenario / base_time '
                    '/ anchor_device；anchor = eval.goals[0].anchor.device_id',
        'pec_home': 'device / persona_id / personalized_operation / room / room_id / '
                    'source_record_id / task；anchor = 记录 id',
        'privacylens': 'name(anchor) / data_subject / data_sender / data_recipient / data_type / '
                       'transmission_principle / user_name / user_email / final_action / '
                       'data_type_concrete / toolkits',
        'toolmem': '逐个 `tool: {json}` 结果（raw_decode 截断容错）取标量字段；'
                   'fact key = <字段>@<实体>，anchor = 实体值；无"均匀属性" -> 只出 needle',
        'toolmem_raw': '会话头 session_id/turns/token_count + tool_call arguments 的标量参数 + '
                       'tool 结果字段；anchor = session_id',
    },
    'needle_assembly': '1 个 needle record + >=min-distractors 个同源填充；填充按 '
                       'sha256("order:<source>:<rid>") 排序后确定性挑选，且必须不含 gold；'
                       'needle 落在第几个 entry 由 seeded RNG 决定',
    'multi_assembly': 'k 个同源 record，每个贡献一个 account 实体（canary 记录块）+ 一个'
                      '"均匀属性"（取值在池内可判别且逐 record 唯一）。属性取值两两互异 '
                      '=> 属性由目标实体唯一确定；删除式脱敏把 k 个 account 压成同一个 '
                      '[REDACTED]，链接随之毁灭（对齐 conflict 基准）',
}


if __name__ == '__main__':
    main()
