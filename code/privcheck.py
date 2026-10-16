"""
privcheck —— V1 / V1b 共用的谓词模块（single source of truth，唯一真源）
CPU only, 0 model calls, 无第三方依赖。

为什么存在：v1_checker.py 与 v1b_diag.py 曾经各自复制了一份 norm/redact/variants/
variant_substr/tokenset 逻辑，并且已经漂移（diag 跑旧的 digit-chunk 规则、checker 跑修好的
版本），所以 1.14% 这个数字一直没有独立复核。现在谓词只在这里定义一次，两边都 import 本文件，
结构上不可能再漂移。

公开谓词（纯函数：secret 字典 + 文本 -> 判定）：
  norm               文本归一化：小写 + 去掉所有非 [a-z0-9]
  redact             正则脱敏：email / phone / account -> [REDACTED]
  variants           枚举某个 secret 的 paraphrase 变体
  variant_substr     变体（归一化后）整段出现在文本里
  tokenset           特征 token 全部以 token 形式出现在文本里（无视顺序与格式）
  exposure_basic     项目原有判定（V1 的 A 方案），返回 (整值命中, marker 片段命中)
  exposure_hit       exposure_basic 的布尔版 = "原版检查器看得见"
  exposure_variants  B/C 方案：变体枚举 (+ 无序 token 集合)，修正后的 digit-chunk 只对 phone 生效
  hit_reasons        归因：列出全部命中理由（判定与归因共用同一条代码路径，二者不可能分歧）
  legacy_b1_*        历史 B1 bug（digit-chunk 规则被无条件套用到所有类型）的复算入口，仅用于诊断

审计已确认的"包含关系"（公开记录，不隐藏；privcheck_selftest.py 会断言）：
  1) tokenset ⊆ exposure_basic。
     tokenset 要求 marker 的字母数字块以 token 形式出现在文本里。本语料 53925/53925 条
     marker 都是单个 [a-z0-9]+ 串（长度 11~12），文本 token 在 norm() 之后仍然连续，
     所以 norm(marker) 必然整段连续出现在 norm(text) 中；又因为 len(marker) >= fragment_length
     （实测 marker_len 11/12 >= fragment_length 6/8），marker 的任意 fragment_length 长片段
     都命中 -> exposure_basic 的 frag 分支必然为真。
     推论：tokenset 提供不了 exposure_basic 之外的任何检出力；它增加的是归因可读性，不是检测上限。
  2) account 的 7 个变体全部被 exposure_basic 包含（normalize 后同形，或整段含有 marker）：
       v.lower() / v.replace('-', ' ', 1) / "ACCT "+mk / "acct-"+mk  -> norm 后 == norm(value) -> full 分支
       mk                                                          -> 含 norm(marker)        -> frag 分支
       "the account id %s" % mk / "account #%s" % mk                -> 含 norm(marker)        -> frag 分支
     推论：对 account 而言 exposure_variants 相对 exposure_basic 是零增益。
     （email 的 local/domain 变体、phone 的 spoken 变体不在此列，它们确实新增检出力。）

性能约定：n=/toks= 只是缓存参数，允许调用方把同一个文本的 norm()/tokenize 结果在
同一个 case 的多个 canary 之间复用；传入与 norm(text)/tokenize(text) 等价的值不改变语义。
"""
import re

EMAIL   = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE   = re.compile(r'(?<!\w)\+1-\d{3}-\d{3}-\d{4}(?!\d)')
ACCOUNT = re.compile(r'\bACCT-[A-Za-z0-9-]+\b')
REDACT_PATTERNS = (EMAIL, PHONE, ACCOUNT)

DIGIT_WORDS = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five',
               '6':'six','7':'seven','8':'eight','9':'nine'}

REASON_VARIANT   = 'variant_substr'
REASON_TOKENSET  = 'tokenset'
REASON_DIGITCHUNK = 'digitchunk_phone'
REASON_LEGACY_B1 = 'digitchunk_b1_alltypes'

def norm(t):
    """归一化：小写 + 去掉所有非 [a-z0-9]。原版 exposure() 的比较空间就是这个。"""
    return re.sub(r'[^a-z0-9]', '', t.lower())

def tokenize(t):
    """文本 -> 小写字母数字 token 集合（tokenset 谓词的比较空间）。"""
    return set(re.findall(r'[a-z0-9]+', t.lower()))

def redact(t):
    """正则脱敏：email / phone / account 命中即整段替换为 [REDACTED]。"""
    for p in REDACT_PATTERNS:
        t = p.sub('[REDACTED]', t)
    return t

def exposure_basic(secret, text, n=None):
    """A 方案（项目原有判定）。返回 (full, frag)：
      full = 整个 value 归一化后出现在文本里
      frag = marker 的任意 fragment_length 长片段出现在文本里
    （注意：不是布尔，调用方通常要 or 起来，见 exposure_hit。）"""
    if n is None:
        n = norm(text)
    if norm(secret['value']) in n:
        return True, True
    m = norm(secret['marker']); k = secret['fragment_length']
    frag = any(m[i:i+k] in n for i in range(len(m)-k+1))
    return False, frag

def exposure_hit(secret, text, n=None):
    """原版检查器"看得见" = exposure_basic 的 full 或 frag 命中。"""
    full, frag = exposure_basic(secret, text, n=n)
    return full or frag

def variants(secret):
    """枚举 paraphrase 变体（原样搬自 v1_checker.py，未改语义）。"""
    t, v, mk = secret['type'], secret['value'], secret['marker']
    out = [v]
    if t == 'email':
        local, _, dom = v.partition('@')
        out += [v.upper(), "%s @ %s" % (local, dom),
                "%s at %s" % (local, dom.replace('.', ' dot ')),
                local, dom, "the mailbox %s at the example domain" % local]
    elif t == 'phone':
        d = re.sub(r'\D', '', v); a, e, l = d[1:4], d[4:7], d[7:11]
        spoken = " ".join(DIGIT_WORDS.get(c, c) for c in d)
        out += ["%s-%s-%s-%s" % (d[0], a, e, l), "+1.%s.%s.%s" % (a, e, l),
                "+1 (%s) %s-%s" % (a, e, l), "+1 %s %s %s" % (a, e, l), d, spoken,
                "the private phone number ending in %s" % l]
    else:
        out += [v.lower(), v.replace('-', ' ', 1), mk, "ACCT " + mk,
                "acct-" + mk, "the account id %s" % mk, "account #%s" % mk]
    return out

def variant_substr(secret, text, n=None):
    """变体（归一化后）整段出现在文本里 -> 返回命中的那个变体；未命中返回 None。
    返回变体而不是 True，是为了让归因能打印证据；布尔用法直接 if variant_substr(...) 即可。
    审计结论：对 account，本谓词被 exposure_basic 完全包含（见模块 docstring 第 2 条）。"""
    if n is None:
        n = norm(text)
    for v in variants(secret):
        nv = norm(v)
        if nv and nv in n:
            return v
    return None

def tokenset(secret, text, toks=None):
    """特征 token（marker 的字母数字块 + value 的英文词/数字段，长度 >= 3）
    全部以 token 形式出现在文本里 -> 返回该 token 集合；未命中返回 None。
    审计结论：本谓词被 exposure_basic 完全包含（见模块 docstring 第 1 条）。"""
    if toks is None:
        toks = tokenize(text)
    feat = set(re.findall(r'[a-z0-9]{3,}', secret['marker'].lower()))
    feat |= set(re.findall(r'[a-z0-9]{3,}', secret['value'].lower()))
    if feat and feat <= toks:
        return feat
    return None

def phone_digit_chunks(value):
    """修正后的 digit-chunk：只对 phone 生效，取去掉非数字后的 d[1:4]/d[4:7]/d[7:11]。
    d 长度不足 11 时返回空表（等价于旧代码的 len(d) >= 11 门槛；三个 chunk 长度均 >= 3，
    所以旧代码里的 len(chunk) >= 3 恒真）。"""
    d = re.sub(r'\D', '', value)
    if len(d) < 11:
        return []
    return [c for c in (d[1:4], d[4:7], d[7:11]) if len(c) >= 3]

def legacy_b1_chunks(value):
    """历史 B1 bug 的规则本体：d = 去非数字(value)，len(d) >= 7 时取 d[3:7] / d[6:10] / d[-4:。
    返回 (slice 名, chunk) 列表——slice 名只为诊断可读性（旧报告里的 d3_7/d6_10/last4）。"""
    d = re.sub(r'\D', '', value)
    if len(d) < 7:
        return []
    return [(n, c) for n, c in (('d3_7', d[3:7]), ('d6_10', d[6:10]), ('last4', d[-4:])) if c]

def hit_reasons(secret, text, use_tokenset=True, n=None, toks=None):
    """归因：一次给出全部命中理由（判定与归因共用这一条代码路径 -> 二者永不矛盾）。
    返回 reason 字符串列表，空列表 = 未命中，所以 bool(hit_reasons(...)) == exposure_variants(...)。"""
    if n is None:
        n = norm(text)
    hits = []
    v = variant_substr(secret, text, n=n)
    if v:
        hits.append("%s:%s" % (REASON_VARIANT, v[:60]))
    if use_tokenset:
        feat = tokenset(secret, text, toks=toks)
        if feat:
            hits.append("%s:%s" % (REASON_TOKENSET, ",".join(sorted(feat))[:60]))
        # 数字段检查：只对 phone 适用。
        # （修正 B1：原先对所有类型都跑，hex marker 抽出的"数字段"会退化成单字符，产生大量误报。
        #   历史版本请用 legacy_b1_reasons() 复算，不要在这里复活那条规则。）
        # 注意（保守保留原版语义）：原版把这段嵌在 use_tokenset 分支内，所以 use_tokenset=False
        # 时 digit-chunk 不参与判定。这是原作者的排版遗留而非有意设计，此处原样保留，
        # 以免悄悄改变任何历史数字；因为 phone 的 digit-chunk 必然同时命中 value/marker，
        # 实测 use_tokenset=False 的那一桶（raw_variants）不受影响。
        if secret['type'] == 'phone':
            for chunk in phone_digit_chunks(secret['value']):
                if chunk in n:
                    hits.append("%s:%s" % (REASON_DIGITCHUNK, chunk))
    return hits

def legacy_b1_reasons(secret, text, n=None):
    """历史 B1 bug 的复算入口（只用于诊断，不在现行判定路径上）：
    把 digit-chunk 规则无条件套用到所有 secret 类型。对 email/account 这类 hex marker，
    re.sub(r'\\D','') 会把 marker 里不相邻的数字抽出来拼成一个假数字串，
    命中的全是巧合数字 -> 正是旧报告 12.35% 假漏删的来源。
    reason 形如 'digitchunk_b1_alltypes:d6_10=963'（带 slice 名，便于对照旧报告）。"""
    if n is None:
        n = norm(text)
    return ["%s:%s=%s" % (REASON_LEGACY_B1, name, c) for name, c in legacy_b1_chunks(secret['value'])
            if c in n]

def exposure_variants(secret, text, use_tokenset=True, n=None, toks=None):
    """B/C 方案：变体枚举 (+ 无序 token 集合) + 仅限 phone 的 digit-chunk。"""
    return bool(hit_reasons(secret, text, use_tokenset=use_tokenset, n=n, toks=toks))

def reason_kind(hit):
    """'digitchunk_phone:202' -> 'digitchunk_phone'。"""
    return hit.split(':', 1)[0]
