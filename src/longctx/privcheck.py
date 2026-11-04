"""privcheck -- the predicate module shared by V1 / V1b (single source of truth).

CPU only, zero model calls, no third-party dependency.

Why it exists: v1_checker.py and v1b_diag.py each used to carry their own copy of
norm/redact/variants/variant_substr/tokenset, and the copies had already drifted
(diag ran the old digit-chunk rule, checker ran the fixed one), so the 1.14%
figure had never been independently re-checked.  The predicates are now defined
here once and both sides import this file, so structural drift is impossible.

Public predicates (pure functions: secret dict + text -> verdict):
  norm               normalisation: lowercase and strip every non-[a-z0-9]
  redact             regex redaction: email / phone / account -> [REDACTED]
  variants           enumerate the paraphrase variants of one secret
  variant_substr     a variant (normalised) occurs in the text as a whole
  tokenset           every characteristic token occurs as a token (any order)
  exposure_basic     the project's original verdict (V1 scheme A),
                     returns (full-value hit, marker-fragment hit)
  exposure_hit       boolean form of exposure_basic = "the original checker sees it"
  exposure_variants  schemes B/C: variant enumeration (+ unordered token set),
                     with the corrected digit-chunk rule applied to phone only
  hit_reasons        attribution: list every reason (verdict and attribution share
                     one code path, so they can never disagree)
  legacy_b1_*        re-computation entry point for the historical B1 bug (the
                     digit-chunk rule applied to every type); diagnostics only

Containments already established by the audit (recorded openly, not hidden;
privcheck_selftest.py asserts them):
  1) tokenset is a subset of exposure_basic.
     tokenset requires the alphanumeric blocks of the marker to occur as tokens.
     In this corpus all 53925/53925 markers are single [a-z0-9]+ strings (length
     11-12) and text tokens remain contiguous after norm(), so norm(marker)
     necessarily occurs as a contiguous substring of norm(text); and because
     len(marker) >= fragment_length (measured 11/12 >= 6/8), every
     fragment_length-long slice of the marker hits -> exposure_basic's frag
     branch is necessarily true.
     Corollary: tokenset adds no detection power beyond exposure_basic; what it
     adds is readable attribution, not a higher detection ceiling.
  2) all seven account variants are contained in exposure_basic (either identical
     after normalisation, or containing the marker as a whole):
       v.lower() / v.replace('-', ' ', 1) / "ACCT "+mk / "acct-"+mk
                                    -> norm() equal to norm(value) -> full branch
       mk                           -> contains norm(marker)       -> frag branch
       "the account id %s" % mk / "account #%s" % mk
                                    -> contains norm(marker)       -> frag branch
     Corollary: for account, exposure_variants has zero gain over exposure_basic.
     (The email local/domain variants and the phone spoken variant are not in
     this list; they do add detection power.)

Performance contract: n= and toks= are cache parameters only.  A caller may reuse
the norm()/tokenize() result of one text across several canaries of the same
case; passing values equivalent to norm(text)/tokenize(text) does not change the
semantics.
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
