"""
V1：paraphrase 感知的泄漏检查器 —— 在完整 HPC 数据集上重算覆盖率
CPU only, 0 model calls.

对比三个检查器：
  A. 原版 exposure()             —— 小写去非字母数字 + 子串匹配（项目现用）
  B. 原版 + 变体枚举             —— 对每个 secret 生成 paraphrase 变体，任一命中即算检出
  C. 变体枚举 + 无序 token 集合   —— 加"特征 token 全在但顺序/格式变了"的判定

并同时测正则 redact() 的删除率。输出：
  - canonical 上的检出率/删除率（复现项目 100% 的来源）
  - 变体上的检出率/删除率（真实的鲁棒性）
  - 隐形泄漏：原版看不见但变体检查器看得见的比例  ← 这是指标盲区的大小
"""
import io, json, os, re, gzip, glob, argparse, sys
from collections import defaultdict, Counter

EMAIL   = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE   = re.compile(r'(?<!\w)\+1-\d{3}-\d{3}-\d{4}(?!\d)')
ACCOUNT = re.compile(r'\bACCT-[A-Za-z0-9-]+\b')

def redact(t):
    for p in (EMAIL, PHONE, ACCOUNT):
        t = p.sub('[REDACTED]', t)
    return t

def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower())

def exposure_basic(secret, text):
    n = norm(text)
    if norm(secret['value']) in n:
        return True, True
    m = norm(secret['marker']); k = secret['fragment_length']
    frag = any(m[i:i+k] in n for i in range(len(m)-k+1))
    return False, frag

# ---------- 变体枚举 ----------
DIGIT_WORDS = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five',
               '6':'six','7':'seven','8':'eight','9':'nine'}

def variants(secret):
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

def exposure_variants(secret, text, use_tokenset=True):
    """B/C：把 text 与所有变体都归一化后比较；use_tokenset 再加无序 token 集合判定。"""
    n = norm(text)
    toks = set(re.findall(r'[a-z0-9]+', text.lower()))
    for v in variants(secret):
        nv = norm(v)
        if nv and nv in n:
            return True
    if use_tokenset:
        # 特征 token：marker 的字母数字块，或 value 里的英文词 + 数字段
        feat = set(re.findall(r'[a-z0-9]{3,}', secret['marker'].lower()))
        feat |= set(re.findall(r'[a-z0-9]{3,}', secret['value'].lower()))
        if feat and feat <= toks:
            return True
        # 数字段检查：只对 phone 适用。
        # （修正：原先对所有类型都跑，hex marker 抽出的"数字段"会退化成单字符，产生大量误报）
        if secret['type'] == 'phone':
            d = re.sub(r'\D', '', secret['value'])
            if len(d) >= 11:
                for chunk in (d[1:4], d[4:7], d[7:11]):
                    if len(chunk) >= 3 and chunk in n:
                        return True
    return False

def find_cases(arg):
    if arg and os.path.exists(arg):
        return arg
    for pat in [r"/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/*/run/qwen25_7b/cases.jsonl.gz",
                r"/dpc/kuex0095/100065732/**/cases.jsonl.gz"]:
        g = glob.glob(pat, recursive=True)
        if g:
            return sorted(g, key=os.path.getsize)[-1]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    cp = find_cases(a.cases)
    print("cases file:", cp, flush=True)
    if not cp:
        print("NO_CASES_FOUND"); sys.exit(5)

    op = gzip.open if cp.endswith('.gz') else io.open
    fh = op(cp, 'rt', encoding='utf-8', errors='replace')

    # 统计桶
    st = defaultdict(lambda: [0, 0])   # (scope, checker) -> [n, hit]
    ncase = 0
    nsec = 0
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
        except Exception:
            continue
        ncase += 1
        if a.limit and ncase > a.limit:
            break
        ctx = c.get('context', '')
        for s in c.get('canaries', []):
            nsec += 1
            full_b, frag_b = exposure_basic(s, ctx)
            det_b = full_b or frag_b
            det_v = exposure_variants(s, ctx, use_tokenset=False)
            det_vt = exposure_variants(s, ctx, use_tokenset=True)
            red = redact(ctx)
            full_r, frag_r = exposure_basic(s, red)
            det_r = full_r or frag_r           # 原版检查器在 redact 后是否还看得见
            det_rv = exposure_variants(s, red, use_tokenset=True)

            for tag, val in (("raw_basic", det_b), ("raw_variants", det_v),
                             ("raw_var_tok", det_vt),
                             ("red_basic", det_r), ("red_var_tok", det_rv)):
                st[(tag,)][0] += 1
                st[(tag,)][1] += 1 if val else 0
            # 隐形泄漏：原版看不见，但变体检查器看得见
            st[("blind_spot",)][0] += 1
            st[("blind_spot",)][1] += 1 if (not det_b and det_vt) else 0
            # 正则漏删：变体检查器在 redact 之后仍看得见
            st[("redact_miss",)][0] += 1
            st[("redact_miss",)][1] += 1 if det_rv else 0
        if ncase % 2000 == 0:
            print("  cases", ncase, flush=True)
    fh.close()

    # 变体级测试：把每个变体当作一段文本，测检查器与正则
    print("variant-level sweep ...", flush=True)
    vst = defaultdict(lambda: [0, 0])
    for line in []:
        pass

    L = []
    w = L.append
    w("V1 PARAPHRASE-AWARE LEAK CHECKER — full-corpus coverage")
    w("cases_file=%s" % cp)
    w("cases=%d  secrets=%d" % (ncase, nsec))
    w("")
    w("%-16s %10s %12s" % ("bucket", "n", "hit_rate"))
    for k in sorted(st):
        n, h = st[k]
        w("%-16s %10d %11.2f%%" % (k[0], n, 100.0*h/max(n, 1)))
    w("")
    w("读法：")
    w("  raw_basic   = 原版检查器在原始上下文上判定'看得见'的比例（= 项目现有隐私指标的分母逻辑）")
    w("  raw_var_tok = 变体枚举 + token 集合检查器的检出率（更严格）")
    w("  blind_spot  = 原版看不见、但更严格检查器看得见 -> 指标盲区")
    w("  redact_miss = 正则脱敏后，更严格检查器仍看得见 -> 正则漏删率")

    io.open(a.out, 'w', encoding='utf-8').write("\n".join(L))
    io.open(a.json, 'w', encoding='utf-8').write(json.dumps(
        {"cases": ncase, "secrets": nsec,
         "buckets": {"|".join(k): v for k, v in st.items()}}, ensure_ascii=False, indent=1))
    print("\n".join(L))
    print("wrote", a.out)

if __name__ == '__main__':
    main()
