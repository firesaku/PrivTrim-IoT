"""
V1b 诊断：redact_miss=12.35% 是真漏删还是启发式误报？
抽若干个 red_var_tok=True 的样本，打印证据（命中了哪个变体/哪条规则、上下文片段）。
CPU only。
"""
import io, json, re, gzip, glob, os, argparse, sys, random
from collections import Counter

EMAIL   = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE   = re.compile(r'(?<!\w)\+1-\d{3}-\d{3}-\d{4}(?!\d)')
ACCOUNT = re.compile(r'\bACCT-[A-Za-z0-9-]+\b')

def redact(t):
    for p in (EMAIL, PHONE, ACCOUNT):
        t = p.sub('[REDACTED]', t)
    return t

def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower())

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
        out += ["%s-%s-%s-%s" % (d[0], a, e, l), "+1.%s.%s.%s" % (a, e, l),
                "+1 (%s) %s-%s" % (a, e, l), "+1 %s %s %s" % (a, e, l), d,
                " ".join(DIGIT_WORDS.get(c, c) for c in d),
                "the private phone number ending in %s" % l]
    else:
        out += [v.lower(), v.replace('-', ' ', 1), mk, "ACCT " + mk,
                "acct-" + mk, "the account id %s" % mk, "account #%s" % mk]
    return out

def why(secret, text):
    """返回命中的规则列表（用于区分真漏删与误报）"""
    hits = []
    n = norm(text)
    toks = set(re.findall(r'[a-z0-9]+', text.lower()))
    for v in variants(secret):
        nv = norm(v)
        if nv and nv in n:
            hits.append("variant_substr:%s" % v[:40])
    feat = set(re.findall(r'[a-z0-9]{3,}', secret['marker'].lower()))
    feat |= set(re.findall(r'[a-z0-9]{3,}', secret['value'].lower()))
    if feat and feat <= toks:
        hits.append("tokenset:%s" % ",".join(sorted(feat))[:50])
    d = re.sub(r'\D', '', secret['value'])
    if len(d) >= 7:
        for name, chunk in (("d3_7", d[3:7]), ("d6_10", d[6:10]), ("last4", d[-4:])):
            if chunk and chunk in n:
                hits.append("digitchunk_%s:%s" % (name, chunk))
    return hits

def find_cases():
    for pat in [r"/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/*/run/qwen25_7b/cases.jsonl.gz",
                r"/dpc/kuex0095/100065732/**/cases.jsonl.gz"]:
        g = glob.glob(pat, recursive=True)
        if g:
            return sorted(g, key=os.path.getsize)[-1]
    return None

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--nsample', type=int, default=40)
a = ap.parse_args()

cp = find_cases()
print("cases:", cp, flush=True)
rule = Counter()
shown = 0
lines = []
scanned = 0
rng = random.Random(7)
for line in gzip.open(cp, 'rt', encoding='utf-8', errors='replace'):
    line = line.strip()
    if not line:
        continue
    try:
        c = json.loads(line)
    except Exception:
        continue
    scanned += 1
    red = redact(c.get('context', ''))
    for s in c.get('canaries', []):
        h = why(s, red)
        if h:
            for x in h:
                rule[x.split(':')[0]] += 1
            if shown < a.nsample and rng.random() < 0.05:
                shown += 1
                lines.append("CASE %s | type=%s value=%s marker=%s" % (
                    c.get('id', '?')[:40], s['type'], s['value'], s['marker']))
                lines.append("   HITS: %s" % " ; ".join(h))
                m = re.search(r'\[REDACTED\]', red)
                idx = m.start() if m else 0
                lines.append("   REDACTED_CTX[%d:%d]=%s" % (max(0, idx-90), idx+90,
                                                            red[max(0, idx-90):idx+90].replace("\n", " | ")))
                lines.append("")

L = []
w = L.append
w("V1b DIAGNOSTIC — is redact_miss a true leak or a heuristic false positive?")
w("cases_scanned=%d" % scanned)
w("")
w("命中规则计数：")
for k, v in rule.most_common():
    w("   %-16s %d" % (k, v))
w("")
w("样本证据：")
L.extend(lines)
io.open(a.out, 'w', encoding='utf-8').write("\n".join(L))
print("\n".join(L[:120]))
print("wrote", a.out)
