"""修正版：分离"正则真的删掉了" vs "检查器根本看不见（被自动记为已移除）"。"""
import io, json, re, os
from collections import defaultdict, Counter

CASES = r"C:\Users\ADMIN\Documents\ChatGPT\Cybersec_IoT\research\token_compression_qa_v1\cases.json"
D = r"D:\CodexData\tmp\privtrim-review-20261015\paraphrase_probe"
os.makedirs(D, exist_ok=True)

EMAIL   = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE   = re.compile(r'(?<!\w)\+1-\d{3}-\d{3}-\d{4}(?!\d)')
ACCOUNT = re.compile(r'\bACCT-[A-Za-z0-9-]+\b')

def redact(t):
    for p in (EMAIL, PHONE, ACCOUNT):
        t = p.sub('[REDACTED]', t)
    return t

def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower())

def exposure(secret, text):
    n = norm(text)
    full = norm(secret['value']) in n
    m = norm(secret['marker']); k = secret['fragment_length']
    frag = any(m[i:i+k] in n for i in range(len(m)-k+1))
    return full, frag

def variants_email(v, mk):
    local, _, dom = v.partition('@')
    return [("0_canonical", v), ("1_case_upper", v.upper()),
            ("2_space_around_at", f"{local} @ {dom}"),
            ("3_at_dot_words", f"{local} at {dom.replace('.', ' dot ')}"),
            ("4_local_part_only", local), ("5_domain_only", dom),
            ("6_paraphrase", f"the contact email ending in {dom.split('.')[-1]}"),
            ("7_paraphrase_w_local", f"the mailbox {local} at the example domain"),
            ("8_mixed_case", local.capitalize() + "@" + dom.title())]

def variants_phone(v, mk):
    d = re.sub(r'\D', '', v); a, e, l = d[1:4], d[4:7], d[7:11]
    sp = " ".join({"0":"zero","1":"one","2":"two","5":"five"}.get(c, c) for c in d)
    return [("0_canonical", v), ("1_no_plus", f"{d[0]}-{a}-{e}-{l}"),
            ("2_dots", f"+1.{a}.{e}.{l}"), ("3_parens", f"+1 ({a}) {e}-{l}"),
            ("4_spaces", f"+1 {a} {e} {l}"), ("5_digits_only", d),
            ("6_spoken", sp), ("7_paraphrase", f"the private phone number ending in {l}")]

def variants_account(v, mk):
    return [("0_canonical", v), ("1_lower", v.lower()),
            ("2_space", v.replace('-', ' ', 1)), ("3_no_prefix", mk),
            ("4_prefix_space", "ACCT " + mk), ("5_paraphrase", f"the account id {mk}"),
            ("6_hash_style", f"account #{mk}"), ("7_reversed_pfx", "acct-" + mk)]

GEN = {'email': variants_email, 'phone': variants_phone, 'account': variants_account}

cases = json.loads(io.open(CASES, encoding='utf-8').read())

# 每类：canonical 与 variants 分开统计"正则真正删除(完整值)"的比例
stat = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # type -> group -> [n, removed_full]
blind = Counter()
per_variant = {}
rows = []

for case in cases:
    for s in case.get('canaries', []):
        t = s['type']
        for name, text in GEN[t](s['value'], s['marker']):
            det_full, _ = exposure(s, text)          # 变体本身能否被检查器看见
            rm_full, rm_frag = exposure(s, redact(text))
            if det_full:
                really_removed = not rm_full          # 看见 → 删没删掉可判定
            else:
                really_removed = None                 # 看不见 → 无法判定，指标自动记"已移除"
                blind[t] += 1
            grp = 'canonical' if name.startswith('0_') else 'variants'
            st = stat[t][grp]
            st[0] += 1
            st[1] += 1 if really_removed else 0
            per_variant.setdefault((t, name), [0, 0, 0])
            pv = per_variant[(t, name)]
            pv[0] += 1
            pv[1] += 1 if really_removed else 0
            pv[2] += 1 if really_removed is None else 0
            rows.append((t, name, det_full, rm_full, rm_frag))

def frac(n, d):
    return (100.0 * n / d) if d else float('nan')

L = []
w = L.append
w("MOTIVATION 摘要（24 案例 / 72 标识符 / %d 个变体样本）" % len(rows))
w("")
w("A. 正则真正删除率（完整值），canonical vs 变体")
w("%-10s %-26s %-26s %s" % ("类型", "canonical", "variants", "落差"))
for t in ['email', 'phone', 'account']:
    c = stat[t]['canonical']; v = stat[t]['variants']
    w("%-10s %-26s %-26s %s" % (
        t,
        "%d/%d = %.1f%%" % (c[1], c[0], frac(c[1], c[0])),
        "%d/%d = %.1f%%" % (v[1], v[0], frac(v[1], v[0])),
        "%.1f pp" % (frac(c[1], c[0]) - frac(v[1], v[0]))))
cn = sum(stat[t]['canonical'][0] for t in stat); cr = sum(stat[t]['canonical'][1] for t in stat)
vn = sum(stat[t]['variants'][0] for t in stat);  vr = sum(stat[t]['variants'][1] for t in stat)
w("%-10s %-26s %-26s %s" % ("合计", "%d/%d = %.1f%%" % (cr, cn, frac(cr, cn)),
                            "%d/%d = %.1f%%" % (vr, vn, frac(vr, vn)),
                            "%.1f pp" % (frac(cr, cn) - frac(vr, vn))))
w("")
w("B. 指标盲区：检查器看不见、因此'无法判定'的样本")
w("%-10s %s" % ("类型", "不可判定数 / 该类型变体数"))
for t in ['email', 'phone', 'account']:
    tot = sum(stat[t][g][0] for g in stat[t])
    w("%-10s %d / %d = %.1f%%" % (t, blind[t], tot, frac(blind[t], tot)))
w("%-10s %d / %d = %.1f%%" % ("合计", sum(blind.values()), len(rows), frac(sum(blind.values()), len(rows))))
w("")
w("（说明：exposure() 对小写去非字母数字后的字符串做子串匹配。当改述把原值打散后，")
w("  匹配失败 → 指标把该条记成'未检出'，而 pin_full/pin_fragment 的分母是'可检出的标识符'，")
w("  所以这些样本既不进泄漏计数、也不进移除计数，等于对指标隐形。）")
w("")
w("C. 逐变体明细（真删=检查器看见且删掉；盲=检查器看不见）")
w("%-9s %-22s %6s %8s %8s" % ("类型", "变体", "n", "真删%", "盲%"))
for (t, name), pv in sorted(per_variant.items()):
    w("%-9s %-22s %6d %8.1f %8.1f" % (t, name, pv[0], frac(pv[1], pv[0]), frac(pv[2], pv[0])))

io.open(os.path.join(D, "SUMMARY.txt"), 'w', encoding='utf-8').write("\n".join(L))
print("ok", len(L), "lines ->", os.path.join(D, "SUMMARY.txt"))
print("blind total:", sum(blind.values()), "of", len(rows))
