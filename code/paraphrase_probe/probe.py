"""
动机实验：量化"正则去敏 + 泄漏检查"在改述/格式变化下的覆盖缺口。
0 次模型调用，纯字符串。

口径完全复刻 research\token_compression_qa_v1\common.py：
  EMAIL / PHONE / ACCOUNT 三条正则；redact() 替换为 [REDACTED]
  normalize_identifier = 小写 + 去掉所有非字母数字
  exposure(secret, text) -> (full, fragment)
"""
import io, json, os, re, sys
from collections import defaultdict, Counter

CASES = r"C:\Users\ADMIN\Documents\ChatGPT\Cybersec_IoT\research\token_compression_qa_v1\cases.json"
OUT   = r"D:\CodexData\tmp\privtrim-review-20261015\paraphrase_probe\REPORT.md"

EMAIL   = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE   = re.compile(r'(?<!\w)\+1-\d{3}-\d{3}-\d{4}(?!\d)')
ACCOUNT = re.compile(r'\bACCT-[A-Za-z0-9-]+\b')

def redact(text):
    for pattern in (EMAIL, PHONE, ACCOUNT):
        text = pattern.sub('[REDACTED]', text)
    return text

def normalize_identifier(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())

def exposure(secret, text):
    normalized = normalize_identifier(text)
    full = normalize_identifier(secret['value']) in normalized
    marker = normalize_identifier(secret['marker'])
    k = secret['fragment_length']
    fragment = any(marker[i:i + k] in normalized for i in range(len(marker) - k + 1))
    return full, fragment

# ---------------------------------------------------------------- 变体生成
def variants_email(v, marker):
    local, _, domain = v.partition('@')
    return [
        ("0_canonical",          v),
        ("1_case_upper",         v.upper()),
        ("2_space_around_at",    f"{local} @ {domain}"),
        ("3_at_dot_words",       f"{local} at {domain.replace('.', ' dot ')}"),
        ("4_local_part_only",    local),
        # 原 5_domain_only = domain：24 例全是 'example.invalid'（distinct=1），不含任何 canary 材料，
        # 对空壳评估"正则删掉了没有"必然虚假通过。换成 build_crossdata.py::variants_for 的 at_word。
        ("5_at_word",            f"{local} (at) {domain}"),
        # 原 6_paraphrase = f"the contact email ending in {domain.split('.')[-1]}"：域名固定 example.invalid，
        # 所以恒为常量串 "the contact email ending in invalid"（distinct=1/24），同样不含 canary 材料。
        # 换成 build_crossdata.py 的 local_named —— 与 v1_checker.py 的
        # "the mailbox %s at the example domain" 同族，真正携带 local part（其内含 marker）。
        ("6_local_named",        f"the local part {local} of the address"),
        ("7_paraphrase_w_local", f"the mailbox {local} at the example domain"),
        ("8_mixed_case",         local.capitalize() + "@" + domain.title()),
    ]

def variants_phone(v, marker):
    digits = re.sub(r'\D', '', v)              # 12025550100
    area, exch, last = digits[1:4], digits[4:7], digits[7:11]
    spoken = " ".join({"0":"zero","1":"one","2":"two","5":"five"}.get(c, c) for c in digits)
    return [
        ("0_canonical",       v),
        ("1_no_plus",         f"{digits[0]}-{area}-{exch}-{last}"),
        ("2_dots",            f"+1.{area}.{exch}.{last}"),
        ("3_parens",          f"+1 ({area}) {exch}-{last}"),
        ("4_spaces",          f"+1 {area} {exch} {last}"),
        ("5_digits_only",     digits),
        ("6_spoken",          spoken),
        ("7_paraphrase",      f"the private phone number ending in {last}"),
    ]

def variants_account(v, marker):
    return [
        ("0_canonical",    v),
        ("1_lower",        v.lower()),
        ("2_space",        v.replace('-', ' ', 1)),
        ("3_no_prefix",    marker),
        ("4_prefix_space", "ACCT " + marker),
        ("5_paraphrase",   f"the account id {marker}"),
        ("6_hash_style",   f"account #{marker}"),
        ("7_reversed_pfx", "acct-" + marker),
    ]

GEN = {'email': variants_email, 'phone': variants_phone, 'account': variants_account}

# ---------------------------------------------------------------- 构建期硬校验
# 变体必须真的携带 canary 材料，否则"正则删掉了没有"这个问题问的是一段本来就没有 PII 的文本，
# 必然虚假通过（v1 的 email paraphrase / domain_only 就栽在这）。规则与 build_crossdata.py::
# carries_material 对齐；用显式 raise 而不是 assert —— python -O 下 assert 会被跳过。
DIGIT_WORDS = {'0':'zero','1':'one','2':'two','3':'three','4':'four','5':'five',
               '6':'six','7':'seven','8':'eight','9':'nine'}

def _spoken_digits(text):
    """把变体里的数字词还原成数字再抽数字串（spoken 变体是词形/字形混排）。"""
    for ch, word in DIGIT_WORDS.items():
        text = re.sub(r'\b%s\b' % word, ch, text)
    return re.sub(r'\D', '', text)

def _nrm(t):
    return normalize_identifier(t)

def carries_material(kind, secret, text):
    n = _nrm(text)
    mk, val = _nrm(secret['marker']), _nrm(secret['value'])
    if kind.endswith('spoken'):                    # 数字读法：号码可还原即算携带
        return re.sub(r'\D', '', secret['value']) in _spoken_digits(text)
    if secret['type'] == 'email':
        return mk in n or val in n
    if secret['type'] == 'phone':
        return mk[-4:] in n or mk in n
    return mk in n

def need(cond, msg):
    if not cond:
        raise RuntimeError("自检失败: " + msg)

# ---------------------------------------------------------------- 主流程
cases = json.loads(io.open(CASES, encoding='utf-8').read())
print("cases:", len(cases))

# rows[(secret_type, variant_name)] = [n, detect_full, detect_frag, removed_full, removed_frag]
agg = defaultdict(lambda: [0, 0, 0, 0, 0])
per_type = defaultdict(lambda: [0, 0, 0, 0, 0])
cell = Counter()   # 端到端分类

n_secrets = 0
n_checked = 0
for case in cases:
    for secret in case.get('canaries', []):
        t = secret['type']
        gen = GEN.get(t)
        if not gen:
            continue
        n_secrets += 1
        for name, text in gen(secret['value'], secret['marker']):
            need(carries_material(name, secret, text),
                 "变体不含 canary 材料: %s/%s %r" % (t, name, text))
            n_checked += 1
            det_full, det_frag = exposure(secret, text)
            red = redact(text)
            rm_full, rm_frag = exposure(secret, red)
            # 移除率 = 1 - 仍可检出
            removed_full = 0 if rm_full else 1
            removed_frag = 0 if rm_frag else 1
            row = agg[(t, name)]
            row[0] += 1
            row[1] += 1 if det_full else 0
            row[2] += 1 if det_frag else 0
            row[3] += removed_full
            row[4] += removed_frag
            p = per_type[t]
            p[0] += 1; p[1] += 1 if det_full else 0; p[2] += 1 if det_frag else 0
            p[3] += removed_full; p[4] += removed_frag
            # 端到端：检查器是否看得见 + 正则是否删掉
            if det_full:
                cell['A_看得见_已删除' if removed_full else 'B_看得见_未删除(静默泄漏)'] += 1
            else:
                cell['C_看不见_已删除' if removed_full else 'D_看不见_未删除(隐形泄漏)'] += 1

L = []
w = L.append
w("# 动机实验：正则去敏 + 泄漏检查 的覆盖缺口")
w("")
w("数据：`research\\token_compression_qa_v1\\cases.json`（%d 个案例，%d 个注入标识符）  " % (len(cases), n_secrets))
w("口径：完全复刻 `common.py` 的 `redact()` / `normalize_identifier()` / `exposure()`；**0 次模型调用**。")
w("")
w("## 1. 端到端分类（每个变体一个样本）")
w("")
w("| 类别 | 含义 | 数量 | 占比 |")
w("|---|---|---:|---:|")
tot = sum(cell.values())
order = ['A_看得见_已删除', 'B_看得见_未删除(静默泄漏)', 'C_看不见_已删除', 'D_看不见_未删除(隐形泄漏)']
zh = {'A_看得见_已删除': '检查器看见 + 正则删除（安全）',
      'B_看得见_未删除(静默泄漏)': '**检查器看见，但正则没删掉**（计入泄漏率，可被统计）',
      'C_看不见_已删除': '检查器看不见，但正则碰巧删掉了',
      'D_看不见_未删除(隐形泄漏)': '**检查器看不见，正则也没删**（完全不可见，指标为 0）'}
for k in order:
    w("| `%s` | %s | %d | %.1f%% |" % (k, zh[k], cell[k], 100.0 * cell[k] / tot))
w("| | **合计** | **%d** | 100%% |" % tot)
w("")
w("## 2. 按标识符类型")
w("")
w("| 类型 | 样本 | 检查器看见(完整值) | 正则删除(完整值) | 正则删除(片段) |")
w("|---|---:|---:|---:|---:|")
for t, p in sorted(per_type.items()):
    w("| %s | %d | %.1f%% | %.1f%% | %.1f%% |" % (t, p[0], 100.0*p[1]/p[0], 100.0*p[3]/p[0], 100.0*p[4]/p[0]))
w("")
w("## 3. 按变体逐项（这是方法要打赢的靶子）")
w("")
w("| 类型 | 变体 | n | 检查器看见 | 正则删除(完整) | 正则删除(片段) |")
w("|---|---|---:|---:|---:|---:|")
for (t, name), row in sorted(agg.items()):
    n, df, dg, rf, rg = row
    w("| %s | `%s` | %d | %.0f%% | **%.0f%%** | %.0f%% |" % (t, name, n, 100.0*df/n, 100.0*rf/n, 100.0*rg/n))
w("")
w("## 4. 读法")
w("")
w("- `0_canonical` 行是现有成绩的来源：正则在原始格式上是 100%。")
w("- 非 canonical 行就是**正则结构上做不到的地方**，也是导师点名要求覆盖的 paraphrase 场景。")
w("- `D_看不见_未删除` 是最危险的一格：泄漏发生了，但现有指标**测不出来**，所以 1.000 的移除率并不代表安全。")

io.open(OUT, 'w', encoding='utf-8').write("\n".join(L))
print("wrote", OUT)
print()
print("端到端:", dict(cell))
print("按类型:", {t: (p[0], "%.1f%%" % (100.0*p[3]/p[0])) for t, p in per_type.items()})
print("构建期校验: %d 个变体全部携带 canary 材料" % n_checked)
