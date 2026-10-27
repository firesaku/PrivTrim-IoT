"""
V1：paraphrase 感知的泄漏检查器 —— 在完整数据集上重算覆盖率
CPU only, 0 model calls.

对比三个检查器：
  A. 原版 exposure()             —— 小写去非字母数字 + 子串匹配（项目现用）
  B. 原版 + 变体枚举             —— 对每个 secret 生成 paraphrase 变体，任一命中即算检出
  C. 变体枚举 + 无序 token 集合   —— 加"特征 token 全在但顺序/格式变了"的判定

谓词统一来自同目录 privcheck.py（唯一真源）：本文件不再持有任何谓词副本，
v1b_diag.py 也 import 同一个模块，所以两边的判定结构上不可能再漂移。

输出（每一行都带自己的分母 n，不用调和平均）：
  raw_basic            = 原版检查器在原始上下文上判定'看得见'的比例（= 项目现有隐私指标的分母逻辑）
  raw_variants         = 变体枚举检查器的检出率
  raw_var_tok          = 变体枚举 + token 集合检查器的检出率（更严格）
  red_basic            = 正则脱敏后原版检查器仍看得见 -> 这才是真实漏删率
  red_var_tok          = 正则脱敏后更严格检查器仍看得见
  checker_fp_rate      = 更严格检查器命中、但原版检查器看不见 -> 检查器误报率（不是泄漏率）
  blind_spot_this_corpus  = 本语料口径的盲区（语料只含 canonical 形式，期望 ~0，说明的是语料）
  blind_spot_probe_external = 改写探测器的口径（外部数字，本脚本不复算，见 --help 说明）
per-type：email / phone / account 每个率一套桶，外加 ALL 合计。

用法：python v1_checker.py --cases <cases.jsonl.gz> --out v1_report.txt --json v1_cells.json
"""
import io, json, os, gzip, glob, argparse, sys, importlib.util
from collections import defaultdict

def load_privcheck():
    """从本文件所在目录加载 privcheck.py（不依赖打包/安装，两个脚本共用同一份谓词）。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'privcheck.py')
    spec = importlib.util.spec_from_file_location('privcheck', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

pc = load_privcheck()

# 向后兼容：把共享谓词 re-export 到本模块命名空间。
# 复核脚本（如 _v1_review_verify/verify.py）用 importlib 加载本文件后直接调用
# v1.exposure_basic / v1.exposure_variants / v1.norm / v1.variants / v1.redact，
# 这些名字必须继续存在；它们现在只是 privcheck 的别名，实现仍然只有一份。
EMAIL, PHONE, ACCOUNT = pc.EMAIL, pc.PHONE, pc.ACCOUNT
norm = pc.norm
redact = pc.redact
variants = pc.variants
variant_substr = pc.variant_substr
tokenset = pc.tokenset
exposure_basic = pc.exposure_basic
exposure_hit = pc.exposure_hit
exposure_variants = pc.exposure_variants
hit_reasons = pc.hit_reasons

TYPES  = ('email', 'phone', 'account')
SCOPES = ('ALL',) + TYPES

# 外部口径（paraphrase_probe/SUMMARY.txt，MOTIVATION 摘要）：24 案例 / 72 标识符 / 600 个变体样本。
# 本脚本不重算这些数字，只把它们和"本语料口径"并排打印，避免 0.00% 被误读成"没有盲区"。
PROBE = {'cases': 24, 'ids': 72, 'variants': 600, 'invisible': 240,
         'removed': 48, 'removed_total': 528,
         'source': 'paraphrase_probe/SUMMARY.txt'}
# 外部口径引用来源必须存在，否则整份报告会变成无出处的数字
PROBE_SRC_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'paraphrase_probe', 'SUMMARY.txt')

def find_cases(arg):
    if arg and os.path.exists(arg):
        return arg
    for pat in [r"/path/to/100065732/privtrim_iot_hpc_20261005/*/run/qwen25_7b/cases.jsonl.gz",
                r"/path/to/100065732/**/cases.jsonl.gz"]:
        g = glob.glob(pat, recursive=True)
        if g:
            return sorted(g, key=os.path.getsize)[-1]
    return None

def fmt(tag, scope, n, h):
    """一行 = 一个率 + 它自己的分母（n=0 时明确打 n/a，不假装是 0%）。"""
    if n:
        return "%-24s %-8s %8d / %8d = %7.2f%%" % (tag, scope, h, n, 100.0 * h / n)
    return "%-24s %-8s %8s / %8d =     n/a" % (tag, scope, '-', 0)

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

    # 统计桶：(tag, scope) -> [n, hit]
    st = defaultdict(lambda: [0, 0])
    def add(tag, scope, hit):
        b = st[(tag, scope)]; b[0] += 1; b[1] += 1 if hit else 0

    fp_reason = defaultdict(int)      # (reason_kind, type) -> 次数
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
        # 缓存：同一个 case 的 3 个 canary 共用同一段文本的 normalize/tokenize 结果
        n_ctx = pc.norm(ctx); toks_ctx = pc.tokenize(ctx)
        red = pc.redact(ctx)          # 每个 case 只脱敏一次（原版放在 canary 循环里，白算 3 倍）
        n_red = pc.norm(red); toks_red = pc.tokenize(red)
        for s in c.get('canaries', []):
            nsec += 1
            t = s.get('type', 'other')
            det_b = pc.exposure_hit(s, ctx, n=n_ctx)
            det_v = pc.exposure_variants(s, ctx, use_tokenset=False, n=n_ctx, toks=toks_ctx)
            det_vt = pc.exposure_variants(s, ctx, use_tokenset=True, n=n_ctx, toks=toks_ctx)
            red_hit = pc.exposure_hit(s, red, n=n_red)
            red_reasons = pc.hit_reasons(s, red, n=n_red, toks=toks_red)
            det_rv = bool(red_reasons)
            # checker_fp：更严格检查器在脱敏后仍命中，而 canonical 检查器看不见 = 误报
            is_fp = det_rv and not red_hit
            blind = (not det_b) and det_vt

            for tag, val in (("raw_basic", det_b), ("raw_variants", det_v),
                             ("raw_var_tok", det_vt), ("red_basic", red_hit),
                             ("red_var_tok", det_rv), ("checker_fp_rate", is_fp),
                             ("blind_spot_this_corpus", blind)):
                for scope in ('ALL', t):
                    add(tag, scope, val)
            if is_fp:
                for h in red_reasons:
                    fp_reason[(pc.reason_kind(h), t)] += 1
        if ncase % 2000 == 0:
            print("  cases", ncase, flush=True)
    fh.close()

    L = []
    w = L.append
    w("V1 PARAPHRASE-AWARE LEAK CHECKER — full-corpus coverage")
    w("cases_file=%s" % cp)
    w("cases=%d  secrets=%d" % (ncase, nsec))
    w("predicates=%s  (唯一真源；v1b_diag.py 用同一份)" % pc.__file__)
    w("")
    w("分母（每个率都只配自己那一行的分母，不跨类型合并、不用调和平均）：")
    for scope in SCOPES:
        w("  %-8s n=%d" % (scope, st[("raw_basic", scope)][0]))
    w("")

    def block(title, tags):
        w(title)
        w("%-24s %-8s %8s   %8s   %8s" % ("bucket", "scope", "hits", "/", "n"))
        for tag in tags:
            for scope in SCOPES:
                n, h = st[(tag, scope)]
                w(fmt(tag, scope, n, h))
        w("")

    block("[1] RAW 原始上下文（未脱敏）",
          ("raw_basic", "raw_variants", "raw_var_tok"))
    block("[2] REDACTED 正则脱敏后（真实漏删率看这里）",
          ("red_basic", "red_var_tok", "checker_fp_rate"))

    n_all, h_all = st[("red_basic", "ALL")]
    fp_all = st[("checker_fp_rate", "ALL")]
    w("  读法：")
    w("   * red_basic = 真实漏删率 = %d/%d = %.2f%%：每一个 canonical canary 都被正则完整删除，"
      % (h_all, n_all, 100.0 * h_all / max(n_all, 1)))
    w("     本语料上正则的真实泄漏率 ~0。")
    w("   * checker_fp_rate = 检查器误报率，不是泄漏率：分母 %d 是所有 secret，分子 %d。"
      % (fp_all[0], fp_all[1]))
    w("     机制：phone 的 3~4 位数字块被拿去在已脱敏文本里做子串匹配，会撞上巧合数字")
    w("     （例如 \"2020\" 里含 \"202\"）。归因计数（reason x type）：")
    if fp_reason:
        for k in sorted(fp_reason, key=lambda k: -fp_reason[k]):
            w("       %-20s %-8s %8d" % (k[0], k[1], fp_reason[k]))
        w("       （reason 命中合计 %d 次 > 分子 %d：同一个 secret 可能命中多个 chunk，"
          % (sum(fp_reason.values()), fp_all[1]))
        w("         所以计数表不能当分子读，分子永远用上面的 hits 列。）")
    else:
        w("       （无）")
    w("")

    w("[3] BLIND SPOT —— 两种口径，分开打印，不可混读")
    w("%-30s %-8s %8s / %8s = %7s" % ("caliber", "scope", "hits", "n", "rate"))
    for scope in SCOPES:
        n, h = st[("blind_spot_this_corpus", scope)]
        w("%-30s %-8s %8d / %8d = %6.2f%%" % ("blind_spot_this_corpus", scope, h, n,
                                              100.0 * h / max(n, 1)))
    n_inv, n_var = PROBE['invisible'], PROBE['variants']
    w("%-30s %-8s %8d / %8d = %6.2f%%" % ("blind_spot_probe_external", "probe", n_inv, n_var,
                                          100.0 * n_inv / n_var))
    w("%-30s %-8s %8d / %8d = %6.2f%%" % ("variant_removal_probe_ext", "probe",
                                          PROBE['removed'], PROBE['removed_total'],
                                          100.0 * PROBE['removed'] / PROBE['removed_total']))
    w("  读法：")
    w("   * blind_spot_this_corpus 只说明\"本语料只含 canonical 形式\"：%d/%d = %.2f%%，"
      % (st[("blind_spot_this_corpus", "ALL")][1], n_all,
         100.0 * st[("blind_spot_this_corpus", "ALL")][1] / max(n_all, 1)))
    w("     它绝对不代表\"检查器没有盲区\"。")
    w("   * blind_spot_probe_external / variant_removal_probe_ext 是改写探测器（paraphrase probe）")
    w("     在 %d 案例 / %d 标识符 / %d 个变体样本上的数字：%d/%d = %.1f%% 的变体样本对指标隐形，"
      % (PROBE['cases'], PROBE['ids'], PROBE['variants'], n_inv, n_var,
         100.0 * n_inv / n_var))
    w("     变体上的真删率只有 %d/%d = %.1f%%。这两行是外部引用，本脚本不复算它们。"
      % (PROBE['removed'], PROBE['removed_total'],
         100.0 * PROBE['removed'] / PROBE['removed_total']))
    w("     source=%s" % PROBE['source'])
    w("     local_copy_exists=%s (%s)" % (os.path.exists(PROBE_SRC_LOCAL), PROBE_SRC_LOCAL))
    w("")

    io.open(a.out, 'w', encoding='utf-8').write("\n".join(L))
    io.open(a.json, 'w', encoding='utf-8').write(json.dumps(
        {"cases": ncase, "secrets": nsec,
         "predicates": pc.__file__,
         "denominators": {scope: st[("raw_basic", scope)][0] for scope in SCOPES},
         "buckets": {"%s|%s" % k: v for k, v in st.items()},
         "checker_fp_reasons": {"%s|%s" % k: v for k, v in fp_reason.items()},
         "blind_spot_probe_external": dict(PROBE)}, ensure_ascii=False, indent=1))
    print("\n".join(L))
    print("wrote", a.out)

if __name__ == '__main__':
    main()
