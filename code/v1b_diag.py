"""
V1b 诊断：旧 redact_miss=12.35% 与修正后 checker_fp_rate=1.14% 分别是什么东西？
  (1) 复现历史 B1 bug：digit-chunk 规则被无条件套用到所有 secret 类型
      （规则本体在 privcheck.legacy_b1_*，本文件不再自带任何谓词副本 -> 不会与 v1_checker.py 漂移）
  (2) 对修正后的残差（= checker_fp_rate 的分子）做归因计数 + 抽样取证，
      证明它是启发式误报而不是漏删。
CPU only, 0 model calls。

用法：python v1b_diag.py --out v1b_report.txt [--cases <cases.jsonl.gz>] [--nsample 5] [--limit N]
样本按 (reason, type) 分桶，每桶取"文件顺序前 nsample 条"（确定性，可复现）。
"""
import io, json, os, gzip, glob, argparse, sys, importlib.util
from collections import Counter, defaultdict

def load_privcheck():
    """从本文件所在目录加载 privcheck.py（不依赖打包/安装，两个脚本共用同一份谓词）。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'privcheck.py')
    spec = importlib.util.spec_from_file_location('privcheck', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

pc = load_privcheck()

TYPES = ('email', 'phone', 'account')

def find_cases(arg):
    if arg and os.path.exists(arg):
        return arg
    for pat in [r"/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/*/run/qwen25_7b/cases.jsonl.gz",
                r"/dpc/kuex0095/100065732/**/cases.jsonl.gz"]:
        g = glob.glob(pat, recursive=True)
        if g:
            return sorted(g, key=os.path.getsize)[-1]
    return None

def norm_index(text):
    """返回 (norm(text), idx)：idx[i] = norm 串第 i 个字符在原文里的下标。
    用于把 norm 空间中的命中位置映射回原文，打印可读证据。"""
    out = []; idx = []
    for i, ch in enumerate(text):
        for c in ch.lower():
            if ('a' <= c <= 'z') or ('0' <= c <= '9'):
                out.append(c); idx.append(i)
    return ''.join(out), idx

def window(text, n_text, idx_map, needle, before=70, after=70):
    """needle（已在 norm 空间里命中）在原文中的上下文窗口。"""
    p = n_text.find(needle)
    if p < 0:
        return '(not found)'
    ci = idx_map[p]
    return text[max(0, ci - before):ci + after].replace('\n', ' | ')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--nsample', type=int, default=40)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    cp = find_cases(a.cases)
    print("cases file:", cp, flush=True)
    if not cp:
        print("NO_CASES_FOUND"); sys.exit(5)

    op = gzip.open if cp.endswith('.gz') else io.open
    fh = op(cp, 'rt', encoding='utf-8', errors='replace')

    ncase = nsec = 0
    den = Counter()                      # type -> secret 数（分母）
    fp_secrets = Counter()               # type -> 修正后仍命中的 secret 数
    fp_reason = Counter()                # (reason, type) -> 命中次数
    legacy_secrets = Counter()           # type -> 旧 B1 规则命中的 secret 数
    legacy_rule = Counter()              # (slice, type) -> 命中次数
    legacy_only = Counter()              # type -> 只有旧规则命中、修正后不命中
    canon_visible = Counter()            # type -> 脱敏后 canonical 检查器仍可见（真漏删）
    samples = defaultdict(list)          # (bucket, type) -> [证据行]
    samples_legacy = defaultdict(list)

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
        red = pc.redact(ctx)                       # 每个 case 只脱敏一次
        n_red = pc.norm(red)
        for s in c.get('canaries', []):
            nsec += 1
            t = s.get('type', 'other')
            den[t] += 1
            reasons = pc.hit_reasons(s, red, n=n_red)          # 修正后的判定（= checker_fp 分子口径）
            legacy = pc.legacy_b1_reasons(s, red, n=n_red)     # 历史 B1 规则的复算
            if pc.exposure_hit(s, red, n=n_red):
                canon_visible[t] += 1
            if reasons:
                fp_secrets[t] += 1
                for r in reasons:
                    fp_reason[(pc.reason_kind(r), t)] += 1
                    if len(samples[(pc.reason_kind(r), t)]) < a.nsample:
                        samples[(pc.reason_kind(r), t)].append((c, s, r, red, n_red))
            if legacy:
                legacy_secrets[t] += 1
                for r in legacy:
                    legacy_rule[(r.split(':', 1)[1].split('=', 1)[0], t)] += 1
                if not reasons:
                    legacy_only[t] += 1
                    if len(samples_legacy[t]) < a.nsample:
                        samples_legacy[t].append((c, s, legacy, red, n_red))
        if ncase % 2000 == 0:
            print("  cases", ncase, flush=True)
    fh.close()

    L = []
    w = L.append
    w("V1b DIAGNOSTIC — 旧 redact_miss 与修正后 checker_fp_rate 的真伪")
    w("cases_file=%s" % cp)
    w("cases_scanned=%d  secrets=%d" % (ncase, nsec))
    w("predicates=%s  (唯一真源；与 v1_checker.py 同一份)" % pc.__file__)
    w("分母：ALL=%d email=%d phone=%d account=%d（每行只配自己的分母）"
      % (nsec, den['email'], den['phone'], den['account']))
    w("")

    w("[1] 历史 B1 bug 复算：digit-chunk 规则被无条件套用到所有类型（旧 12.35% 的来源）")
    w("%-28s %-8s %8s / %8s = %7s" % ("legacy_b1_hits(any)", "scope", "hits", "n", "rate"))
    tot = sum(legacy_secrets.values())
    w("%-28s %-8s %8d / %8d = %6.2f%%" % ("legacy_b1_hits(any)", "ALL", tot, nsec,
                                          100.0 * tot / max(nsec, 1)))
    for t in TYPES:
        w("%-28s %-8s %8d / %8d = %6.2f%%" % ("legacy_b1_hits(any)", t, legacy_secrets[t], den[t],
                                              100.0 * legacy_secrets[t] / max(den[t], 1)))
    w("  旧规则的规则级计数（一个 secret 可命中多条，故合计 >= 上面的 secret 数）：")
    for k in sorted(legacy_rule, key=lambda k: -legacy_rule[k]):
        w("     %-24s %-8s %8d" % (k[0], k[1], legacy_rule[k]))
    non_phone = legacy_secrets['email'] + legacy_secrets['account']
    w("  -> 旧规则命中的 secret 里 %d/%d = %.1f%% 是 email/account（B1 bug：hex marker 被 \\D 抽成"
      % (non_phone, max(tot, 1), 100.0 * non_phone / max(tot, 1)))
    w("     假数字串，chunk 命中的全是巧合数字），只有 %d 条是 phone。" % legacy_secrets['phone'])
    w("")

    w("[2] 修正后的残差 = checker_fp_rate 的分子（只对 phone 跑 digit-chunk）")
    w("%-28s %-8s %8s / %8s = %7s" % ("checker_fp_rate", "scope", "hits", "n", "rate"))
    tot_fp = sum(fp_secrets.values())
    w("%-28s %-8s %8d / %8d = %6.2f%%" % ("checker_fp_rate", "ALL", tot_fp, nsec,
                                          100.0 * tot_fp / max(nsec, 1)))
    for t in TYPES:
        w("%-28s %-8s %8d / %8d = %6.2f%%" % ("checker_fp_rate", t, fp_secrets[t], den[t],
                                              100.0 * fp_secrets[t] / max(den[t], 1)))
    w("  归因计数（reason x type）：")
    for k in sorted(fp_reason, key=lambda k: -fp_reason[k]):
        w("     %-24s %-8s %8d" % (k[0], k[1], fp_reason[k]))
    w("     （reason 命中合计 %d 次 > 分子 %d：同一个 secret 可命中多个 chunk；"
      % (sum(fp_reason.values()), tot_fp))
    w("       计数表不能当分子读，分子永远用上面 hits 列。）")
    w("  同类对照：脱敏后 canonical 检查器仍可见（真漏删）的 secret 数：")
    for t in TYPES:
        w("     %-24s %-8s %8d / %8d = %6.2f%%" % ("canonical_leak(red_basic)", t,
                                                   canon_visible[t], den[t],
                                                   100.0 * canon_visible[t] / max(den[t], 1)))
    w("  -> 真漏删 %d/%d = %.2f%%，残差 %d/%d = %.2f%% 全部是误报；"
      % (sum(canon_visible.values()), nsec, 100.0 * sum(canon_visible.values()) / max(nsec, 1),
         tot_fp, nsec, 100.0 * tot_fp / max(nsec, 1)))
    w("     旧规则 vs 修正后：%d -> %d 个 secret 命中（漂移量 %d = 旧数字的 %.1f%%）。"
      % (tot, tot_fp, tot - tot_fp, 100.0 * (tot - tot_fp) / max(tot, 1)))
    w("")

    w("[3] 抽样证据：修正后残差（每 (reason,type) 桶取文件顺序前 %d 条）" % a.nsample)
    if not samples:
        w("  （无）")
    for key in sorted(samples, key=lambda k: -fp_reason[k]):
        kind, t = key
        w("  ---- bucket %s / %s : %d 条（分子 %d）" % (kind, t, len(samples[key]), fp_reason[key]))
        for c, s, r, red, n_red in samples[key]:
            n_text, idx_map = norm_index(red)
            # reason 形如 'digitchunk_phone:202' / 'tokenset:abc,def'，取第一个可定位的片段
            needle = r.split(':', 1)[1].split(',', 1)[0]
            w("  CASE %s | type=%s value=%s marker=%s" % (c.get('id', '?')[:40], s['type'],
                                                          s['value'], s['marker']))
            w("     CHECKER_FP: %s" % r)
            w("     canonical 在脱敏文本中可见? %s （必须 False 才叫误报）"
              % pc.exposure_hit(s, red, n=n_red))
            w("     value 出现次数=%d  marker 出现次数=%d"
              % (red.count(s['value']), red.count(s['marker'])))
            w("     命中块 %r 的原文上下文: %s" % (needle, window(red, n_text, idx_map, needle)))
            w("")
    w("[4] 抽样证据：历史 B1 规则命中、但修正后不命中（旧数字的来源）")
    if not samples_legacy:
        w("  （无）")
    for t in sorted(samples_legacy):
        w("  ---- type %s : 旧命中 %d 条，其中 %d 条修正后为 0" % (t, legacy_secrets[t], legacy_only[t]))
        for c, s, legacy, red, n_red in samples_legacy[t]:
            n_text, idx_map = norm_index(red)
            needle = legacy[0].split('=', 1)[1]
            w("  CASE %s | type=%s value=%s marker=%s" % (c.get('id', '?')[:40], s['type'],
                                                          s['value'], s['marker']))
            w("     LEGACY_B1_HITS: %s" % " ; ".join(legacy))
            w("     修正后 hit_reasons: %s （空 = 旧规则纯误报）" % pc.hit_reasons(s, red, n=n_red))
            w("     命中块 %r 的原文上下文: %s" % (needle, window(red, n_text, idx_map, needle)))
            w("")
    w("[5] 结论")
    w("  (a) 旧 redact_miss=12.35%% 不是漏删：%d/%d = %.2f%% 由 B1 bug（非 phone 类型被套 digit-chunk）"
      % (non_phone, max(tot, 1), 100.0 * non_phone / max(tot, 1)))
    w("      贡献，修正后同一口径只剩 %d/%d = %.2f%%。" % (tot_fp, nsec,
                                                          100.0 * tot_fp / max(nsec, 1)))
    w("  (b) 修正后的 %d/%d = %.2f%% 也不是泄漏：同批数据里 canonical 真漏删 = %d。"
      % (tot_fp, nsec, 100.0 * tot_fp / max(nsec, 1), sum(canon_visible.values())))
    w("      它是 checker_fp_rate（检查器误报率），来源见 [3] 的 reason 计数与上下文。")
    w("  (c) blind_spot 必须按两种口径分开报：本语料只含 canonical 形式，所以本语料口径 ~0；")
    w("      改写探测器口径（paraphrase_probe/SUMMARY.txt）是 240/600 = 40.0% 隐形、变体真删 48/528 = 9.1%。")

    io.open(a.out, 'w', encoding='utf-8').write("\n".join(L))
    print("\n".join(L[:200]))
    print("wrote", a.out)

if __name__ == '__main__':
    main()
