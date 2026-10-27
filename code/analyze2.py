"""分析两件事:
  1) pt-sweep (1482353) 的预算下探结果 —— 0.25 是不是端点效应
  2) pt-b8chk (1482354) 的 batch=8 与 batch=2 是否逐例等价
"""
import json, io, os, collections
RD = '/dpc/kuex0095/100069961_ku/privtrim'


def load(p):
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in io.open(p, encoding='utf-8') if x.strip()]


def num(r, k):
    v = r.get(k)
    return float(v) if isinstance(v, (int, float)) else 0.0


def agg(rows):
    out = collections.defaultdict(list)
    for r in rows:
        out[(r.get('arm'), r.get('ratio'))].append(r)
    return out


# ---------------- 1. sweep ----------------
print('=' * 78)
print('1. 预算下探 sweep')
print('=' * 78)
srows, missing = [], []
for s in range(32):
    p = '%s/out/sweep_s%d_percase.jsonl' % (RD, s)
    if not os.path.exists(p):
        missing.append(s); continue
    srows += load(p)
print('shards: %d/32   missing: %s' % (32 - len(missing), missing or 'none'))
print('records: %d   cases: %d' % (len(srows), len(set(r['case_key'] for r in srows))))

a = agg(srows)
print('\n%-22s %6s %5s %8s %8s %9s %8s' % ('arm', 'ratio', 'n', 'F1', 'EM', 'tgt_pres', 'leak'))
print('-' * 72)
for k in sorted(a, key=lambda x: (str(x[0]), -1 if x[1] is None else float(x[1]))):
    v = a[k]; n = len(v)
    f1 = sum(num(r, 'f1') for r in v) / n
    em = sum(num(r, 'em') for r in v) / n
    tp = sum(num(r, 'tgt_entity_present') for r in v) / n
    lk = sum(1 for r in v if num(r, 'ctx_leak_value') > 0 or num(r, 'out_leak_value') > 0) / n
    print('%-22s %6s %5d %8.4f %8.4f %9.3f %8.3f' % (k[0], k[1], n, f1, em, tp, lk))

# 合并 max32 已有预算点,画完整曲线
print('\n--- alias_budget_link 完整预算曲线 (sweep 新增 + max32 已有) ---')
mrows = load(RD + '/out/max32_percase.jsonl')
curve = {}
for src, rr in (('max32', mrows), ('sweep', srows)):
    for r in rr:
        if r.get('arm') == 'alias_budget_link' and r.get('ratio') is not None:
            curve.setdefault(r['ratio'], []).append(num(r, 'f1'))
print('%8s %8s %8s' % ('budget', 'F1', 'n'))
for ratio in sorted(curve):
    v = curve[ratio]
    print('%8.2f %8.4f %8d' % (ratio, sum(v) / len(v), len(v)))

# ---------------- 2. batch8 vs batch2 ----------------
print('\n' + '=' * 78)
print('2. batch=8 对照 (分片 0)')
print('=' * 78)
b8 = load(RD + '/out/b8_s0_percase.jsonl')
b2 = load(RD + '/out/max32_s0_percase.jsonl')
print('batch8 rows=%d   batch2 rows=%d' % (len(b8), len(b2)))
if b8 and b2:
    def idx(rows):
        return {(r['case_key'], r.get('arm'), r.get('ratio')): r for r in rows}
    i8, i2 = idx(b8), idx(b2)
    common = set(i8) & set(i2)
    print('matched keys: %d' % len(common))
    same_pred = same_f1 = same_em = 0
    diffs = []
    for k in common:
        r8, r2_ = i8[k], i2[k]
        if (r8.get('pred') or '') == (r2_.get('pred') or ''):
            same_pred += 1
        else:
            diffs.append(k)
        if abs(num(r8, 'f1') - num(r2_, 'f1')) < 1e-9:
            same_f1 += 1
        if num(r8, 'em') == num(r2_, 'em'):
            same_em += 1
    n = len(common)
    print('identical pred : %d/%d  (%.3f%%)' % (same_pred, n, 100.0 * same_pred / n))
    print('identical f1   : %d/%d' % (same_f1, n))
    print('identical em   : %d/%d' % (same_em, n))
    if diffs:
        print('first 5 differing keys:')
        for k in diffs[:5]:
            print('   %s  batch8=%r  batch2=%r' % (k, i8[k].get('pred'), i2[k].get('pred')))
    # 均值差异
    for arm in ['full', 'alias', 'alias_budget_link', 'budget_cluster_link']:
        v8 = [num(r, 'f1') for k, r in i8.items() if k[1] == arm]
        v2 = [num(r, 'f1') for k, r in i2.items() if k[1] == arm]
        if v8 and v2:
            print('  %-22s batch8 F1=%.4f (n=%d)  batch2 F1=%.4f (n=%d)  d=%+.4f'
                  % (arm, sum(v8) / len(v8), len(v8), sum(v2) / len(v2), len(v2),
                     sum(v8) / len(v8) - sum(v2) / len(v2)))
