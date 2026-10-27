"""① ② ③ 三组实验的汇总分析。"""
import json, io, glob, collections

RD = '/dpc/kuex0095/100069961_ku/privtrim'


def load(pat):
    out = []
    for p in sorted(glob.glob(RD + '/' + pat)):
        for x in io.open(p, encoding='utf-8'):
            if x.strip():
                out.append(json.loads(x))
    return out


def f1(r):
    v = r.get('f1')
    return float(v) if isinstance(v, (int, float)) else 0.0


def avg(rows, key, arm, ratio=None):
    v = [f1(r) for r in rows
         if r.get('arm') == arm and (ratio is None or r.get('ratio') == ratio)]
    return (sum(v) / len(v), len(v)) if v else (None, 0)


# ---------------- ① 秩泄漏修复对 F1 的影响 ----------------
print('=' * 74)
print('① A3: 加 PT_ALIAS_KEY 后 F1 是否下降（同 300 例）')
print('=' * 74)
rk = load('out/rkfix_s*_percase.jsonl')
maxr = load('out/max32_percase.jsonl')
keys = set(r['case_key'] for r in rk)
base = [r for r in maxr if r['case_key'] in keys]
print('样本: rkfix=%d 行 / %d 例；无 key 基线=%d 行' % (len(rk), len(keys), len(base)))
print('\n%-22s %10s %10s %10s' % ('arm', 'F1(无key)', 'F1(有key)', 'Δ'))
print('-' * 56)
for arm, ratio in [('full', None), ('alias', None), ('alias_budget_link', 0.5)]:
    a, na = avg(base, 'f1', arm, ratio)
    b, nb = avg(rk, 'f1', arm, ratio)
    if a is None or b is None:
        print('%-22s %10s %10s' % (arm, a, b)); continue
    print('%-22s %10.4f %10.4f %+10.4f   (n=%d/%d)' % (arm, a, b, b - a, na, nb))

# ---------------- ② INTACT 问题同步 ----------------
print('\n' + '=' * 74)
print('② INTACT：问题同步前后（同 100 例）')
print('=' * 74)
ev0 = load('out/evint_s*_percase.jsonl')       # 未同步
ev1 = load('out/evsync_s*_percase.jsonl')      # 已同步
print('未同步 %d 行 / 已同步 %d 行' % (len(ev0), len(ev1)))
print('\n%-18s %10s %10s %10s' % ('arm', 'F1(未同步)', 'F1(已同步)', 'Δ'))
print('-' * 52)
for arm in ['ext_intact', 'ext_full', 'full']:
    a, na = avg(ev0, 'f1', arm)
    b, nb = avg(ev1, 'f1', arm)
    sa = '%.4f' % a if a is not None else '—'
    sb = '%.4f' % b if b is not None else '—'
    d = '%+.4f' % (b - a) if (a is not None and b is not None) else '—'
    print('%-18s %10s %10s %10s   (n=%d/%d)' % (arm, sa, sb, d, na, nb))

# ---------------- ③ 跨模型：Qwen vs Mistral ----------------
print('\n' + '=' * 74)
print('③ 跨模型泛化：Qwen2.5-7B vs Mistral-7B-Instruct-v0.3（同 300 例）')
print('=' * 74)
xm = load('out/xm_s*_percase.jsonl')
keys2 = set(r['case_key'] for r in xm)
base2 = [r for r in maxr if r['case_key'] in keys2]
print('样本: mistral=%d 行 / %d 例；qwen=%d 行' % (len(xm), len(keys2), len(base2)))
print('\n%-22s %12s %12s %10s' % ('arm', 'Qwen F1', 'Mistral F1', 'Δ'))
print('-' * 60)
for arm in ['full', 'alias', 'alias_budget', 'alias_budget_link']:
    ratio = None if arm in ('full', 'alias') else 0.5
    a, na = avg(base2, 'f1', arm, ratio)
    b, nb = avg(xm, 'f1', arm, ratio)
    if a is None or b is None:
        print('%-22s %12s %12s' % (arm, a, b)); continue
    print('%-22s %12.4f %12.4f %+10.4f' % (arm, a, b, b - a))

print('\n--- 机制差值（关键：link 的增益是否跨模型成立）---')
for name, rows, ratio in [('Qwen', base2, 0.5), ('Mistral', xm, 0.5)]:
    lbl, _ = avg(rows, 'f1', 'alias_budget_link', 0.5)
    ab, _ = avg(rows, 'f1', 'alias_budget', 0.5)
    al, _ = avg(rows, 'f1', 'alias', None)
    fu, _ = avg(rows, 'f1', 'full', None)
    if None in (lbl, ab, al, fu):
        continue
    print('  %-8s link 增益(abl - ab) = %+0.4f   |  alias - full = %+0.4f'
          % (name, lbl - ab, al - fu))
