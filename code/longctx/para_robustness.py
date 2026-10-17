"""提问方式稳健性：改写后臂间排序是否保持不变。

对照关系
--------
  A = aliasq（原问题，9 臂 × 3 预算，batch 2，gpu-10-4）
  B = pallA（机械改写变体 A，**同样 9 臂 × 3 预算，batch 2**）
两者只差问题文本，所以可以直接逐格比。

主判据（不是绝对 F1）
--------------------
  (1) `budget_cluster_link` 是否仍显著高于 `budget_cluster`
  (2) `alias_budget_link` 是否仍显著高于 `alias_budget`
  (3) 两条 link 臂是否都接近或超过 `full`
  (4) 整体排序是否基本保持
另附成对 bootstrap（同一批案例、同一重采样）给出差值的置信区间。

注意：绝对 F1 不可比（问题文本不同）；可比的是**同一问法内的臂间差**。
"""
import argparse
import collections
import io
import json
import random
import statistics as st


def load(p):
    o = json.load(io.open(p, encoding='utf-8'))
    return {k: dict(n=v['n'], f1=v['f1'] / max(1, v['n']), tok=v['tok'] / max(1, v['n']),
                    gic=v['ans'] / max(1, v['n']), leak=v['cv'] / max(1, v['cn']),
                    tent=v['tent'] / max(1, v['n'])) for k, v in o['all'].items()}


def percase(p):
    d = collections.defaultdict(dict)
    for l in io.open(p, encoding='utf-8'):
        l = l.strip()
        if not l:
            continue
        r = json.loads(l)
        d[r['case_key']]['%s@%s' % (r['arm'], 'ref' if r.get('ratio') is None else r['ratio'])] = r['f1']
    return d


def boot(diffs, B=4000, seed=20261015):
    rng = random.Random(seed)
    n = len(diffs)
    m = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        m.append(s / n)
    m.sort()
    return st.mean(diffs), m[int(0.025 * B)], m[int(0.975 * B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--original', required=True, help='aliasq cells.json')
    ap.add_argument('--rephrased', required=True, help='pallA cells.json')
    ap.add_argument('--orig-percase', default='')
    ap.add_argument('--reph-percase', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()

    A = load(a.original)
    B = load(a.rephrased)
    keys = [k for k in A if k in B]
    L = []
    L.append('提问方式稳健性：原问题 vs 机械改写变体 A（同 9 臂 × 3 预算配置）')
    L.append('%-28s %10s %10s %9s' % ('arm|budget', 'original', 'rephrased', 'delta'))
    for k in sorted(keys, key=lambda x: -A[x]['f1']):
        L.append('%-28s %10.4f %10.4f %+9.4f' % (k, A[k]['f1'], B[k]['f1'],
                                                 B[k]['f1'] - A[k]['f1']))

    L.append('')
    L.append('主判据（臂间差，而非绝对 F1）：')

    def d(o, x, y):
        return (o[x]['f1'] - o[y]['f1']) if x in o and y in o else float('nan')

    pairs = [('budget_cluster_link|0.5', 'budget_cluster|0.5', '锚点增益（原始串）'),
             ('alias_budget_link|0.5', 'alias_budget|0.5', '锚点增益（别名）'),
             ('budget_cluster_link|0.5', 'full|None', 'link 相对不压缩'),
             ('alias_budget_link|0.5', 'full|None', 'alias link 相对不压缩')]
    for x, y, name in pairs:
        if x in A and y in A and x in B and y in B:
            L.append('  %-24s 原 %+.4f → 改写 %+.4f   %s'
                     % (name, d(A, x, y), d(B, x, y),
                        '同号且量级可比' if (d(A, x, y) > 0) == (d(B, x, y) > 0)
                        else '*** 符号改变 ***'))
        else:
            L.append('  %-24s 缺臂，跳过（%s / %s）' % (name, x, y))

    if a.orig_percase and a.reph_percase:
        PA, PB = percase(a.orig_percase), percase(a.reph_percase)
        L.append('')
        L.append('成对 bootstrap（同一批案例；差值 = 改写 − 原）:')
        for x, y, name in pairs:
            kx = x.replace('|None','@ref').replace('|0.5','@0.5').replace('|0.25','@0.25').replace('|0.75','@0.75')
            ky = y.replace('|None','@ref').replace('|0.5','@0.5').replace('|0.25','@0.25').replace('|0.75','@0.75')
            pairs2 = []
            for kk in PB:
                if kx in PB[kk] and ky in PB[kk] and kx in PA.get(kk, {}) and ky in PA[kk]:
                    pairs2.append((PB[kk][kx] - PB[kk][ky]) - (PA[kk][kx] - PA[kk][ky]))
            if len(pairs2) < 5:
                L.append('  %-24s 配对数不足(%d)' % (name, len(pairs2)))
                continue
            m, lo, hi = boot(pairs2)
            L.append('  %-24s n=%3d  Δ差=%+.4f [%+.4f, %+.4f]  %s'
                     % (name, len(pairs2), m, lo, hi,
                        'CI 不含 0（改写改变了该差值）' if (lo > 0 or hi < 0)
                        else 'CI 含 0（该差值在改写后保持）'))

    txt = '\n'.join(L)
    print(txt)
    if a.out:
        io.open(a.out, 'w', encoding='utf-8', newline='\n').write(txt + '\n')
        print('\n[para] wrote %s' % a.out)
    if a.json_out:
        io.open(a.json_out, 'w', encoding='utf-8').write(json.dumps(
            dict(original=A, rephrased=B), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
