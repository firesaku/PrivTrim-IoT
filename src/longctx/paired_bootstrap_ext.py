"""成对 bootstrap：外部基线对比的置信区间。

为什么必须做
------------
"我们的 0.5500 高于 provence 的 0.4750" 这种话在 40 个案例上不能直接断言 ——
40 例的案例级波动足以吞掉 0.075 的差距。判定必须基于**成对**重采样
（同一批案例、同一次重采样里两个臂用同一组案例），而不是各自独立算 CI。

口径
----
- 单位是**案例**（臂内部是确定性的贪心解码，所以不确定性只来自"抽到哪些案例"）；
- 重采样 B 次，每次有放回抽 n 个案例，算均值的差；取 2.5%/97.5% 分位；
- 差值 CI 不含 0 → 该对比在案例级重采样下稳健。

数据来源
--------
内部臂：r3 的 percase.jsonl（case_key / arm / ratio / f1）
外部臂：本地打分的 scored.jsonl（case_key / arm / f1）
"""
import argparse
import collections
import io
import json
import random
import statistics as st


def load_pairs(internal_path, ext_path, ref_arm, ref_ratio, case_kind):
    per = collections.defaultdict(dict)
    for line in io.open(internal_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('case_kind') != case_kind:
            continue
        if r['arm'] == ref_arm and r.get('ratio') == ref_ratio:
            per[r['case_key']]['ref'] = r['f1']
    for line in io.open(ext_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        per[r['case_key']][r['arm']] = r['f1']
    return per


def boot(diffs, B, seed=20261015):
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * B)]
    hi = means[int(0.975 * B)]
    return st.mean(diffs), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--internal', required=True)
    ap.add_argument('--external', required=True)
    ap.add_argument('--ref-arm', default='budget_cluster_link')
    ap.add_argument('--ref-ratio', type=float, default=0.5)
    ap.add_argument('--case-kind', default='multi_entity')
    ap.add_argument('--challengers', default='ext_provence,ext_cpc,ext_llmlingua2,budget_random')
    ap.add_argument('--B', type=int, default=4000)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    per = load_pairs(a.internal, a.external, a.ref_arm, a.ref_ratio, a.case_kind)
    have = {k: v for k, v in per.items() if 'ref' in v}
    n_ref = len(have)
    lines = []
    lines.append('成对 bootstrap（%s 案例，B=%d，参照 = %s@%s）'
                 % (a.case_kind, a.B, a.ref_arm, a.ref_ratio))
    lines.append('参照臂有结果的案例数 = %d' % n_ref)
    for ch in [c for c in a.challengers.split(',') if c]:
        if ch in ('budget_random', 'budget_cluster'):
            # 内部挑战臂：标签形如 arm@ratio，从 r3 的 percase 里取
            arm, _, ratio = ch.partition('@')
            ratio = float(ratio) if ratio else 0.5
            per2 = collections.defaultdict(dict)
            for line in io.open(a.internal, encoding='utf-8'):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get('case_kind') != a.case_kind:
                    continue
                if r['arm'] == a.ref_arm and r.get('ratio') == a.ref_ratio:
                    per2[r['case_key']]['ref'] = r['f1']
                if r['arm'] == arm and r.get('ratio') == ratio:
                    per2[r['case_key']]['ch'] = r['f1']
            pairs = [(v['ref'], v['ch']) for v in per2.values()
                     if 'ref' in v and 'ch' in v]
            label = '%s@%s' % (arm, ratio)
        else:
            pairs = [(v['ref'], v[ch]) for k, v in have.items() if ch in v]
            label = ch
        if len(pairs) < 5:
            lines.append('  %-26s 配对数不足（%d），不报' % (label, len(pairs)))
            continue
        diffs = [x - y for x, y in pairs]
        m, lo, hi = boot(diffs, a.B)
        sig = 'CI 不含 0' if (lo > 0 or hi < 0) else '**CI 含 0，不显著**'
        lines.append('  %-26s n=%2d  Δ=%.4f  [%+.4f, %+.4f]  %s'
                     % (label, len(pairs), m, lo, hi, sig))
    txt = '\n'.join(lines)
    print(txt)
    if a.out:
        io.open(a.out, 'w', encoding='utf-8', newline='\n').write(txt + '\n')
        print('\n[boot] wrote %s' % a.out)


if __name__ == '__main__':
    main()
