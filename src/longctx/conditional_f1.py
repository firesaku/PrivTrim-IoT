"""条件 F1：只在"答案确实还在上下文里"的案例上比较（F1 | gold_in_ctx = 1）。

为什么必须报这个
----------------
各臂的 `gold_in_ctx` 差别很大（例如 `provence` 0.90、`llmlingua2` 0.70、
`budget_cluster` 在 multi_entity 上是 1.00）。直接比总体 F1 会把两件事混在一起：

  (a) 该臂**把答案丢了** —— 结构上不可能答对，这不是"策略差"，是"证据没了"；
  (b) 该臂**保住了答案但模型用不上** —— 这才是选择策略本身的问题。

条件 F1 把 (a) 摘出去，只看 (b)。两个数一起报才诚实：
  - 总体 F1   = 端到端可用性（含丢答案的代价）
  - F1|gic=1  = 纯策略质量
  - 覆盖率    = gic 的比例，说明条件样本有多大，避免拿极小子集说事。
"""
import argparse
import collections
import io
import json
import sys


def load(path):
    return [json.loads(l) for l in io.open(path, encoding='utf-8') if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--percase', action='append', required=True,
                    help='可多次给：name=path')
    ap.add_argument('--min-n', type=int, default=5,
                    help='条件样本小于该值不报，避免拿极小子集说事')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    lines = []
    for spec in a.percase:
        name, _, path = spec.partition('=')
        rows = load(path)
        agg = collections.defaultdict(lambda: dict(n=0, f1=0.0, cov=0, cf1=0.0, cn=0,
                                                   tok=0, ctok=0))
        for r in rows:
            key = '%s@%s' % (r['arm'], 'ref' if r.get('ratio') is None else r['ratio'])
            s = agg[key]
            s['n'] += 1
            s['f1'] += r['f1']
            s['tok'] += r.get('ctx_tokens') or 0
            if r.get('ans_in_ctx'):
                s['cov'] += 1
                s['cf1'] += r['f1']
                s['cn'] += 1
                s['ctok'] += r.get('ctx_tokens') or 0
        lines.append('=== %s (cases=%d) ===' % (name, len(rows)))
        for key in sorted(agg, key=lambda k: -agg[k]['f1'] / max(1, agg[k]['n'])):
            s = agg[key]
            n = max(1, s['n'])
            cov = s['cov']
            if cov < a.min_n:
                lines.append('  %-26s totalF1=%.4f  (gic=%d/%d 样本不足，不报条件 F1)'
                             % (key, s['f1'] / n, cov, s['n']))
                continue
            lines.append('  %-26s totalF1=%.4f tok=%.0f | gic=%d/%d (%.0f%%) '
                         'F1|gic=%.4f tok|gic=%.0f'
                         % (key, s['f1'] / n, s['tok'] / n, cov, s['n'], 100.0 * cov / s['n'],
                            s['cf1'] / cov, s['ctok'] / cov))
    txt = '\n'.join(lines)
    print(txt)
    if a.out:
        io.open(a.out, 'w', encoding='utf-8', newline='\n').write(txt + '\n')
        print('\n[condf1] wrote %s' % a.out)


if __name__ == '__main__':
    main()
