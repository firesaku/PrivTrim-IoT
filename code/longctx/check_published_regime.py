"""核对已发布 batch1 基准的结构：可查询对象与需保护对象是否相交。

这不是复现他们的实验，而是检查他们的**设计**能不能表达"敏感且相关"那一格：
答案本身要求使用必须被保护的值。

产出：逐项计数 + 结论。对应 docs/18。
"""
import argparse
import io
import json
import re
import statistics as st

WS = re.compile(r'\s+')


def nz(s):
    return WS.sub(' ', s or '').strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True, help='batch1 cases.json')
    ap.add_argument('--summary', default='', help='batch1 summary.json（可选）')
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()

    cases = json.load(io.open(a.cases, encoding='utf-8'))
    n = len(cases)
    q_hit = ans_hit = src_hit = with_can = 0
    positions = []
    for c in cases:
        cans = [nz(x.get('value')) for x in (c.get('canaries') or []) if x.get('value')]
        if not cans:
            continue
        with_can += 1
        q = nz(c.get('question'))
        ans = ' '.join(nz(x.get('text')) for x in (c.get('answers') or []))
        src = nz(c.get('source_context'))
        if any(v in q for v in cans):
            q_hit += 1
        if any(v in ans for v in cans):
            ans_hit += 1
        if any(v in src for v in cans):
            src_hit += 1
        ctx = c.get('context') or ''
        for x in (c.get('canaries') or []):
            i = ctx.find(x.get('value') or '')
            if i >= 0:
                positions.append(i / max(1, len(ctx)))

    out = dict(n_cases=n, cases_with_canary=with_can,
               question_contains_canary=q_hit,
               answer_contains_canary=ans_hit,
               canary_also_in_source_context=src_hit,
               n_canary_occurrences=len(positions))
    if positions:
        out['canary_relative_position'] = dict(min=min(positions), median=st.median(positions),
                                              max=max(positions))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print()
    if q_hit == 0 and ans_hit == 0 and src_hit == 0:
        print('VERDICT: 可查询对象与需保护对象完全不相交 —— '
              '"敏感且相关"那一格在设计上不存在。')
    else:
        print('VERDICT: 存在交集（q=%d ans=%d src=%d），需重新审视结论。'
              % (q_hit, ans_hit, src_hit))

    if a.summary:
        s = json.load(io.open(a.summary, encoding='utf-8'))
        rows = s['tables']['matched']
        print()
        print('published matched-track results (n=%d):' % n)
        print('%-18s %7s %10s %9s' % ('method', 'F1', 'retention', 'pin_full'))
        for r in rows:
            print('%-18s %7.4f %10.3f %9.3f'
                  % (r['method'], r['f1'], r['retention'], r.get('pin_full', -1)))
        out['published'] = [{k: r.get(k) for k in ('method', 'f1', 'retention', 'pin_full')}
                            for r in rows]

    if a.json_out:
        io.open(a.json_out, 'w', encoding='utf-8').write(
            json.dumps(out, ensure_ascii=False, indent=1))
        print('\n[regime] wrote %s' % a.json_out)


if __name__ == '__main__':
    main()
