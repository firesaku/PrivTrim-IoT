"""提示词口径的隐私表：上下文泄漏 ∪ 问题泄漏。

为什么需要它（复查 C1）
-----------------------
`leak_of(case, text)` 只对**上下文**做暴露检测。但模型真正读到的是 `问题 + 上下文`，
而问题里往往原样写着被查询的实体（就是 `target_entity`，就是 canary）。

实测（112 例）：不开 `--alias-question` 时**全部 21 个臂格**在问题侧泄漏 40/112；
所以已发布过的那些 `leakV = 0.00%` 都是**上下文口径**的数字，
在提示词口径上并不成立。

本脚本给出正确的表：逐例把两侧并起来，得到**提示词层**的泄漏率。

数据来源
--------
  percase.jsonl  : 逐例 `ctx_leak_value` / `ctx_leak_frag` / `n_canaries` → 上下文侧
  cases.jsonl    : canaries 与 question → 问题侧（用 privcheck.exposure_basic 现算）
  该次运行的臂集合 : 从 percase 里读，并按 `--alias-question` 复现问题的别名化规则

问题侧的规则必须与 harness 完全一致：只有 `ARMS_ALIAS` 且开了 flag 的臂才改写问题。
"""
import argparse
import collections
import io
import json
import os
import sys

ARMS_ALIAS = ['alias', 'alias_budget', 'alias_budget_link']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--percase', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--privcheck-dir', required=True)
    ap.add_argument('--v4-dir', required=True)
    ap.add_argument('--alias-question', action='store_true')
    ap.add_argument('--label', default='run')
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()

    sys.path.insert(0, a.privcheck_dir)
    sys.path.insert(0, a.v4_dir)
    import privcheck
    import v4_router

    cases = {c['key']: c for c in
             (json.loads(l) for l in io.open(a.cases, encoding='utf-8') if l.strip())}
    rows = [json.loads(l) for l in io.open(a.percase, encoding='utf-8') if l.strip()]

    qcache = {}

    def qleak(case_key, arm):
        """问题侧的 (value_hits, frag_hits)。"""
        k = (case_key, arm in ARMS_ALIAS and a.alias_question)
        if k in qcache:
            return qcache[k]
        c = cases.get(case_key)
        if not c:
            return (0, 0)
        q = c.get('question') or ''
        if arm in ARMS_ALIAS and a.alias_question:
            q = v4_router.apply_replacements(q, c.get('canaries') or [],
                                             v4_router.Aliaser('alias'))
        v = f = 0
        for x in (c.get('canaries') or []):
            fv, ff = privcheck.exposure_basic(x, q)
            v += 1 if fv else 0
            f += 1 if ff else 0
        qcache[k] = (v, f)
        return v, f

    agg = collections.defaultdict(lambda: dict(n=0, ctx_cases=0, q_cases=0, prompt_cases=0,
                                               ncan=0, ctx_v=0, ctx_f=0, q_v=0, q_f=0))
    for r in rows:
        arm = r['arm']
        ratio = r.get('ratio')
        key = '%s@%s' % (arm, 'ref' if ratio is None else ratio)
        s = agg[key]
        s['n'] += 1
        s['ncan'] += r.get('n_canaries') or 0
        cv = r.get('ctx_leak_value') or 0
        cf = r.get('ctx_leak_frag') or 0
        qv, qf = qleak(r['case_key'], arm)
        s['ctx_v'] += cv
        s['ctx_f'] += cf
        s['q_v'] += qv
        s['q_f'] += qf
        ctx_hit = (cv > 0 or cf > 0)
        q_hit = (qv > 0 or qf > 0)
        s['ctx_cases'] += 1 if ctx_hit else 0
        s['q_cases'] += 1 if q_hit else 0
        s['prompt_cases'] += 1 if (ctx_hit or q_hit) else 0

    print('PRIVACY[%s] alias_question=%s  (per-arm: cases_leaking / n ; ctx vs question vs prompt)'
          % (a.label, a.alias_question))
    out = []
    for key in sorted(agg, key=lambda k: (k.split('@')[0], k.split('@')[1])):
        s = agg[key]
        n = max(1, s['n'])
        out.append('%s n=%d ctx=%d/%d q=%d/%d PROMPT=%d/%d'
                   % (key, s['n'], s['ctx_cases'], s['n'], s['q_cases'], s['n'],
                      s['prompt_cases'], s['n']))
    print(' ;; '.join(out))
    if a.json_out:
        io.open(a.json_out, 'w', encoding='utf-8').write(json.dumps(
            dict(label=a.label, alias_question=a.alias_question,
                 rows={k: v for k, v in agg.items()}), ensure_ascii=False, indent=1))
        print('[privacy] wrote %s' % a.json_out)


if __name__ == '__main__':
    main()
