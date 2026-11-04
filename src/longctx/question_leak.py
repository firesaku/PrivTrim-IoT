"""问题侧泄漏：修正"零泄漏"这个声明。

问题
----
`leak_of(case, text)` 只对**上下文**做 `privcheck.exposure_basic`，从不看问题字符串。
但模型真正读到的提示词是 `问题 + 上下文`。于是：

  - `delete` / `delete_budget` 报 `leakV = 0.00%`，
  - 作业 A 的 `alias_budget_link`（未开 --alias-question）也报 `0.00%`，

而它们的**问题里原样带着被查询的实体串** —— 恰恰是四象限里"敏感且相关"那一格，
也就是整篇论文的核心。所以那个 0.00% 是**上下文口径**的数字，不是提示词口径。

本脚本（模型无关，不需要 GPU）
------------------------------
对每条臂按它自己的问题策略重建问题字符串，逐 canary 做暴露检测，给出：
  q_leak_v / q_leak_f   问题侧的整值 / 片段泄漏
  ctx_leak_v            上下文侧（从 cells.json 读，已有）
  prompt_leak_min       提示词层泄漏的**下界** = 问题侧（因为上下文侧的并集只会更大）

问题策略
--------
  --alias-question 关：所有臂都用原始问题（= 作业 A）
  --alias-question 开：仅 ARMS_ALIAS 用同一份别名映射改写后的问题，其余臂仍用原始问题
本脚本两种策略都算，直接给出对照。
"""
import argparse
import io
import json
import os
import sys

ARMS_ALIAS = ['alias', 'alias_budget', 'alias_budget_link']
ARMS = ['full', 'delete', 'alias', 'budget_cluster', 'budget_cluster_link',
        'budget_random', 'delete_budget', 'alias_budget', 'alias_budget_link']
BUDGETS = [0.25, 0.5, 0.75]


def load_cases(path):
    return [json.loads(l) for l in io.open(path, encoding='utf-8') if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--privcheck-dir', required=True)
    ap.add_argument('--v4-dir', required=True)
    ap.add_argument('--cells', default='', help='可选：读上下文侧泄漏做对照')
    ap.add_argument('--json-out', required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.privcheck_dir)
    sys.path.insert(0, a.v4_dir)
    import privcheck
    import v4_router

    cases = load_cases(a.cases)

    def question_for(case, arm, alias_question):
        q = case.get('question') or ''
        if not alias_question or arm not in ARMS_ALIAS:
            return q
        return v4_router.apply_replacements(q, case.get('canaries') or [],
                                            v4_router.Aliaser('alias'))

    rows = []
    for alias_question in (False, True):
        for arm in ARMS:
            ratios = [None] if arm in ('full', 'delete', 'alias') else BUDGETS
            for ratio in ratios:
                n_can = qv = qf = cases_with_leak = 0
                examples = []
                for c in cases:
                    cans = c.get('canaries') or []
                    if not cans:
                        continue
                    n_can += len(cans)
                    q = question_for(c, arm, alias_question)
                    hit_v = hit_f = 0
                    for x in cans:
                        full, frag = privcheck.exposure_basic(x, q)
                        hit_v += 1 if full else 0
                        hit_f += 1 if frag else 0
                    qv += hit_v
                    qf += hit_f
                    if hit_v or hit_f:
                        cases_with_leak += 1
                        if len(examples) < 2:
                            examples.append(c['key'])
                rows.append(dict(arm=arm, ratio=ratio, alias_question=alias_question,
                                 n_cases=len(cases), n_canaries=n_can,
                                 q_leak_value=qv, q_leak_frag=qf,
                                 cases_with_question_leak=cases_with_leak,
                                 examples=examples))

    # 上下文侧对照（若给了 cells.json）
    ctx = {}
    if a.cells and os.path.isfile(a.cells):
        o = json.load(io.open(a.cells, encoding='utf-8'))
        for k, v in o['all'].items():
            arm, _, ratio = k.partition('|')
            ctx[(arm, None if ratio in ('None', '') else float(ratio))] = (
                v['cv'], v['cf'], v['cn'])

    out = []
    for r in rows:
        cv, cf, cn = ctx.get((r['arm'], r['ratio']), (None, None, None))
        out.append('%s@%s flag=%-5s  q_leak: value=%d frag=%d cases=%d/%d  '
                   'ctx_leak: value=%s frag=%s of %s'
                   % (r['arm'], ('ref' if r['ratio'] is None else r['ratio']),
                      r['alias_question'], r['q_leak_value'], r['q_leak_frag'],
                      r['cases_with_question_leak'], r['n_cases'],
                      cv, cf, cn))
    # 单行输出，避免显示层折叠
    print('QUESTION-LEAK :: ' + ' ;; '.join(out))
    io.open(a.json_out, 'w', encoding='utf-8').write(
        json.dumps(dict(rows=rows), ensure_ascii=False, indent=1))
    print('[qleak] wrote %s' % a.json_out)


if __name__ == '__main__':
    main()
