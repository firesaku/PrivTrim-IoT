"""按公开数据集拆开看：同一个结论在每个来源上都成立吗？

为什么单独做这一张表
--------------------
`docs/17` 的主表是 112 例的汇总。评审最自然的一问是：**结论是不是只靠某一个数据集撑起来的？**
本脚本把既有产物按 `来源` 拆开重算（不新增数字来源），并排显示"我们 / 纯冗余度 / 已发布方法"
在每个公开数据集上的表现。

数据来源（全部只读既有产物）
- `--percase`：内部臂逐例结果（含 `source` / `case_kind`），如 `fixres/aliasq_qwen/percase.jsonl`
- `--ext`：已发布方法在同一批多实体案例上的 `cells`，如 `extbl_multi/scored.json`
- 来源清单：`data/MANIFEST_longctx.json`

口径：F1 = 该来源内 `f1` 总和 / 案例数（与主表一致）；`n` = 案例数；`gic` = 答案串是否仍在上下文里。
两个子集分母不同（全部 112 例 vs 多实体 40 例），所以**两张表分别看，不混用**。
"""
import argparse
import collections
import io
import json
import os

KINDS = ('all', 'multi_entity')


def from_percase(path, kind):
    agg = collections.defaultdict(lambda: dict(n=0, f1=0.0, gic=0))
    for line in io.open(path, encoding='utf-8'):
        if not line.strip():
            continue
        r = json.loads(line)
        if kind != 'all' and r.get('case_kind') != kind:
            continue
        k = (r['source'], r['arm'], str(r['ratio']))
        a = agg[k]
        a['n'] += 1
        a['f1'] += (r.get('f1') or 0.0)
        a['gic'] += int(r.get('ans_in_ctx') or 0)
    out = {}
    for k, a in agg.items():
        n = max(1, a['n'])
        out[k] = dict(n=a['n'], f1=a['f1'] / n, gic=a['gic'] / n)
    return out


def from_ext(paths):
    out = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        o = json.load(io.open(path, encoding='utf-8'))
        for k, v in (o.get('cells') or {}).items():
            p = k.split('|')
            if len(p) != 3:
                continue
            n = max(1, v['n'])
            out.setdefault((p[0], p[1], p[2]), dict(n=v['n'], f1=v['f1'] / n, gic=v['ans'] / n))
    return out


def cell(tab, src, arm, ratio):
    return tab.get((src, arm, ratio))


def total(tab, srcs, arm, ratio):
    rows = [tab[(s, arm, ratio)] for s in srcs if (s, arm, ratio) in tab]
    if not rows:
        return None, 0
    return sum(r['f1'] * r['n'] for r in rows) / sum(r['n'] for r in rows), sum(r['n'] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--percase', required=True)
    ap.add_argument('--ext', nargs='*', default=[], help='已发布方法的 cells 产物（可多个，如 scored.json mk.json）')
    ap.add_argument('--ratio', default='0.5')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    R = a.ratio

    I = from_percase(a.percase, 'all')
    M = from_percase(a.percase, 'multi_entity')
    E = from_ext(a.ext)

    srcs = sorted({k[0] for k in I})
    msrc = sorted({k[0] for k in M})
    L = []
    L.append('# 按公开数据集拆开（`per_source_table.py` 生成，可复跑）')
    L.append('')
    L.append('## A. 全部 112 例（Qwen2.5-7B，作业 10602439，预算 %s）' % R)
    L.append('')
    L.append('| 来源（公开数据集） | n | `full` 不压缩 | `budget_cluster` 纯冗余度 | `budget_cluster_link` 我们 | `alias_budget_link` 我们 | `budget_random` |')
    L.append('|---|---|---|---|---|---|---|')
    for s in srcs:
        n = (cell(I, s, 'full', 'None') or {}).get('n')
        vals = [cell(I, s, arm, r) for arm, r in (
            ('full', 'None'), ('budget_cluster', R), ('budget_cluster_link', R),
            ('alias_budget_link', R), ('budget_random', R))]
        L.append('| `%s` | %s | %s |' % (s, n, ' | '.join(
            '--' if v is None else '%.4f' % v['f1'] for v in vals)))
    t = [total(I, srcs, arm, r) for arm, r in (
        ('full', 'None'), ('budget_cluster', R), ('budget_cluster_link', R),
        ('alias_budget_link', R), ('budget_random', R))]
    L.append('| **合计** | %d | %s |' % (t[0][1], ' | '.join('%.4f' % x[0] for x in t)))
    L.append('')
    L.append('## B. 40 个多实体案例（同一批案例上跑已发布方法；分母是该来源在多实体子集里的案例数）')
    L.append('')
    L.append('| 来源 | n | `budget_cluster_link` 我们 | `ext_provence` | `ext_llmlingua2` | `ext_mask_quito_x` | `budget_cluster` 纯冗余度 |')
    L.append('|---|---|---|---|---|---|---|')
    for s in msrc:
        n = (cell(M, s, 'budget_cluster_link', R) or cell(M, s, 'full', 'None') or {}).get('n')
        vals = [cell(M, s, 'budget_cluster_link', R), cell(E, s, 'ext_provence', R),
                cell(E, s, 'ext_llmlingua2', R), cell(E, s, 'ext_mask_quito_x', R),
                cell(M, s, 'budget_cluster', R)]
        L.append('| `%s` | %s | %s |' % (s, n, ' | '.join(
            '--' if v is None else '%.4f' % v['f1'] for v in vals)))
    tm = [total(M, msrc, 'budget_cluster_link', R), total(E, msrc, 'ext_provence', R),
          total(E, msrc, 'ext_llmlingua2', R),
          total(E, msrc, 'ext_mask_quito_x', R), total(M, msrc, 'budget_cluster', R)]
    L.append('| **合计** | %d | %s |' % (tm[0][1], ' | '.join(
        '--' if x[0] is None else '%.4f' % x[0] for x in tm)))
    L.append('')
    L.append('## C. 校验（必须与主表一致）')
    L.append('')
    L.append('- A 合计：`full` %.4f / `budget_cluster` %.4f / `budget_cluster_link` %.4f / '
             '`alias_budget_link` %.4f / `budget_random` %.4f'
             % (t[0][0], t[1][0], t[2][0], t[3][0], t[4][0]))
    L.append('- B 合计：我们 %.4f / provence %.4f / llmlingua2 %.4f / mask_quito_x %.4f / '
             '`budget_cluster` %.4f' % tuple(x[0] if x[0] is not None else float('nan') for x in tm))

    txt = '\n'.join(L)
    print('CHK_A full=%.6f cluster=%.6f link=%.6f aliaslink=%.6f random=%.6f' % (
        t[0][0], t[1][0], t[2][0], t[3][0], t[4][0]))
    print('CHK_B ours=%.4f prov=%.4f ll2=%.4f mkq=%.4f cluster=%.4f' % tuple(
        x[0] if x[0] is not None else -1 for x in tm))
    print('SOURCES_A=%s' % ','.join(srcs))
    print('SOURCES_B=%s' % ','.join(msrc))
    if a.out:
        io.open(a.out, 'w', encoding='utf-8', newline='\n').write(txt + '\n')
        print('[per_source] wrote %s' % a.out)


if __name__ == '__main__':
    main()
