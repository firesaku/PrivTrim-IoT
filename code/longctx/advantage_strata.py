"""优势分层：我们的领先出现在哪一类案例上？（用既有产物，不新增数字来源）

动机
----
汇总表里对 `provence` 只是"总量领先"，逐来源看还是 2/5 胜 3/5 负（`docs/17` §3.6）。
按 YF 的读法，这时应该**找出有解释力的优势场景**：把案例按"答案在上下文里的位置"和
"实体数"分层，看领先集中在哪一层、为什么。

判据（先定后算）
- 位置：`answer_char_offset / len(context)`（越大＝答案越靠后，越容易被尾部裁剪伤到）。
- 实体数：`n_entities`。
- 分层内比 F1（该层内 f1 总和 / 案例数），并统计"该层第一名是谁"。

输入（全部只读既有产物）
- `--cases`  ：`data/longctx_cases.jsonl`（案例元信息：来源、实体数、答案偏移）
- `--internal`: `fixres/aliasq_qwen/percase.jsonl`（内部臂逐例）
- `--ext`    ：`extbl_multi/scored.jsonl`（已发布方法逐例，同一批多实体案例）
"""
import argparse
import collections
import io
import json

OURS = 'alias_budget_link'
EXT_KEEP = ('ext_provence', 'ext_llmlingua2', 'ext_cpc', 'ext_mask_quito_x',
            'ext_mask_llmlingua2', 'ext_quito_x', 'ext_dac')
INT_KEEP = ('full', 'alias', 'delete', 'budget_cluster', 'budget_cluster_link',
            'alias_budget_link', 'alias_budget', 'budget_random', 'delete_budget')
RATIO = '0.5'


def load_cases(path):
    out = {}
    for line in io.open(path, encoding='utf-8'):
        if not line.strip():
            continue
        c = json.loads(line)
        ctx = c.get('context') or ''
        pos = c.get('answer_char_offset')
        if pos is None:
            pos = c.get('entity_offset')
        rel = (float(pos) / max(1, len(ctx))) if pos is not None else None
        out[c['key']] = dict(source=c.get('source'), kind=c.get('case_kind'),
                             n_entities=c.get('n_entities'), n_canaries=len(c.get('canaries') or []),
                             rel=rel, ctx_chars=len(ctx),
                             region_start=c.get('target_region_start'),
                             region_end=c.get('target_region_end'))
    return out


def load_rows(path, keep, kinds=('multi_entity',)):
    rows = []
    for line in io.open(path, encoding='utf-8'):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('arm') not in keep:
            continue
        if kinds and r.get('case_kind') not in kinds:
            continue
        if str(r.get('ratio')) != RATIO:
            continue
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--internal', required=True)
    ap.add_argument('--ext', nargs='*', default=[])
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    C = load_cases(a.cases)
    rows = load_rows(a.internal, INT_KEEP)
    for p in a.ext:
        rows += load_rows(p, EXT_KEEP)

    by_case = collections.defaultdict(dict)
    for r in rows:
        by_case[r['case_key']][r['arm']] = r.get('f1') or 0.0
    cases = sorted(k for k in by_case if k in C and C[k]['rel'] is not None)
    arms = sorted({r['arm'] for r in rows})

    def strat(cases, key, cuts, names):
        out = collections.OrderedDict()
        for nm in names:
            out[nm] = [k for k in cases if cuts(nm, C[k][key])]
        return out

    pos_strata = collections.OrderedDict([
        ('答案在头部（<1/3）', [k for k in cases if C[k]['rel'] < 1 / 3]),
        ('答案在中部（1/3–2/3）', [k for k in cases if 1 / 3 <= C[k]['rel'] < 2 / 3]),
        ('答案在尾部（≥2/3）', [k for k in cases if C[k]['rel'] >= 2 / 3]),
    ])
    ent_strata = collections.OrderedDict([
        ('实体数 = 3（n=8）', [k for k in cases if (C[k]['n_entities'] or 0) <= 3]),
        ('实体数 = 5（n=31）', [k for k in cases if (C[k]['n_entities'] or 0) == 5]),
        ('实体数 ≥ 6（n=1）', [k for k in cases if (C[k]['n_entities'] or 0) >= 6]),
    ])

    def table(strata, arms):
        L = ['| 分层 | n | ' + ' | '.join('`%s`' % a for a in arms) + ' | 该层第一 |',
             '|---|---|' + '---|' * (len(arms) + 1)]
        first = collections.Counter()
        for nm, ks in strata.items():
            if not ks:
                continue
            vals = {}
            for arm in arms:
                xs = [by_case[k][arm] for k in ks if arm in by_case[k]]
                vals[arm] = (sum(xs) / len(xs)) if xs else None
            best = max((v, k) for k, v in vals.items() if v is not None)[1]
            first[best] += 1
            L.append('| %s | %d | %s | **%s** |' % (
                nm, len(ks), ' | '.join('--' if vals[x] is None else '%.4f' % vals[x] for x in arms),
                best))
        return L, first

    L = []
    L.append('# 优势分层：领先出现在哪一类案例上')
    L.append('')
    L.append('预算 @%s；分层内 F1 = 该层 f1 总和 / 案例数；案例 = 40 个多实体案例中'
             '元信息齐全的 %d 个。' % (RATIO, len(cases)))
    L.append('')
    L.append('## A. 按答案位置分层')
    L.append('')
    ta, fa = table(pos_strata, arms)
    L += ta
    L.append('')
    L.append('## B. 按实体数分层')
    L.append('')
    tb, fb = table(ent_strata, arms)
    L += tb
    L.append('')
    L.append('## C. 分层第一名的计数')
    L.append('')
    for k, v in sorted(fa.items(), key=lambda kv: -kv[1]):
        L.append('- 位置分层：`%s` 第一 %d 层' % (k, v))
    for k, v in sorted(fb.items(), key=lambda kv: -kv[1]):
        L.append('- 实体数分层：`%s` 第一 %d 层' % (k, v))

    txt = '\n'.join(L) + '\n'
    if a.out:
        io.open(a.out, 'w', encoding='utf-8', newline='\n').write(txt)
    # 控制台只打要点，避免长表被折叠
    tail = pos_strata['答案在尾部（≥2/3）']
    head = pos_strata['答案在头部（<1/3）']
    def mean(ks, arm):
        xs = [by_case[k][arm] for k in ks if arm in by_case[k]]
        return sum(xs) / len(xs) if xs else float('nan')
    print('cases=%d arms=%d' % (len(cases), len(arms)))
    print('STRATUM_SIZES head=%d mid=%d tail=%d | ent3=%d ent5=%d ent6=%d' % (
        len(head), len(pos_strata['答案在中部（1/3–2/3）']), len(tail),
        len(ent_strata['实体数 = 3（n=8）']), len(ent_strata['实体数 = 5（n=31）']),
        len(ent_strata['实体数 ≥ 6（n=1）'])))
    print('ENT3 top4: ' + ' '.join('%s=%.4f' % (x, mean(ent_strata['实体数 = 3（n=8）'], x)) for x in
                                   sorted(arms, key=lambda x: -mean(ent_strata['实体数 = 3（n=8）'], x))[:4]))
    print('ENT5 top4: ' + ' '.join('%s=%.4f' % (x, mean(ent_strata['实体数 = 5（n=31）'], x)) for x in
                                   sorted(arms, key=lambda x: -mean(ent_strata['实体数 = 5（n=31）'], x))[:4]))
    print('TAIL top5: ' + ' '.join('%s=%.4f' % (x, mean(tail, x)) for x in
                                   sorted(arms, key=lambda x: -mean(tail, x))[:5]))
    print('HEAD top5: ' + ' '.join('%s=%.4f' % (x, mean(head, x)) for x in
                                   sorted(arms, key=lambda x: -mean(head, x))[:5]))
    print('FIRST_place_counts position=%s entity=%s' % (dict(fa), dict(fb)))


if __name__ == '__main__':
    main()
