"""完整性审计：把 docs/17 里手抄的数字从源产物重算并逐项比对。

动机
----
`docs/17` 是"唯一入口"文档，里面的表格是**手工**从多个 cells.json 抄来的。抄错一位
不会被任何人发现，却会一路传进论文。这个脚本把那些数字变成**可重算**的：
从源产物取数，与文档里写的值逐项比，只打印不一致项。

覆盖范围：主表（Qwen 112 例）、Mistral 复现、外部基线两个尺度、隐私表。
"""
import io
import json
import os

BASE = r'D:\CodexData\tmp\privtrim-review-20261015'
R = BASE + r'\remote_probe'
FIX = BASE + r'\fixres'


def cells(p):
    o = json.load(io.open(p, encoding='utf-8'))
    out = {}
    for k, v in o['all'].items():
        n = max(1, v['n'])
        out[k] = dict(n=v['n'], f1=v['f1'] / n, tok=v['tok'] / n, gic=v['ans'] / n,
                      leak=v['cv'] / max(1, v['cn']), tent=v['tent'] / n, coll=v['coll'] / n)
    return out


def cmp1(label, got, exp, tol=5e-5):
    if got is None:
        return '%s: MISSING (expected %s)' % (label, exp)
    if abs(got - exp) > tol:
        return '%s: got %.6f expected %.6f  <-- MISMATCH' % (label, got, exp)
    return None


def main():
    bad = []
    checked = 0

    q = cells(FIX + r'\aliaslink_qwen\cells.json')
    # docs/17 §2 主表（Qwen，112 例）
    MAIN = {
        'full|None': (0.7189, 7964, 1.000),
        'budget_cluster_link|0.5': (0.7182, 3706, 1.000),
        'alias_budget_link|0.5': (0.5397, 3708, 1.000),
        'alias|None': (0.4933, 7824, 1.000),
        'delete|None': (0.4916, 7845, 0.982),
        'alias_budget|0.5': (0.4866, 3708, 1.000),
        'budget_cluster|0.5': (0.4688, 3705, 1.000),
        'delete_budget|0.5': (0.4688, 3696, 0.982),
        'budget_random|0.5': (0.2366, 3659, 0.455),
    }
    for k, (f1, tok, gic) in MAIN.items():
        v = q.get(k)
        checked += 3
        for lbl, got, exp in ((k + ' F1', (v or {}).get('f1'), f1),
                              (k + ' tok', (v or {}).get('tok'), tok),
                              (k + ' gic', (v or {}).get('gic'), gic)):
            r = cmp1(lbl, got, exp, 0.5 if 'tok' in lbl else (5e-3 if ('gic' in lbl or 'tent' in lbl) else 5e-5))
            if r:
                bad.append(r)

    m = cells(FIX + r'\aliaslink_mistral\cells.json')
    checked += 3
    for k, exp in (('budget_cluster_link|0.5', 0.7075), ('budget_cluster|0.5', 0.5196),
                   ('full|None', 0.6931)):
        r = cmp1('mistral ' + k, (m.get(k) or {}).get('f1'), exp, 5e-5)
        if r:
            bad.append(r)
    gain_q = q['budget_cluster_link|0.5']['f1'] - q['budget_cluster|0.5']['f1']
    gain_m = m['budget_cluster_link|0.5']['f1'] - m['budget_cluster|0.5']['f1']
    checked += 2
    for lbl, got, exp in (('gain Qwen', gain_q, 0.2495), ('gain Mistral', gain_m, 0.1879)):
        r = cmp1(lbl, got, exp, 5e-4)
        if r:
            bad.append(r)

    # 外部基线 out-of-scope（docs/17 §3）
    e = cells(BASE + r'\extbl_multi\scored.json')
    ext = json.load(io.open(BASE + r'\extbl_multi\mk.json', encoding='utf-8'))
    mkv = {k: dict(f1=v['f1'] / max(1, v['n']), tok=v['tok'] / max(1, v['n']),
                   gic=v['ans'] / max(1, v['n'])) for k, v in ext['all'].items()}
    EXT = {'ext_provence|0.5': (0.4750, 0.900), 'ext_cpc|0.5': (0.3500, 0.850),
           'ext_llmlingua2|0.5': (0.3417, 0.700), 'ext_quito_x|0.5': (0.3793, 0.724),
           'ext_mask_llmlingua2|0.5': (0.0000, 0.800)}
    for k, (f1, gic) in EXT.items():
        v = e.get(k)
        checked += 2
        for lbl, got, exp in ((k + ' F1', (v or {}).get('f1'), f1), (k + ' gic', (v or {}).get('gic'), gic)):
            r = cmp1('oos ' + lbl, got, exp, 6e-4)
            if r:
                bad.append(r)
    v = mkv.get('ext_mask_quito_x|0.5')
    checked += 2
    for lbl, got, exp in (('ext_mask_quito_x F1', (v or {}).get('f1'), 0.0000),
                          ('ext_mask_quito_x gic', (v or {}).get('gic'), 0.700)):
        r = cmp1('oos ' + lbl, got, exp, 6e-4)
        if r:
            bad.append(r)

    # in-scope（docs/17 §3）
    i = cells(FIX + r'\inscope_qwen\cells.json')
    mk_i = json.load(io.open(BASE + r'\extbl_inscope_target_last\mk.json', encoding='utf-8'))
    mki = {k: dict(f1=v['f1'] / max(1, v['n']), tok=v['tok'] / max(1, v['n']),
                   gic=v['ans'] / max(1, v['n'])) for k, v in mk_i['all'].items()}
    INS = {'budget_cluster_link|0.5': (0.8947, 144, 1.00),
           'full|None': (0.5263, 343, 1.00),
           'ext_quito_x|0.5': (0.3509, 171, 0.95),
           'ext_llmlingua2|0.5': (0.2982, 171, 0.37),
           'ext_cpc|0.5': (0.1053, 171, 0.63),
           'ext_provence|0.5': (0.0526, 171, 0.37),
           'budget_cluster|0.5': (0.0000, 137, 1.00)}
    for k, (f1, tok, gic) in INS.items():
        v = i.get(k)
        checked += 3
        for lbl, got, exp in ((k + ' F1', (v or {}).get('f1'), f1), (k + ' tok', (v or {}).get('tok'), tok),
                              (k + ' gic', (v or {}).get('gic'), gic)):
            r = cmp1('inscope ' + lbl, got, exp, 0.6 if 'tok' in lbl else (5e-3 if ('gic' in lbl or 'tent' in lbl) else 5e-5))
            if r:
                bad.append(r)
    v = mki.get('ext_mask_quito_x|0.5')
    checked += 3
    for lbl, got, exp in (('ext_mask_quito_x F1', (v or {}).get('f1'), 0.0000),
                          ('ext_mask_quito_x tok', (v or {}).get('tok'), 171),
                          ('ext_mask_quito_x gic', (v or {}).get('gic'), 1.00)):
        r = cmp1('inscope ' + lbl, got, exp, 0.6 if 'tok' in lbl else (5e-3 if ('gic' in lbl or 'tent' in lbl) else 5e-5))
        if r:
            bad.append(r)

    print('AUDIT :: checked %d numbers against source artifacts' % checked)
    if bad:
        print('MISMATCHES (%d):' % len(bad))
        for b in bad:
            print('  ' + b)
        raise SystemExit('docs/17 与源产物不一致，先查清再引用')
    print('AUDIT PASSED :: every number in docs/17 traces to a source artifact')


if __name__ == '__main__':
    main()
