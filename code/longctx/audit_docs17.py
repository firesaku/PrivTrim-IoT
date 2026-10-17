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
import re

BASE = r'D:\CodexData\tmp\privtrim-review-20261015'
R = BASE + r'\remote_probe'
FIX = BASE + r'\fixres'
REP = R + r'\results\reports'


def cells(p):
    o = json.load(io.open(p, encoding='utf-8'))
    out = {}
    for k, v in o['all'].items():
        n = max(1, v['n'])
        out[k] = dict(n=v['n'], f1=v['f1'] / n, tok=v['tok'] / n, gic=v['ans'] / n,
                      leak=v['cv'] / max(1, v['cn']), tent=v['tent'] / n, coll=v['coll'] / n,
                      ov=v.get('ov', 0))
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

    # docs/17 §3.5 提问方式稳健性（两个变体）
    pa = cells(R + r'\results\cells\longctx_paraA_cells.json')
    pb = cells(R + r'\results\cells\longctx_paraB_cells.json')
    AB = {'full|None': (0.6831, 0.6575),
          'budget_cluster_link|0.5': (0.7441, 0.7170),
          'alias_budget_link|0.5': (0.7262, 0.7170),
          'budget_cluster|0.5': (0.5402, 0.4420),
          'alias_budget|0.5': (0.5848, 0.4598),
          'alias|None': (0.6618, 0.6385),
          'delete|None': (0.4739, 0.4705)}
    for k, (fa, fb) in AB.items():
        for v, got, exp in (('A', (pa.get(k) or {}).get('f1'), fa),
                            ('B', (pb.get(k) or {}).get('f1'), fb)):
            checked += 1
            r = cmp1('para%s %s' % (v, k), got, exp, 5e-5)
            if r:
                bad.append(r)
    # 判据是臂间差：文档里写的四个判据与两个派生量都必须等于重算值
    DERIVED = [
        ('para raw gain A', pa['budget_cluster_link|0.5']['f1'] - pa['budget_cluster|0.5']['f1'], 0.2039),
        ('para raw gain B', pb['budget_cluster_link|0.5']['f1'] - pb['budget_cluster|0.5']['f1'], 0.2750),
        ('para alias gain A', pa['alias_budget_link|0.5']['f1'] - pa['alias_budget|0.5']['f1'], 0.1414),
        ('para alias gain B', pb['alias_budget_link|0.5']['f1'] - pb['alias_budget|0.5']['f1'], 0.2571),
        ('para link-full A', pa['budget_cluster_link|0.5']['f1'] - pa['full|None']['f1'], 0.0610),
        ('para link-full B', pb['budget_cluster_link|0.5']['f1'] - pb['full|None']['f1'], 0.0594),
        ('para aliaslink-full A', pa['alias_budget_link|0.5']['f1'] - pa['full|None']['f1'], 0.0432),
        ('para aliaslink-full B', pb['alias_budget_link|0.5']['f1'] - pb['full|None']['f1'], 0.0594),
        ('para full drop A', pa['full|None']['f1'] - q['full|None']['f1'], -0.0358),
        ('para full drop B', pb['full|None']['f1'] - q['full|None']['f1'], -0.0614),
    ]
    for lbl, got, exp in DERIVED:
        checked += 1
        r = cmp1(lbl, got, exp, 5e-4)
        if r:
            bad.append(r)
    # 报告里的 bootstrap 行：文档引用的 Δ/CI 必须与报告原文逐位一致
    EXP_BOOT = {
        'A': [('锚点增益（原始串）', 112, -0.0456, -0.1170, 0.0258),
              ('锚点增益（别名）', 112, -0.0902, -0.1696, -0.0188),
              ('link 相对不压缩', 112, 0.0617, -0.0107, 0.1344),
              ('alias link 相对不压缩', 112, 0.0170, -0.0611, 0.0934)],
        'B': [('锚点增益（原始串）', 112, 0.0255, -0.0191, 0.0778),
              ('锚点增益（别名）', 112, 0.0255, -0.0115, 0.0676),
              ('link 相对不压缩', 112, 0.0601, -0.0054, 0.1280),
              ('alias link 相对不压缩', 112, 0.0333, -0.0283, 0.1007)],
    }
    BOOT_PAT = re.compile(r'(锚点增益（原始串）|锚点增益（别名）|link 相对不压缩|alias link 相对不压缩)'
                          r'\s+n=\s*(\d+)\s+Δ差=([+-][\d.]+)\s+\[([+-][\d.]+), ([+-][\d.]+)\]')
    for v in ('A', 'B'):
        got = {}
        for line in io.open(REP + r'\longctx_para%s_robustness.txt' % v, encoding='utf-8'):
            m = BOOT_PAT.search(line)
            if m:
                got[m.group(1)] = (int(m.group(2)), float(m.group(3)),
                                   float(m.group(4)), float(m.group(5)))
        for name, n, d, lo, hi in EXP_BOOT[v]:
            checked += 1
            g = got.get(name)
            if g is None:
                bad.append('boot %s %s: MISSING' % (v, name))
            elif g != (n, d, lo, hi):
                bad.append('boot %s %s: got %s expected %s  <-- MISMATCH'
                           % (v, name, g, (n, d, lo, hi)))

    # docs/17 §2.1 预算曲线（@0.10/@0.15 来自作业 10602732，其余来自 aliasq 10602439）
    xb = cells(R + r'\results\cells\longctx_xbud_cells.json')
    qa = cells(FIX + r'\aliasq_qwen\cells.json')
    CURVE = {'budget_cluster_link': (0.5161, 0.5563, 0.6545, 0.7182, 0.7191),
             'alias_budget_link': (0.5339, 0.5830, 0.6813, 0.7450, 0.7191),
             'budget_cluster': (0.3750, 0.3125, 0.4107, 0.4688, 0.4955),
             'alias_budget': (0.3929, 0.3393, 0.4643, 0.5134, 0.6384),
             'budget_random': (0.0446, 0.0625, 0.1250, 0.2366, 0.4602),
             'delete_budget': (0.3750, 0.3125, 0.4018, 0.4688, 0.4598)}
    for arm, vals in CURVE.items():
        for ratio, exp in zip(('0.1', '0.15', '0.25', '0.5', '0.75'), vals):
            k = '%s|%s' % (arm, ratio)
            src = xb if ratio in ('0.1', '0.15') else qa
            checked += 1
            r = cmp1('curve %s' % k, (src.get(k) or {}).get('f1'), exp, 5e-5)
            if r:
                bad.append(r)
    for ratio, exp_tok in (('0.1', 391.0), ('0.15', 681.0), ('0.25', 1633.0),
                           ('0.5', 3708.0), ('0.75', 5579.0)):
        k = 'alias_budget_link|%s' % ratio
        src = xb if ratio in ('0.1', '0.15') else qa
        checked += 1
        r = cmp1('curve tok %s' % k, (src.get(k) or {}).get('tok'), exp_tok, 0.6)
        if r:
            bad.append(r)
    # 极端预算下我们的臂不得超预算；长度保持臂必须与 aliasq 逐位相同
    for ratio in ('0.1', '0.15'):
        for arm in ('budget_cluster_link', 'alias_budget_link', 'budget_cluster',
                    'alias_budget', 'delete_budget'):
            checked += 1
            ov = ((xb.get('%s|%s' % (arm, ratio)) or {}).get('ov'))
            if ov is None:
                bad.append('xbud %s|%s: MISSING' % (arm, ratio))
            elif abs(ov) > 0.5:
                bad.append('xbud %s|%s overshoot=%s (docs 写 0)' % (arm, ratio, ov))
    for k in ('full|None', 'alias|None', 'delete|None'):
        checked += 1
        r = cmp1('xbud invariance ' + k, (xb.get(k) or {}).get('f1'),
                 (qa.get(k) or {}).get('f1'), 1e-9)
        if r:
            bad.append(r)

    # docs/17 §2 第二模型段（Mistral + --alias-question，作业 10602656）
    mq = cells(R + r'\results\cells\longctx_mistralq_cells.json')
    MQ = {'alias_budget_link|0.5': (0.7343, 4150.0, 0.0000, 1.000, 1.000),
          'full|None': (0.6931, 8897.0, None, 1.000, 1.000),
          'budget_cluster_link|0.5': (0.7075, 4145.0, 0.0603, 1.000, 1.000),
          'budget_cluster|0.5': (0.5196, 4142.0, None, 0.643, 1.000),
          'budget_random|0.5': (0.2631, 4096.0, None, 0.821, 0.464),
          'alias_budget_link|0.25': (0.7172, 1820.0, 0.0000, 0.982, 0.911),
          'alias_budget_link|0.75': (0.7120, 6236.0, 0.0000, 1.000, 1.000)}
    for k, (f1, tok, leak, tent, gic) in MQ.items():
        v = mq.get(k) or {}
        for lbl, got, exp, tol in (('F1', v.get('f1'), f1, 5e-5), ('tok', v.get('tok'), tok, 0.6),
                                   ('tent', v.get('tent'), tent, 1e-3), ('gic', v.get('gic'), gic, 1e-3),
                                   ('ctxLeak', v.get('leak'), leak, 1e-4)):
            if exp is None:
                continue
            checked += 1
            r = cmp1('mistralq %s %s' % (k, lbl), got, exp, tol)
            if r:
                bad.append(r)
    checked += 3
    for lbl, got, exp in (
            ('mistralq alias gain', mq['alias_budget_link|0.5']['f1'] - mq['budget_cluster|0.5']['f1'], 0.2146),
            ('mistralq link gain', mq['budget_cluster_link|0.5']['f1'] - mq['budget_cluster|0.5']['f1'], 0.1879),
            ('mistralq vs full', mq['alias_budget_link|0.5']['f1'] - mq['full|None']['f1'], 0.0412)):
        r = cmp1(lbl, got, exp, 5e-4)
        if r:
            bad.append(r)
    m0 = cells(FIX + r'\aliaslink_mistral\cells.json')
    ndiff = sum(1 for k in set(m0) & set(mq)
                if k.split('|')[0] not in ('alias', 'alias_budget', 'alias_budget_link')
                and abs(m0[k]['f1'] - mq[k]['f1']) > 1e-9)
    checked += 1
    if ndiff != 0:
        bad.append('mistralq 非别名格差异 %d 处（docs 写 0）' % ndiff)

    print('AUDIT :: checked %d numbers against source artifacts' % checked)
    if bad:
        print('MISMATCHES (%d):' % len(bad))
        for b in bad:
            print('  ' + b)
        raise SystemExit('docs/17 与源产物不一致，先查清再引用')
    print('AUDIT PASSED :: every number in docs/17 traces to a source artifact')


if __name__ == '__main__':
    main()
