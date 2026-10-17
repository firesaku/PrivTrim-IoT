"""机械改写问题表面形式，用于"提问方式稳健性"检查。

为什么做这个
------------
我们的问题是**模板化**的（真实日志记录 + 模板查询）。审稿人会问：结论是不是只在
这一种问法下成立？

本脚本用**纯规则**（不调用任何模型，所以不存在"用模型生成评测题"的污染问题）
把问题改写成另外两种自然问法，实体值与属性名**逐字保留**，因此 gold 答案不变。

三种问法的对应
--------------
原版:   "In the log above, which persona id is recorded for the account ACCT-X?"
变体 A: "For the account ACCT-X in the log above, what is the recorded persona id?"
变体 B: "Which persona id is associated with the account ACCT-X in the log above?"

原版:   "In the log above, what is the query type recorded for entry 0?"
变体 A: "For entry 0 in the log above, what is the recorded query type?"
变体 B: "What query type is associated with entry 0 in the log above?"

自检：改写前后必须满足
  1) 目标实体串仍在问题里；
  2) gold 答案串**不在**问题里（否则等于泄题）；
  3) 属性名仍在问题里；
  4) 改写后的问题与原问题**不同**（确实改了）。
不满足的案例会报出来，不静默通过。
"""
import argparse
import io
import json
import re
import sys

# 原版两种句式
RE_MULTI = re.compile(r'^In the log above, which (.+?) is recorded for the account (.+?)\?$')
RE_NEEDLE = re.compile(r'^In the log above, what is the (.+?) recorded for (.+?)\?$')


def rephrase(q, variant):
    m = RE_MULTI.match(q)
    if m:
        attr, ent = m.group(1), m.group(2)
        if variant == 'A':
            return 'For the account %s in the log above, what is the recorded %s?' % (ent, attr)
        return 'Which %s is associated with the account %s in the log above?' % (attr, ent)
    m = RE_NEEDLE.match(q)
    if m:
        attr, anchor = m.group(1), m.group(2)
        if variant == 'A':
            return 'For %s in the log above, what is the recorded %s?' % (anchor, attr)
        return 'What %s is associated with %s in the log above?' % (attr, anchor)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--variant', default='B', choices=['A', 'B'])
    ap.add_argument('--report', default='')
    a = ap.parse_args()

    cases = [json.loads(l) for l in io.open(a.cases, encoding='utf-8') if l.strip()]
    out, bad, skipped, preexisting = [], [], 0, []
    for c in cases:
        q = c.get('question') or ''
        q2 = rephrase(q, a.VARIANT if False else a.variant)
        if q2 is None:
            skipped += 1
            out.append(c)          # 无法匹配的句式保持原样，不算改写
            continue
        te = c.get('target_entity') or ''
        gold = (c.get('answers') or [''])[0]
        # 既有缺陷：**原问题**里就含 gold —— 与改写无关，单独报告，不拦截
        if gold and gold in q:
            preexisting.append((c['key'], c.get('anchor_kind'), c.get('attribute_key'), gold))
        # 改写引入的问题：必须为空
        if q2 == q:
            bad.append((c['key'], 'unchanged'))
        if te and te not in q2:
            bad.append((c['key'], 'target_entity lost'))
        if gold and gold in q2 and gold not in q:
            bad.append((c['key'], 'GOLD LEAKED BY REPHRASING'))
        d = dict(c)
        d['question'] = q2
        d['question_original'] = q
        d['question_variant'] = a.variant
        out.append(d)

    io.open(a.out, 'w', encoding='utf-8', newline='\n').write(
        '\n'.join(json.dumps(x, ensure_ascii=False) for x in out) + '\n')
    print('REPHRASE :: variant=%s cases=%d rephrased=%d skipped=%d rephrase_problems=%d '
          'preexisting_defects=%d'
          % (a.variant, len(cases), len(cases) - skipped, skipped, len(bad),
             len(preexisting)))
    if preexisting:
        print('  PRE-EXISTING benchmark defects (gold already in the ORIGINAL question; '
              'not caused by rephrasing, flagged for the record):')
        for p in preexisting:
            print('    %s  anchor_kind=%s  attribute_key=%s  gold=%s' % p)
    if bad:
        print('  rephrase-induced problems (must be empty): %s' % bad[:6])
        raise SystemExit('改写自检未通过，不允许用于实验')
    # 抽一条对照给人看
    for c in out:
        if c.get('question_original'):
            print('  原: %s' % c['question_original'])
            print('  改: %s' % c['question'])
            print('  gold=%s  target_entity=%s' % ((c.get('answers') or [''])[0],
                                                   c.get('target_entity')))
            break
    if a.report:
        io.open(a.report, 'w', encoding='utf-8').write(
            json.dumps([dict(key=c['key'], orig=c.get('question_original'),
                             new=c['question']) for c in out if c.get('question_original')],
                       ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
