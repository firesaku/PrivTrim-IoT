"""位置覆盖率：外部压缩器到底"看见"了长上下文的哪些位置？

要解决的问题
------------
quito_x (T5) / provence (DeBERTa) / llmlingua2 (xlm-R) 的底层模型窗口都是 512。
把一个 18,282 token 的上下文喂进去，如果它只按前 512 token 做决策、其余原样返回，
那么预算对账表里 `matched.tokens == target` 依然成立，看起来一切正常 ——
于是"基线在多实体案例上失效"就可能只是"它根本没看见后面 97% 的上下文"。

这两种解释天差地别：
  - 若它全局剪枝仍失效  → 领域级发现（真正想证明的）
  - 若它只看了前 512     → 能力边界，不能算"被公平地打败"

方法（model-free，不改任何基线代码）
------------------------------------
1. 原上下文按字符位置切成 10 等份（decile 0..9），并为原文建**词三元组 -> decile** 索引。
2. 压缩后文本切成句子（与 harness 的 split_units 同一正则）。
3. 每个输出句子的词三元组逐个投票：命中原文某个 decile 就投一票；
   取票数最高的 decile 作为该句的归属。
   —— 为什么用三元组投票而不是连续子串锚定：
      llmlingua2 / dac / quito_x 是**按 token 丢**的，输出不再是原文的连续子串，
      连续锚定会 100% 匹配失败（实测过：llmlingua2 unmatched=100%）。
      三元组只要求"三个词连续出现过"，丢一个词只破坏局部三元组，其余仍能定位。
4. 按句子词数加权，统计输出在各 decile 上的分布。
5. 同时算 gold 答案在原文的哪个 decile —— 若 gold 在 decile 8 而某方法输出 100%
   落在 decile 0，那它**结构上不可能答对**，不是"策略不好"。

判读
----
  deciles_covered = 输出覆盖到的 decile 个数（1..10）
  frac_d0         = 落在 decile 0 的输出占比
  at_gold         = 落在 gold 所在 decile 的输出占比（0 = 结构上不可能答对）
  unmatched       = 回溯不到位置的句子占比（越高越要人工看）

  - 全局剪枝：deciles_covered 高、frac_d0 明显 < 1
  - 只看前窗：deciles_covered == 1、frac_d0 ≈ 1

自检（--self-check）
--------------------
用恒等映射与"随机丢 30% 词"的映射跑同一套度量：
  恒等  -> deciles_covered 必须 = 10、unmatched 必须 = 0
  丢词  -> deciles_covered 仍应 = 10、unmatched 必须低
任何一条不满足，说明度量本身有问题，**不许**拿它去判基线。
"""
import argparse
import collections
import io
import json
import os
import random
import re
import sys

SPLIT = re.compile(r'(?<=[.!?])\s+|\n+')
WS = re.compile(r'\s+')
WORD = re.compile(r"[0-9a-zA-Z_@.\-]+")


def norm(s):
    return WS.sub(' ', s).strip().lower()


def sentences(text):
    return [p.strip() for p in SPLIT.split(text) if len(p.strip()) >= 2]


def words_of(s):
    return WORD.findall(norm(s))


def decile_of(pos, bounds):
    for d in range(len(bounds) - 1):
        if bounds[d] <= pos < bounds[d + 1]:
            return d
    return len(bounds) - 2


def decile_bounds(n, k=10):
    return [(i * n) // k for i in range(k + 1)]


def build_trigram_index(text, k=10):
    """word-trigram -> {decile: 出现次数}；同时返回全文词数与 decile 边界（按字符位置）。"""
    bounds = decile_bounds(len(text), k)
    idx = collections.defaultdict(collections.Counter)
    # 逐词扫描并记录该词在原文中的字符位置，据此判定它属于哪个 decile
    for m in WORD.finditer(text):
        pass
    toks = [(m.group(0), m.start()) for m in WORD.finditer(text)]
    ws = [t[0].lower() for t in toks]
    for i in range(len(ws) - 2):
        d = decile_of(toks[i][1], bounds)
        idx[(ws[i], ws[i + 1], ws[i + 2])][d] += 1
    return idx, bounds, len(ws)


def coverage_one(text, out_text, k=10):
    idx, bounds, nwords = build_trigram_index(text, k)
    per = collections.Counter()
    unmatched = 0
    total = 0
    for s in sentences(out_text):
        ws = words_of(s)
        w = max(1, len(ws))
        total += w
        if len(ws) < 3:
            unmatched += w
            continue
        votes = collections.Counter()
        for i in range(len(ws) - 2):
            c = idx.get((ws[i], ws[i + 1], ws[i + 2]))
            if c:
                for d, n in c.items():
                    votes[d] += n
        if not votes:
            unmatched += w
            continue
        per[votes.most_common(1)[0][0]] += w
    frac = [per[d] / total if total else 0.0 for d in range(k)]
    return dict(frac=frac, unmatched_frac=(unmatched / total if total else 0.0),
                deciles_covered=sum(1 for x in frac if x > 0.01),
                frac_top_decile=frac[0], total_weight=total)


def gold_deciles(case):
    text = case['context']
    bounds = decile_bounds(len(text))
    low = text.lower()
    out = set()
    for g in (case.get('answers') or []):
        g2 = norm(g)
        if not g2:
            continue
        i = low.find(g2)
        if i < 0:
            continue
        out.add(decile_of(i, bounds))
    return sorted(out)


def self_check(cases, k=10):
    """度量自检：恒等映射必须 10/10；随机丢 30% 词仍须 >=8。不满足就炸。"""
    print('=== SELF-CHECK（度量本身是否可信）===')
    ok = True
    rng = random.Random(20261017)
    for name, mutate in (('identity', lambda t: t),
                         ('drop30%_words', lambda t: ' '.join(
                             w for w in t.split() if rng.random() > 0.30))):
        covs = []
        for c in cases[:3]:
            covs.append(coverage_one(c['context'], mutate(c['context']), k))
        cov = sum(x['deciles_covered'] for x in covs) / len(covs)
        un = sum(x['unmatched_frac'] for x in covs) / len(covs)
        top = sum(x['frac_top_decile'] for x in covs) / len(covs)
        print('  %-14s deciles=%.2f  unmatched=%.1f%%  frac_d0=%.1f%%'
              % (name, cov, 100 * un, 100 * top))
        if name == 'identity' and (cov < 9.5 or un > 0.02):
            ok = False
            print('  *** FAIL: 恒等映射必须覆盖 10 个 decile 且 unmatched≈0')
        if name.startswith('drop') and cov < 7.5:
            ok = False
            print('  *** FAIL: 丢词后仍应覆盖 >=8 个 decile')
    if not ok:
        raise SystemExit('度量自检未通过，不允许用于判断基线')
    print('  SELF-CHECK PASSED')
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--extdir', default='')
    ap.add_argument('--methods', default='')
    ap.add_argument('--json-out', default='')
    ap.add_argument('--self-check', action='store_true')
    a = ap.parse_args()

    cases = [json.loads(l) for l in io.open(a.cases, encoding='utf-8') if l.strip()]
    by_key = {c['key']: c for c in cases}

    if a.self_check:
        self_check(cases)
        if not a.extdir:
            return
    if not a.extdir:
        raise SystemExit('需要 --extdir')

    methods = [m for m in a.methods.split(',') if m] or sorted(
        d for d in os.listdir(a.extdir) if os.path.isdir(os.path.join(a.extdir, d)))
    rows = []
    for m in methods:
        md = os.path.join(a.extdir, m)
        if not os.path.isdir(md):
            continue
        for fn in sorted(os.listdir(md)):
            if not fn.endswith('.json'):
                continue
            r = json.loads(io.open(os.path.join(md, fn), encoding='utf-8').read())
            if r.get('status') != 'ok':
                continue
            txt = ((r.get('matched') or {}).get('text') or '')
            case = by_key.get(r.get('id'))
            if not txt or case is None:
                continue
            cov = coverage_one(case['context'], txt)
            rows.append(dict(key=r.get('id'), method=m, case_kind=case.get('case_kind'),
                             orig_tokens=r.get('original_tokens'),
                             matched_tokens=(r.get('matched') or {}).get('tokens'),
                             target_tokens=r.get('target_tokens'),
                             gold_deciles=gold_deciles(case), **cov))

    agg = collections.defaultdict(lambda: dict(n=0, covered=0.0, top=0.0, un=0.0,
                                               gold_hit=0.0, ns=[], first_only=0))
    for r in rows:
        s = agg[r['method']]
        s['n'] += 1
        s['covered'] += r['deciles_covered']
        s['top'] += r['frac_top_decile']
        s['un'] += r['unmatched_frac']
        s['ns'].append(r['deciles_covered'])
        if r['deciles_covered'] <= 1:
            s['first_only'] += 1
        if r['gold_deciles']:
            s['gold_hit'] += sum(r['frac'][d] for d in r['gold_deciles'])

    w = ['位置覆盖率（context 按字符位置切 10 等份；输出按词数加权）',
         '%-17s %4s %11s %10s %10s %10s %10s %s'
         % ('method', 'n', 'deciles', 'frac_d0', 'unmatched', 'at_gold', 'only1', 'min/max')]
    for m in sorted(agg):
        s = agg[m]
        n = max(1, s['n'])
        w.append('%-17s %4d %11.2f %9.1f%% %9.1f%% %9.1f%% %10d %d/%d'
                 % (m, s['n'], s['covered'] / n, 100 * s['top'] / n, 100 * s['un'] / n,
                    100 * s['gold_hit'] / n, s['first_only'], min(s['ns']), max(s['ns'])))
    w.append('')
    w.append('判读：deciles≈10 = 全局剪枝；deciles==1 且 frac_d0≈100% = 只看了开头一段。')
    w.append('     at_gold = 输出落在 gold 所在 decile 的比例（0 = 结构上不可能答对）。')
    w.append('     only1 = 该方法的案例里"只覆盖 1 个 decile"的个数。')
    print('\n'.join(w))
    if a.json_out:
        io.open(a.json_out, 'w', encoding='utf-8').write(
            json.dumps(dict(rows=rows,
                            summary={k: {kk: vv for kk, vv in v.items() if kk != 'ns'}
                                     for k, v in agg.items()}),
                       ensure_ascii=False, indent=1))
        print('\n[coverage] wrote %s (rows=%d)' % (a.json_out, len(rows)))


if __name__ == '__main__':
    main()
