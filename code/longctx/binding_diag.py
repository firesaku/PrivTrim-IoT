"""绑定诊断：目标实体和它的属性还"同处一段"吗？

为什么现有列不够
----------------
`collision` 对不含删除/别名语义的臂（含全部外部基线）**恒为 0**，没有信息量。
`tgt_entity_present` / `tgt_attr_present` 只回答"两个串还在不在"，**不回答它们是否还绑在一起**。

而多实体题的结构是：题目点名一个实体，要在 k 个几乎同形的区域里找**它**的属性。
所以只要"实体串"和"它的属性串"被压缩分散到不同片段，模型即使两个都看得见也无法绑定 ——
实测 `budget_cluster` 正是这种情况：`gold_in_ctx = 1.000` 而 F1 = **0.0000**。

度量
----
对每条臂文本：
  ent_hits   目标实体表面串（含别名形式）出现的位置（字符偏移）
  gold_hits   gold 答案串出现的位置
  min_dist   所有 (实体, 答案) 对的最小字符距离；任一为空则 None
  co_located min_dist <= 该案例的目标区域长度（target_region_end - start）
             —— 在原文里这个条件必然成立（答案就在该区域内），
                所以 co_located 掉下去只可能是压缩造成的。

输出按臂汇总：both_present / co_located / min_dist 中位数，供与 F1 对照。
"""
import argparse
import collections
import io
import json
import os
import re
import statistics as st
import sys

WS = re.compile(r'\s+')


def nz(s):
    return WS.sub(' ', s).strip().lower()


def all_positions(hay, needle):
    out = []
    i = hay.find(needle)
    while i >= 0:
        out.append(i)
        i = hay.find(needle, i + 1)
    return out


def load_cases(path):
    return {c['key']: c for c in (json.loads(l) for l in io.open(path, encoding='utf-8') if l.strip())}


def alias_token_of(case, v4):
    """按 entity_diag 的口径（先按 canary 列表顺序预播）取目标实体的别名 token。"""
    al = v4.Aliaser('alias')
    for _c in (case.get('canaries') or []):
        al.token(_c)
    cmap = {c['value']: c for c in (case.get('canaries') or []) if c['type'] == 'account'}
    te = case.get('target_entity')
    return al.token(cmap[te]) if te in cmap else te


def diag_one(case, text, v4):
    low = nz(text)
    rl = max(1, (case.get('target_region_end') or 0) - (case.get('target_region_start') or 0))
    te = case.get('target_entity') or ''
    forms = {nz(te)} if te else set()
    try:
        forms.add(nz(alias_token_of(case, v4)))
    except Exception:
        pass
    forms.discard('')
    golds = [nz(g) for g in (case.get('answers') or []) if nz(g)]

    ent_hits = sorted({p for f in forms for p in all_positions(low, f)})
    gold_hits = sorted({p for g in golds for p in all_positions(low, g)})
    if not ent_hits or not gold_hits:
        return dict(ent_present=int(bool(ent_hits)), gold_present=int(bool(gold_hits)),
                    both_present=0, min_dist=None, co_located=0, region_len=rl)
    d = min(abs(e - g) for e in ent_hits for g in gold_hits)
    return dict(ent_present=1, gold_present=1, both_present=1, min_dist=d,
                co_located=int(d <= rl), region_len=rl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--internal', default='', help='--dump-arms 产出的 JSONL')
    ap.add_argument('--extdir', default='', help='ext_compress.py 的输出目录')
    ap.add_argument('--v4-dir', required=True, help='含 v4_router.py 的目录')
    ap.add_argument('--json-out', required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.v4_dir)
    import v4_router as v4  # noqa: E402

    cases = load_cases(a.cases)
    recs = []

    if a.internal:
        for line in io.open(a.internal, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            c = cases.get(o['key'])
            if c is None:
                continue
            lab = o['arm'] + ('' if o.get('ratio') is None else '@%.2f' % o['ratio'])
            recs.append(dict(key=o['key'], arm=lab, kind='internal',
                             ratio=o.get('ratio'), **diag_one(c, o['text'], v4)))

    if a.extdir:
        for m in sorted(os.listdir(a.extdir)):
            p = os.path.join(a.extdir, m)
            if not os.path.isdir(p):
                continue
            for fn in sorted(os.listdir(p)):
                if not fn.endswith('.json'):
                    continue
                r = json.loads(io.open(os.path.join(p, fn), encoding='utf-8').read())
                if r.get('status') != 'ok':
                    continue
                txt = ((r.get('matched') or {}).get('text') or '')
                c = cases.get(r.get('id'))
                if not txt or c is None:
                    continue
                recs.append(dict(key=r['id'], arm='ext_' + m, kind='external',
                                 ratio=0.5, **diag_one(c, txt, v4)))

    # 原文参照：co_located 必须在 100%
    for k, c in cases.items():
        recs.append(dict(key=k, arm='ORIGINAL', kind='reference', ratio=None,
                         **diag_one(c, c['context'], v4)))

    agg = collections.defaultdict(lambda: dict(n=0, both=0, co=0, dists=[]))
    for r in recs:
        s = agg[r['arm']]
        s['n'] += 1
        s['both'] += r['both_present']
        s['co'] += r['co_located']
        if r['min_dist'] is not None:
            s['dists'].append(r['min_dist'])

    w = []
    w.append('绑定诊断（%d 个案例）：目标实体与它的属性是否还同处一段' % len(cases))
    w.append('%-26s %4s %12s %12s %14s' % ('arm', 'n', 'both_present', 'co_located', 'min_dist_p50'))
    for arm in sorted(agg, key=lambda x: -agg[x]['co'] / max(1, agg[x]['n'])):
        s = agg[arm]
        n = max(1, s['n'])
        w.append('%-26s %4d %11.1f%% %11.1f%% %14s'
                 % (arm, s['n'], 100 * s['both'] / n, 100 * s['co'] / n,
                    ('%.0f' % st.median(s['dists'])) if s['dists'] else 'n/a'))
    txt = '\n'.join(w)
    print(txt)
    io.open(a.json_out, 'w', encoding='utf-8').write(json.dumps(
        dict(rows=recs, summary={k: {kk: vv for kk, vv in v.items() if kk != 'dists'}
                                 for k, v in agg.items()}), ensure_ascii=False, indent=1))
    print('\n[binding] wrote %s (rows=%d)' % (a.json_out, len(recs)))


if __name__ == '__main__':
    main()
