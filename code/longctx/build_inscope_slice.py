"""构造 in-scope 对照切片：把**真正的多实体任务**压进 <=512 token。

为什么需要它
------------
已发布 batch1 的 `PROTOCOL.md` 要求上下文**完整装入 Provence 的 512-token 窗口**；
其实测 96 例是 186–430 token、**没有一例超过 512**。而 longctx 是 4k–18k token，
超出已验证范围 36–43 倍。所以"已发布方法在多实体任务上失效"必须配一个 in-scope 对照。

第一版为什么被推翻（必读）
--------------------------
第一版直接取"整个目标区域 + 一个干扰区域"，再裁到 512。结果 25 例里**只有 2 例**真的
含 >=2 个实体 —— 因为一个区域约 400–3700 token，目标区域单独就把 512 占满了，
干扰区域被裁成空串。那些切片其实是**单实体 needle 题**，却被元数据标成 `n_entities=2`。
若不查这一步，就会把"已发布方法在 in-scope 上也失效"写成结论，而证据是错的任务类型。
→ 教训：**切片的元数据声明必须由切片内容本身验证**，不能由构造意图决定。

本版构造
--------
1. 目标窗口：以 `entity_offset` 与 `answer_char_offset` 的最小/最大值为中心，
   各留 margin，**裁剪到目标区域范围内** → 必然同时含目标实体与 gold 属性值。
2. 干扰窗口：另取一个在原文中出现的实体值，以其偏移为中心开同样大小的窗口，
   裁到它所属的区域范围内 → 该窗口含另一个实体（通常也含它的属性值，构成竞争候选）。
3. `ctx = 干扰窗口 + 目标窗口`（顺序与原区域顺序一致，目标在后），
   安全裁剪**只作用于干扰段**（BPE 会在拼接边界合并，必须用实测 token 数二分）。
4. 硬自检（不通过就丢）：`<=budget` 且 **>=2 个不同实体值在文本中** 且目标实体在 且 gold 在。
"""
import argparse
import io
import json
import re
import sys

WS = re.compile(r'\s+')


def nt_of(tk):
    def nt(s):
        return len(tk.encode(s, add_special_tokens=False))
    return nt


def fit_prefix(text, sep, tail, nt, budget):
    """取 text 的最长前缀，使 text[:k] + sep + tail <= budget（实测 token 判据）。"""
    if nt(sep + tail) > budget:
        return ''
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if nt(text[:mid] + sep + tail) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--budget', type=int, default=512)
    ap.add_argument('--margin', type=int, default=260, help='目标窗口在实体/答案两侧各留的字符数')
    ap.add_argument('--dist-margin', type=int, default=260)
    ap.add_argument('--report', default='')
    ap.add_argument('--order', default='target_last', choices=['target_last', 'target_first'],
                    help='目标窗口放前还是放后。发布的 prefix_budget 是**砍尾**的，'
                         '所以这个顺序决定了"预算不够时先牺牲谁"，必须两种都跑才能把'
                         '"长度效应"与"裁剪位置效应"分开。')
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    nt = nt_of(tk)

    cases = [json.loads(l) for l in io.open(a.cases, encoding='utf-8') if l.strip()]
    cases = [c for c in cases if c.get('case_kind') == 'multi_entity']

    out, rows = [], []
    for c in cases:
        ctx0 = c['context']
        rs, re_ = c.get('target_region_start'), c.get('target_region_end')
        eo, ao = c.get('entity_offset'), c.get('answer_char_offset')
        te = c.get('target_entity') or ''
        gold = (c.get('answers') or [''])[0]
        if None in (rs, re_, eo, ao) or not te or not gold:
            rows.append(dict(key=c['key'], status='DROP', reason='missing offsets'))
            continue
        # 1) 目标窄窗：覆盖实体与答案，且不越出目标区域
        lo = max(rs, min(eo, ao) - a.margin)
        hi = min(re_, max(eo, ao) + len(gold) + a.margin)
        tgt = ctx0[lo:hi]
        if nt(tgt) > a.budget:
            tgt = tgt[:bisect_chars(tgt, nt, a.budget)]
        if te not in tgt or gold not in tgt:
            rows.append(dict(key=c['key'], status='DROP', reason='target window lost entity/gold'))
            continue
        # 2) 干扰窄窗：另取一个实体值
        others = [e for e in (c.get('entity_values') or []) if e != te and e in ctx0]
        dist = ''
        for ev in sorted(others, key=lambda x: ctx0.find(x)):
            p = ctx0.find(ev)
            d_lo = max(0, p - a.dist_margin)
            d_hi = min(len(ctx0), p + a.dist_margin)
            # 裁到该实体所属的区域，避免把一个区域切成两半
            for (a0, a1, _rid) in (c.get('entry_spans') or []):
                if a0 <= p < a1:
                    d_lo, d_hi = max(a0, d_lo), min(a1, d_hi)
                    break
            cand = ctx0[d_lo:d_hi]
            if a.order == 'target_last':
                # 干扰段在前、是"头"，受约束的是它的最长**前缀**
                dist = fit_prefix(cand, '\n', tgt, nt, a.budget)
            else:
                # 目标段在前、干扰段在后，受约束的是干扰段的长度（仍取其前缀，但头是 tgt）
                dist = fit_tail(cand, tgt, '\n', nt, a.budget)
            if dist.strip():
                break
        if a.order == 'target_last':
            ctx = (dist + '\n' + tgt) if dist.strip() else tgt
        else:
            ctx = (tgt + '\n' + dist) if dist.strip() else tgt
        # 3) 硬自检：必须真的是多实体
        present = [e for e in (c.get('entity_values') or []) if e in ctx]
        if len(present) < 2 or te not in ctx or gold not in ctx or nt(ctx) > a.budget:
            rows.append(dict(key=c['key'], status='DROP',
                             reason='selfcheck failed: n_ent=%d te=%s gold=%s tok=%d'
                                    % (len(present), te in ctx, gold in ctx, nt(ctx))))
            continue
        d = dict(c)
        d['context'] = ctx
        d['key'] = c['key'].replace('longctx-', 'inscope-')
        d['inscope_slice'] = dict(source_key=c['key'], token_budget=a.budget, tokens=nt(ctx),
                                  n_entities_present=len(present),
                                  note='narrow window around entity+gold, plus one distractor window')
        out.append(d)
        rows.append(dict(key=d['key'], status='ok', tokens=nt(ctx), n_entities=len(present)))

    io.open(a.out, 'w', encoding='utf-8', newline='\n').write(
        '\n'.join(json.dumps(x, ensure_ascii=False) for x in out) + '\n')
    ok = [r for r in rows if r['status'] == 'ok']
    toks = sorted(r['tokens'] for r in ok)
    nents = sorted(r['n_entities'] for r in ok)
    print('INSCOPE :: ok=%d dropped=%d  tokens min/med/max=%s/%s/%s  n_entities min/med/max=%s/%s/%s'
          % (len(ok), len(rows) - len(ok), toks[0] if toks else '-',
             toks[len(toks) // 2] if toks else '-', toks[-1] if toks else '-',
             nents[0] if nents else '-', nents[len(nents) // 2] if nents else '-',
             nents[-1] if nents else '-'))
    if len(ok) != len(rows):
        print('INSCOPE :: drop reasons %s'
              % [r['reason'] for r in rows if r['status'] != 'ok'][:4])
    if a.report:
        io.open(a.report, 'w', encoding='utf-8').write(json.dumps(rows, ensure_ascii=False, indent=1))


def fit_tail(text, head, sep, nt, budget):
    """取 text 的最长前缀，使 head + sep + text[:k] <= budget（头固定、尾受约束）。"""
    if nt(head + sep) > budget:
        return ''
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if nt(head + sep + text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def bisect_chars(text, nt, budget):
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if nt(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


if __name__ == '__main__':
    main()
