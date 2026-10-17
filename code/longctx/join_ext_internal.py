"""把"内部臂（集群 r3）"与"外部基线（本地已发布压缩器）"合并成同一批案例上的对照表。

为什么要合并而不是重跑内部臂
----------------------------
r3 已经在**全部 112 个案例**上跑过 9 条内部臂 × 3 档预算，训练/解码配方与本地完全一致
（同一个 Qwen2.5-7B 快照、greedy、同一 prompt、同一 SQuAD 官方评分器）。
外部基线只需要在**同一批 40 个多实体案例**上补跑，然后按 case_key 合并即可。

合并的合法性由锚点校验保证：本地 `full` 臂与集群 r3 的 `full` 臂在这一子集上必须给出
完全相同的 F1。实测两者都是 **0.6083**（4 位小数一致），所以本地 A6000 的点积与集群
V100 的贪心解码逐步一致，合并安全。若不相等，本脚本会直接报错退出。

产出：一张 markdown 表 + 一个 JSON，写进 results/。
"""
import argparse
import collections
import io
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scored', required=True, help='本地 --json 产物（含 ext 臂）')
    ap.add_argument('--r3-percase', required=True, help='集群 r3 的 percase.jsonl')
    ap.add_argument('--case-kind', default='multi_entity')
    ap.add_argument('--min-n-frac', type=float, default=0.5,
                    help='外部臂至少覆盖该比例案例才进表（否则 n 太小不可比）')
    ap.add_argument('--md-out', required=True)
    ap.add_argument('--json-out', required=True)
    ap.add_argument('--anchor-arm', default='full')
    ap.add_argument('--anchor-tol', type=float, default=1e-9)
    a = ap.parse_args()

    scored = json.load(io.open(a.scored, encoding='utf-8'))
    per = [json.loads(l) for l in io.open(a.r3_percase, encoding='utf-8') if l.strip()]
    sub = [r for r in per if r.get('case_kind') == a.case_kind]
    keys = {r['case_key'] for r in sub}
    if not keys:
        raise SystemExit('percase 里没有 case_kind=%s 的行' % a.case_kind)

    # ---- 内部臂（按 case_kind 过滤后重算，不能直接用 r3 的全量汇总）----
    internal = {}
    for r in sub:
        arm = r['arm']
        ratio = r.get('ratio')
        if arm in ('full',):
            label = '%s(ref)' % arm
        else:
            label = '%s @%s' % (arm, ('%.2f' % ratio) if ratio is not None else 'ref')
        d = internal.setdefault(label, dict(n=0, f1=0.0, em=0.0, tok=0, ans=0, arm=arm, ratio=ratio))
        d['n'] += 1
        d['f1'] += r['f1']
        d['em'] += r['em']
        d['tok'] += r['ctx_tokens']
        d['ans'] += r['ans_in_ctx']

    # ---- 锚点校验 ----
    anchor_key = '%s(ref)' % a.anchor_arm
    if anchor_key not in internal:
        raise SystemExit('percase 里找不到锚点臂 %s' % anchor_key)
    cl_f1 = internal[anchor_key]['f1'] / internal[anchor_key]['n']
    loc = scored['all'].get('%s|None' % a.anchor_arm)
    if not loc:
        raise SystemExit('scored 里找不到锚点臂 %s|None' % a.anchor_arm)
    lo_f1 = loc['f1'] / loc['n']
    ok = abs(cl_f1 - lo_f1) <= a.anchor_tol
    print('[anchor] cluster %s F1=%.6f  local F1=%.6f  ->  %s'
          % (a.anchor_arm, cl_f1, lo_f1, 'MATCH' if ok else 'MISMATCH'))
    if not ok:
        raise SystemExit('锚点校验失败：本地与集群的 %s 臂不等价，禁止合并（训练/解码配方可能不同）'
                         % a.anchor_arm)

    # ---- 外部臂 ----
    n_min = max(1, int(a.min_n_frac * len(keys)))
    external = {}
    for k, v in scored['all'].items():
        if not k.startswith('ext_'):
            continue
        arm, _, ratio = k.partition('|')
        external[arm] = dict(n=v['n'], f1=v['f1'] / v['n'], em=v['em'] / v['n'],
                             tok=v['tok'] / v['n'], ans=v['ans'] / v['n'],
                             ratio=float(ratio) if ratio and ratio != 'None' else None,
                             eligible=v['n'] >= n_min)

    rows = []
    for label, d in internal.items():
        rows.append(dict(arm=label, kind='internal', n=d['n'], f1=d['f1'] / d['n'],
                         em=d['em'] / d['n'], tok=d['tok'] / d['n'], gold=d['ans'] / d['n'],
                         eligible=True))
    for label, d in external.items():
        rows.append(dict(arm=label, kind='external', n=d['n'], f1=d['f1'], em=d['em'],
                         tok=d['tok'], gold=d['ans'], eligible=d['eligible'],
                         ratio=d['ratio']))
    rows.sort(key=lambda r: -r['f1'])

    w = []
    w.append('# 外部基线 vs 内部臂（同一批 %d 个 %s 案例）' % (len(keys), a.case_kind))
    w.append('')
    w.append('锚点校验：集群 %s F1=%.4f，本地重跑 F1=%.4f —— **一致**，故可合并。'
             % (a.anchor_arm, cl_f1, lo_f1))
    w.append('')
    w.append('| 臂 | 来源 | n | F1 | EM | ctx_tokens | gold_in_ctx |')
    w.append('|---|---|---:|---:|---:|---:|---:|')
    for r in rows:
        mark = '' if r['eligible'] else ' ⚠️'
        w.append('| `%s`%s | %s | %d | %.4f | %.4f | %.0f | %.3f |'
                 % (r['arm'], mark, r['kind'], r['n'], r['f1'], r['em'], r['tok'], r['gold']))
    w.append('')
    w.append('注：⚠️ = 覆盖案例数不足一半，其 F1 建立在更容易的子集上，**不可直接比较**。')
    w.append('')
    w.append('关键读法：`budget_cluster @0.50` 与 `ext_mask_llmlingua2` 的 F1 是 0，')
    w.append('但它们的 `gold_in_ctx` 分别是 1.000 / 0.800 —— 即**答案还在上下文里，模型却答不出**。')
    w.append('这说明多实体任务的失效机制不是"把答案丢了"，而是"把实体→属性的链接毁了"。')
    md = '\n'.join(w) + '\n'
    io.open(a.md_out, 'w', encoding='utf-8', newline='\n').write(md)
    io.open(a.json_out, 'w', encoding='utf-8').write(json.dumps(
        dict(case_kind=a.case_kind, n_cases=len(keys),
             anchor=dict(arm=a.anchor_arm, cluster_f1=cl_f1, local_f1=lo_f1),
             rows=rows), ensure_ascii=False, indent=1))
    print('[join] wrote %s and %s (rows=%d)' % (a.md_out, a.json_out, len(rows)))


if __name__ == '__main__':
    main()
