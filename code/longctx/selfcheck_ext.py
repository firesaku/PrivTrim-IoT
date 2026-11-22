"""外部方法产物的自查：预算、脱敏、gold 保留三项，与评分产物交叉印证。

为什么要有这个
--------------
`ext_mask_quito.py` 是新加的包装器（复刻 mask_quito_x 语义）。它声称做了三件事：
  1. 先 redact、再压缩；
  2. 预算仍按**原文** token 数算（`budget = original_tokens // 2`）；
  3. 压缩后不超过预算。
这三条必须在产物上验，而不是靠读代码相信。而且它们与 harness 的诊断列有可交叉印证的关系：

    gold 仍在的比例  == 评分产物的 `gold_in_ctx` (gic)
    实体残留为 0     == 评分产物的 `tgt_entity_present` (tent) 为 0

如果这两个等式对不上，说明包装器与 harness 对同一件事的看法不一致 —— 那必须先查清楚，
不能把分数拿去用。实测（40 个多实体案例）：gold 28/40 与 gic 0.700 一致、
实体残留 0 与 tent 0.00 一致。
"""
import argparse
import io
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--extdir', required=True, help='含 <method>/ 子目录')
    ap.add_argument('--method', required=True)
    ap.add_argument('--cells', default='', help='可选：同一次运行的评分产物，用于交叉印证')
    a = ap.parse_args()

    cases = {c['key']: c for c in
             (json.loads(l) for l in io.open(a.cases, encoding='utf-8') if l.strip())}
    d = os.path.join(a.extdir, a.method)
    if not os.path.isdir(d):
        raise SystemExit('目录不存在: %s' % d)

    ok = err = over = ent_gone = ent_left = gold_kept = gold_lost = 0
    for f in sorted(os.listdir(d)):
        if not f.endswith('.json'):
            continue
        r = json.load(io.open(os.path.join(d, f), encoding='utf-8'))
        if r.get('status') != 'ok':
            err += 1
            continue
        ok += 1
        c = cases.get(r.get('id'))
        t = (r.get('matched') or {}).get('text') or ''
        tgt = r.get('target_tokens') or 0
        got = (r.get('matched') or {}).get('tokens') or 0
        if got > tgt:
            over += 1
        te = (c or {}).get('target_entity') or ''
        g = ((c or {}).get('answers') or [''])[0]
        if te:
            if te in t:
                ent_left += 1
            else:
                ent_gone += 1
        if g:
            if g in t:
                gold_kept += 1
            else:
                gold_lost += 1

    print('SELFCHECK[%s] ok=%d err=%d' % (a.method, ok, err))
    print('  budget: matched>target = %d  (must be 0)' % over)
    print('  entity: redacted=%d residual=%d  (residual must be 0 for a masking method)'
          % (ent_gone, ent_left))
    print('  gold  : kept=%d lost=%d  (%.3f kept)'
          % (gold_kept, gold_lost, gold_kept / max(1, gold_kept + gold_lost)))

    if a.cells:
        o = json.load(io.open(a.cells, encoding='utf-8'))
        arm = 'ext_%s|0.5' % a.method
        v = o.get('all', {}).get(arm)
        if v:
            gic = v['ans'] / max(1, v['n'])
            tent = v['tent'] / max(1, v['n'])
            print('  cross-check vs scoring artifact (%s): gic=%.3f tent=%.3f' % (arm, gic, tent))
            gold_ratio = gold_kept / max(1, gold_kept + gold_lost)
            ok1 = abs(gold_ratio - gic) < 1e-6
            ok2 = (ent_left == 0 and tent == 0.0) or (ent_left > 0 and tent > 0)
            print('    gold kept ratio == gic ?  %s (%.3f vs %.3f)'
                  % ('YES' if ok1 else '*** NO ***', gold_ratio, gic))
            print('    entity residual vs tent ?  %s (residual=%d, tent=%.3f)'
                  % ('CONSISTENT' if ok2 else '*** INCONSISTENT ***', ent_left, tent))
            if not (ok1 and ok2):
                raise SystemExit('包装器与 harness 诊断不一致，先查清楚再用分数')
    print('SELFCHECK PASSED')


if __name__ == '__main__':
    main()
