"""补跑 mask_quito_x：先 redact，再用 quito_x 压缩（沿用已发布代码的预算约定）。

为什么单独写
------------
`mask_quito_x` **不在** batch1 的 `METHODS` 里 —— 它来自更早的
`research/token_compression_explore_v1`（`results.sqlite` 里的 `mask_quito_x` /
`mask_quito_x_paper_sum` 两条 track 就是那次跑的）。语义与 batch1 里的
`mask_llmlingua2` 同构：

    compress.py:111   text = redact(c['context']) if method == 'mask_llmlingua2' else c['context']

即"先脱敏、再压缩"，而预算 `budget = c['original_tokens'] // 2` 仍按**原文**长度算。

复刻方式（不重实现任何算法）
----------------------------
把案例的 `context` 换成 `redact(原文)`、**保留 `original_tokens` 为原文的计数**，
然后调用已发布的 `compress_case(c, 'quito_x', ...)`。这样：
  - 压缩器用的是已发布的 quito_x（原码）；
  - 脱敏用的是已发布的 `redact`（原码）；
  - 预算约定与 mask_llmlingua2 逐字一致。
产物与 `ext_compress.py` 同格式，所以下游 `ext_to_arms.py` 直接可用。

为什么不改 ext_compress.py：那个脚本已经过对抗式复查，不想为了一个方法再动它。
"""
import argparse
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

B1_DEFAULT = r'D:\Shijie Zeng\CodeX\cybersec_IoT\research\token_compression_batch1_v1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch1', default=B1_DEFAULT)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--outdir', required=True, help='沿用 ext_compress.py 的目录约定')
    ap.add_argument('--keys', default='')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true', help='只建文本与自检，不压缩')
    a = ap.parse_args()

    b1 = Path(a.batch1).resolve()
    sys.path.insert(0, str(b1))
    import common as C          # noqa: E402
    import compress as K        # noqa: E402

    cases = [json.loads(l) for l in io.open(a.cases, encoding='utf-8') if l.strip()]
    if a.keys:
        want = set(a.keys.split(','))
        cases = [c for c in cases if c['key'] in want or c.get('id') in want]
    if a.limit:
        cases = cases[:a.limit]
    if not cases:
        raise SystemExit('没有选中任何案例')

    tok = C.get_tokenizer()
    adapted = []
    for c in cases:
        raw = c['context']
        red = C.redact(raw)
        if red == raw:
            print('[warn] %s: redact 未改变文本（无 canary 命中？）' % c.get('key'))
        adapted.append(dict(id=c.get('key') or c.get('id'),
                            question=c['question'],
                            context=red,                      # 压缩的是脱敏后文本
                            original_tokens=C.count(tok, raw),  # 预算按**原文**计数
                            case_kind=c.get('case_kind'),
                            dataset=c.get('dataset')))

    if a.dry_run:
        n_red = sum(1 for x, c in zip(adapted, cases) if x['context'] != c['context'])
        print('DRYRUN :: cases=%d redacted_differs=%d  '
              '原始 token p50=%d  脱敏后 token p50=%d'
              % (len(adapted), n_red,
                 sorted(x['original_tokens'] for x in adapted)[len(adapted) // 2],
                 sorted(C.count(tok, x['context']) for x in adapted)[len(adapted) // 2]))
        return

    out = Path(a.outdir) / 'mask_quito_x'
    out.mkdir(parents=True, exist_ok=True)
    run, cache, stats, info = K.load_method('quito_x', tok)
    print('[mask_quito_x] loaded: %s' % json.dumps(info, ensure_ascii=False)[:200], flush=True)
    done = err = skip = 0
    t0 = time.time()
    for x in adapted:
        fp = out / (x['id'] + '.json')
        if fp.exists():
            skip += 1
            continue
        try:
            r = K.compress_case(x, 'quito_x', run, cache, stats, tok)
        except Exception:
            r = dict(id=x['id'], method='mask_quito_x', status='error',
                     error=traceback.format_exc(), original_tokens=x['original_tokens'])
            err += 1
        r['method'] = 'mask_quito_x'          # 标注真实方法名
        r['masked'] = True
        fp.write_text(json.dumps(r, ensure_ascii=False), encoding='utf-8')
        done += 1
        print('[mask_quito_x] %d/%d status=%s matched=%s target=%s %.1fs'
              % (done + skip, len(adapted), r.get('status'),
                 (r.get('matched') or {}).get('tokens'), x['original_tokens'] // 2,
                 time.time() - t0), flush=True)
    print('[mask_quito_x] DONE done=%d skip=%d err=%d %.1fs' % (done, skip, err, time.time() - t0))


if __name__ == '__main__':
    main()
