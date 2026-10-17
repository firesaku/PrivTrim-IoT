"""把 ext_compress.py 的逐例产物转换成 e2e_longctx_r4.py --ext-arms 需要的 JSONL。

按对抗式复查的要求，转换端必须做到：
  C2/C3  失败也出行（status=error 的案例不静默消失），并对每个 (case, method)
         打上来源标记，让"覆盖不全"在产物里可见而不是只在 console。
  C4     每条记录带 trim / original_tokens / target_tokens / code_sha256 / warnings，
         使"我们的预算适配器砍了多少"进得了对比表。
  M2     把 matplotlib 之外的 warning（尤其 provence 的 split_context）记下来。
  m5     键名用 key（同时接受 id，避免拼错就崩）。

两条臂：
  ext_<m>        matched.text —— 被 prefix_budget 裁到 <= budget（等预算，fill=100%）
  extn_<m>       native.text  —— 算法在预算内自己选出的最大输出（不裁剪，fill 可能 <100%）
两条都出，是因为单看任一条都会误导：
  - 只看 matched：等于让所有方法都被我们裁到恰好 fill=100%，而内部臂只有 92~93%，
    基线反而多得约 7% token（对基线有利，需说明）。
  - 只看 native：基线可能严重欠用预算（batch1 里 provence 欠 26%），
    那就变成"用更少的 token 比"，同样要说明。
"""
import argparse
import collections
import io
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extdir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--methods', default='', help='默认取 extdir 下所有子目录')
    ap.add_argument('--code-sha', default='', help='压缩代码 sha256（可留空）')
    a = ap.parse_args()

    methods = [m for m in a.methods.split(',') if m] or sorted(
        d for d in os.listdir(a.extdir)
        if os.path.isdir(os.path.join(a.extdir, d)))
    rows = []
    stat = collections.OrderedDict()
    for m in methods:
        md = os.path.join(a.extdir, m)
        if not os.path.isdir(md):
            stat[m] = dict(n=0, ok=0, err=0, native_ok=0, missing_dir=True)
            continue
        s = dict(n=0, ok=0, err=0, native_ok=0, trim_p50=None, warnings=0)
        trims = []
        for fn in sorted(os.listdir(md)):
            if not fn.endswith('.json'):
                continue
            r = json.loads(io.open(os.path.join(md, fn), encoding='utf-8').read())
            key = r.get('id') or r.get('key')
            if not key:
                continue
            s['n'] += 1
            matched = r.get('matched') or {}
            native = r.get('native') or {}
            prov = dict(original_tokens=r.get('original_tokens'),
                        target_tokens=r.get('target_tokens'),
                        trim=r.get('adapter_trimmed_tokens'),
                        native_tokens=native.get('tokens'),
                        native_within_cap=r.get('native_within_cap'),
                        code_sha=a.code_sha or r.get('code_sha256'),
                        warnings=r.get('warnings') or [])
            if r.get('status') != 'ok' or not matched.get('text'):
                s['err'] += 1
                # 失败也出行：让"这个 (case, method) 没结果"在对比里可见
                rows.append(dict(key=key, method=m, text='', status='error',
                                 error=(r.get('error') or '')[:300], **prov))
                continue
            s['ok'] += 1
            if r.get('adapter_trimmed_tokens'):
                trims.append(r['adapter_trimmed_tokens'])
            if len(prov['warnings']) > 0:
                s['warnings'] += 1
            rows.append(dict(key=key, method=m, text=matched['text'], status='ok', **prov))
            # native：只在它确实没超预算时才是合法对照
            if native.get('text') and native.get('tokens') is not None \
                    and r.get('target_tokens') and native['tokens'] <= r['target_tokens']:
                s['native_ok'] += 1
                rows.append(dict(key=key, method=m + '_native', text=native['text'],
                                 status='ok', **prov))
        if trims:
            trims.sort()
            s['trim_p50'] = trims[len(trims) // 2]
        stat[m] = s

    io.open(a.out, 'w', encoding='utf-8', newline='\n').write(
        '\n'.join(json.dumps(x, ensure_ascii=False) for x in rows) + '\n')

    print('wrote %s rows=%d' % (a.out, len(rows)))
    for m, s in stat.items():
        print('  %-18s n=%-4d ok=%-4d err=%-4d native_eligible=%-4d trim_p50=%s'
              % (m, s['n'], s['ok'], s['err'], s['native_ok'],
                 s.get('trim_p50') if s.get('trim_p50') is not None else '-'))
    # 覆盖率断言：每个方法都应有 n 条 matched 行（含失败），否则说明产物不全
    for m, s in stat.items():
        if s['n'] == 0:
            print('  *** WARN: method %s 没有任何产物（目录缺失或全空）' % m, file=sys.stderr)


if __name__ == '__main__':
    main()
