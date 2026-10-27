"""外部基线：把已发布的 6 个压缩方法原样跑到长上下文基准上。

设计原则
--------
**不重实现任何压缩算法。** 本脚本直接 import batch1 原目录的 `common.py` 与
`compress.py`，复用它的 `load_method(method, tok)` / `compress_case(...)`，
只做一件事：把长上下文案例的 schema 适配成 batch1 期望的字段。

为什么要适配：batch1 的案例用 `id` / `question` / `original_tokens`，
长上下文案例用 `key` / `question` / （无 original_tokens，需现场用 tokenizer 实测）。
这是字段名差异，不是算法差异。

预算口径（必须诚实说明）
------------------------
`compress.py` 里 `budget = n // 2` 是**写死的 0.5**，即
`target = floor(0.5 * nt(context))`。
本项目的预算臂用 `B = round(ratio * nt(full_context))`。
两者在偶数上完全相同，奇数上最多差 1 token。
所以本脚本跑出来的就是**精确的 0.5 档对标**，无需改动任何已发布代码。

产物
----
`<outdir>/<method>/<case_id>.json`，每个文件是 `compress_case` 的原样返回，
其中 `matched.text` 是 token 数 <= budget 的压缩后上下文（可直接送模型打分），
`native.text` 是算法自身在最优 rate 下的原生输出（可能超预算）。
`matched` 是等预算对标的正确选择；两者都保留，便于事后核查。
"""
# --- HPC_LOGIN_GUARD_V1 : 禁止在登录节点运行（2026-10-27 违规事故后加入）---
import os as _os_guard, sys as _sys_guard
if not _os_guard.environ.get("SLURM_JOB_ID"):
    _sys_guard.stderr.write(
        "\n[BLOCKED] 该脚本会加载模型/做重计算，必须在 Slurm 作业内运行。\n"
        "  检测不到 SLURM_JOB_ID -> 当前很可能是 HPC 登录节点。\n"
        "  正确做法:  sbatch <你的.slurm>   （优先 --partition=prod）\n"
        "  参见 AGENTS.md 规则 1。\n\n")
    _sys_guard.exit(97)
# --- end HPC_LOGIN_GUARD_V1 ---
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

B1_DEFAULT = r'D:\Shijie Zeng\CodeX\cybersec_IoT\research\token_compression_batch1_v1'
ALL_METHODS = ['full', 'quito_x', 'dac', 'provence', 'cpc', 'llmlingua2', 'mask_llmlingua2']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch1', default=B1_DEFAULT,
                    help='已发布 batch1 代码目录（含 common.py / compress.py）')
    ap.add_argument('--cases', required=True, help='longctx_cases.jsonl')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--methods', default=','.join(ALL_METHODS))
    ap.add_argument('--keys', default='', help='逗号分隔的 case key，仅跑这些（smoke 用）')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 个案例（smoke 用）')
    a = ap.parse_args()

    b1 = Path(a.batch1).resolve()
    if not (b1 / 'common.py').is_file():
        raise SystemExit('batch1/common.py 不存在: %s' % b1)
    sys.path.insert(0, str(b1))
    import common as C          # noqa: E402  已发布原码
    import compress as K        # noqa: E402  已发布原码

    methods = [m for m in a.methods.split(',') if m]
    bad = [m for m in methods if m not in K.METHODS]
    if bad:
        raise SystemExit('未知方法 %s；可用：%s' % (bad, K.METHODS))

    cases = [json.loads(l) for l in open(a.cases, encoding='utf-8') if l.strip()]
    if a.keys:
        want = set(a.keys.split(','))
        cases = [c for c in cases if c['key'] in want]
    if a.limit:
        cases = cases[:a.limit]
    if not cases:
        raise SystemExit('没有选中任何案例')

    print('[ext] batch1=%s' % b1, flush=True)
    print('[ext] cases=%d methods=%s' % (len(cases), methods), flush=True)
    tok = C.get_tokenizer()

    # ---- schema 适配：只改字段名，不改内容 ----
    adapted = []
    for c in cases:
        ctx = c['context']
        adapted.append(dict(
            id=c['key'],
            question=c['question'],
            context=ctx,
            original_tokens=C.count(tok, ctx),
            # 以下仅用于事后分组统计，不参与压缩
            case_kind=c.get('case_kind'),
            dataset=c.get('dataset'),
            canaries=c.get('canaries'),
        ))
    nts = sorted(x['original_tokens'] for x in adapted)
    print('[ext] context tokens: min=%d median=%d max=%d  目标预算=floor(0.5*n)'
          % (nts[0], nts[len(nts) // 2], nts[-1]), flush=True)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / 'ext_manifest.json'

    for method in methods:
        md = out / method
        md.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            run, cache, stats, info = K.load_method(method, tok)
        except Exception:
            tb = traceback.format_exc()
            (out / ('LOAD_ERROR_%s.txt' % method)).write_text(tb, encoding='utf-8')
            print('[ext] %-16s LOAD FAILED -> LOAD_ERROR_%s.txt' % (method, method), flush=True)
            continue
        print('[ext] %-16s loaded in %.1fs  info=%s'
              % (method, time.time() - t0, json.dumps(info, ensure_ascii=False)[:220]), flush=True)

        done = skip = err = 0
        for i, c in enumerate(adapted):
            fp = md / (c['id'] + '.json')
            if fp.exists():
                skip += 1
                continue
            try:
                r = K.compress_case(c, method, run, cache, stats, tok)
            except Exception:
                r = dict(id=c['id'], method=method, status='error',
                         error=traceback.format_exc(), original_tokens=c['original_tokens'])
                err += 1
            fp.write_text(json.dumps(r, ensure_ascii=False), encoding='utf-8')
            done += 1
            if done % 10 == 0 or i == len(adapted) - 1:
                mt = (r.get('matched') or {}).get('tokens')
                print('[ext] %-16s %3d/%d status=%s matched=%s target=%s %.1fs'
                      % (method, i + 1, len(adapted), r.get('status'), mt,
                         c['original_tokens'] // 2, time.time() - t0), flush=True)

        print('[ext] %-16s DONE done=%d skip=%d err=%d  %.1fs'
              % (method, done, skip, err, time.time() - t0), flush=True)
        # 清掉显存，避免下一个方法 OOM
        try:
            import torch
            del run
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ---- 汇总 ----
    rows = []
    for method in methods:
        md = out / method
        for fp in sorted(md.glob('*.json')):
            try:
                r = json.loads(fp.read_text(encoding='utf-8'))
            except Exception:
                continue
            m = r.get('matched') or {}
            nat = r.get('native') or {}
            rows.append(dict(
                key=r.get('id'), method=method, status=r.get('status'),
                original_tokens=r.get('original_tokens'),
                target_tokens=r.get('target_tokens'),
                matched_tokens=m.get('tokens'),
                native_tokens=nat.get('tokens'),
                native_within_cap=r.get('native_within_cap'),
                adapter_trimmed_tokens=r.get('adapter_trimmed_tokens'),
                seconds=r.get('compression_seconds'),
                error=(r.get('error') or '')[:400],
            ))
    summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print('[ext] wrote %s rows=%d' % (summary_path, len(rows)), flush=True)


if __name__ == '__main__':
    main()
