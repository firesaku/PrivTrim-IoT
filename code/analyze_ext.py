"""外部基线 vs 内部臂的最终对比表（数据到齐后一条命令出结果）。

数据来源:
  内部臂  : out/max32_percase.jsonl        (2728 例 x 21)
  外部臂  : out/extev_s*_percase.jsonl     (由 exteval.slurm 产出，ext_<method> / ext_<method>_native)

用法: cd /dpc/kuex0095/100069961_ku/privtrim && env/bin/python analyze_ext.py
      可选参数 1: 覆盖外部臂文件的 glob（用于在真实数据到齐前用 dry-run 产物验证本脚本）
"""
import json, io, os, sys, glob, collections, math

RD = '/dpc/kuex0095/100069961_ku/privtrim'
EXT_GLOB = sys.argv[1] if len(sys.argv) > 1 else RD + '/out/*ev*_s*_percase.jsonl'
BOOT = 4000
try:
    import numpy as np
    rng = np.random.default_rng(20261021)
except ImportError:
    np = None


def load(p):
    return [json.loads(x) for x in io.open(p, encoding='utf-8') if x.strip()]


def num(r, k):
    v = r.get(k)
    return float(v) if isinstance(v, (int, float)) else 0.0


ext_rows = []
_shards = sorted(glob.glob(EXT_GLOB))
for p in _shards:
    ext_rows += load(p)
print('external eval rows: %d  (%d shards)' % (len(ext_rows), len(_shards)))
if not ext_rows:
    print('还没有 extev 产物 —— 等 1483706 跑完再执行')
    raise SystemExit(0)

maxrows = load(RD + '/out/max32_percase.jsonl')
ext_keys = set(r['case_key'] for r in ext_rows)
print('cases evaluated with ext arms: %d' % len(ext_keys))

# 内部臂，只保留同一批案例。
# 口径: 外部基线是由 compress.py 的 `budget = n//2` 压出来的，**固定 50%**。
#       所以主对比必须用我们的 @0.50 臂，才是等预算 apples-to-apples；
#       @0.20 是我们的最优工作点，只能作为附加行出现，不能当主对比，
#       否则就是拿 20% 压缩率去比人家的 50%，审稿人一句话就能拆掉。
internal_at50 = [r for r in maxrows
                 if r['case_key'] in ext_keys
                 and r.get('arm') == 'alias_budget_link' and r.get('ratio') == 0.50]
# 0.20 这个预算点只在 sweep 里（max32 只有 0.25/0.5/0.75）
sweep_rows = []
for _p in sorted(glob.glob(RD + '/out/sweep_s*_percase.jsonl')):
    sweep_rows += load(_p)
internal_at20 = [r for r in sweep_rows
                 if r['case_key'] in ext_keys
                 and r.get('arm') == 'alias_budget_link' and r.get('ratio') == 0.20]
internal_alias = [r for r in maxrows
                  if r['case_key'] in ext_keys and r.get('arm') == 'full' and r.get('ratio') is None]

MAIN_REF = 'OURS@0.50(equal)'
EXTRA_REF = 'OURS@0.20(best)'

# 统一用「臂名字符串」做键：外部臂各案例的 ratio 固定，避免 (arm, ratio) 对不齐
# （ext_full 的 ratio 实测是 0.5 而不是 None，早期版本因此漏掉了它）
by_case = collections.defaultdict(dict)
for r in ext_rows:
    by_case[r['case_key']][r['arm']] = r
for r in internal_at50:
    by_case[r['case_key']][MAIN_REF] = r
for r in internal_at20:
    by_case[r['case_key']][EXTRA_REF] = r
for r in internal_alias:
    by_case[r['case_key']]['full(max32)'] = r

arm_names = sorted({r['arm'] for r in ext_rows})
pairs = arm_names + [MAIN_REF, EXTRA_REF, 'full(max32)']

print('\n%-26s %6s %8s %8s %9s %9s' % ('arm', 'n', 'F1', 'EM', 'tgt_pres', 'leak'))
print('-' * 74)
stats = {}
for k in pairs:
    v = [d[k] for d in by_case.values() if k in d]
    if not v:
        continue
    n = len(v)
    f1 = sum(num(r, 'f1') for r in v) / n
    em = sum(num(r, 'em') for r in v) / n
    tp = sum(num(r, 'tgt_entity_present') for r in v) / n
    lk = sum(1 for r in v if num(r, 'ctx_leak_value') > 0 or num(r, 'out_leak_value') > 0) / n
    stats[k] = [num(r, 'f1') for r in v]
    print('%-26s %6d %8.4f %8.4f %9.3f %9.3f' % (k, n, f1, em, tp, lk))

# 与内部臂的配对检验（只在该臂与对照都有值的案例上）
ref = MAIN_REF
if ref in stats:
    print('\n=== 配对对比: 各外部臂 vs %s（等预算，这才是论文主表） ===' % ref)
    for k in pairs:
        if k == ref or k not in stats:
            continue
        d = []
        for case, m in by_case.items():
            if k in m and ref in m:
                d.append(num(m[k], 'f1') - num(m[ref], 'f1'))
        if len(d) < 10:
            continue
        arr = np.array(d)
        lo = hi = float('nan')
        if np is not None:
            idx = rng.integers(0, len(arr), size=(BOOT, len(arr)))
            bs = arr[idx].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
        print('  %-26s n=%-5d dF1=%+.4f  95%%CI[%+.4f,%+.4f]  wins=%d losses=%d' %
              (k, len(d), arr.mean(), lo, hi,
               int((arr > 1e-12).sum()), int((arr < -1e-12).sum())))

# dac 覆盖告警
dac_keys = set(r['case_key'] for r in ext_rows if r['arm'] == 'ext_dac')
if dac_keys and len(dac_keys) < len(ext_keys):
    print('\n!! 覆盖告警: ext_dac 只有 %d/%d 例，且历史上只处理过 <=8790 token 的案例。'
          % (len(dac_keys), len(ext_keys)))
    print('   它的 F1 不能与其他臂并列解读，必须先做长度/来源分层再比。')
