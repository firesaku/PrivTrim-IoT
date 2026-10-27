"""外部基线全量进度总览 + 续跑指引（零 GPU，随时可跑）。

用法:
  cd /dpc/kuex0095/100069961_ku/privtrim && env/bin/python status_ext.py

回答四个问题:
  1. 五个压缩方法各完成到哪 / 还差多少
  2. 压缩产物覆盖了多少案例
  3. 外部评测跑了多少、每个臂覆盖多少例
  4. 下一步该提交什么（直接给出可复制的命令）
"""
import json, io, os, glob, collections

RD = '/dpc/kuex0095/100069961_ku/privtrim'
EXT = RD + '/out/ext_full'
MINI = ['quito_x', 'provence', 'cpc', 'llmlingua2', 'mask_llmlingua2']

cases = [json.loads(x) for x in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if x.strip()]
allkeys = set(c['key'] for c in cases)
N = len(cases)
print('数据集: longctx_max  案例数=%d\n' % N)

# ---- 1. 压缩进度 ----
print('%-18s %8s %8s %7s' % ('method', 'done', 'remain', 'pct'))
print('-' * 45)
cover = {}
for m in MINI:
    d = os.path.join(EXT, m)
    ks = set(f[:-5] for f in os.listdir(d) if f.endswith('.json')) if os.path.isdir(d) else set()
    ks &= allkeys
    cover[m] = ks
    print('%-18s %8d %8d %6.1f%%' % (m, len(ks), N - len(ks), 100.0 * len(ks) / N))

dac = os.path.join(EXT, 'dac')
dacn = len([f for f in os.listdir(dac) if f.endswith('.json')]) if os.path.isdir(dac) else 0
print('%-18s %8d %8d %6.1f%%   (已暂缓)' % ('dac', dacn, N - dacn, 100.0 * dacn / N))

full_cover = set.intersection(*cover.values())
print('\n五个方法都覆盖到的案例: %d / %d (%.1f%%)' % (len(full_cover), N, 100.0 * len(full_cover) / N))

# ---- 2. 评测进度 ----
ev = collections.defaultdict(set)
files = sorted(glob.glob(RD + '/out/extev_s*_percase.jsonl')) + \
        sorted(glob.glob(RD + '/out/extevfull_s*_percase.jsonl'))
for p in files:
    for x in io.open(p, encoding='utf-8'):
        if x.strip():
            r = json.loads(x)
            ev[r['arm']].add(r['case_key'])

print('\n评测产物文件: %d 个' % len(files))
if ev:
    print('%-26s %8s %8s' % ('arm', 'cases', 'of total'))
    print('-' * 46)
    for a in sorted(ev):
        print('%-26s %8d %7.1f%%' % (a, len(ev[a]), 100.0 * len(ev[a]) / N))
    done_any = set().union(*ev.values())
    print('\n有任一外部臂评测结果的案例: %d / %d' % (len(done_any), N))
else:
    print('尚无评测产物。')

# ---- 3. 判据 ----
need_comp = [m for m in MINI if len(cover[m]) < N]
need_eval = N - len(set().union(*ev.values())) if ev else N
print('\n' + '=' * 60)
if need_comp:
    print('还缺压缩: %s' % ', '.join('%s(%d)' % (m, N - len(cover[m])) for m in need_comp))
else:
    print('压缩: 五个方法都已覆盖全量 ✓')
if need_eval:
    print('还缺评测: %d 例' % need_eval)
else:
    print('评测: 已覆盖全量 ✓')
    print('下一步 -> env/bin/python analyze_ext.py')

# ---- 4. 续跑命令 ----
print('\n--- 续跑命令 ---')
if need_comp:
    total = sum(N - len(cover[m]) for m in need_comp)
    print('  env/bin/python make_extc_plan.py        # 重算剩余分片（自带覆盖自检）')
    print('  sbatch extcomp2.slurm                   # 预计 %d 个案例待压缩' % total)
if need_comp or need_eval:
    print('  注意: extcomp2.slurm 与 extevalfull.slurm 都可直接重跑，幂等。')
