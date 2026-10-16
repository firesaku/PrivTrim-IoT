"""
把我们的逐例结果与已发布基线（mask_quito_x / full）逐例 join，做同案例对比。
在 Almesbar 上直接读共享盘上的 results.sqlite（只读）。

审计修复（2026-10-16，全部为加法式、默认响亮）：
  1) my 侧唯一性：JSONL 里若有重复 (dataset, case_key, arm) 行（例如重跑追加），旧版会把
     重复行重复计入 my 侧均值**和**逐例配对差，静默扭曲 dEM/dF1。现在默认 raise SystemExit；
     确需合并用 --dedup（保留最后一条，并打印数值冲突的行）。
  2) join 覆盖率：my 侧有、DB 侧无的 case 旧版静默跳过、从不统计，于是标着 "on the SAME cases"
     的表其实可能换了总体。现在逐数据集 / 逐 baseline 方法统计覆盖率并写进报告头部；
     覆盖率 <100% 时报告里打 NOT-A-MATCHED-COMPARISON 警告，且退出码非 0
     （加 --allow-partial 才以 0 退出继续用报告）。
  3) DB 侧重复键（同 dataset/case_key/method 多行）旧版静默 last-wins。现在计数并打印 DB-DUP；
     若重复行的 em/f1 不一致（数值确实被 last-wins 改掉）退出码非 0。
CLI 兼容：默认路径常量与输出文件名都没变，新增的都是可选开关。
  python join_baseline.py [--dedup] [--allow-partial] [--mine P] [--db P] [--out P]
退出码：0 正常 / 2 参数错 / 3 覆盖率 <100% / 4 DB 重复键数值冲突。
"""
import io, json, os, sqlite3, sys
from collections import defaultdict

B = '/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005'
RD = '/dpc/kuex0095/100065732/privtrim_v2c_20261015'
DB = os.path.join(B, 'run/qwen25_7b/results.sqlite')
MINE = os.path.join(RD, 'e2eout/e2e2_percase.jsonl')
OUT = os.path.join(RD, 'e2eout/join_report.txt')
MT = ['full', 'mask_quito_x', 'mask_cpc', 'cpc', 'quito_x']

def parse_argv(argv):
    """不改动原有调用方式（run_join.sh 无参数），只加可选开关。"""
    o = dict(dedup=False, allow_partial=False, mine=MINE, db=DB, out=OUT)
    flags = {'--dedup': 'dedup', '--allow-partial': 'allow_partial'}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('-h', '--help'):
            print(__doc__); sys.exit(0)
        elif a in flags:
            o[flags[a]] = True; i += 1
        elif a in ('--mine', '--db', '--out'):
            if i + 1 >= len(argv):
                print("缺参数值: %s" % a); sys.exit(2)
            o[a[2:]] = argv[i + 1]; i += 2
        else:
            print("未知参数: %s" % a); print(__doc__); sys.exit(2)
    return o

ARGS = parse_argv(sys.argv[1:])
DB, MINE, OUT = ARGS['db'], ARGS['mine'], ARGS['out']

mine, lino = [], []
ln = 0
with io.open(MINE, encoding='utf-8') as fh:
    for line in fh:
        ln += 1
        line = line.strip()
        if line:
            mine.append(json.loads(line)); lino.append(ln)
print("mine rows:", len(mine))

# ------------------------------------------------ 1) my 侧唯一性断言（审计 major）
first, dups = {}, []
for i, r in enumerate(mine):
    k = (r['dataset'], r['case_key'], r['arm'])
    if k in first:
        dups.append((k, first[k], i))
    else:
        first[k] = i
if dups:
    def _sf(i):
        return (mine[i]['em'], mine[i]['f1'])
    conf = [d for d in dups if _sf(d[1]) != _sf(d[2])]
    print("!! DUPLICATE (dataset,case_key,arm): %d 组重复 / 共 %d 行冗余；数值冲突 %d 组"
          % (len(dups), len(dups), len(conf)))
    for k, i0, i1 in dups[:10]:
        print("   %s  line#%d vs line#%d  em/f1 %s/%s vs %s/%s%s"
              % (k, lino[i0], lino[i1], mine[i0]['em'], mine[i0]['f1'],
                 mine[i1]['em'], mine[i1]['f1'],
                 "  <-- CONFLICT" if _sf(i0) != _sf(i1) else ""))
    if len(dups) > 10:
        print("   ... 其余 %d 组省略" % (len(dups) - 10))
    if not ARGS['dedup']:
        raise SystemExit("重复行会按重复次数重复计权（my 侧均值 + 逐例配对差）-> 拒绝出报告；"
                         "确认要合并请加 --dedup（保留最后一条）")
    print("!! --dedup：按 (dataset,case_key,arm) 保留最后一条，丢弃 %d 行" % len(dups))
    keep = {}
    for i, r in enumerate(mine):
        keep[(r['dataset'], r['case_key'], r['arm'])] = i
    idx = sorted(keep.values())
    mine, lino = [mine[i] for i in idx], [lino[i] for i in idx]
    print("   dedup 后 mine rows:", len(mine))

mykeys = {}
for r in mine:
    mykeys.setdefault((r['dataset'], r['case_key']), set()).add(r['arm'])
print("unique (dataset,case_key):", len(mykeys))
ds_keys = defaultdict(set)
for (ds, ck) in mykeys:
    ds_keys[ds].add(ck)

con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
cur = con.cursor()
base = defaultdict(dict)                       # (ds,ck) -> method -> (em,f1)
dbdup = defaultdict(list)                      # (ds,ck,method) -> 被覆盖掉的旧值
WANT = ('full', 'mask_quito_x', 'mask_cpc', 'cpc', 'quito_x')
cur.execute("SELECT dataset, case_key, method, em, f1 FROM metrics WHERE rate=? AND track=? AND method IN (%s)"
            % ','.join('?'*len(WANT)),
            (0.5, 'matched', *WANT))
n = 0
for ds, ck, method, em, f1 in cur:
    if ck in ds_keys.get(ds, ()):
        if method in base[(ds, ck)]:           # 审计 minor：以前这里静默 last-wins
            dbdup[(ds, ck, method)].append(base[(ds, ck)][method])
        base[(ds, ck)][method] = (em, f1)
        n += 1
print("baseline rows matched:", n)
# full 的 rate 是 1.0，单独取
cur.execute("SELECT dataset, case_key, method, em, f1 FROM metrics WHERE rate=? AND track=? AND method='full'",
            (1.0, 'matched'))
nf = 0
for ds, ck, method, em, f1 in cur:
    if ck in ds_keys.get(ds, ()):
        if method in base[(ds, ck)]:
            dbdup[(ds, ck, method)].append(base[(ds, ck)][method])
        base[(ds, ck)]['full'] = (em, f1)
        nf += 1
print("baseline full rows matched:", nf)

# ------------------------------------------------ 3) DB 侧重复键报告（审计 minor）
db_conf = []
if dbdup:
    for k in sorted(dbdup):
        vals = dbdup[k] + [base[k[:2]][k[2]]]
        if len(set(vals)) > 1:
            db_conf.append((k, vals))
    print("!! DB-DUP: %d 个 (dataset,case_key,method) 在 DB 里有多行（原为静默 last-wins），"
          "冗余 %d 行；其中 em/f1 冲突 %d 个" % (len(dbdup), sum(len(v) for v in dbdup.values()), len(db_conf)))
    for k, vals in db_conf[:10]:
        print("   %s  多值 %s（last-wins 取最后一个）" % (k, vals))
else:
    print("DB-DUP: none")

# ------------------------------------------------ 2) join 覆盖率（审计 major）
our_keys = set(mykeys)
matched = {k for k in our_keys if base.get(k)}
cov_ds = {}
for ds in sorted(ds_keys):
    cov_ds[ds] = (sum(1 for ck in ds_keys[ds] if base.get((ds, ck))), len(ds_keys[ds]))
cov_m = {m: sum(1 for k in our_keys if m in base.get(k, {})) for m in MT}
miss = sorted(our_keys - matched)[:10]
print("JOIN COVERAGE: %d/%d (%.1f%%) of my (dataset,case_key) have any DB baseline row"
      % (len(matched), len(our_keys), 100.0*len(matched)/max(len(our_keys), 1)))
for ds in sorted(cov_ds):
    ok, tot = cov_ds[ds]
    print("   %-12s %5d/%5d  %.1f%%" % (ds, ok, tot, 100.0*ok/max(tot, 1)))
for m in MT:
    print("   方法 %-14s %5d/%5d" % (m, cov_m[m], len(our_keys)))
if miss:
    print("   未匹配样例（前 %d 个）: %s" % (len(miss), miss))
COV_FULL = len(matched) == len(our_keys) and len(our_keys) > 0

# 聚合
def agg(pairs):
    if not pairs:
        return None
    n = len(pairs)
    return n, sum(p[0] for p in pairs)/n, sum(p[1] for p in pairs)/n

rows = []
mine_by = defaultdict(list)
for r in mine:
    mine_by[(r['dataset'], r['arm'])].append((r['em'], r['f1']))

L = []
w = L.append
w("JOIN REPORT — my arms vs published baselines on the SAME cases")
w("my rows=%d  keys=%d  baseline matched rows=%d (method@rate0.5) + %d (full@1.0)" % (
    len(mine), len(mykeys), n, nf))
w("JOIN COVERAGE (审计新增，前两节标题写 on the SAME cases，必须看这行):")
w("  overall %d/%d = %.1f%%" % (len(matched), len(our_keys), 100.0*len(matched)/max(len(our_keys), 1)))
for ds in sorted(cov_ds):
    ok, tot = cov_ds[ds]
    w("  %-12s %5d/%5d = %.1f%%" % (ds, ok, tot, 100.0*ok/max(tot, 1)))
w("  by method: " + "  ".join("%s=%d/%d" % (m, cov_m[m], len(our_keys)) for m in MT))
if not COV_FULL:
    w("!! COVERAGE < 100% —— 下表 NOT-A-MATCHED-COMPARISON，不是同一批 case 上的对比 !!")
if dbdup:
    w("!! DB-DUP: %d 个 (dataset,case_key,method) 在 results.sqlite 里有多行（last-wins），冲突 %d 个"
      % (len(dbdup), len(db_conf)))
w("")
w("%-10s %-18s %7s %9s %9s" % ("dataset", "arm", "n", "EM", "F1"))
for ds in sorted({r['dataset'] for r in mine}):
    for arm in ['full', 'filtered_full', 'win_extractive', 'win_cluster', 'mask_extractive', 'mask_cluster']:
        a = agg(mine_by.get((ds, arm)))
        if a:
            w("%-10s %-18s %7d %8.2f%% %8.2f%%" % (ds, arm, a[0], 100*a[1], 100*a[2]))
w("")
w("=== 已发布基线（同一批 case 上重算）===")
w("%-10s %-18s %7s %9s %9s" % ("dataset", "baseline", "n", "EM", "F1"))
for ds in sorted({r['dataset'] for r in mine}):
    for m in ['full', 'mask_quito_x', 'mask_cpc', 'cpc', 'quito_x']:
        pairs = [base[k][m] for k in base if k[0] == ds and m in base[k]]
        a = agg(pairs)
        if a:
            w("%-10s %-18s %7d %8.2f%% %8.2f%%" % (ds, m, a[0], 100*a[1], 100*a[2]))
w("")
w("=== 逐例配对差（我的臂 − mask_quito_x），仅取两边都有的 case ===")
w("    该表覆盖：mask_quito_x %d/%d = %.1f%%   逐数据集: %s" % (
    cov_m['mask_quito_x'], len(our_keys), 100.0*cov_m['mask_quito_x']/max(len(our_keys), 1),
    "  ".join("%s %d/%d" % (ds, sum(1 for ck in ds_keys[ds] if 'mask_quito_x' in base.get((ds, ck), {})),
                            len(ds_keys[ds])) for ds in sorted(ds_keys))))
w("%-10s %-18s %7s %11s %11s" % ("dataset", "arm", "n", "dEM(pp)", "dF1(pp)"))
for ds in sorted({r['dataset'] for r in mine}):
    for arm in ['win_extractive', 'win_cluster', 'mask_extractive', 'mask_cluster']:
        d = []
        for r in mine:
            if r['dataset'] != ds or r['arm'] != arm:
                continue
            b = base.get((ds, r['case_key']), {}).get('mask_quito_x')
            if b:
                d.append((r['em'] - b[0], r['f1'] - b[1]))
        if d:
            k = len(d)
            w("%-10s %-18s %7d %+10.2f %+10.2f" % (
                ds, arm, k, 100*sum(x[0] for x in d)/k, 100*sum(x[1] for x in d)/k))

io.open(OUT, 'w', encoding='utf-8').write("\n".join(L))
print("\n".join(L))
print("wrote", OUT)

# ------------------------------------------------ 退出码（响亮）
rc = 0
if not COV_FULL and not ARGS['allow_partial']:
    print("!! COVERAGE < 100%：报告已写出但不构成 matched comparison；"
          "确认接受部分覆盖请加 --allow-partial")
    rc = 3
if db_conf:
    print("!! DB 侧重复键数值冲突：baseline 数值被 last-wins 改掉，请先核对 results.sqlite")
    rc = max(rc, 4)
if rc:
    print("EXIT %d" % rc)
    sys.exit(rc)
print("JOIN_OK coverage=100%% db_dup=%d" % len(dbdup))
