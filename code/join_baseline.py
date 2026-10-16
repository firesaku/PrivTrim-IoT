"""
把我们的逐例结果与已发布基线（mask_quito_x / full）逐例 join，做同案例对比。
在 Almesbar 上直接读共享盘上的 results.sqlite（只读）。
"""
import io, json, os, sqlite3, sys
from collections import defaultdict

B = '/dpc/kuex0095/100065732/privtrim_iot_hpc_20261005/PrivTrim_IoT_HPC_20261005'
RD = '/dpc/kuex0095/100065732/privtrim_v2c_20261015'
DB = os.path.join(B, 'run/qwen25_7b/results.sqlite')
MINE = os.path.join(RD, 'e2eout/e2e2_percase.jsonl')
OUT = os.path.join(RD, 'e2eout/join_report.txt')

mine = []
with io.open(MINE, encoding='utf-8') as fh:
    for line in fh:
        line = line.strip()
        if line:
            mine.append(json.loads(line))
print("mine rows:", len(mine))

mykeys = {}
for r in mine:
    mykeys.setdefault((r['dataset'], r['case_key']), set()).add(r['arm'])
print("unique (dataset,case_key):", len(mykeys))
ds_keys = defaultdict(set)
for (ds, ck) in mykeys:
    ds_keys[ds].add(ck)

con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
cur = con.cursor()
base = defaultdict(dict)     # (ds,ck) -> method -> (em,f1)
WANT = ('full', 'mask_quito_x', 'mask_cpc', 'cpc', 'quito_x')
cur.execute("SELECT dataset, case_key, method, em, f1 FROM metrics WHERE rate=? AND track=? AND method IN (%s)"
            % ','.join('?'*len(WANT)),
            (0.5, 'matched', *WANT))
n = 0
for ds, ck, method, em, f1 in cur:
    if ck in ds_keys.get(ds, ()):
        base[(ds, ck)][method] = (em, f1)
        n += 1
print("baseline rows matched:", n)
# full 的 rate 是 1.0，单独取
cur.execute("SELECT dataset, case_key, method, em, f1 FROM metrics WHERE rate=? AND track=? AND method='full'",
            (1.0, 'matched'))
nf = 0
for ds, ck, method, em, f1 in cur:
    if ck in ds_keys.get(ds, ()):
        base[(ds, ck)]['full'] = (em, f1)
        nf += 1
print("baseline full rows matched:", nf)

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
