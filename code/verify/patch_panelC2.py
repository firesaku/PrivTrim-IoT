"""三处修正：
1. 面板 C 的 kind 匹配：by_case_kind 的键是 'multi_entity'/'needle'，
   等值判断漏掉了 multi（needle 恰好匹配上，所以图看起来"部分正常"）。
   改为前缀匹配。
2. 面板 A 的 'flat region 50-75%' 文字与图例重叠 -> 移到坐标区左上方空白处。
3. 面板 C 标题过长撞到 D 标题 -> 缩短。
"""
import io
import re

P = r'D:\CodexData\tmp\privtrim-review-20261015\hpc\plot_longctx.py'
s = io.open(P, encoding='utf-8').read()
orig = s

# --- 1. kind 前缀匹配 -------------------------------------------------------
old = """    kind, arm = parts[0], parts[1]
    if kind not in ('multi', 'needle') or arm not in ('budget_cluster', 'budget_cluster_link'):
        continue
    agg[(kind, arm)] = (v['f1'] / v['n'], v['n'])"""
new = """    kind, arm = parts[0], parts[1]
    # 注意：by_case_kind 的键是 'multi_entity' / 'needle'，必须前缀匹配。
    # 用等值判断会静默漏掉 multi（而 needle 恰好匹配），得到一张"部分正确"的图。
    kk = ('multi' if kind.startswith('multi')
          else 'needle' if kind.startswith('needle') else None)
    if kk is None or arm not in ('budget_cluster', 'budget_cluster_link'):
        continue
    agg[(kk, arm)] = (v['f1'] / v['n'], v['n'])"""
assert old in s, 'kind match block not found'
s = s.replace(old, new, 1)

# --- 2. flat region 标注位置 ------------------------------------------------
old2 = "ax.annotate('flat region 50-75%', xy=((x50 + x75) / 2, 0.115), ha='center',"
new2 = "ax.annotate('flat region 50-75%', xy=((x50 + x75) / 2, 0.775), ha='center',"
assert old2 in s, 'flat region annotate not found'
s = s.replace(old2, new2, 1)

# --- 3. 面板 C 标题缩短 -----------------------------------------------------
old3 = "ax.set_title('C  Anchor rule: gain concentrated in multi-entity cases',"
new3 = "ax.set_title('C  Gain by case type',"
assert old3 in s, 'panel C title not found'
s = s.replace(old3, new3, 1)

assert s != orig
io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
import py_compile
py_compile.compile(P, doraise=True)
print('OK patched 3 spots')
for pat in ("kind.startswith('multi')", '0.775', "C  Gain by case type"):
    print('  check %-32s %s' % (pat, pat in s))
