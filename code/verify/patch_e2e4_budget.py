"""Apply the corrected exact-trim fix to e2e4.py's drop_to_budget.

Same defect as e2e_longctx.py: the accounting quantity (sum of per-unit token
counts) is not an upper bound on the joined text, because the separators
themselves cost tokens. e2e4 has no hard budget assertion, so its overshoot was
silent rather than fatal.

The fix uses the CORRECTED trim (advance a cursor along `order`), NOT the
tail-trim I first prescribed -- `order` deliberately sorts answer-bearing units
last, so trimming from its tail deletes the answer first and breaks
monotonicity (measured: gold present at 0.25, deleted at 0.50, back at 0.75).

This does not change e2e4's published conclusion: both `budget_cluster` and
`budget_random` call this function, so they were biased the same way, and the
achieved token counts were printed. But it removes a known accounting error.
"""
import io
import re
import shutil

P = r'D:\CodexData\tmp\privtrim-review-20261015\hpc\e2e4.py'

src = io.open(P, encoding='utf-8').read()
if 'def drop_to_budget(' not in src:
    raise SystemExit('drop_to_budget not found in e2e4.py -- aborting, do not guess')

start = src.index('def drop_to_budget(')
m = re.compile(r'^def ', re.M).search(src, start + 1)
end = m.start() if m else len(src)
old = src[start:end]

NEW = '''def drop_to_budget(units, budget, nt, order):
    """按 order 丢单元，直到**发射文本本身**的 token 数 <= budget。

    记账量必须是被断言的那个量。早期版本用"逐单元 token 数之和"当记账量并声称它
    是空格拼接文本的上界——**这个说法是错的**：拼接会插入分隔空格，每个分隔符自身
    可能占一个 token，所以 nt(" ".join(sel)) 可以比 sum(nt(u) for u in sel) 大。
    实测在 e2e_longctx 的 longctx-toolmem-needle-0003 上正好超 1 个 token
    （got=5378 B=5377）。本文件没有预算硬断言，所以那个超标是静默的。

    因此先用和做一次预剪（便宜），再用**真实拼接文本**做精确修剪。

    精确修剪**沿 order 用游标继续前进**，而不是从 order 尾部往前丢：order 故意把
    含答案的单元排在最后，从尾部丢等于先删答案，并且会破坏"预算越大保留越多"的
    单调性（实测 longctx-toolmem_raw-needle-0002：0.25 存活、0.50 被删、0.75 又回来）。
    游标方案使被丢弃集合始终是 order 的前缀，因而嵌套、因而单调。
    """
    utok = [nt(u) for u in units]
    total = sum(utok)
    keep = [True] * len(units)
    cursor = 0
    for i in order:
        if total <= budget:
            break
        keep[i] = False
        total -= utok[i]
        cursor += 1

    for _ in range(len(units) + 1):
        sel = [u for k, u in enumerate(units) if keep[k]]
        if len(sel) <= 1 or nt(" ".join(sel)) <= budget:
            break
        while cursor < len(order) and not keep[order[cursor]]:
            cursor += 1
        if cursor >= len(order):
            break                      # 无可再丢，交给调用方判断
        keep[order[cursor]] = False
        cursor += 1
    return " ".join(u for k, u in enumerate(units) if keep[k])

'''

shutil.copyfile(P, P + '.bak_prefixcount')
src = src[:start] + NEW + src[end:]
io.open(P, 'w', encoding='utf-8', newline='\n').write(src)

import py_compile
py_compile.compile(P, doraise=True)
print('OK patched e2e4.py (backup at e2e4.py.bak_prefixcount)')
print('OLD head:', old.split(chr(10))[1][:70] if len(old.split(chr(10))) > 1 else '')
print('has cursor:', 'cursor' in NEW)
