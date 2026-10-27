"""给所有会加载模型的重脚本插入 Slurm 守卫（幂等，可重复执行）。

守卫逻辑：检测不到 SLURM_JOB_ID 就立刻退出。
这样无论用什么解释器、什么路径、谁调用它，都**不可能**在登录节点上跑起来。

只保护重脚本；make_* / build_* / analyze_* 这类纯 JSON 处理的轻脚本不保护
（它们是允许在登录节点运行的）。
"""
import io, os, re, sys

RD = '/path/to/project'
TARGETS = [
    'run_pc.py',
    'gen_guiding.py',
    'ext_compress2.py',
    'run_intact.py',
    'code/longctx/e2e_longctx.py',
    'code/longctx/e2e_longctx_r2.py',
    'code/longctx/e2e_longctx_r3.py',
    'code/longctx/e2e_longctx_r4.py',
    'code/longctx/e2e_longctx_r5.py',
    'code/longctx/e2e_longctx_r5full.py',
    'code/longctx/e2e_longctx_r5full_ORIG.py',
    'code/longctx/ext_compress.py',
]

MARK = 'HPC_LOGIN_GUARD_V1'
GUARD = '''# --- {mark} : 禁止在登录节点运行（2026-10-27 违规事故后加入）---
import os as _os_guard, sys as _sys_guard
if not _os_guard.environ.get("SLURM_JOB_ID"):
    _sys_guard.stderr.write(
        "\\n[BLOCKED] 该脚本会加载模型/做重计算，必须在 Slurm 作业内运行。\\n"
        "  检测不到 SLURM_JOB_ID -> 当前很可能是 HPC 登录节点。\\n"
        "  正确做法:  sbatch <你的.slurm>   （优先 --partition=prod）\\n"
        "  参见 AGENTS.md 规则 1。\\n\\n")
    _sys_guard.exit(97)
# --- end {mark} ---
'''.format(mark=MARK)

done, skipped, missing = [], [], []
for rel in TARGETS:
    p = os.path.join(RD, rel)
    if not os.path.exists(p):
        missing.append(rel); continue
    src = io.open(p, encoding='utf-8', errors='replace').read()
    if MARK in src:
        skipped.append(rel); continue
    lines = src.splitlines(keepends=True)
    # 插在 shebang / 编码声明 / 模块 docstring 之后
    idx = 0
    if lines and lines[0].startswith('#!'):
        idx = 1
    if idx < len(lines) and re.match(r'^#.*coding[:=]', lines[idx]):
        idx += 1
    # 跳过开头的 docstring
    if idx < len(lines) and lines[idx].lstrip().startswith(('"""', "'''")):
        q = lines[idx].lstrip()[:3]
        if lines[idx].count(q) >= 2:
            idx += 1
        else:
            idx += 1
            while idx < len(lines) and q not in lines[idx]:
                idx += 1
            idx += 1
    new = ''.join(lines[:idx]) + GUARD + ''.join(lines[idx:])
    io.open(p, 'w', encoding='utf-8', newline='\n').write(new)
    done.append(rel)

print('已加守卫 :', len(done))
for x in done:
    print('   +', x)
print('已存在跳过:', len(skipped))
for x in skipped:
    print('   =', x)
print('文件不存在:', len(missing))
for x in missing:
    print('   ?', x)
