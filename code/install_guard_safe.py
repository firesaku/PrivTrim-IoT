"""给指定脚本插入 Slurm 守卫（安全版）。

安全措施：
  1. 写入前校验 st_uid == 当前用户（绝不改别人的文件）
  2. 拒绝修改符号链接（避免顺着链接写到别处）
  3. 首次修改前在脚本同目录 .guard_bak/ 下留备份
  4. 幂等（已含 MARK 则跳过）
"""
import io, os, re, sys, shutil, time

ME = os.getuid()
MARK = 'HPC_LOGIN_GUARD_V1'
GUARD = '''# --- {mark} : 禁止在登录节点运行（2026-10-27 违规事故后加入）---
import os as _os_guard, sys as _sys_guard
if not _os_guard.environ.get("SLURM_JOB_ID"):
    _sys_guard.stderr.write(
        "\\n[BLOCKED] 该脚本会做重计算，必须在 Slurm 作业内运行。\\n"
        "  检测不到 SLURM_JOB_ID -> 当前很可能是 HPC 登录节点。\\n"
        "  正确做法:  sbatch <你的.slurm>   （优先 --partition=prod）\\n"
        "  参见 AGENTS.md 规则 1。\\n\\n")
    _sys_guard.exit(97)
# --- end {mark} ---
'''.format(mark=MARK)

TARGETS = sys.argv[1:]
if not TARGETS:
    print('usage: install_guard_safe.py <script.py> [...]'); raise SystemExit(2)

ok, skipped, refused, missing = [], [], [], []
for p in TARGETS:
    if not os.path.exists(p):
        missing.append(p); continue
    st = os.lstat(p)
    if os.path.islink(p):
        refused.append((p, '是符号链接')); continue
    if st.st_uid != ME:
        refused.append((p, '属主不是本账号 (uid=%d)' % st.st_uid)); continue
    src = io.open(p, encoding='utf-8', errors='replace').read()
    if MARK in src:
        skipped.append(p); continue

    # 备份
    bakdir = os.path.join(os.path.dirname(p) or '.', '.guard_bak')
    os.makedirs(bakdir, exist_ok=True)
    bak = os.path.join(bakdir, os.path.basename(p) + '.%s' % time.strftime('%Y%m%d_%H%M%S'))
    shutil.copy2(p, bak)

    lines = src.splitlines(keepends=True)
    idx = 0
    if lines and lines[0].startswith('#!'):
        idx = 1
    if idx < len(lines) and re.match(r'^#.*coding[:=]', lines[idx]):
        idx += 1
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
    # 语法自检：改完必须还能编译
    try:
        compile(new, p, 'exec')
    except SyntaxError as e:
        refused.append((p, '插入后语法错误，已放弃: %s' % e)); continue
    io.open(p, 'w', encoding='utf-8', newline='\n').write(new)
    ok.append((p, bak))

print('已加守卫 : %d' % len(ok))
for p, b in ok:
    print('   +', p)
    print('     备份:', b)
print('已存在跳过: %d' % len(skipped))
for p in skipped:
    print('   =', p)
print('拒绝修改 : %d' % len(refused))
for p, why in refused:
    print('   !', p, '->', why)
print('不存在   : %d' % len(missing))
for p in missing:
    print('   ?', p)
