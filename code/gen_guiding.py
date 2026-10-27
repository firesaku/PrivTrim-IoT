"""为 Perception Compressor 生成 guiding questions（query_dict）。

作者仓库只带了 NQ 版本，未发布生成脚本，因此这里用本地 Qwen2.5-7B 生成，
每个问题 3 条子问题（代码里 query_dict[query][:3]）。
这是**文档化的偏离**：子问题由本地模型而非原作者所用模型生成。
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
import json, io, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RD = '/dpc/kuex0095/100069961_ku/privtrim'
IN = RD + '/data/pc_query_dict_questions.json'
OUT = RD + '/data/pc_query_dict.json'

qs = json.load(io.open(IN, encoding='utf-8'))
print('questions =', len(qs))

import glob
m = glob.glob(RD + '/hf_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*')[0]
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float16).to('cuda')
model.eval()

SYS = ('You are a search assistant. Given a question, write 3 short sub-questions '
       'that would help find the answer inside a long document. '
       'Output exactly 3 lines, each a sub-question, no numbering, no extra text.')

out = {}
for i, q in enumerate(qs):
    msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': q}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt').to('cuda')
    with torch.no_grad():
        o = model.generate(ids, max_new_tokens=96, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    txt = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
    lines = [l.strip(' -*0123456789.') for l in txt.splitlines() if l.strip()]
    lines = [l for l in lines if len(l) > 5][:3]
    while len(lines) < 3:
        lines.append(q)
    out[q] = lines
    if (i + 1) % 50 == 0:
        print('  %d/%d' % (i + 1, len(qs)))

json.dump(out, io.open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print('wrote', OUT, 'entries =', len(out))
for q in list(out)[:2]:
    print('  Q:', q)
    for s in out[q]:
        print('     -', s)
