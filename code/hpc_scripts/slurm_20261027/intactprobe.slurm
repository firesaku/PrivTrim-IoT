#!/bin/bash
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpu
#SBATCH --job-name=pt-intact
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --chdir=/path/to/project/repos/text_sanitization
#SBATCH --output=/path/to/project/logs/intact-%j.out
#SBATCH --error=/path/to/project/logs/intact-%j.err
#
# INTACT 探针：在我们的 3 个案例上跑 replacements.py 的 LLM 生成分支。
#
# 依据（从源码读出）:
#   - 数据格式是 TAB: doc.annotations[ann].entity_mentions[{entity_type,start_offset,end_offset,...}]
#   - entity_type 属于 PERSON/CODE/DATETIME 时走**规则**路径（编号/日期），
#     其余类型才会调 LLM。我们的 canary 映射成 EMAIL/ID/PHONE，强制走 LLM。
#   - 模型默认 mistralai/Mistral-7B-Instruct-v0.2；缓存里只有 v0.3，用 v0.3（需在结果里注明）
#   - 4-bit 量化由 llm_inference_hf.bnb_config 提供（bitsandbytes 已装）
#
# 只跑 replacements.py（候选生成），不跑 guess.py（推断式选择），
# 因为 guess 还要再跑一轮 LLM 推断，先确认生成这一环能通。
set -uo pipefail
module load intel_h200_gpu 2>/dev/null || true
RD=/path/to/project
export PYTHONNOUSERSITE=1
export HF_HOME=$RD/hf_cache
export HF_HUB_CACHE=$RD/hf_cache
export TRANSFORMERS_CACHE=$RD/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
PY=$RD/env-ext/bin/python

echo "=== INTACT probe host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== input"
ls -l $RD/data/intact_probe3.json

T0=$(date +%s)
timeout -k 60 1000 $PY -u $RD/run_intact.py --cases $RD/data/intact_probe3.json --model mistralai/Mistral-7B-Instruct-v0.3 --out $RD/data/intact_probe3_out.json --limit 3
RC=$?
echo "=== rc=$RC elapsed=$(( $(date +%s) - T0 ))s"
echo "=== output summary"
$PY - <<'PY'
import json, io, os
p = '/path/to/project/data/intact_probe3_out.json'
if not os.path.exists(p):
    print('  no output file'); raise SystemExit(0)
docs = json.load(io.open(p, encoding='utf-8'))
print('  docs=%d' % len(docs))
for d in docs:
    ann = list(d['annotations'].values())[0]
    ms = ann['entity_mentions']
    kinds = {}
    n_with = 0
    for m in ms:
        for k in m:
            if k.startswith('replacements_'):
                kinds[k] = kinds.get(k, 0) + 1
                if m[k]:
                    n_with += 1
    print('   %-36s ents=%d  repl_fields=%s  nonempty=%d'
          % (d['doc_id'][:36], len(ms), kinds, n_with))
    for m in ms[:3]:
        for k in m:
            if k.startswith('replacements_'):
                print('      %-10s %-28s -> %s' % (k, m['span_text'][:28], str(m[k])[:90]))
PY
exit 0
