"""在 LongBench 上运行 Perception Compressor（NAACL 2025 Findings）。

原则：**不修改作者代码**。本脚本只 import 它的模块并调用三个公开函数，
流程与它 main() 里的完全一致：

    get_rank_results -> budget_control -> Semi_Guided_Iterative_Compression

输出写成本项目的 ext 口径：out/ext_full/perception/<key>.json
    {"status":"ok", "matched":{"text":压缩后上下文}, "native":{"text":同左}, ...}
（该方法是单变体，没有 matched/native 之分，两者写同一文本。）

偏离说明（须写进论文）：
  - guiding questions 由本地 Qwen2.5-7B 生成，作者只发布了 NQ 版本
  - 上下文按段落切分为多个 ctx（作者原数据是检索段落，结构相近）
  - compression_rate=2 对应保留 50% token，对齐我们的等预算协议
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
import sys, os, json, io, argparse, time, glob
import torch

RD = '/dpc/kuex0095/100069961_ku/privtrim'
sys.path.insert(0, RD + '/pc_stubs')                    # icecream/seaborn/pynvml 桩
sys.path.insert(0, RD + '/repos/PerceptionCompressor')  # 作者原码
import PerceptionCompressor as PC
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

INSTRUCTION = ("Write a high-quality answer for the given question using only the "
               "provided search results (some of which might be irrelevant).")


def first(pattern):
    g = glob.glob(pattern)
    if not g:
        raise SystemExit('missing model: ' + pattern)
    return g[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default=RD + '/data/pc_qa.jsonl')
    ap.add_argument('--query-dict', default=RD + '/data/pc_query_dict.json')
    ap.add_argument('--outdir', default=RD + '/out/ext_full/perception')
    ap.add_argument('--rate', type=float, default=2.0, help='压缩率；2 = 保留 50%% token')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    # 作者 run_NQ_example.sh 里的默认超参，原样沿用
    ap.add_argument('--k_1', type=float, default=0.4)
    ap.add_argument('--tau_o', type=float, default=0.2)
    ap.add_argument('--k_2', type=float, default=0.1)
    ap.add_argument('--coarse_control_coefficient', type=float, default=1.0)
    ap.add_argument('--segment_size', type=int, default=200)
    ap.add_argument('--reverse', action='store_true')   # 作者 argparse 里有，默认 False
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    llama2 = first(RD + '/hf_cache/models--meta-llama--Llama-2-7b-hf/snapshots/*')
    llama3 = first(RD + '/hf_cache/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/*')
    sbert = first(RD + '/hf_cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*')

    # CPU 上 fp16 多数算子不支持，按设备选 dtype（作者原配置是 cuda + fp16）
    ddtype = torch.float16 if dev == 'cuda' else torch.float32
    print('loading models... device=%s dtype=%s' % (dev, ddtype), flush=True)
    model = LlamaForCausalLM.from_pretrained(llama2, torch_dtype=ddtype).to(dev)
    tokenizer = LlamaTokenizer.from_pretrained(llama2)
    bert = AutoModel.from_pretrained(sbert, torch_dtype=ddtype).to(dev)
    bert_tokenizer = AutoTokenizer.from_pretrained(sbert)
    cnt_tokenizer = AutoTokenizer.from_pretrained(llama3, use_fast=False)
    print('models ready on', dev, flush=True)

    qd = json.load(io.open(a.query_dict, encoding='utf-8'))
    cases = [json.loads(x) for x in io.open(a.cases, encoding='utf-8') if x.strip()]
    mine = cases[a.shard::a.nshards]
    if a.limit:
        mine = mine[:a.limit]
    print('shard=%d/%d cases=%d query_dict=%d' % (a.shard, a.nshards, len(mine), len(qd)), flush=True)

    ok = fail = 0
    for i, ex in enumerate(mine):
        key = ex['key']
        out_path = os.path.join(a.outdir, key + '.json')
        if os.path.exists(out_path):
            ok += 1
            continue
        t0 = time.time()
        try:
            query = ex['question']
            org_docs = ['Document [%d](Title:%s)%s' % (j + 1, c['title'], c['text'])
                        for j, c in enumerate(ex['ctxs'])]
            idx = PC.get_rank_results(
                org_docs, query, 'perception_compressor', 'after',
                [0] * len(org_docs), INSTRUCTION, qd,
                bert, bert_tokenizer, model, tokenizer)
            rerank_docs = ['Document [%d](Title:%s)%s' % (e[0], ex['ctxs'][e[0]]['title'],
                                                          ex['ctxs'][e[0]]['text'])
                           for e in idx]
            retain_docs, r, target_token = PC.budget_control(
                INSTRUCTION, rerank_docs, query, a.rate, tokenizer,
                a.coarse_control_coefficient)
            compressed_context = PC.Semi_Guided_Iterative_Compression(
                retain_docs, query, model, tokenizer, a, r=r)

            ctx_tokens = len(tokenizer(compressed_context).input_ids)
            org_tokens = len(tokenizer(ex['ctxs'][0]['text']).input_ids) if len(ex['ctxs']) == 1 else None
            rec = {
                'status': 'ok',
                'key': key,
                'method': 'perception',
                'matched': {'text': compressed_context},
                'native': {'text': compressed_context},
                'compression_rate': a.rate,
                'ctx_tokens': ctx_tokens,
                'target_token': target_token,
                'secs': round(time.time() - t0, 1),
            }
            with io.open(out_path, 'w', encoding='utf-8') as f:
                json.dump(rec, f, ensure_ascii=False)
            ok += 1
            if (i + 1) % 5 == 0 or i < 3:
                print('  %d/%d  %s  ctx_tokens=%d  %.1fs'
                      % (i + 1, len(mine), key, ctx_tokens, time.time() - t0), flush=True)
        except Exception as e:
            fail += 1
            with io.open(os.path.join(a.outdir, key + '.json'), 'w', encoding='utf-8') as f:
                json.dump({'status': 'error', 'key': key, 'error': '%s: %s' % (type(e).__name__, str(e)[:300])}, f)
            print('  FAIL %s -> %s: %s' % (key, type(e).__name__, str(e)[:160]), flush=True)
    print('DONE ok=%d fail=%d' % (ok, fail), flush=True)


if __name__ == '__main__':
    main()
