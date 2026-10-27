"""把 INTACT 的替代结果应用回原文，写成 out/ext_full/intact/<key>.json。

写成这个格式后，build_ext_arms.py / e2e_longctx_r5full.py --ext-arms
可以**完全不动**地消费它（arm 名会是 ext_intact），和其余 8 个外部基线走同一条评测链路。

替换选择：优先 replacements_mistral，其次 replacements_rulebased，取候选列表第一个。
（INTACT 的 guess.py 会再跑一轮推断攻击挑选候选，本脚本不跑，必须在论文里注明。）
"""
import json, io, os, sys, glob

RD = '/dpc/kuex0095/100069961_ku/privtrim'
sys.path.insert(0, RD + '/research/token_compression_batch1_v1')
import common as C

tok = C.get_tokenizer()
ntok = lambda s: C.count(tok, s)

OUTDIR = RD + '/out/ext_full/intact'
os.makedirs(OUTDIR, exist_ok=True)

cases = {json.loads(x)['key']: json.loads(x)
         for x in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if x.strip()}

files = sorted(glob.glob(RD + '/data/intact_sample_s*_out.json'))
print('input result files:', len(files))
done = leak_b = leak_a = 0
tok_b = tok_a = 0
ents = filled = 0

for p in files:
    for d in json.load(io.open(p, encoding='utf-8')):
        key = d['doc_id']
        text = d['text']
        ann = list(d['annotations'].values())[0]
        ms = sorted(ann['entity_mentions'], key=lambda m: -m['start_offset'])
        nf = 0
        for m in ms:
            ents += 1
            reps = None
            for k in ('replacements_mistral', 'replacements_rulebased'):
                v = m.get(k)
                if v:
                    reps = v if isinstance(v, list) else [v]
                    break
            if not reps:
                continue
            nf += 1
            filled += 1
            rep = reps[0] if reps[0] else ''
            text = text[:m['start_offset']] + rep + text[m['end_offset']:]

        c = cases.get(key)
        cans = [x['value'] for x in (c.get('canaries') or [])] if c else []
        lb = sum(1 for v in cans if v in d['text'])
        la = sum(1 for v in cans if v in text)
        leak_b += lb
        leak_a += la
        tb, ta = ntok(d['text']), ntok(text)
        tok_b += tb
        tok_a += ta

        n_orig = ntok(c['context']) if c else tb
        B = n_orig // 2
        rec = dict(id=key, method='intact', status='ok',
                   original_tokens=n_orig, target_tokens=B,
                   matched=dict(text=text, tokens=ta),
                   native=None,
                   intent='anonymize',
                   protocol='INTACT replacements.py, Mistral-7B-v0.3 fp16, '
                            'email/phone->MISC, account->CODE, 取候选[0]（未跑 guess.py）')
        with io.open(os.path.join(OUTDIR, key + '.json'), 'w', encoding='utf-8') as f:
            json.dump(rec, f, ensure_ascii=False)
        done += 1

print('docs written      : %d' % done)
print('entities          : %d  (filled %d)' % (ents, filled))
print('canary leak       : %d -> %d' % (leak_b, leak_a))
print('tokens            : %d -> %d  (%.2f%% retained)' % (tok_b, tok_a, 100.0 * tok_a / tok_b))
