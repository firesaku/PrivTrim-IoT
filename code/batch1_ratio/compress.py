import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse, functools, time, traceback, shutil
from common import *

def cache_forward(model):
    original=model.forward; cache={};stats={'calls':0,'hits':0}
    def forward(*args,**kwargs):
        inputs=kwargs.get('input_ids',args[0] if args else None)
        mask=kwargs.get('attention_mask',args[1] if len(args)>1 else None)
        sig=(tuple(inputs.shape),inputs.detach().cpu().numpy().tobytes(),None if mask is None else mask.detach().cpu().numpy().tobytes(),repr({k:v for k,v in kwargs.items() if k not in ['input_ids','attention_mask']}))
        if sig not in cache:
            stats['calls']+=1;cache[sig]=original(*args,**kwargs)
        else:stats['hits']+=1
        return cache[sig]
    # preserve the forward signature used by Transformers/PEFT where relevant.
    model.forward=functools.wraps(original)(forward)
    return cache,stats

def load_method(method,tok):
    import torch
    cache={};stats={};model=None
    info={'device':'cuda:0','dtype':'float32','training_updates':0}
    if method=='quito_x':
        sys.path.insert(0,str(CODE/'quito_x'))
        from quito.compressor import T5Compressor
        import quito.utils as qu
        # Official utils calls this missing helper. Implement the specified descending-score ranking.
        qu.get_sorted_ids=lambda scores: sorted(range(len(scores)),key=lambda i:float(scores[i]),reverse=True)
        compressor=T5Compressor(model_path('google/flan-t5-small'),sigma=1)
        model=compressor.model;orig=compressor.t5_attention_score
        def attn(doc,query):
            k=(doc,query)
            if k not in cache:cache[k]=orig(doc,query)
            return cache[k]
        compressor.t5_attention_score=attn
        def run(text,query,rate):return compressor.compress(text,query,ratio=rate,word=True)
        info.update(checkpoint='google/flan-t5-small',sigma=1,word=True,implementation='author T5Compressor, cached frozen attention',patches=['define missing get_sorted_ids as stable descending importance sort'])
    elif method=='dac':
        sp=importlib.util.spec_from_file_location('dac_author',CODE/'dac/compressor.py');module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
        compressor=module.PromptCompressor(model_path('Qwen/Qwen2-0.5B-Instruct'),device_map='cuda',
            model_config={'trust_remote_code':False,'local_files_only':True,'torch_dtype':torch.float32})
        model=compressor.model
        def run(text,query,rate):
            return compressor.compress(text,compress_ratio=1-min(rate,.999999),method='dynamic_attn_ppl',fusion='additive',alpha=.8,dyn_time=None,preserve_punct=False)['compressed_text']
        info.update(checkpoint='Qwen/Qwen2-0.5B-Instruct',dyn_time='author adaptive default',alpha=.8,fusion='additive',query_used=False)
    elif method=='provence':
        from transformers import AutoModel
        model=AutoModel.from_pretrained(model_path('naver/provence-reranker-debertav3-v1'),local_files_only=True,trust_remote_code=True).cuda()
        cache,stats=cache_forward(model)
        def run(text,query,rate):
            return model.process(query,text,title=None,always_select_title=False,threshold=1-rate,enable_warnings=True)['pruned_context']
        info.update(checkpoint='naver/provence-reranker-debertav3-v1',title=None,always_select_title=False,custom_code='reviewed pinned modeling_provence.py; no changes')
    elif method=='cpc':
        # Stage only Python sources; disable an unused training-only metric download.
        stage=HERE/'cpc_inference_source'
        for f in (CODE/'cpc').rglob('*.py'):
            dest=stage/f.relative_to(CODE/'cpc');dest.parent.mkdir(parents=True,exist_ok=True)
            text=f.read_text(encoding='utf-8')
            if f.relative_to(CODE/'cpc').as_posix()=='util/util.py':
                text=text.replace('metric = evaluate.load("accuracy")','metric = None  # Unused training metric; offline inference only.')
            if f.relative_to(CODE/'cpc').as_posix()=='util/preprocessing.py':
                text=text.replace('                sentence_embedding_type=self.sentence_embedding_type,\n','')
            dest.write_text(text,encoding='utf-8')
        (stage/'training/__init__.py').write_text('# Local CPC package; avoid collision with unrelated installed training package.\n',encoding='utf-8')
        sys.path.insert(0,str(stage))
        from model.model import load_model_and_tokenizer
        from prompt_compressor import PromptCompressorCPC
        from util.preprocessing import SamplePreprocessor
        from util.util import SentenceEmbeddingType
        cfg=read(CODE/'cpc/configs/cpc-1.0-llama.json')
        cfg.update(model_name_or_path=model_path('unsloth/Llama-3.2-1B-Instruct'),do_train=False,do_eval=False,attn_implementation='sdpa',output_dir=str(HERE/'unused_training_output'))
        cpath=HERE/'cpc_inference_config.json'
        if not cpath.exists():write_new(cpath,cfg)
        model,ctok=load_model_and_tokenizer(str(cpath),str(HERE/'cpc_tokenizer_compat'),model_path('deadcode99/cpc-1.0-llama-1b-ds-v5-iter66-lora-bidirectional-attn'))
        references=read(HERE/'cpc_tokenizer_reference.json')
        for ref in references:
            assert ctok.encode(ref['text'],add_special_tokens=False)==ref['plain']
            assert ctok.encode(ref['text'],add_special_tokens=True)==ref['special']
        compressor=PromptCompressorCPC.__new__(PromptCompressorCPC)
        compressor.model=model;compressor.tokenizer=ctok
        compressor.processor=SamplePreprocessor(ctok,max_context_len=6144,use_question_as_suffix=False,sentence_embedding_type=SentenceEmbeddingType.AVG)
        compressor.openai_tokenizer=tok
        cache,stats=cache_forward(model)
        def run(text,query,rate):return compressor.compress(text,query,compression_target_tokens=int(count(tok,text)*rate))
        info.update(dtype='bfloat16',checkpoint='official CPC 1B LoRA with unsloth/Llama-3.2-1B-Instruct',backend='sdpa, author bidirectional mask',patches=['unused training evaluate.load disabled','remove unsupported sentence_embedding_type keyword from tokenizer call; retain sole AVG pooling implementation','local training/__init__.py avoids installed package name collision','tokenizer JSON BPE merges serialization converted, IDs verified against current official tokenizer','config local paths, no_train/no_eval, flash_attention_2 -> sdpa'],tokenizer_equivalence_checks=len(references)*2,budget_tokenizer='Qwen2.5 downstream')
    elif method in ['llmlingua2','mask_llmlingua2']:
        from llmlingua import PromptCompressor
        compressor=PromptCompressor(str(LINGUA2),device_map='cuda',use_llmlingua2=True,
            model_config={'local_files_only':True,'trust_remote_code':False,'torch_dtype':torch.float32})
        model=compressor.model;cache,stats=cache_forward(model)
        def run(text,query,rate):return compressor.compress_prompt([text],rate=rate,use_context_level_filter=False)['compressed_prompt']
        info.update(checkpoint=str(LINGUA2),use_context_level_filter=False)
    elif method=='full':
        def run(text,query,rate):return text
    else:raise ValueError(method)
    if model is not None:model.eval();model.requires_grad_(False)
    return run,cache,stats,info

def prefix_budget(tok,text,budget):
    ids=tok.encode(text,add_special_tokens=False)
    output=tok.decode(ids[:budget],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    while count(tok,output)>budget:
        output=output[:-1]
    return output

def compress_case(c,method,run,cache,stats,tok):
    import torch
    cache.clear();stats.update(calls=0,hits=0)
    text=redact(c['context']) if method=='mask_llmlingua2' else c['context']
    n=c['original_tokens'];budget=max(int(round(float(__import__('os').environ.get('PT_EXT_RATIO','0.5'))*n)),1);trials=[]
    torch.cuda.synchronize();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def trial(rate):
        t=time.perf_counter()
        with torch.inference_mode():out=run(text,c['question'],rate)
        torch.cuda.synchronize()
        r=dict(rate=rate,text=out,tokens=count(tok,out),seconds=time.perf_counter()-t)
        trials.append(r);return r
    if method=='full':
        under=over=dict(text=text,tokens=n,rate=1,seconds=0)
        matched=text
    else:
        # Include the full native range; no task labels or predictions affect search.
        trial(.000001);trial(1.)
        lo,hi=.000001,1.
        for i in range(12):
            rate=.5 if i==0 else (lo+hi)/2
            r=trial(rate)
            if r['tokens']>budget:hi=rate
            else:lo=rate
            if r['tokens']==budget:break
        below=[r for r in trials if r['tokens']<=budget]
        above=[r for r in trials if r['tokens']>=budget]
        if not above:raise ValueError('No native output reaches target; do not pad or substitute another algorithm')
        # Some sentence algorithms always retain at least one sentence. Keep their
        # smallest real output and explicitly flag a cap failure; never invent an empty native result.
        under=max(below,key=lambda r:r['tokens']) if below else min(trials,key=lambda r:r['tokens'])
        over=min(above,key=lambda r:r['tokens'])
        matched=prefix_budget(tok,over['text'],budget)
    torch.cuda.synchronize()
    result=dict(id=c['id'],method=method,status='ok',original_tokens=n,target_tokens=budget,
        trials=trials,native=under,native_within_cap=method=='full' or under['tokens']<=budget,matched=dict(text=matched,tokens=count(tok,matched)),
        adapter_source=over,adapter_trimmed_tokens=over['tokens']-count(tok,matched),
        compression_seconds=time.perf_counter()-start,forward_cache_stats=dict(stats),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    if method!='full':assert result['matched']['tokens']<=budget
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('method',choices=METHODS);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    import torch,transformers
    torch.set_num_threads(4);torch.manual_seed(0)
    tok=get_tokenizer();dest=HERE/('smoke' if args.smoke else 'compressed')/args.method
    if args.smoke:dest=dest/time.strftime('%Y%m%d_%H%M%S')
    dest.mkdir(parents=True,exist_ok=True)
    if args.smoke:cases=[{k:c[k] for k in ['id','question','context','original_tokens']} for c in read(OLD/'cases.json')[:2]]
    else:cases=read(HERE/'inputs.json')
    t=time.perf_counter()
    try:
        run,cache,stats,info=load_method(args.method,tok)
        info.update(load_seconds=time.perf_counter()-t,torch=torch.__version__,transformers=transformers.__version__,code_sha256=sha(__file__))
        env=dest/'runtime.json'
        if not env.exists():write_new(env,info)
    except Exception:
        write_new(dest/'load_error.json',{'error':traceback.format_exc()});raise
    for i,c in enumerate(cases):
        path=dest/(c['id']+'.json')
        if path.exists():continue
        try:r=compress_case(c,args.method,run,cache,stats,tok)
        except Exception:r={'id':c['id'],'method':args.method,'status':'error','error':traceback.format_exc()}
        write_new(path,r)
        print(args.method,i+1,len(cases),r['status'],r.get('native',{}).get('tokens'),r.get('matched',{}).get('tokens'),round(r.get('compression_seconds',0),2),flush=True)
        if r['status']=='error':raise RuntimeError(r['error'])
if __name__=='__main__':main()
