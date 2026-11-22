import time
from common import *
from selection import compression_path

def pending_jobs():
    jobs=[]
    for case in read(HERE/'cases.json'):
        for method in METHODS:
            cid=case['id']+'__'+method
            comp=read(compression_path(case['id'],method))
            if comp['status']!='ok':continue
            if (HERE/'task'/(cid+'.json')).exists():continue
            if (HERE/'requests'/(cid+'.json')).exists():raise RuntimeError('Unresolved reserved call; no automatic retry')
            jobs.append({'id':cid,'case':case['id'],'method':method,'messages':messages(case,comp['text'])})
    return jobs

def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(4);torch.manual_seed(0)
    tok=AutoTokenizer.from_pretrained(str(QWEN),local_files_only=True,padding_side='left')
    ts=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(str(QWEN),local_files_only=True,torch_dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
    load_seconds=time.perf_counter()-ts
    deadline=time.monotonic()+600
    print('Model loaded; waiting for all frozen compression records before generation.',flush=True)
    while not all(compression_path(c['id'],m).exists() for c in read(HERE/'cases.json') for m in METHODS):
        if time.monotonic()>deadline:raise TimeoutError('Compression records incomplete; no generation reservations made')
        time.sleep(5)
    jobs=pending_jobs()
    write_new(HERE/'runtime.json',{'load_seconds':load_seconds,'torch':torch.__version__,
        'transformers':transformers.__version__,'code_sha256':sha(__file__),
        'common_sha256':sha(HERE/'common.py'),'config_sha256':sha(HERE/'config.json'),
        'compression_selection_sha256':sha(HERE/'compression_selection.json')})
    cfg=read(HERE/'config.json')['decode'];print('QA model ready',len(jobs),'calls',flush=True)
    for start in range(0,len(jobs),cfg['batch_size']):
        batch=jobs[start:start+cfg['batch_size']]
        texts=[tok.apply_chat_template(j['messages'],tokenize=False,add_generation_prompt=True) for j in batch]
        enc=tok(texts,return_tensors='pt',padding=True,truncation=False).to('cuda:0')
        assert enc.input_ids.shape[1]+cfg['max_new_tokens']<=8192
        for j,text in zip(batch,texts):
            write_new(HERE/'requests'/(j['id']+'.json'),dict(j,rendered_prompt=text,runtime_sha256=sha(HERE/'runtime.json')))
        torch.cuda.synchronize();ts=time.perf_counter()
        with torch.inference_mode():
            outputs=model.generate(**enc,max_new_tokens=cfg['max_new_tokens'],do_sample=False,pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize();seconds=time.perf_counter()-ts
        eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos]
        for i,j in enumerate(batch):
            ids=outputs[i,enc.input_ids.shape[1]:].tolist();stop=next((n+1 for n,x in enumerate(ids) if x in eos),len(ids));ids=ids[:stop]
            output=tok.decode(ids,skip_special_tokens=True)
            write_new(HERE/'task'/(j['id']+'.json'),{'id':j['id'],'case':j['case'],'method':j['method'],'output':output,
                'input_tokens':int(enc.attention_mask[i].sum()),'output_tokens':len(ids),'ended_with_eos':bool(ids and ids[-1] in eos),
                'batch_seconds':seconds,'batch_index':start//cfg['batch_size'],'generation_seconds_allocated':seconds/len(batch)})
        print('QA',start+len(batch),'/',len(jobs),'seconds',round(seconds,2),flush=True)
    write_new(HERE/'complete.json',{'task_calls':len(jobs)})

if __name__=='__main__':main()
