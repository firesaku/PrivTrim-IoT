"""Uniform short-context adaptation, frozen before any QA task generation."""
import sys
import time
import traceback
from common import *

def main():
    assert not (HERE/'task').exists() and not (HERE/'requests').exists()
    old=[read(p) for p in (HERE/'compressed').glob('*__longllmlingua_gpt2.json')]
    assert len(old)==24
    selection={'folders':{m:('compressed_long50' if m=='longllmlingua_gpt2' else 'compressed') for m in METHODS},
        'reason':'default iterative_size=200 left five short inputs completely uncompressed; frozen all-case adaptation before any task generation',
        'longllmlingua_iterative_size':50,'old_iterative_size':200,'preflight_errors':[r['id'] for r in old if r['status']!='ok'],
        'unchanged':'all 24 cases, GPT2 weights, common tokenizer, target/tolerance, other compressors, question conditioning and 12 candidate search',
        'base_config_sha256':sha(HERE/'config.json'),'code_sha256':sha(__file__),
        'no_task_outcomes_existed_at_selection':True}
    write_new(HERE/'compression_selection.json',selection)
    sys.path.append(r'D:\Shijie Zeng\CodeX\cybersec_IoT\.venv\Lib\site-packages')
    import torch
    from transformers import AutoTokenizer
    from llmlingua import PromptCompressor
    torch.set_num_threads(4);torch.manual_seed(0)
    tok=AutoTokenizer.from_pretrained(str(QWEN),local_files_only=True)
    count=lambda s:len(tok.encode(s,add_special_tokens=False))
    compressor=PromptCompressor(str(GPT2),device_map='cpu',model_config={'local_files_only':True,'trust_remote_code':False,'torch_dtype':torch.float32})
    for c in read(HERE/'cases.json'):
        ts=time.perf_counter();n=count(c['context']);target=n//2;lo,hi=0.001,0.999;trials=[]
        try:
            for step in range(12):
                rate=0.5 if step==0 else (lo+hi)/2;begin=time.perf_counter()
                output=compressor.compress_prompt([c['context']],question=c['question'],
                    target_token=max(1,int(compressor.get_token_length(c['context'])*rate)),
                    condition_in_question='after',condition_compare=True,rank_method='longllmlingua',
                    concate_question=False,use_context_level_filter=False,use_sentence_level_filter=False,
                    iterative_size=50)['compressed_prompt']
                tokens=count(output);trials.append({'rate':rate,'tokens':tokens,'text':output,'seconds':time.perf_counter()-begin})
                if tokens>target:hi=rate
                else:lo=rate
            eligible=[r for r in trials if r['tokens']<=target]
            if not eligible:raise RuntimeError('No candidate within target budget')
            chosen=max(eligible,key=lambda x:x['tokens']);actual=chosen['tokens']
            result={'id':c['id'],'method':'longllmlingua_gpt2','status':'ok','text':chosen['text'],'original_tokens':n,
                'actual_tokens':actual,'retention':actual/n,'budget_matched':0.48<=actual/n<=0.5,
                'compression_seconds':time.perf_counter()-ts,'trials':trials,'iterative_size':50,
                'selection_sha256':sha(HERE/'compression_selection.json')}
        except Exception:
            result={'id':c['id'],'method':'longllmlingua_gpt2','status':'error','error':traceback.format_exc(),
                'trials':trials,'compression_seconds':time.perf_counter()-ts}
        write_new(HERE/'compressed_long50'/(c['id']+'__longllmlingua_gpt2.json'),result)
        print('long50',c['id'],result.get('retention'),result['status'],flush=True)

if __name__=='__main__':main()
