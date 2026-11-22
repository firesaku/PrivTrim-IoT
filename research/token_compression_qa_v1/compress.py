import argparse
import importlib.util
import sys
import time
import traceback
from common import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=METHODS);args=parser.parse_args()
    import torch
    from transformers import AutoTokenizer,GPT2Tokenizer,GPT2LMHeadModel
    torch.set_num_threads(4);torch.manual_seed(0)
    tok=AutoTokenizer.from_pretrained(str(QWEN),local_files_only=True)
    count=lambda text:len(tok.encode(text,add_special_tokens=False));method=args.method
    if method in ['llmlingua2','mask_llmlingua2']:
        from llmlingua import PromptCompressor
        compressor=PromptCompressor(str(LINGUA2),device_map='cpu',use_llmlingua2=True,
            model_config={'local_files_only':True,'trust_remote_code':False,'torch_dtype':torch.float32})
        def run(text,query,rate):
            if method=='mask_llmlingua2':text=redact(text)
            return compressor.compress_prompt([text],rate=rate,use_context_level_filter=False)['compressed_prompt']
    elif method=='longllmlingua_gpt2':
        sys.path.append(r'D:\Shijie Zeng\CodeX\cybersec_IoT\.venv\Lib\site-packages')
        from llmlingua import PromptCompressor
        compressor=PromptCompressor(str(GPT2),device_map='cpu',model_config={'local_files_only':True,'trust_remote_code':False,'torch_dtype':torch.float32})
        def run(text,query,rate):
            return compressor.compress_prompt([text],question=query,target_token=max(1,int(compressor.get_token_length(text)*rate)),
                condition_in_question='after',condition_compare=True,rank_method='longllmlingua',concate_question=False,
                use_context_level_filter=False,use_sentence_level_filter=False,iterative_size=200)['compressed_prompt']
    elif method=='selective_context_token':
        path=ROOT/'external/Selective_Context/selective_context.py'
        spec=importlib.util.spec_from_file_location('sc_author',path);mod=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=mod;spec.loader.exec_module(mod);mod.DEVICE='cpu'
        class LocalSC(mod.SelectiveContext):
            def _prepare_model(self):
                self.tokenizer=GPT2Tokenizer.from_pretrained(str(GPT2),local_files_only=True)
                self.model=GPT2LMHeadModel.from_pretrained(str(GPT2),local_files_only=True).to('cpu').eval()
                self.max_token_length=self.model.config.n_positions;self.get_self_information=self._get_self_info_via_gpt2
        compressor=LocalSC();original=compressor._lexical_unit;cache={}
        def cached(sents):
            key=tuple(sents)
            if key not in cache:cache.clear();cache[key]=original(sents)
            return cache[key]
        compressor._lexical_unit=cached
        def run(text,query,rate):return compressor(text,reduce_ratio=1-rate,reduce_level='token')[0]
    for c in read(HERE/'cases.json'):
        path=HERE/'compressed'/(c['id']+'__'+method+'.json')
        if path.exists():continue
        text=c['context'];n=count(text);target=n//2;trials=[];ts=time.perf_counter()
        try:
            if method=='full':chosen=text
            else:
                lo,hi=0.001,0.999
                for step in range(12):
                    rate=0.5 if step==0 else (lo+hi)/2
                    begin=time.perf_counter();output=run(text,c['question'],rate);tokens=count(output)
                    trials.append({'rate':rate,'tokens':tokens,'text':output,'seconds':time.perf_counter()-begin})
                    if tokens>target:hi=rate
                    else:lo=rate
                eligible=[r for r in trials if r['tokens']<=target]
                if not eligible:raise RuntimeError('No candidate within common target budget')
                chosen=max(eligible,key=lambda r:r['tokens'])['text']
            actual=count(chosen)
            result={'id':c['id'],'method':method,'status':'ok','text':chosen,'original_tokens':n,'actual_tokens':actual,
                'retention':actual/n,'budget_matched':method=='full' or 0.48<=actual/n<=0.5,
                'compression_seconds':time.perf_counter()-ts,'trials':trials,'code_sha256':sha(__file__),
                'config_sha256':sha(HERE/'config.json')}
        except Exception:
            result={'id':c['id'],'method':method,'status':'error','error':traceback.format_exc(),'trials':trials,
                'compression_seconds':time.perf_counter()-ts}
        write_new(path,result)
        print(c['id'],method,result.get('retention'),result['status'],flush=True)

if __name__=='__main__':main()
