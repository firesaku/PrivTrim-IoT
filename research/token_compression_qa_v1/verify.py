"""Final deterministic checks, source receipt and non-overwriting archive."""
import shutil
import time
from common import *
from selection import compression_path

def main():
    from transformers import AutoTokenizer
    cfg=read(HERE/'config.json');assert cfg['code_sha256']==sha(HERE/'common.py')
    assert cfg['data_sha256']==sha(ROOT/'external/squad_v11/dev-v1.1.json')
    assert cfg['scorer_sha256']==sha(SCORER)
    cases=read(HERE/'cases.json');summary=read(HERE/'summary.json')
    tok=AutoTokenizer.from_pretrained(str(QWEN),local_files_only=True)
    count=lambda text:len(tok.encode(text,add_special_tokens=False))
    ncomp=0;expected_calls=0;issues=[]
    for c in cases:
        assert all(a['text'] in c['context'] for a in c['answers'])
        for m in METHODS:
            comp=read(compression_path(c['id'],m));ncomp+=1
            if comp['status']!='ok':issues.append({'case':c['id'],'method':m,'type':'compression_error'});continue
            assert count(comp['text'])==comp['actual_tokens']
            assert count(c['context'])==comp['original_tokens']
            if m!='full':assert comp['actual_tokens']<=comp['original_tokens']//2
            if not comp['budget_matched']:issues.append({'case':c['id'],'method':m,'type':'budget_underfill','retention':comp['retention']})
            cid=c['id']+'__'+m;req=read(HERE/'requests'/(cid+'.json'));result=read(HERE/'task'/(cid+'.json'))
            assert req['messages']==messages(c,comp['text'])
            rendered=tok.apply_chat_template(req['messages'],tokenize=False,add_generation_prompt=True)
            assert req['rendered_prompt']==rendered
            assert result['input_tokens']==count(rendered)
            expected_calls+=1
    assert len(list((HERE/'task').glob('*.json')))==expected_calls
    assert len(list((HERE/'requests').glob('*.json')))==expected_calls
    initial=[read(p) for p in (HERE/'compressed').glob('*__longllmlingua_gpt2.json')]
    receipt={'cases':len(cases),'distinct_articles':len({c['title'] for c in cases}),'selected_compression_records':ncomp,
        'task_calls':expected_calls,'paired_common_budget_cases':len(summary['paired_ids']),
        'actual_token_counts_verified':True,'task_messages_and_tokens_verified':True,'answers_preserved':True,
        'truncated_task_outputs':summary['truncated_outputs'],'budget_or_compression_issues':issues,
        'initial_long200_errors_preserved':sum(r['status']=='error' for r in initial),
        'source_config_and_dataset_unchanged':True,'code_hashes':{p.name:sha(p) for p in HERE.glob('*.py')},
        'generation_seconds':sum(read(p)['generation_seconds_allocated'] for p in (HERE/'task').glob('*.json'))}
    write_new(HERE/'verification.json',receipt)
    destroot=Path(r'D:\Shijie Zeng\CodeX\cybersec_IoT')
    target=destroot/'research/token_compression_qa_v1';assert not target.exists()
    shutil.copytree(HERE,target)
    source_target=destroot/'external/squad_v11'
    if not source_target.exists():shutil.copytree(ROOT/'external/squad_v11',source_target)
    for p in HERE.rglob('*'):
        if p.is_file():assert sha(p)==sha(target/p.relative_to(HERE))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
