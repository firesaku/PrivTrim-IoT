import hashlib, importlib.util, json, re, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'research/token_compression_qa_v1'
spec=importlib.util.spec_from_file_location('old_common',OLD/'common.py')
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
QWEN=old.QWEN; LINGUA2=old.LINGUA2
read=old.read; sha=old.sha; write_new=old.write_new
exposure=old.exposure; messages=old.messages; scorer=old.scorer; redact=old.redact
CODE=ROOT/'literature-search-20260921-token-compression/code'
ALGORITHMS=['quito_x','dac','provence','cpc','llmlingua2']
METHODS=['full']+ALGORITHMS+['mask_llmlingua2']
GROUP={m:('training_free' if m in ['quito_x','dac'] else 'trained_compressor') for m in ALGORITHMS}
GROUP.update(full='reference',mask_llmlingua2='rule_control')
def model_path(repo):
    m=read(HERE/'model_manifest.json')[repo]
    assert m['status']=='ok', repo
    return m['path']
def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(QWEN),local_files_only=True)
def count(tok,text):return len(tok.encode(text,add_special_tokens=False))
def key(c,m,track):return c['id']+'__'+m+'__'+track
def harmonic(a,b):return 2*a*b/(a+b) if a+b else 0.
