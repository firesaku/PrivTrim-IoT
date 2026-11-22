"""Serialization-only tokenizer bridge from tokenizers 0.20+ to 0.19."""
from common import *
import shutil
from transformers import AutoTokenizer
src=Path(model_path('deadcode99/cpc-1.0-llama-1b-tokenizer'))
dst=HERE/'cpc_tokenizer_compat';dst.mkdir(exist_ok=True)
for f in src.glob('*.json'):shutil.copyfile(f,dst/f.name)
data=read(src/'tokenizer.json')
assert all(isinstance(x,list) and len(x)==2 and all(' ' not in y for y in x) for x in data['model']['merges'])
data['model']['merges']=[' '.join(x) for x in data['model']['merges']]
(dst/'tokenizer.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')
tok=AutoTokenizer.from_pretrained(str(src),local_files_only=True)
texts=['Hello world.\nA number: 123. Café café. <|end_of_text|>','_','<|begin_of_text|>']
for c in read(HERE/'inputs.json'):
    texts.extend([c['context'],c['question']])
    texts.extend(c['context'].splitlines())
refs=[{'text':t,'plain':tok.encode(t,add_special_tokens=False),'special':tok.encode(t,add_special_tokens=True)} for t in texts]
write_new(HERE/'cpc_tokenizer_reference.json',refs)
write_new(HERE/'cpc_tokenizer_conversion.json',{'source_sha256':sha(src/'tokenizer.json'),'converted_sha256':sha(dst/'tokenizer.json'),
    'conversion':'BPE merges [left,right] -> left + ASCII space + right; vocabulary and all other fields identical',
    'reference_tokenizer':__import__('tokenizers').__version__,'reference_tests':len(refs)})
print('Converted serialization; reference tests',len(refs))
