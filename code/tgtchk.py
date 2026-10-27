import json, io

RD = '/dpc/kuex0095/100069961_ku/privtrim'
cases = [json.loads(x) for x in io.open(RD + '/data/longctx_max.jsonl', encoding='utf-8') if x.strip()][:3]
for c in cases:
    print('=== %s' % c['key'])
    print('  question     :', c['question'][:140])
    print('  target_entity:', c.get('target_entity'))
    print('  answers      :', c.get('answers'))
    print('  canaries     :', [(x.get('type'), x.get('value')) for x in (c.get('canaries') or [])][:4])
    te = c.get('target_entity') or ''
    print('  target in canaries? ', te in [x.get('value') for x in (c.get('canaries') or [])])
    print('  target in context?  ', te in c['context'])
    print()
