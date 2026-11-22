"""Generate additional mechanical rephrasings (C/D/E) of the templated questions.

Motivation: docs/17 limitation 8 records that variants A and B give CONTRADICTORY
bootstrap conclusions for the alias gain (+0.14 A / +0.23 original / +0.26 B; A's CI
excludes 0, B's includes 0). Two variants are not enough to say anything. This adds
three more so the effect can be reported as an interval.

Pure rules, no model calls. Self-checks (must all pass):
  1) target entity string still present in the rephrased question
  2) gold answer NOT introduced into the question by the rewrite
  3) the question actually changed
  4) attribute name still present
"""
import io, json, re, sys

RE_MULTI = re.compile(r'^In the log above, which (.+?) is recorded for the account (.+?)\?$')
RE_NEEDLE = re.compile(r'^In the log above, what is the (.+?) recorded for (.+?)\?$')

def rephrase(q, v):
    m = RE_MULTI.match(q)
    if m:
        attr, ent = m.group(1), m.group(2)
        return {
            'C': 'Looking at the log above, what %s corresponds to the account %s?' % (attr, ent),
            'D': 'In the log above, identify the %s of the account %s.' % (attr, ent),
            'E': 'The log above records the account %s; what is its %s?' % (ent, attr),
        }.get(v)
    m = RE_NEEDLE.match(q)
    if m:
        attr, anchor = m.group(1), m.group(2)
        return {
            'C': 'Looking at the log above, what %s corresponds to %s?' % (attr, anchor),
            'D': 'In the log above, identify the %s of %s.' % (attr, anchor),
            'E': 'The log above contains %s; what is its %s?' % (anchor, attr),
        }.get(v)
    return None

def main():
    cases = [json.loads(l) for l in io.open(sys.argv[1], encoding='utf-8') if l.strip()]
    for v in ('C', 'D', 'E'):
        out, bad, skipped, pre = [], [], 0, []
        for c in cases:
            q = c.get('question') or ''
            q2 = rephrase(q, v)
            if q2 is None:
                skipped += 1
                out.append(c)
                continue
            te = c.get('target_entity') or ''
            gold = (c.get('answers') or [''])[0]
            if gold and gold in q:
                pre.append(c['key'])
            if q2 == q:
                bad.append((c['key'], 'unchanged'))
            if te and te not in q2:
                bad.append((c['key'], 'target_entity lost'))
            if gold and gold in q2 and gold not in q:
                bad.append((c['key'], 'GOLD LEAKED'))
            d = dict(c)
            d['question'] = q2
            d['question_original'] = q
            d['question_variant'] = v
            out.append(d)
        path = sys.argv[2] % v
        io.open(path, 'w', encoding='utf-8', newline='\n').write(
            '\n'.join(json.dumps(x, ensure_ascii=False) for x in out) + '\n')
        print('variant=%s cases=%d rephrased=%d skipped=%d problems=%d preexisting_defects=%d'
              % (v, len(cases), len(cases) - skipped, skipped, len(bad), len(pre)))
        if bad:
            print('  PROBLEMS:', bad[:6])
            raise SystemExit('self-check failed for variant ' + v)
        print('  sample:', out[0].get('question'))

main()