"""Build conflict-scenario cases: the answer needs MULTIPLE sensitive entities
to stay distinguishable.

WHY THIS EXISTS
---------------
The project's privacy front end only ever DELETES (`[REDACTED]`). The design
docs call the "sensitive AND task-relevant" quadrant the one nobody has
covered, but there was no case where it could be measured, because on the
existing corpora the sensitive values are never needed to answer.

The interesting conflict is NOT "the answer is the secret" -- any
desensitisation destroys that answer, so nothing can win. It is:

    several DISTINCT sensitive entities appear, the question is about ONE of
    them, and the answer requires linking that one entity to another sentence.

Under deletion every entity becomes the same token `[REDACTED]`, so the link is
destroyed and the model must guess among k candidates. Under a STABLE ALIAS
(`[ACCOUNT-A]` vs `[ACCOUNT-B]`) the link survives while the value still never
appears. That is the whole point of the fourth quadrant, and this builder makes
it measurable.

Output: conflict_cases.jsonl (frozen, sha256-stamped) with
  mode = single | multi      (how many distinct sensitive entities)
so the single-entity control shows deletion is fine when there is nothing to
disambiguate, and multi shows it collapse.

Usage:
    python build_conflict_cases.py --out conflict_cases.jsonl \
        --per-mode 120 --n-entities 3 --seed 20261016
"""
import argparse
import hashlib
import io
import json
import os
import random

DEPTS = ['Security', 'Facilities', 'Networking', 'Payroll', 'Logistics',
         'Research', 'Compliance', 'FieldOps']
SITES = ['Building 4', 'Warehouse B', 'Remote DC', 'Lab 2', 'Annex 7']
DEVICES = ['sensor-gw-11', 'camera-north-3', 'hvac-ctrl-02', 'badge-rdr-09']

DISTRACTORS = [
    'heartbeat ok from {d} (rtt {ms} ms)',
    'firmware check on {d} reported no drift',
    'scheduled job prune-logs finished in {ms} ms',
    'config sync for {d} completed with no conflicts',
    'daily digest queued for 06:00 local time',
    'link monitor saw {ms} ms jitter on the uplink',
]


def acct(rng):
    return 'ACCT-%06x' % rng.randrange(16 ** 6)


def email_for(rng):
    return 'user%04d@example.invalid' % rng.randrange(10000)


def phone_for(rng):
    return '+1-202-555-%04d' % rng.randrange(10000)


def build_case(rng, n_entities, mode, cid):
    """One case. All entities share ONE type (accounts) so that type-only
    generalisation is maximally unhelpful -- it cannot disambiguate them."""
    n = 1 if mode == 'single' else n_entities
    ents = [acct(rng) for _ in range(n)]
    while len(set(ents)) < n:                     # keep them distinct
        ents = [acct(rng) for _ in range(n)]
    attrs = rng.sample(DEPTS, n)
    pair = list(zip(ents, attrs))
    rng.shuffle(pair)

    alert_ids = ['A-%04d' % rng.randrange(10000) for _ in range(n)]
    while len(set(alert_ids)) < n:
        alert_ids = ['A-%04d' % rng.randrange(10000) for _ in range(n)]

    # the alert lines name an account; the directory lines map account->dept.
    # the two lists are shuffled INDEPENDENTLY so the mapping cannot be
    # recovered from position once the account tokens are all identical.
    alarm_pairs = list(zip(alert_ids, ents))
    rng.shuffle(alarm_pairs)
    dir_lines = [('dirsvc: %s mapped to department %s' % (e, a)) for e, a in pair]
    rng.shuffle(dir_lines)

    lines = ['[2026-03-14 02:11:07] authd: session opened for user svc-ops from 10.4.2.19']
    for aid, ent in alarm_pairs:
        lines.append('[2026-03-14 02:11:%02d] alertd: ALERT %s raised by account %s (severity %s)'
                     % (rng.randrange(10, 59), aid, ent, rng.choice(['high', 'medium', 'low'])))
    for _ in range(rng.randrange(2, 4)):
        lines.append('[2026-03-14 02:12:%02d] mon: %s' % (
            rng.randrange(10, 59),
            rng.choice(DISTRACTORS).format(d=rng.choice(DEVICES), ms=rng.randrange(5, 900))))
    for dl in dir_lines:
        lines.append('[2026-03-14 02:13:%02d] %s' % (rng.randrange(10, 59), dl))
    for _ in range(rng.randrange(1, 3)):
        lines.append('[2026-03-14 02:14:%02d] mon: %s' % (
            rng.randrange(10, 59),
            rng.choice(DISTRACTORS).format(d=rng.choice(DEVICES), ms=rng.randrange(5, 900))))
    rng.shuffle(lines[1:])                        # break positional alignment

    target_alert = alarm_pairs[0][0]
    target_ent = alarm_pairs[0][1]
    target_attr = dict(pair)[target_ent]

    context = '\n'.join(lines)
    question = ('In the log above, which department is the account that raised '
                'ALERT %s mapped to?' % target_alert)

    canaries = [{'type': 'account', 'value': e, 'marker': e.split('-')[1],
                 'fragment_length': 6} for e in ents]
    return {
        'key': 'conflict-%s-%04d' % (mode, cid),
        'dataset': 'conflict',
        'source': 'synth-iot-log',
        'mode': mode,
        'n_entities': n,
        'context': context,
        'question': question,
        'answers': [target_attr],
        'canaries': canaries,
        'target_entity': target_ent,
        'target_alert': target_alert,
        'target_attribute': target_attr,
        'all_pairs': pair,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-mode', type=int, default=120)
    ap.add_argument('--n-entities', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20261016)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    recs = []
    for mode in ('single', 'multi'):
        for i in range(a.per_mode):
            recs.append(build_case(rng, a.n_entities, mode, i))

    # every case must be answerable from its own context and have distinct entities
    for r in recs:
        assert r['target_attribute'] in r['context'], r['key']
        assert r['target_entity'] in r['context'], r['key']
        vals = [c['value'] for c in r['canaries']]
        assert len(set(vals)) == len(vals), r['key']
        assert len(vals) == r['n_entities'], r['key']

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    body = '\n'.join(json.dumps(r, ensure_ascii=False) for r in recs) + '\n'
    io.open(a.out, 'w', encoding='utf-8', newline='\n').write(body)
    sha = hashlib.sha256(body.encode('utf-8')).hexdigest()

    print('wrote %s' % a.out)
    print('cases=%d (single=%d, multi=%d)  n_entities=%d  seed=%d'
          % (len(recs), a.per_mode, a.per_mode, a.n_entities, a.seed))
    print('sha256=%s' % sha)
    print('guess-baseline for multi = 1/%d = %.1f%%'
          % (a.n_entities, 100.0 / a.n_entities))


if __name__ == '__main__':
    main()
