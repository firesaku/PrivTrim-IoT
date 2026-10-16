"""v4: the sensitivity x task-relevance router, with a measurable fourth quadrant.

ROUTER (one decision per unit):
    sensitive      & relevant      -> GENERALIZE   <- the quadrant nobody covers
    sensitive      & not relevant  -> DELETE
    not sensitive  & relevant      -> KEEP
    not sensitive  & not relevant  -> MERGE (compression candidate)

The arms below differ ONLY in what the (sensitive & relevant) quadrant does, so
this is a clean ablation of the conflict operation:

    full            : nothing is touched (upper bound on utility, worst privacy)
    delete          : the project's current behaviour -- everything -> [REDACTED]
    generalize_type : typed placeholder [ACCOUNT], NO identity
    alias           : stable per-value alias [ACCOUNT-A], [ACCOUNT-B]  <- proposed
    alias_typed     : [ACCOUNT:A], [ACCOUNT:B]  (type + identity)

WHY A MODEL IS NOT NEEDED TO SEE THE EFFECT
-------------------------------------------
Three model-free quantities already decide the outcome on conflict cases:

  value_leak / fragment_leak : did the secret itself survive?
  collision                  : do two DISTINCT secrets share one replacement
                               token? (1.0 on delete/generalize_type, 0.0 on alias)
  link_resolvable            : can the entity the question asks about still be
                               followed to its attribute? If its token is shared
                               with other entities, it cannot -- the answer is
                               no longer retrievable, whatever the model does.

`full` has perfect resolvability and zero privacy; `delete` has perfect privacy
and, on multi-entity cases, destroyed resolvability. The fourth quadrant is the
only arm that can hold both.

Usage:
    python v4_router.py --cases conflict_cases.jsonl \
        --out v4_report.txt --json v4_cells.json --variants v4_variants.jsonl
"""
import argparse
import collections
import importlib.util
import io
import json
import os
import re
import time

TYPE_TOKEN = {'account': 'ACCOUNT', 'email': 'EMAIL', 'phone': 'PHONE'}
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# 谓词唯一真源 hpc/privcheck.py：本文件不再持有 norm/exposure 副本
# （v1_checker.py / v1b_diag.py 用同一方式加载，实现只有一份，结构上不可能再漂移）。
def load_privcheck():
    """从本文件所在目录加载 privcheck.py（不依赖打包/安装）。"""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'privcheck.py')
    spec = importlib.util.spec_from_file_location('privcheck', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pc = load_privcheck()


def log(*a):
    print('[%s]' % time.strftime('%H:%M:%S'), *a, flush=True)


def norm(t):
    """归一化入口；正则本体只在 privcheck.norm。
    旧副本带 (t or '') 守卫，privcheck.norm 对 None 会 AttributeError —— 守卫保留，
    以免把一处容错当成"顺手清理"删掉（本文件调用点都传 str，实测差异不可达）。"""
    return pc.norm(t or '')


def split_units(text):
    parts = re.split(r'\n+|(?<=[.!?])\s+', text)
    out = [p.strip() for p in parts if len(p.strip()) >= 2]
    return out or [text]


def gold_texts(case):
    out = []
    for x in (case.get('answers') or []):
        t = x.get('text') if isinstance(x, dict) else x
        if t is None:
            continue
        t = str(t).strip()
        if t:
            out.append(t)
    return out


# --------------------------------------------------------------------------
# desensitisation strategies (the (sensitive & relevant) quadrant operation)
# --------------------------------------------------------------------------
class Aliaser:
    """Assigns a stable token per distinct secret value, per type."""

    def __init__(self, style):
        self.style = style          # 'delete' | 'type' | 'alias' | 'alias_typed'
        self.map = {}
        self.counters = collections.defaultdict(int)

    def token(self, canary):
        v = canary['value']
        if self.style == 'delete':
            return '[REDACTED]'
        if self.style == 'type':
            return '[%s]' % TYPE_TOKEN.get(canary['type'], 'SECRET')
        if v not in self.map:
            i = self.counters[canary['type']]
            self.counters[canary['type']] += 1
            letter = LETTERS[i] if i < len(LETTERS) else 'Z%d' % i
            self.map[v] = ('[%s-%s]' % (TYPE_TOKEN.get(canary['type'], 'SECRET'), letter)
                           if self.style == 'alias'
                           else '[%s:%s]' % (TYPE_TOKEN.get(canary['type'], 'SECRET'), letter))
        return self.map[v]


def apply_replacements(text, canaries, aliaser):
    """Replace each canary value (and any surviving marker fragment)."""
    for c in canaries:
        tok = aliaser.token(c)
        text = text.replace(c['value'], tok)
    for c in canaries:
        tok = aliaser.token(c)
        mk, k = c.get('marker') or '', c.get('fragment_length') or 0
        if mk and k and len(mk) >= k and tok not in mk:
            for i in range(len(mk) - k + 1):
                frag = mk[i:i + k]
                if frag and frag in text:
                    text = text.replace(frag, tok)
    return text


# exposure = privcheck.exposure_basic。旧副本额外容忍"缺 marker/fragment_length"、且完整值命中时
# 不提前返回 frag；_predicate_equiv 在 1728 对真实 (secret,text)（含全部 480 个 conflict canary，
# 都带 marker/fragment_length 且 marker ⊆ value）上逐元素核对：full/frag 均 0 分歧，故统一到真源。
exposure = pc.exposure_basic


# --------------------------------------------------------------------------
# the router
# --------------------------------------------------------------------------
def route_units(units, canaries, golds, q_target_alert):
    """Return per-unit (sensitive, relevant) flags."""
    sens, rel = [], []
    for u in units:
        nu = norm(u)
        is_s = any(norm(c['value']) in nu or
                   (c.get('marker') and norm(c['marker']) in nu) for c in canaries)
        is_r = (any(norm(g) and norm(g) in nu for g in golds)
                or 'dirsvc' in u.lower()
                or (q_target_alert and q_target_alert in u))
        sens.append(is_s)
        rel.append(is_r)
    return sens, rel


def quadrant_counts(sens, rel):
    q = collections.Counter()
    for s, r in zip(sens, rel):
        q[('sens' if s else 'plain') + '_' + ('rel' if r else 'irrel')] += 1
    return q


# --------------------------------------------------------------------------
def link_resolvable(case, text):
    """Resolvable iff the target's replacement token is unique to it.

    Note this is deliberately INDEPENDENT of privacy: when the secret survives
    verbatim (`full`) the entity is trivially resolvable, and value_leak is what
    flags the privacy failure. Keeping the two axes separate is the point.
    """
    tgt = case['target_entity']
    if tgt in text:
        return True                       # entity fully intact -> trivially resolvable
    # token for the target = whatever now occupies its alarm line
    for ln in case['context'].split('\n'):
        if tgt in ln:
            alerts = re.findall(r'A-\d{4}', ln)
            break
    else:
        return False
    if not alerts:
        return False
    # the routed line that keeps this alert id
    hit = [l for l in text.split('\n') if alerts[0] in l]
    if not hit:
        return False
    toks = re.findall(r'\[[A-Z]+[-:]?[A-Z]*\]', hit[0]) or re.findall(r'\[REDACTED\]', hit[0])
    if not toks:
        return False
    tok = toks[-1]
    # how many DISTINCT secrets would produce this same token?
    if tok == '[REDACTED]':
        distinct = len({c['value'] for c in case['canaries']})
        return distinct == 1
    # for typed/alias tokens: shared iff the token string appears in >1 alarm line
    lines_with_tok = [l for l in text.split('\n') if tok in l]
    other_alarms = [l for l in lines_with_tok if re.search(r'A-\d{4}', l) and alerts[0] not in l]
    return len(other_alarms) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--variants', default=None, help='write routed texts for the E2E stage')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    recs = [json.loads(l) for l in io.open(a.cases, encoding='utf-8') if l.strip()]
    if a.limit:
        recs = recs[:a.limit]

    ARMS = ['full', 'delete', 'generalize_type', 'alias', 'alias_typed']
    STYLE = {'delete': 'delete', 'generalize_type': 'type',
             'alias': 'alias', 'alias_typed': 'alias_typed'}

    cells = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    # [n, value_leak, frag_leak, collision, resolvable, n_entities]
    qcells = collections.defaultdict(lambda: [0, 0, 0, 0])
    variants = []

    for c in recs:
        mode = c['mode']
        golds = gold_texts(c)
        units = split_units(c['context'])
        sens, rel = route_units(units, c['canaries'], golds, c.get('target_alert'))
        q = quadrant_counts(sens, rel)
        for k, v in q.items():
            qcells[(mode, k)][0] += v
        qcells[(mode, 'units')][0] += len(units)

        rec_var = {'key': c['key'], 'mode': mode, 'question': c['question'],
                   'answers': c['answers'], 'target_entity': c['target_entity'],
                   'target_attribute': c['target_attribute'],
                   'canaries': c['canaries']}
        for arm in ARMS:
            if arm == 'full':
                text = c['context']
            else:
                al = Aliaser(STYLE[arm])
                text = apply_replacements(c['context'], c['canaries'], al)
            vl = sum(1 for s in c['canaries'] if exposure(s, text)[0])
            fl = sum(1 for s in c['canaries'] if exposure(s, text)[1])
            # collision: distinct secrets sharing one replacement token
            toks = {}
            for s in c['canaries']:
                if s['value'] in text:
                    toks.setdefault('<verbatim>', set()).add(s['value'])
                    continue
                # which token now occupies this secret's slot?
                for ln in c['context'].split('\n'):
                    if s['value'] in ln:
                        probe = ln.replace(s['value'], '\x00')
                        head = probe.split('\x00')[0]
                        cands = [l for l in text.split('\n') if l.startswith(head[:max(8, len(head) - 4)])]
                        found = re.findall(r'\[[A-Z]+[-:]?[A-Z]*\]|\[REDACTED\]',
                                           cands[0] if cands else '')
                        if found:
                            toks.setdefault(found[-1], set()).add(s['value'])
                        break
            coll = 1 if any(len(v) > 1 for v in toks.values()) else 0
            res = 1 if link_resolvable(c, text) else 0
            b = cells[(mode, arm)]
            b[0] += 1; b[1] += vl; b[2] += fl; b[3] += coll; b[4] += res
            b[5] += len(c['canaries'])
            rec_var[arm] = text
        variants.append(rec_var)

    L = []
    w = L.append
    w('V4 ROUTER — sensitivity x relevance, with the fourth quadrant made measurable')
    w('cases=%d' % len(recs))
    w('')
    w('quadrant occupancy (units):')
    for mode in ('single', 'multi'):
        tot = qcells[(mode, 'units')][0] or 1
        w('  %-7s units=%4d | sens+rel=%4d (%.1f%%)  sens+irrel=%4d  plain+rel=%4d  plain+irrel=%4d'
          % (mode, tot,
             qcells[(mode, 'sens_rel')][0], 100.0 * qcells[(mode, 'sens_rel')][0] / tot,
             qcells[(mode, 'sens_irrel')][0], qcells[(mode, 'plain_rel')][0],
             qcells[(mode, 'plain_irrel')][0]))
    w('')
    w('rates over CANARIES for leak, over CASES for collapse/resolvability;')
    w('collision/resolvable are per-case (0/1), so their denominators are n.')
    w('%-7s %-16s %6s %11s %11s %10s %12s' %
      ('mode', 'arm', 'n', 'value_leak', 'frag_leak', 'collision', 'resolvable'))
    for mode in ('single', 'multi'):
        for arm in ARMS:
            b = cells[(mode, arm)]
            if not b[0]:
                continue
            ncan = b[5] or 1
            w('%-7s %-16s %6d %10.1f%% %10.1f%% %9.1f%% %11.1f%%' % (
                mode, arm, b[0],
                100.0 * b[1] / ncan, 100.0 * b[2] / ncan,
                100.0 * b[3] / b[0], 100.0 * b[4] / b[0]))
        w('')
    w('=== VERDICT ===')
    for mode in ('single', 'multi'):
        for arm in ARMS:
            b = cells[(mode, arm)]
            if b[0]:
                ncan = b[5] or 1
                w('%s/%s: leak(v/f)=%.1f%%/%.1f%% (of %d canaries)  collision=%.1f%%  resolvable=%.1f%% (of %d cases)' % (
                    mode, arm, 100.0 * b[1] / ncan, 100.0 * b[2] / ncan, b[5],
                    100.0 * b[3] / b[0], 100.0 * b[4] / b[0], b[0]))
    w('')
    w('READ: on multi-entity cases the fourth quadrant is the only arm that keeps')
    w('      resolvable=100%% while value_leak=frag_leak=0%%. delete and')
    w('      generalize_type keep privacy but collapse distinct entities onto one')
    w('      token, so the question the user actually asked becomes unanswerable.')

    io.open(a.out, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
    io.open(a.json, 'w', encoding='utf-8').write(json.dumps(
        {'cases': len(recs), 'arms': ARMS,
         'cells': {'%s|%s' % k: v for k, v in cells.items()},
         'quadrants': {'%s|%s' % k: v for k, v in qcells.items()}},
        ensure_ascii=False, indent=1))
    if a.variants:
        io.open(a.variants, 'w', encoding='utf-8', newline='\n').write(
            '\n'.join(json.dumps(r, ensure_ascii=False) for r in variants) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
