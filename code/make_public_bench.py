"""把 LongBench 转成本项目的案例格式，并注入合成 PII canary。

设计原则（为了对外部效度，不改动公开数据本身）:
  - context / question / answers **完全保持 LongBench 原样**，不拼接、不改写
    -> 效用（F1）是在公开基准上测的，不是自建题
  - 只**追加**一段 device-log 块，里面放合成的 PII（account / email / phone）
    -> 泄漏指标（canary 是否残留）才有对象可测
  - case_kind 设为 'single'，使 evaluator 跳过 multi_entity 专用的实体诊断
    （那些诊断需要 entity_values/target_entity，公开数据没有）

局限（必须写进论文）:
  - 问题不引用注入的 canary，所以 link 机制的增益在这批数据上体现不出来
    -> 这里测的是"压缩是否保值 + 脱敏是否止漏"，不是 link 消融
  - canary 是合成的，不是真实 PII 分布
"""
import argparse, json, io, hashlib, random, os

RD = '/dpc/kuex0095/100069961_ku/privtrim'
LB = RD + '/data/longbench/data'
TASKS = ['multifieldqa_en', 'qasper', 'hotpotqa']


def mk_canaries(case_seed, n_rec=5, rng=None):
    """生成 n_rec 条 device-log 记录及其 canary 列表。"""
    rng = rng or random.Random(case_seed)
    recs, cans = [], []
    for i in range(n_rec):
        h = hashlib.sha256(('%s|%d' % (case_seed, i)).encode()).hexdigest()[:12]
        acct = 'ACCT-%s' % h
        email = 'contact.%s@example.invalid' % h[:10]
        phone = '+1-202-555-%04d' % (1000 + rng.randrange(9000))
        dev = 'sensor_%s_%02d' % (h[:4], rng.randrange(64))
        recs.append('{"device": "%s", "account": "%s", "contact": "%s", "phone": "%s"}'
                    % (dev, acct, email, phone))
        for t, v in (('account', acct), ('email', email), ('phone', phone)):
            cans.append(dict(type=t, value=v, marker=v, fragment_length=min(8, len(v))))
    return recs, cans


def build(per_task, out_path, seed=20261024):
    rng = random.Random(seed)
    rows = []
    for task in TASKS:
        p = '%s/%s.jsonl' % (LB, task)
        if not os.path.exists(p):
            print('  missing', p); continue
        src = [json.loads(x) for x in io.open(p, encoding='utf-8') if x.strip()]
        picked = src[:per_task]
        for i, r in enumerate(picked):
            key = 'lb-%s-%04d' % (task, i)
            recs, cans = mk_canaries(key, rng=rng)
            block = '\n----- device log -----\n' + '\n'.join(recs) + '\n----- end log -----\n'
            rows.append(dict(
                key=key,
                source=task,
                case_kind='single',
                dataset='LongBench',
                context=r['context'] + block,
                question=r['input'],
                answers=list(r['answers']),
                canaries=cans,
            ))
    with io.open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-task', type=int, default=100)
    ap.add_argument('--out', default=RD + '/data/public_lb.jsonl')
    a = ap.parse_args()
    rows = build(a.per_task, a.out)
    print('wrote %d cases -> %s' % (len(rows), a.out))
    by = {}
    for r in rows:
        by.setdefault(r['source'], []).append(len(r['context']))
    for t, ls in by.items():
        print('  %-18s n=%-4d 上下文长度 min=%d max=%d' % (t, len(ls), min(ls), max(ls)))
    r = rows[0]
    print('\n样例 key      :', r['key'])
    print('     question :', r['question'][:100])
    print('     answers  :', r['answers'])
    print('     canaries : %d 条，前 3 条 = %s' % (
        len(r['canaries']), [(c['type'], c['value']) for c in r['canaries'][:3]]))
    print('     注入块   :', r['context'][-220:].replace('\n', ' | '))


if __name__ == '__main__':
    main()
