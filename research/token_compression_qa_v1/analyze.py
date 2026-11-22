import csv
import itertools
import random
import statistics
from common import *
from selection import compression_path

def harmonic(a,b):return 0. if a+b==0 else 2*a*b/(a+b)

def main():
    cfg=read(HERE/'config.json');assert sha(SCORER)==cfg['scorer_sha256']
    metric=scorer();cases=read(HERE/'cases.json');rows=[];predictions={m:{} for m in METHODS}
    selected={'version':'1.1','data':[{'title':c['title'],'paragraphs':[{'context':c['source_context'],
        'qas':[{'id':c['id'],'question':c['question'],'answers':c['answers']}]}]} for c in cases]}
    paired=[]
    for c in cases:
        comps={m:read(compression_path(c['id'],m)) for m in METHODS}
        matched=all(r['status']=='ok' and r['budget_matched'] for r in comps.values())
        if matched:paired.append(c['id'])
        for m,comp in comps.items():
            cid=c['id']+'__'+m
            if comp['status']!='ok':
                predictions[m][c['id']]=''
                rows.append({'case':c['id'],'title':c['title'],'method':m,'status':'compression_error','paired':False,
                    'retention':None,'budget_matched':False,'em':0.,'f1':0.,'pin_full':None,'pin_fragment':None,
                    'pout_full':None,'pout_fragment':None,'joint_full':0.,'joint_fragment':0.,'input_tokens':0,'output_tokens':0,
                    'compression_seconds':comp['compression_seconds'],'truncated':False})
                continue
            task=read(HERE/'task'/(cid+'.json'));prediction=task['output'].strip();predictions[m][c['id']]=prediction
            em=max(metric.compute_exact(a['text'],prediction) for a in c['answers'])
            f1=max(metric.compute_f1(a['text'],prediction) for a in c['answers'])
            input_text='\n'.join(msg['content'] for msg in messages(c,comp['text']))
            pin=[exposure(s,input_text) for s in c['canaries']];pout=[exposure(s,prediction) for s in c['canaries']]
            pfull=1-statistics.mean(p[0] for p in pin);pfrag=1-statistics.mean(p[1] for p in pin)
            rows.append({'case':c['id'],'title':c['title'],'method':m,'status':'ok','paired':matched,
                'retention':comp['retention'],'budget_matched':comp['budget_matched'],'em':em,'f1':f1,
                'pin_full':pfull,'pin_fragment':pfrag,'pout_full':1-statistics.mean(p[0] for p in pout),
                'pout_fragment':1-statistics.mean(p[1] for p in pout),'joint_full':harmonic(f1,pfull),
                'joint_fragment':harmonic(f1,pfrag),'input_tokens':task['input_tokens'],'output_tokens':task['output_tokens'],
                'compression_seconds':comp['compression_seconds'],'truncated':not task['ended_with_eos']})
    # Independent aggregate cross-check through the author's full scoring function.
    official={}
    for m in METHODS:
        exact,f1=metric.get_raw_scores(selected['data'],predictions[m])
        rr=[r for r in rows if r['method']==m]
        assert all(r['em']==exact[r['case']] and abs(r['f1']-f1[r['case']])<1e-12 for r in rr)
        official[m]=metric.make_eval_dict(exact,f1)
    tables={}
    for scope in ['all24','paired_budget_matched']:
        table=[]
        for m in METHODS:
            rr=[r for r in rows if r['method']==m and (scope=='all24' or r['paired'])]
            s={'method':m,'n':len(rr),'budget_matched':sum(r['budget_matched'] for r in rr)}
            for key in ['retention','em','f1','pin_full','pin_fragment','pout_full','pout_fragment','joint_full','joint_fragment','input_tokens','output_tokens','compression_seconds']:
                s[key]=None if not rr or any(r[key] is None for r in rr) else statistics.mean(r[key] for r in rr)
            table.append(s)
        tables[scope]=table
    pairs=[];rng=random.Random(20260921)
    for a,b in itertools.combinations(METHODS[1:],2):
        ar={r['case']:r for r in rows if r['method']==a and r['paired']}
        br={r['case']:r for r in rows if r['method']==b and r['paired']}
        if not ar:continue
        for key in ['f1','joint_full','joint_fragment']:
            ds=[ar[k][key]-br[k][key] for k in ar]
            boot=sorted(statistics.mean(rng.choices(ds,k=len(ds))) for _ in range(5000))
            pairs.append({'a':a,'b':b,'metric':key,'n':len(ds),'difference_points':100*statistics.mean(ds),
                'exploratory_bootstrap95_points':[100*boot[124],100*boot[4874]]})
    result={'config_sha256':sha(HERE/'config.json'),'compression_selection_sha256':sha(HERE/'compression_selection.json'),'rows':rows,'tables':tables,'paired_ids':paired,
        'official_scorer_crosscheck':official,'exploratory_paired_intervals':pairs,
        'truncated_outputs':sum(r['truncated'] for r in rows)}
    write_new(HERE/'summary.json',result)
    with (HERE/'per_case.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def pct(v):return 'NA' if v is None else f'{100*v:.1f}'
    lines=['# 客观问答与可控标识符残留验证','',
        'Qwen2.5-7B-Instruct；SQuAD 1.1 dev 的 24 个固定哈希问题，来自 24 篇文章。每题添加 3 个虚构标识符，位置在前/中/后轮换。',
        '**这是一套人工构造的隐私代理测试，不是 SQuAD 原生隐私标注，也不能替代 PrivacyLens 的语义泄露测量。** 任务正确性由标准答案计算 EM/F1，不使用 LLM 评分。','',
        f'主要配对比较包含全部压缩算法均达到 48%–50% 保留率的 {len(paired)}/24 条。预算筛选在查看任务成绩前定义，所有 24 条结果另列。','',
        '| 算法 | n | 保留率 | EM | F1 | 完整值去除 | 片段去除 | J（完整值） | J（片段） |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in tables['paired_budget_matched']:
        lines.append(f"| {s['method']} | {s['n']} | {pct(s['retention'])}% | {pct(s['em'])} | {pct(s['f1'])} | {pct(s['pin_full'])} | {pct(s['pin_fragment'])} | {pct(s['joint_full'])} | {pct(s['joint_fragment'])} |")
    lines+=['','## 全部 24 条（含预算偏差）','','| 算法 | 达标数 | 平均保留率 | EM | F1 | 完整值去除 | 片段去除 | J（完整值） |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in tables['all24']:
        lines.append(f"| {s['method']} | {s['budget_matched']}/24 | {pct(s['retention'])}% | {pct(s['em'])} | {pct(s['f1'])} | {pct(s['pin_full'])} | {pct(s['pin_fragment'])} | {pct(s['joint_full'])} |")
    lines+=['','## 口径与限制','',
        '- 完整值去除：原始虚构邮箱、电话、账号值在小写化和移除标点/空白后不可完整匹配的比例。这只是确定性的完整值残留代理，不等于语义不可恢复。',
        '- 片段去除：邮箱/账号的随机核心至少连续 6 个字符、电话至少连续 8 位数字均不再残留的比例。它是额外保守诊断，也不是正式隐私保证。',
        '- J 为逐例 F1 和对应输入去除率的调和平均，再平均；全部分数 0–100。输出残留另存 summary.json。',
        '- LLMLingua-2 使用原始 MeetingBank 模型；LongLLMLingua 为 GPT2 小模型适配、单上下文，迭代块为 50；Selective Context 为作者 token 级实现。没有用近似算法冒充原实现。',
        '- 初次 LongLLMLingua 的 200-token 块在 5 条短输入上未压缩。生成任何任务答案前，另行冻结了 50-token 块配置，统一重跑该方法全部 24 条；旧输出和错误完整保留于 compressed/，采用的版本见 compression_selection.json 和 compressed_long50/。',
        '- mask_llmlingua2 是固定的常规格式正则脱敏，再用相同 LLMLingua-2 压缩；预算始终按未脱敏的原始输入计算，算法不读取金标标识符列表。它是简单对照，不是新算法或论文贡献。',
        '- 虚构标识符与问答任务无关且格式规范，规则脱敏在这里有明显的先验优势。尚未测试任务必需信息与隐私冲突、自然敏感属性、攻击者推断或格式混淆。',
        '- 只有一个下游模型、一个保留率和 24 道开发题；公开问答可能与预训练数据重合。不能据此宣称泛化优势或 SOTA。',
        '- 评分器源代码来自官方 SQuAD v2 脚本的 EM/F1 函数，应用于有答案的 v1.1 样本；所选金标归一化后均非空。逐条分数已与作者 get_raw_scores 独立交叉检查。',
        '- 全部压缩器在 CPU 上执行，推理 GPU 与另一个作业共享；耗时只作本次成本记录，不据此比较端到端速度。',
        '- 源数据及改造说明遵循源数据 CC BY-SA 4.0 归属要求。原始数据、脚本版本和哈希见 config.json 与 external/squad_v11/provenance.json。','',
        '来源：[SQuAD 官方项目](https://rajpurkar.github.io/SQuAD-explorer/)、[固定版本评分脚本](https://github.com/rajpurkar/SQuAD-explorer/blob/09eac9971f46889fa057ff2c870bf71092ba9d55/evaluate-v2.0.py)。']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'paired_n':len(paired),'tables':tables,'truncated':result['truncated_outputs']},indent=2))

if __name__=='__main__':main()
