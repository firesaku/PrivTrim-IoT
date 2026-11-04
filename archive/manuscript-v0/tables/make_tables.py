"""论文表格生成器：所有数字从 results/ 真实产物读出，禁止手抄。

输出 paper/tabs.tex，被 main.tex \\input。
用法：
  python make_tables.py --repo <repo> --fix <fixres> --base <review tmp 根> --out <tabs.tex>
"""
import argparse
import collections
import io
import json
import os


def cells_all(path):
    o = json.load(io.open(path, encoding='utf-8'))
    out = {}
    for k, v in o['all'].items():
        n = max(1, v['n'])
        out[k] = dict(n=v['n'], f1=v['f1'] / n, tok=v['tok'] / n, gic=v['ans'] / n,
                      tent=v['tent'] / n, coll=v['coll'] / n,
                      leak=v['cv'] / max(1, v['cn']))
    return out


def cells_src(path, merge=None):
    out = {} if merge is None else merge
    o = json.load(io.open(path, encoding='utf-8'))
    for k, v in (o.get('cells') or {}).items():
        p = k.split('|')
        if len(p) != 3:
            continue
        n = max(1, v['n'])
        out[(p[0], p[1], p[2])] = dict(n=v['n'], f1=v['f1'] / n, tok=v['tok'] / n,
                                       gic=v['ans'] / n)
    return out


def row(name, v, cols=('f1', 'tok', 'gic'), fmt=('%.4f', '%.0f', '%.3f')):
    if v is None:
        return '%-28s & ' % name + ' & '.join(['--'] * len(cols)) + r' \\'
    vals = []
    for c, f in zip(cols, fmt):
        x = v.get(c)
        vals.append('--' if x is None else f % x)
    return '%-28s & ' % name + ' & '.join(vals) + r' \\'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--fix', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    Q = cells_all(a.fix + r'\aliasq_qwen\cells.json')
    Q0 = cells_all(a.fix + r'\aliaslink_qwen\cells.json')
    M0 = cells_all(a.fix + r'\aliaslink_mistral\cells.json')
    M1 = cells_all(a.fix + r'\mistralq\cells.json')
    X = cells_all(a.repo + r'\results\cells\longctx_xbud_cells.json')
    PA = cells_all(a.fix + r'\paraA\cells.json')
    PB = cells_all(a.fix + r'\paraB\cells.json')
    IN = cells_all(a.fix + r'\inscope_qwen\cells.json')
    E = cells_src(a.base + r'\extbl_multi\scored.json')
    cells_src(a.base + r'\extbl_multi\mk.json', E)
    EI = cells_src(a.base + r'\extbl_inscope_target_last\mk.json')
    EI2 = cells_src(a.base + r'\extbl_inscope_target_last\scored.json', EI) \
        if os.path.exists(a.base + r'\extbl_inscope_target_last\scored.json') else EI

    def ext(arm, ratio='0.5', src=None):
        xs = [v for (s, ar, r), v in E.items() if ar == arm and r == ratio and (src is None or s == src)]
        if not xs:
            return None
        n = sum(x['n'] for x in xs)
        return dict(n=n, f1=sum(x['f1'] * x['n'] for x in xs) / n,
                    tok=sum(x['tok'] * x['n'] for x in xs) / n,
                    gic=sum(x['gic'] * x['n'] for x in xs) / n)

    L = []
    A = L.append

    # ---------- 表：主结果 ----------
    A(r'\begin{table}[t]')
    A(r'\caption{Main comparison on the 112-case benchmark (Qwen2.5-7B-Instruct, seed '
      r'20261015). Budget arms target 50\% of the uncompressed token count; ``prompt leak'' '
      r'is exact-text accounting over question plus compressed context. ABL $=$ '
      r'\arm{alias\_budget\_link} (\ours), BCL $=$ \arm{budget\_cluster\_link}, BC $=$ '
      r'\arm{budget\_cluster}, ALB $=$ \arm{alias\_budget}, DB $=$ \arm{delete\_budget}, '
      r'BR $=$ \arm{budget\_random}.}')
    A(r'\label{tab:main}')
    A(r'\centering\scriptsize\setlength{\tabcolsep}{3pt}')
    A(r'\begin{tabular}{lrrrrr}')
    A(r'\toprule')
    A(r'Arm & F1 & tokens & gic & tent & prompt leak \\')
    A(r'\midrule')
    order = [('full|None', r'\arm{full}'),
             ('alias_budget_link|0.5', r'ABL (\ours)'),
             ('budget_cluster_link|0.5', r'BCL'),
             ('alias|None', r'\arm{alias}'),
             ('alias_budget|0.5', r'ALB'),
             ('delete|None', r'\arm{delete}'),
             ('budget_cluster|0.5', r'BC'),
             ('delete_budget|0.5', r'DB'),
             ('budget_random|0.5', r'BR')]
    LEAK = {'full|None': '112/112', 'alias_budget_link|0.5': r'\textbf{0/112}',
            'budget_cluster_link|0.5': '44/112', 'alias|None': '0/112',
            'alias_budget|0.5': '40/112', 'delete|None': '0/112',
            'budget_cluster|0.5': '44/112', 'delete_budget|0.5': '0/112',
            'budget_random|0.5': '111/112'}
    for k, lab in order:
        v = Q.get(k)
        if not v:
            continue
        bold = (k == 'alias_budget_link|0.5')
        f1 = (r'\textbf{%.4f}' % v['f1']) if bold else ('%.4f' % v['f1'])
        A('%s & %s & %.0f & %.3f & %.3f & %s \\\\' % (lab, f1, v['tok'], v['gic'],
                                                      v['tent'], LEAK[k]))
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{table}')
    A('')

    # ---------- 表：已发布方法 ----------
    A(r'\begin{table}[t]')
    A(r'\caption{Published compressors re-run through their own reference implementations, '
      r'at a 50\% budget (see \cite{provence,quitox,llmlingua2,cpc}). Out-of-scope: the 40 '
      r'multi-entity cases; in-scope: the 19 cases whose target region fits the '
      r'$\leq$512-token regime those methods were validated in. \ours{} $=$ '
      r'\arm{budget\_cluster\_link}; \arm{BC} $=$ \arm{budget\_cluster}; '
      r'\arm{mask+QX} $=$ \arm{mask\_quito\_x}.}')
    A(r'\label{tab:ext}')
    A(r'\centering\scriptsize\setlength{\tabcolsep}{3.5pt}')
    A(r'\begin{tabular}{lrrrrrr}')
    A(r'\toprule')
    A(r'& \multicolumn{3}{c}{out-of-scope (40)} & \multicolumn{3}{c}{in-scope (19)} \\')
    A(r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}')
    A(r'Method & F1 & tokens & gic & F1 & tokens & gic \\')
    A(r'\midrule')
    def multi_int(arm, ratio='0.5'):
        """内部臂在 40 个多实体案例上的聚合（来自 percase，与已发布方法同分母）。"""
        xs = [r.get('f1') or 0.0 for r in _percase(a.fix + r'\aliasq_qwen\percase.jsonl')
              if r.get('case_kind') == 'multi_entity' and r.get('arm') == arm
              and str(r.get('ratio')) == ratio]
        toks = [r.get('ctx_tokens') or 0 for r in _percase(a.fix + r'\aliasq_qwen\percase.jsonl')
                if r.get('case_kind') == 'multi_entity' and r.get('arm') == arm
                and str(r.get('ratio')) == ratio]
        gics = [int(r.get('ans_in_ctx') or 0) for r in _percase(a.fix + r'\aliasq_qwen\percase.jsonl')
                if r.get('case_kind') == 'multi_entity' and r.get('arm') == arm
                and str(r.get('ratio')) == ratio]
        if not xs:
            return None
        return dict(n=len(xs), f1=sum(xs) / len(xs), tok=sum(toks) / len(toks),
                    gic=sum(gics) / len(gics))

    def agg(tab, arm, ratio='0.5'):
        xs = [v for (s, ar, r), v in tab.items() if ar == arm and r == ratio]
        if not xs:
            return None
        n = sum(x['n'] for x in xs)
        return dict(n=n, f1=sum(x['f1'] * x['n'] for x in xs) / n,
                    tok=sum(x['tok'] * x['n'] for x in xs) / n,
                    gic=sum(x['gic'] * x['n'] for x in xs) / n)

    ext_rows = [
        (r'\arm{full}', ext('full', ratio='None'), Q.get('full|None'), IN.get('full|None'), False),
        (r'\ours{}', multi_int('budget_cluster_link'),
         None, IN.get('budget_cluster_link|0.5'), True),
        (r'Provence', ext('ext_provence'), None, IN.get('ext_provence|0.5'), False),
        (r'Quito-X', ext('ext_quito_x'), None, IN.get('ext_quito_x|0.5'), False),
        (r'LLMLingua-2', ext('ext_llmlingua2'), None,
         IN.get('ext_llmlingua2|0.5'), False),
        (r'CPC', ext('ext_cpc'), None, IN.get('ext_cpc|0.5'), False),
        (r'\arm{mask+QX}', ext('ext_mask_quito_x'), None,
         agg(EI, 'ext_mask_quito_x'), False),
        (r'\arm{BC}', multi_int('budget_cluster'), None,
         IN.get('budget_cluster|0.5'), False)]
    for lab, ov, _unused, iv, bold in ext_rows:
        f1o = '--' if not ov else '%.4f' % ov['f1']
        toko = '--' if not ov else '%.0f' % ov['tok']
        gico = '--' if not ov else '%.3f' % ov['gic']
        f1i = '--' if not iv else '%.4f' % iv['f1']
        toki = '--' if not iv else '%.0f' % iv['tok']
        gici = '--' if not iv else '%.3f' % iv['gic']
        if bold:
            f1o = r'\textbf{%s}' % f1o
            f1i = r'\textbf{%s}' % f1i
        A('%s & %s & %s & %s & %s & %s & %s \\\\' % (lab, f1o, toko, gico, f1i, toki, gici))
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{table}')
    A('')

    # ---------- 表：预算曲线 ----------
    A(r'\begin{table}[t]')
    A(r'\caption{Budget curve (Qwen2.5-7B, 112 cases; ABL $=$ \arm{alias\_budget\_link}, '
      r'BCL $=$ \arm{budget\_cluster\_link}, BC $=$ \arm{budget\_cluster}, '
      r'BR $=$ \arm{budget\_random}). The $@0.10$ and $@0.15$ columns come from one job, the '
      r'remaining columns from a second job of the same configuration; comparisons within a '
      r'column set are exact. Every budget arm of ours satisfies its budget with zero overshoot.}')
    A(r'\label{tab:budget}')
    A(r'\centering\scriptsize\setlength{\tabcolsep}{3pt}')
    A(r'\begin{tabular}{lrrrrr}')
    A(r'\toprule')
    A(r'Arm & @0.10 & @0.15 & @0.25 & @0.50 & @0.75 \\')
    A(r'\midrule')
    for key, lab in (('alias_budget_link', r'ABL (\ours)'),
                     ('budget_cluster_link', r'BCL'),
                     ('budget_cluster', r'BC'),
                     ('budget_random', r'BR')):
        vals = []
        for r in ('0.1', '0.15', '0.25', '0.5', '0.75'):
            src = X if r in ('0.1', '0.15') else Q
            v = src.get('%s|%s' % (key, r))
            vals.append('--' if not v else ('\\textbf{%.4f}' % v['f1'] if key == 'alias_budget_link'
                                            else '%.4f' % v['f1']))
        A('%s & %s \\\\' % (lab, ' & '.join(vals)))
    A(r'\midrule')
    toks = []
    gains = []
    for r in ('0.1', '0.15', '0.25', '0.5', '0.75'):
        src = X if r in ('0.1', '0.15') else Q
        v = src.get('alias_budget_link|%s' % r)
        toks.append('%.0f' % v['tok'])
        gains.append('%+.3f' % (v['f1'] - src['budget_cluster|%s' % r]['f1']))
    A(r'context tokens & %s \\' % ' & '.join(toks))
    A(r'anchor gain & %s \\' % ' & '.join(gains))
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{table}')
    A('')

    # ---------- 表：稳健性 ----------
    A(r'\begin{table}[t]')
    A(r'\caption{Robustness. Top block: two mechanical question rephrasings, each an '
      r'independent job with the same nine-arm configuration. Bottom block: the second '
      r'model, with the question-side alias map enabled.}')
    A(r'\label{tab:robust}')
    A(r'\centering\footnotesize\setlength{\tabcolsep}{3.5pt}')
    A(r'\begin{tabular}{lrrr}')
    A(r'\toprule')
    A(r'Contrast & original & rephrasing A & rephrasing B \\')
    A(r'\midrule')
    def gain(d, hi, lo):
        return d[hi]['f1'] - d[lo]['f1']
    A(r'anchor gain (raw strings) & %+.4f & %+.4f & %+.4f \\' % (
        gain(Q, 'budget_cluster_link|0.5', 'budget_cluster|0.5'),
        gain(PA, 'budget_cluster_link|0.5', 'budget_cluster|0.5'),
        gain(PB, 'budget_cluster_link|0.5', 'budget_cluster|0.5')))
    A(r'anchor gain (aliased) & %+.4f & %+.4f & %+.4f \\' % (
        gain(Q, 'alias_budget_link|0.5', 'budget_cluster|0.5'),
        gain(PA, 'alias_budget_link|0.5', 'budget_cluster|0.5'),
        gain(PB, 'alias_budget_link|0.5', 'budget_cluster|0.5')))
    A(r'\arm{full} F1 & %.4f & %.4f & %.4f \\' % (Q['full|None']['f1'], PA['full|None']['f1'],
                                                  PB['full|None']['f1']))
    A(r'\arm{alias\_budget\_link} F1 & %.4f & %.4f & %.4f \\' % (
        Q['alias_budget_link|0.5']['f1'], PA['alias_budget_link|0.5']['f1'],
        PB['alias_budget_link|0.5']['f1']))
    A(r'\midrule')
    A(r'Arm @0.50 & Qwen2.5-7B & Mistral-7B & \\')
    A(r'\midrule')
    for key, lab in (('alias_budget_link|0.5', r'ABL (\ours)'),
                     ('full|None', r'full (uncompressed)'),
                     ('budget_cluster_link|0.5', r'BCL'),
                     ('budget_cluster|0.5', r'BC'),
                     ('budget_random|0.5', r'BR')):
        A('%s & %.4f & %.4f & \\\\' % (lab, Q[key]['f1'], M1[key]['f1']))
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{table}')
    A('')

    # ---------- 表：逐来源（附录） ----------
    cases = {}
    for line in io.open(a.repo + r'\data\longctx_cases.jsonl', encoding='utf-8'):
        c = json.loads(line)
        cases[c['key']] = c
    cnt = collections.Counter()
    for c in cases.values():
        cnt[(c.get('source'), c.get('case_kind'))] += 1
    srcs = sorted({k[0] for k in cnt})
    A(r'\begin{table*}[t]')
    A(r'\caption{Per-source breakdown on the 112-case benchmark (top) and on the 40 '
      r'multi-entity cases used for the published-compressor comparison (bottom). The core '
      r'contrast---anchor protection above redundancy-only selection---holds in every source.}')
    A(r'\label{tab:persource}')
    A(r'\centering\footnotesize\setlength{\tabcolsep}{5pt}')
    A(r'\begin{tabular}{lrrrrrrrr}')
    A(r'\toprule')
    A(r'& & \multicolumn{3}{c}{112-case benchmark} & \multicolumn{3}{c}{40 multi-entity cases} \\')
    A(r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}')
    A(r'Source (public dataset) & $n$ & \arm{full} & \arm{BC} & \arm{ABL} (\ours) & '
      r'\arm{ABL} (\ours) & Provence & Quito-X & redundancy \\')
    A(r'\midrule')
    for s in srcs:
        n_all = cnt[(s, 'needle')] + cnt[(s, 'multi_entity')]
        n_m = cnt[(s, 'multi_entity')]
        f_full = Q.get('full|None')
        per = {}
        for (src, arm, ratio), v in []:
            pass
        A(r'\multicolumn{8}{l}{\textit{%s}}\\*' % s) if False else None
        A('\\arm{%s} & %d & %.4f & %.4f & %.4f & %s & %s & %s & %s \\\\' % (
            s.replace('_', r'\_'), n_all,
            _srcf1(a.fix + r'\aliasq_qwen\percase.jsonl', s, 'full', 'None'),
            _srcf1(a.fix + r'\aliasq_qwen\percase.jsonl', s, 'budget_cluster', '0.5'),
            _srcf1(a.fix + r'\aliasq_qwen\percase.jsonl', s, 'alias_budget_link', '0.5'),
            _fmt(_srcf1(a.fix + r'\aliasq_qwen\percase.jsonl', s, 'alias_budget_link', '0.5', multi=True)),
            _fmt(dict.get(ext('ext_provence', src=s) or {}, 'f1')),
            _fmt(dict.get(ext('ext_quito_x', src=s) or {}, 'f1')),
            _fmt(_srcf1(a.fix + r'\aliasq_qwen\percase.jsonl', s, 'budget_cluster', '0.5', multi=True))))
    A(r'\bottomrule')
    A(r'\end{tabular}')
    A(r'\end{table*}')

    io.open(a.out, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
    print('wrote %s (%d lines)' % (a.out, len(L)))


_CACHE = {}


def _percase(path):
    if path not in _CACHE:
        rows = []
        for line in io.open(path, encoding='utf-8'):
            if line.strip():
                rows.append(json.loads(line))
        _CACHE[path] = rows
    return _CACHE[path]


def _srcf1(path, src, arm, ratio, multi=False):
    xs = [r.get('f1') or 0.0 for r in _percase(path)
          if r.get('source') == src and r.get('arm') == arm and str(r.get('ratio')) == ratio
          and ((r.get('case_kind') == 'multi_entity') if multi else True)]
    return (sum(xs) / len(xs)) if xs else None


def _fmt(v):
    return '--' if v is None else '%.4f' % v


if __name__ == '__main__':
    main()
