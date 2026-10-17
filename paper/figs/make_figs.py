"""论文用图：全部数字来自 results/ 下的真实产物，不手填。

输出：paper/figs/fig2_pareto.pdf, fig3_mechanism.pdf, fig4_strata_budget.pdf
用法：
  python make_figs.py --repo <repo root> --fix <fixres 根> --ext <extbl_multi 目录>
"""
import argparse
import io
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_OURS = '#2f5d8a'      # 我们（深蓝）
C_OURS2 = '#7ba7cc'     # 我们的第二条臂（浅蓝）
C_BASE = '#b0b0b0'      # 已发布方法（灰）
C_BAD = '#c0504d'       # 失效臂（暗红）
C_REF = '#4f4f4f'       # 不压缩参照（深灰）
plt.rcParams.update({'font.size': 8, 'axes.grid': True, 'grid.alpha': 0.25,
                     'grid.linewidth': 0.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 300})


def load_cells(path):
    o = json.load(io.open(path, encoding='utf-8'))
    out = {}
    for k, v in o['all'].items():
        n = max(1, v['n'])
        out[k] = dict(n=v['n'], f1=v['f1'] / n, tok=v['tok'] / n, gic=v['ans'] / n,
                      tent=v['tent'] / n, leak=v['cv'] / max(1, v['cn']))
    return out


def load_cells_by_src(path):
    o = json.load(io.open(path, encoding='utf-8'))
    out = {}
    for k, v in (o.get('cells') or {}).items():
        p = k.split('|')
        if len(p) != 3:
            continue
        n = max(1, v['n'])
        out[(p[0], p[1], p[2])] = dict(n=v['n'], f1=v['f1'] / n, gic=v['ans'] / n,
                                       tok=v['tok'] / n, leak=v['cv'] / max(1, v['cn']))
    return out


def ext_point(E, arm, ratio='0.5'):
    xs = [v for (s, a, r), v in E.items() if a == arm and r == ratio]
    if not xs:
        return None
    n = sum(x['n'] for x in xs)
    return dict(n=n, f1=sum(x['f1'] * x['n'] for x in xs) / n,
                tok=sum(x['tok'] * x['n'] for x in xs) / n,
                leak=sum((x.get('leak') or 0) * x['n'] for x in xs) / n,
                gic=sum(x['gic'] * x['n'] for x in xs) / n)


def fig2_pareto(Q, E, out):
    """F1 vs 上下文 token：隐私-效用-成本的位置一图看清（40 多实体案例）。"""
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    f_ref, t_ref = Q['full|None']['f1'], Q['full|None']['tok']
    ax.axhline(f_ref, color=C_REF, lw=0.8, ls='--')
    ax.scatter([t_ref], [f_ref], color=C_REF, marker='P', s=34, zorder=4,
               edgecolor='white', linewidth=0.6, label='uncompressed (leak 100%)')
    for arm, lab, col, mk in (
            ('alias_budget_link', 'ours: alias+anchor (leak 0%)', C_OURS, 'o'),
            ('budget_cluster_link', 'ours: anchor only (leak 6%)', C_OURS2, 's'),
            ('budget_cluster', 'redundancy only (leak 3%)', C_BAD, 'v'),
            ('budget_random', 'random drop (leak 51%)', C_BAD, '^')):
        v = Q.get(arm + '|0.5')
        if v:
            ax.scatter([v['tok']], [v['f1']], color=col, marker=mk, s=32, zorder=4,
                       edgecolor='white', linewidth=0.6, label=lab)
    for arm, lab in (('ext_provence', 'Provence'), ('ext_llmlingua2', 'LLMLingua-2'),
                     ('ext_cpc', 'CPC'), ('ext_mask_quito_x', 'mask+Quito-X')):
        v = ext_point(E, arm)
        if v:
            ax.scatter([v['tok']], [v['f1']], color=C_BASE, marker='D', s=26, zorder=3,
                       edgecolor='white', linewidth=0.6, label=lab)
    ax.annotate('F1 of the uncompressed prompt', (t_ref, f_ref), textcoords='offset points',
                xytext=(-118, -14), fontsize=6.2, color=C_REF,
                arrowprops=dict(arrowstyle='-', color=C_REF, lw=0.5))
    ax.set_xlabel('context tokens fed to the model')
    ax.set_ylabel('F1 @50% budget')
    ax.set_ylim(0, 0.86)
    ax.set_xlim(0, 9400)
    ax.legend(fontsize=5.4, frameon=False, loc='lower left', handletextpad=0.4,
              borderpad=0.1, labelspacing=0.25)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'), dpi=300)
    plt.close(fig)
    print('wrote', out)


def fig3_mechanism(Q, E, out):
    """机制：答案还在上下文里（gic=1）却答不对 —— 纯冗余度与 mask 类方法落在右下角。"""
    pts = []
    for arm, lab, col, mk in (
            ('alias_budget_link', 'ours: alias+anchor', C_OURS, 'o'),
            ('budget_cluster_link', 'ours: anchor only', C_OURS2, 's'),
            ('budget_cluster', 'redundancy only', C_BAD, 'v'),
            ('budget_random', 'random drop', C_BAD, '^'),
            ('full', 'uncompressed', C_REF, 'P')):
        v = Q.get(arm + '|None') if arm in ('full',) else Q.get(arm + '|0.5')
        if v:
            pts.append((v['gic'], v['f1'], lab, col, mk))
    for arm, lab in (('ext_mask_quito_x', 'mask+Quito-X'), ('ext_mask_llmlingua2', 'mask+LLMLingua-2'),
                     ('ext_provence', 'Provence'), ('ext_llmlingua2', 'LLMLingua-2'), ('ext_cpc', 'CPC')):
        xs = [v for (s, a, r), v in E.items() if a == arm and r == '0.5']
        if not xs:
            continue
        n = sum(x['n'] for x in xs)
        pts.append((sum(x['gic'] * x['n'] for x in xs) / n,
                    sum(x['f1'] * x['n'] for x in xs) / n, lab, C_BASE, 'D'))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for gic, f1, lab, col, mk in pts:
        ax.scatter([gic], [f1], color=col, marker=mk, s=30, zorder=3,
                   edgecolor='white', linewidth=0.6)
    seen = {}
    for gic, f1, lab, col, mk in sorted(pts, key=lambda x: -x[1]):
        dy = seen.get(round(gic, 1), 0)
        seen[round(gic, 1)] = dy + 1
        ax.annotate(lab, (gic, f1), textcoords='offset points', xytext=(4, 3 - 8 * dy),
                    fontsize=6.5, color='#333333')
    ax.axvspan(0.95, 1.02, color='#eeeeee', zorder=0)
    ax.text(0.985, 0.03, 'answer still in context', rotation=90, fontsize=6,
            color='#777777', ha='center')
    ax.set_xlabel('gold span still present in compressed context (gic)')
    ax.set_ylabel('F1 @50%')
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.03, 0.82)
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'), dpi=300)
    plt.close(fig)
    print('wrote', out)


def fig4_strata_budget(Q, M, S, out):
    """左：预算曲线；右：按答案位置分层（我们 vs 已发布）。"""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax = axes[0]
    xs = [0.10, 0.15, 0.25, 0.50, 0.75]
    series = (('alias_budget_link', 'ours: alias+anchor', C_OURS, 'o'),
              ('budget_cluster_link', 'ours: anchor only', C_OURS2, 's'),
              ('budget_cluster', 'redundancy only', C_BAD, 'v'),
              ('budget_random', 'random drop', C_BAD, '^'))
    for arm, lab, col, mk in series:
        ys, ts = [], []
        for r in xs:
            v = Q.get('%s|%s' % (arm, r))
            if v:
                ys.append(v['f1'])
                ts.append(v['tok'])
        if ys:
            ax.plot(ts, ys, marker=mk, color=col, lw=1.3, ms=4, label=lab)
    v = Q.get('full|None')
    ax.axhline(v['f1'], color=C_REF, lw=0.8, ls='--')
    ax.text(700, v['f1'] + 0.015, 'uncompressed', color=C_REF, fontsize=6.5)
    ax.set_xscale('log')
    ax.set_xlabel('context tokens (log scale)')
    ax.set_ylabel('F1')
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=6, frameon=False, loc='lower right')

    ax = axes[1]
    names = ['head\n(<1/3)', 'middle', 'tail\n(>2/3)']
    lab_series = [('alias_budget_link', 'ours: alias+anchor', C_OURS),
                  ('budget_cluster_link', 'ours: anchor only', C_OURS2),
                  ('ext_provence', 'Provence', C_BASE),
                  ('ext_quito_x', 'Quito-X', C_BASE)]
    import numpy as np
    w = 0.2
    for i, (arm, lab, col) in enumerate(lab_series):
        vals = []
        for j in range(3):
            vals.append(S.get((j, arm)))
        xs2 = [j + (i - 1.5) * w for j in range(3)]
        ax.bar(xs2, [0 if v is None else v for v in vals], width=w, color=col, label=lab,
               edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel('F1 @50%')
    ax.set_ylim(0, 0.9)
    ax.legend(fontsize=6, frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'), dpi=300)
    plt.close(fig)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--fix', required=True)
    ap.add_argument('--ext', required=True)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    Q = load_cells(a.repo + r'\results\cells\longctx_aliaslink_cells.json')
    Qa = load_cells(a.fix + r'\aliasq_qwen\cells.json')
    E = load_cells_by_src(a.ext + r'\scored.json')
    E.update(load_cells_by_src(a.ext + r'\mk.json'))
    M = load_cells(a.fix + r'\mistralq\cells.json')

    # 预算曲线用 aliasq（0.25/0.5/0.75）+ xbud（0.10/0.15）
    X = load_cells(a.repo + r'\results\cells\longctx_xbud_cells.json')
    Qc = dict(Qa)
    for k, v in X.items():
        if k.endswith('|0.1') or k.endswith('|0.15'):
            Qc[k] = v
    Qc['full|None'] = Qa['full|None']
    Qc['alias|None'] = Qa['alias|None']

    # 优势分层：位置层 × 臂（从产物 jsonl 现算，保证与报告一致）
    S = {}
    cases = {}
    for line in io.open(a.repo + r'\data\longctx_cases.jsonl', encoding='utf-8'):
        c = json.loads(line)
        if c.get('case_kind') != 'multi_entity':
            continue
        off = c.get('answer_char_offset')
        if off is None:
            off = c.get('entity_offset')     # 与 advantage_strata.py 同口径
        rel = (off / max(1, len(c['context']))) if off is not None else None
        cases[c['key']] = rel
    rows = []
    for p in (a.fix + r'\aliasq_qwen\percase.jsonl', a.ext + r'\scored.jsonl', a.ext + r'\mk.jsonl'):
        for line in io.open(p, encoding='utf-8'):
            r = json.loads(line)
            if r.get('case_kind') != 'multi_entity' or str(r.get('ratio')) != '0.5':
                continue
            rows.append(r)
    for r in rows:
        rel = cases.get(r['case_key'])
        if rel is None:
            continue
        j = 0 if rel < 1 / 3 else (1 if rel < 2 / 3 else 2)
        S.setdefault((j, r['arm']), []).append(r.get('f1') or 0.0)
    S = {k: sum(v) / len(v) for k, v in S.items() if v}

    fig2_pareto(Q, E, a.outdir + r'\fig2_pareto.pdf')
    fig3_mechanism(Q, E, a.outdir + r'\fig3_mechanism.pdf')
    fig4_strata_budget(Qc, M, S, a.outdir + r'\fig4_strata_budget.pdf')
    print('tails: ours=%.4f provence=%.4f quito=%.4f n_opts=%d' % (
        S.get((2, 'alias_budget_link'), float('nan')), S.get((2, 'ext_provence'), float('nan')),
        S.get((2, 'ext_quito_x'), float('nan')), len(S)))


if __name__ == '__main__':
    main()
