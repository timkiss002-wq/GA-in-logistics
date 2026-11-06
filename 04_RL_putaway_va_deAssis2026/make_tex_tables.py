"""Sinh các bảng LaTeX của bài v6 từ results/lp/summary.json (không chép số bằng tay)."""
import json
import numpy as np, pandas as pd

S = json.load(open('results/lp/summary.json'))
f3 = lambda x: f'{x:.3f}'
pm = lambda m, s, d=3: f'{m:.{d}f}$\\pm${s:.{d}f}'
out = {}

# Bảng kết quả chính tại λ = 0,5
rows = []
for r in S['table_main']:
    if 'J_sd' in r and r['J_sd'] == r['J_sd'] and r['policy'].startswith('LP'):
        rows.append(f"\\midrule\n\\textbf{{LP-RL (ours)}} & {pm(r['MI'], r['MI_sd'])} & {r['T']:,.0f}$\\pm${r['T_sd']:.0f} & \\textbf{{{pm(r['J'], r['J_sd'])}}} \\\\")
    else:
        rows.append(f"{r['policy']} & {r['MI']:.3f} & {r['T']:,.0f} & {r['J']:.3f} \\\\")
out['main'] = r"""\begin{table}[t]
\caption{Test results at $\lambda=0.5$ (2,913 waves). LP-RL: mean $\pm$ s.d.\ over 5 training seeds}
\label{tab:main}
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrr@{}}
\toprule
Policy & Mixing $MI$ & Route $T$ & $J(0.5)$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\\[2pt]\raggedright\scriptsize Random and the de Assis agents: mean of 5 seeds. $T$ in dataset units per wave.
\end{table}"""

# Bảng theo λ
rows = []
for r in S['table_lambda']:
    win = r['better'] == 5
    b = (lambda x: f'\\textbf{{{x}}}') if win else (lambda x: x)
    rows.append(f"{b(f'{r['lam']:.1f}')} & {r['best']} & {r['J_best']:.3f} & {pm(r['J_lp'], r['J_lp_sd'])} & "
                f"{b(f'${r['dJ']:+.3f}\\pm{r['dJ_sd']:.3f}$')} & {r['better']} / {r['worse']} \\\\")
out['lambda'] = r"""\begin{table}[t]
\caption{LP-RL vs.\ the best rule-based baseline at each $\lambda$ on the test waves ($\Delta J<0$ favours LP-RL; 5 seeds)}
\label{tab:lambda}
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}rlrrrc@{}}
\toprule
$\lambda$ & Best baseline & $J$ base & $J$ LP-RL & $\Delta J$ & Better/worse \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\\[2pt]\raggedright\scriptsize Better / worse: seeds with a significant difference (paired Wilcoxon over waves, Holm over all 55 tests, $\alpha=0.05$).
\end{table}"""

# Bảng ablation
rows = []
for r in S['table_ablation']:
    rows.append(f"{r['variant']} & ${r['d0.2']:+.3f}$ & ${r['d0.5']:+.3f}$ & ${r['d0.8']:+.3f}$ & ${r['mean']:+.3f}\\pm{r['sd']:.3f}$ \\\\")
out['ablation'] = r"""\begin{table}[t]
\caption{Ablation on the test waves ($\Delta J$ = variant $-$ LP-RL, same seed; positive means the removed part helps; 2 seeds)}
\label{tab:ablation}
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Variant & $\lambda=0.2$ & $0.5$ & $0.8$ & Mean $\pm$ s.d. \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""

# Bảng trọng số học được
NAMES = {'free': 'free slots', 'sameSKU': 'same SKU in $l$', 'sameModel': 'same model in $l$', 'MI': '$MI_l$ before',
         'dMI': 'change of $MI_l$', 'dist': 'distance to depot', 'cut': 'cut in SKU distance',
         'cutDemand': 'cut $\\times$ SKU demand', 'heat': 'demand already in $l$'}
rows = []
for j, r in enumerate(S['table_theta'], 1):
    rows.append(f"{j} & {NAMES[r['feature']]} & ${r['w0']:+.2f}\\pm{r['w0_sd']:.2f}$ & ${r['w05']:+.2f}\\pm{r['w05_sd']:.2f}$ & ${r['w1']:+.2f}\\pm{r['w1_sd']:.2f}$ \\\\")
out['theta'] = r"""\begin{table}[t]
\caption{Learned effective weights $w_j(\lambda)=\theta_j+\lambda\theta_{d+j}$ on standardised features (mean $\pm$ s.d.\ over 5 seeds). A positive weight attracts the unit to locations with a high value of the feature}
\label{tab:theta}
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}rlrrr@{}}
\toprule
\# & Feature & $\lambda=0$ & $\lambda=0.5$ & $\lambda=1$ \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""

# Bảng overfit
fit = pd.DataFrame(S['table_fit'])
sb = pd.DataFrame(S.get('split_baselines', []))
rows = [f"LP-RL & {pm(fit.train.mean(), fit.train.std())} & {pm(fit.val.mean(), fit.val.std())} & {pm(fit.test.mean(), fit.test.std())} \\\\"]
for n in ['Random', 'Nearest', 'GreedyJ']:
    q = sb[(sb.policy == n) & (np.isclose(sb.lam, 0.5))].set_index('split').J
    rows.append(f"{n} & {q['train']:.3f} & {q['val']:.3f} & {q['test']:.3f} \\\\")
out['fit'] = r"""\begin{table}[t]
\caption{$J(0.5)$ on the three parts of the data, each run from $S_0$. LP-RL: mean $\pm$ s.d.\ over 5 seeds}
\label{tab:fit}
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lccc@{}}
\toprule
Policy & Training (4,494 waves) & Validation (300) & Test (2,913) \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
json.dump(out, open('results/lp/tex_tables.json', 'w'), indent=1)
for k, v in out.items(): print(f'%%% {k}\n{v}\n')
