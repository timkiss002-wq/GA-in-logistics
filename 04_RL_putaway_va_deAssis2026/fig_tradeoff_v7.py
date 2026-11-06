"""Bài v7b – hình đánh đổi MI–lộ trình với MỌI phương pháp (từ results/v7/compare_all.csv)."""
import pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8})
d = pd.read_csv('results/v7/compare_all.csv')
fig, ax = plt.subplots(figsize=(3.45, 2.9))
pts = [('Random', 'Random (current)', 's', '#888780'), ('Nearest', 'Nearest', '^', '#2a78d6'), ('ABC', 'ABC class-based', 'v', '#2a78d6'),
       ('Turnover', 'Turnover', 'D', '#2a78d6'), ('Scattered', 'Scattered', 'P', '#1baf7a'), ('Correlated', 'Correlated', 'X', '#eda100'),
       ('deAssis heuristic', 'de Assis heur.', 'o', '#e87ba4'), ('deAssis supervised', 'de Assis sup.', 'h', '#e87ba4'),
       ('deAssis adaptive', 'de Assis adapt.', 'p', '#e87ba4')]
for k, lab, mk, c in pts:
    r = d[(d.method == k) & (d.lam == 0.5)].iloc[0]
    ax.scatter(r['T'], r.MI, marker=mk, s=22, color=c, edgecolor='white', linewidth=0.6, label=lab, zorder=3)
for k, lab, c, ls in [('GreedyJ', r'GreedyJ ($\lambda$=0..1)', '#eb6834', '--'), ('Deep RL (rank+assign)', r'Deep RL ($\lambda$=0..1)', '#6250d6', ':'),
                      ('LP-RL', r'LP-RL ($\lambda$=0..1)', '#e34948', '-')]:
    g = d[d.method == k].sort_values('lam')
    ax.plot(g['T'], g.MI, ls=ls, color=c, lw=1.4, marker='.', ms=4, label=lab, zorder=4)
ax.set_xlabel('route length per wave $T$ (dataset units)'); ax.set_ylabel('mixing index $MI$')
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.grid(color='#e1e0d9', lw=0.5); ax.set_axisbelow(True)
ax.legend(fontsize=5.6, ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.27))
fig.tight_layout(); fig.savefig('results/v7/fig_tradeoff.pdf'); fig.savefig('results/v7/fig_tradeoff.png', dpi=170)
