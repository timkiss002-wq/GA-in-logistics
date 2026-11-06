"""Bài v7 – Hình thời gian lấy hàng tương đối (đi lại + tìm hàng + phần khác) ở λ = 0,5. Từ results/v7/summary_v7.json."""
import json
import matplotlib as mpl; import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8})
S = json.load(open('results/v7/summary_v7.json'))['time_lam05']
lab = {'Random': 'Random (current)', 'Nearest': 'Nearest', 'Turnover': 'Turnover', 'GreedyJ': 'GreedyJ',
       'DA_heuristic': 'de Assis heur.', 'DA_supervised': 'de Assis sup.', 'DA_adaptive': 'de Assis adapt.', 'LP-RL': 'LP-RL', 'Gnone-Class': 'Gnone class-based', 'Gnone-Ded': 'Gnone dedicated', 'Gnone-Hyb': 'Gnone hybrid', 'WB-Scatter': 'Weidinger-Boysen', 'Wanis-Score': 'Wanis score'}
rows = sorted(S, key=lambda r: -r['tau'])
fig, ax = plt.subplots(figsize=(3.4, 3.0))
for i, r in enumerate(rows):
    tr, se, ot = 0.5 * (1 - r['dT']), 0.2 * (1 - r['dS']), 0.3
    x = 0
    for v, c, h in [(tr, '#2a78d6', None), (se, '#eb6834', '////'), (ot, '#c3c2b7', None)]:
        ax.barh(i, v - 0.004, left=x, height=0.62, color=c, hatch=h, edgecolor='white', linewidth=0)
        x += v
    ax.text(x + 0.01, i, f"{r['tau']:.3f}", va='center', fontsize=7)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab[r['method']] for r in rows])
ax.set_xlim(0, 1.12); ax.set_xlabel(r'picking time $\tau$ relative to random storage')
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.grid(axis='x', color='#e1e0d9', lw=0.6); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#2a78d6', label='travel'), Patch(facecolor='#eb6834', hatch='////', edgecolor='white', label='search'),
                   Patch(color='#c3c2b7', label='other')], ncol=3, loc='lower center', bbox_to_anchor=(0.45, 1.0), frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig('results/v7/fig_time.pdf'); fig.savefig('results/v7/fig_time.png', dpi=160)
