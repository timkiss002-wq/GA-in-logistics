"""v9: đường học (J kiểm định theo số quyết định) của 5 seed LP-RL gọn (lpS). Ghi results/v7/fig_curve_lpS.pdf"""
import json, os, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8})
fig, ax = plt.subplots(figsize=(3.45, 2.3))
for i in range(5):
    m = json.load(open(f'models_lp/lpS_s{i}.json')); col = f'C{i}'
    for r, pc in enumerate(m.get('probe_curves', [])):
        pc = pd.DataFrame(pc); ch = r == m['chosen_restart']
        ax.plot(pc.step / 1000, pc.val_J, lw=0.8 if ch else 0.5, ls='-' if ch else ':', color=col, alpha=1 if ch else 0.6)
    c = pd.DataFrame(m['curve']); off = m.get('probe', 0)
    ax.plot((c.step + off) / 1000, c.val_J, lw=0.9, color=col, label=f'seed {i}')
sb = pd.read_csv('results/lp/split_baselines.csv')
for n in ['Nearest', 'GreedyJ']:
    ax.axhline(sb[(sb.policy == n) & (sb.split == 'val')].J.mean(), color='k', ls='--', lw=0.6)
ax.text(405, 0.832, 'Nearest,\nGreedyJ', fontsize=5.5, va='center')
ax.axvline(100, color='grey', lw=0.5); ax.text(50, 0.912, 'short starts', ha='center', fontsize=6); ax.text(250, 0.912, 'continuation of the kept start', ha='center', fontsize=6)
ax.set_xlabel('Training decisions (thousands)'); ax.set_ylabel('Validation $J$')
ax.set_ylim(0.80, 0.92); ax.set_xlim(0, 460)
ax.legend(fontsize=5.5, frameon=False, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.07), handlelength=1.2); ax.grid(lw=0.3, alpha=0.5)
fig.tight_layout(); fig.savefig('results/v7/fig_curve_lpS.pdf'); fig.savefig('results/v7/fig_curve_lpS.png', dpi=200)
