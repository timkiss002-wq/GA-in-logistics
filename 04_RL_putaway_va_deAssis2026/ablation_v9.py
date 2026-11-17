"""v9: bảng ablation so với LP-RL gọn (lpS). ΔJ = biến thể − LP-RL (cùng seed khi so 2 seed; trung bình 5 seed khi so cấu hình cũ).
Ghi results/v7/ablation_v9.csv"""
import numpy as np, pandas as pd, os
from lp_rl import MI0, T0
L = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
def J(tag, lam):
    f = f'results/v7/base/LP-{tag}_{lam}_s0.npz'
    if not os.path.exists(f): return np.nan
    h = np.load(f)['hist']; return lam * h[:, 0].mean() / MI0 + (1 - lam) * h[:, 2].mean() / T0
rows = []
for name, pre, seeds in [('gamma0', 'abS_gamma0', range(5)), ('nomix', 'abS_nomix', [0, 1]), ('nolam', 'abS_nolam', [0, 1]),
                         ('+split', 'ab_notref', range(5)), ('+split+Tref (v7)', 'lp', range(5))]:
    for lam in L:
        d = [J(f'{pre}_s{s}', lam) - J(f'lpS_s{s}', lam) for s in seeds]
        rows.append(dict(variant=name, lam=lam, dJ=np.nanmean(d), n=int(np.sum(~np.isnan(d)))))
df = pd.DataFrame(rows); df.to_csv('results/v7/ablation_v9.csv', index=False)
p = df.pivot(index='variant', columns='lam', values='dJ'); p['mean'] = p.mean(1)
print(p[[0.0, 0.2, 0.5, 0.8, 1.0, 'mean']].round(3))
print(df.groupby('variant').n.min())
