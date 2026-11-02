"""Kiểm định ước lượng nhanh so với GRO trên 300 wave × 15 chính sách đã chạy."""
import glob, os, time
import numpy as np, pandas as pd
from scipy import stats
from tour_fast import tour_len
I = dict(np.load('inst.npz', allow_pickle=True)); XY = I['cell_xy']
rows = []
for f in sorted(glob.glob('results/gro_*.csv')):
    name = os.path.basename(f)[4:-4]; g = pd.read_csv(f); z = np.load(f'results/picks_{name}.npz'); p = z['ptr']
    for w, d in zip(g.wave, g.dist):
        rows.append(dict(policy=name, wave=w, gro=d, fast=tour_len(XY[z['cells'][p[w]:p[w + 1]]])))
df = pd.DataFrame(rows)
print(f'tương quan theo wave: Pearson {stats.pearsonr(df.gro, df.fast)[0]:.4f} | fast/GRO TB {np.mean(df.fast / df.gro):.4f} | max {np.max(df.fast / df.gro):.4f}')
m = df.groupby('policy')[['gro', 'fast']].mean()
print(f'thứ hạng chính sách: Spearman {stats.spearmanr(m.gro, m.fast)[0]:.4f}'); print(m.round(1).to_string())
t = time.time(); [tour_len(XY[np.random.choice(len(XY), 23)]) for _ in range(1000)]; print(f'{(time.time() - t):.3f} ms/wave')
