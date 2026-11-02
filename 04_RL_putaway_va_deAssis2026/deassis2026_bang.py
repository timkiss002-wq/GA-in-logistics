"""Bảng và hình tái hiện de Assis 2026, đặt cạnh số của bài (Bảng 2, Hình 3, Hình 4)."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = 'results/deassis2026/'
df = pd.read_csv(R + 'ket_qua_5seed.csv'); tr = json.load(open(R + 'trace.json')); sens = pd.read_csv(R + 'do_nhay.csv')
paper = {'Baseline (Random)': (0.918, 100.0, None), 'heuristic': (0.830, 99.66, '≈ 2.500'),
         'supervised': (0.884, 99.71, '≈ 2.000'), 'adaptive': (0.822, 99.48, '≈ 2.500')}
out = ['## Bảng R1. Tái hiện Bảng 2 của de Assis et al. (2026) sau 10.000 lô (nhóm: TB ± ĐLC 5 seed, 1.093 ô vùng ngẫu nhiên)\n',
       '| Agent | MI (bài) | MI (nhóm) | % critical MI_l > 0,8 (bài) | % critical (nhóm) | số hành động thực hiện (bài, đọc từ Hình 4) | thực hiện (nhóm) | chờ duyệt (nhóm) |',
       '|---|---|---|---|---|---|---|---|',
       '| Baseline (Random, kho đầy) | 0,918 | 0,9147 | 100,0 | 100,0 | – | – | – |',
       '| Sau khi giải phóng 20% | – | 0,9036 | – | 99,6 | – | – | – |']
for m in ['heuristic', 'supervised', 'adaptive']:
    d = df[df['mode'] == m]; p = paper[m]
    out.append(f'| {m} | {p[0]:.3f} | {d.MI.mean():.4f} ± {d.MI.std():.4f} | {p[1]:.2f} | {d.crit.mean():.2f} ± {d.crit.std():.2f} | '
               f'{p[2]} | {d.executed.mean():.0f} | {d.deferred.mean():.0f} |')
out.append('\nThứ tự MI trong bài: adaptive < heuristic < supervised. Của nhóm: supervised < adaptive < heuristic.\n')
out.append('## Bảng R2. Độ nhạy theo các tham số bài không công bố (TB 3 seed)\n')
out.append('| Biến thể | heuristic MI | supervised MI | adaptive MI | heuristic % critical |\n|---|---|---|---|---|')
for v, g in sens.groupby('bien_the', sort=False):
    gg = g.set_index('mode')
    f = lambda m, c='MI': f'{gg.loc[m, c]:.4f}' if m in gg.index else '–'
    out.append(f'| {v} | {f("heuristic")} | {f("supervised")} | {f("adaptive")} | {f("heuristic", "crit") if "heuristic" in gg.index else "–"} |')
open(R + 'bang_tai_hien.md', 'w', encoding='utf-8').write('\n'.join(out)); print('\n'.join(out))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
col = {'heuristic': '#4C78A8', 'supervised': '#E45756', 'adaptive': '#54A24B'}
for m in col:
    T = np.array([[x[1] for x in tr[f'{m}_{s}']] for s in range(5)]); E = np.array([[x[2] for x in tr[f'{m}_{s}']] for s in range(5)])
    t = [x[0] for x in tr[f'{m}_0']]
    ax[0].plot(t, T.mean(0), color=col[m], label=f'{m} (nhóm)'); ax[1].plot(t, E.mean(0), color=col[m], label=f'{m} (nhóm)')
for m, v in [('heuristic', .830), ('supervised', .884), ('adaptive', .822)]:
    ax[0].axhline(v, color=col[m], ls=':', lw=1); ax[0].text(10000, v, f' bài: {v}', color=col[m], fontsize=7, va='center')
ax[0].set_title('MI trung bình theo số lô (tương ứng Hình 3)'); ax[0].set_xlabel('số lô hàng về'); ax[0].legend(fontsize=8)
ax[1].set_title('Số hành động thực hiện cộng dồn (tương ứng Hình 4)'); ax[1].set_xlabel('số lô hàng về')
ax[1].axhline(2500, color='gray', ls=':', lw=1); ax[1].text(10000, 2500, ' bài ≈ 2.500', fontsize=7, va='center')
for a in ax: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(R + 'hinh_tai_hien.png', dpi=160)
