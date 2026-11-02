"""
Tái hiện de Assis 2026 theo CÁCH ĐỌC 2 (xem results/deassis2026/nguyen_nhan_lech.md):
  - Ngưỡng θ tác động như SÀN ở mức critical: không cất vào ô nếu MI_l sau khi cất < θ_min (mặc định 0,8),
    tuỳ chọn thêm TRẦN θ_max (bài viết "MI_l sau <= θ + ς").
  - Lô supervised chuyển người duyệt được người vận hành cất vào ô trống ngẫu nhiên.
  - Trạng thái đầu: 847 ô (quy mô bài) đổ ngẫu nhiên đầy theo tần suất mã/cỡ thực.
Bước 1 (--calib): lưới nhỏ θ_min × θ_max để chọn giá trị khớp Bảng 2 của bài nhất (bài không công bố θ, ς).
Bước 2 (mặc định): chạy 5 seed với giá trị đã chọn, ghi bảng so sánh với bài và hình như Hình 3, Hình 4.
"""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from deassis2026_repl import load, run, initial, DEFAULT
from deassis2026_giathuyet import synthetic, PAPER, score

R = 'results/deassis2026/'
base = load('all'); syn = synthetic(base)

if '--calib' in sys.argv:
    rows = []
    for tmin in [0.78, 0.80, 0.82]:
        for tmax in [1.0, 0.92, 0.90, 0.88]:
            P = dict(DEFAULT); P.update(theta_min=tmin, theta=tmax, deferred='operator'); res = {}
            for m in PAPER:
                rs = [run(syn, m, s, P, trace_every=0) for s in range(2)]
                res[m] = (np.mean([r['MI'] for r in rs]), np.mean([r['crit'] for r in rs]), np.mean([r['executed'] for r in rs]))
            e, ok = score(res)
            rows.append(dict(theta_min=tmin, theta_max=tmax, sai_lech=e, dung_thu_tu=ok,
                             **{f'{m}_{k}': res[m][i] for m in PAPER for i, k in enumerate(['MI', 'crit', 'exec'])}))
            print(f'θ_min {tmin} θ_max {tmax} | H {res["heuristic"][0]:.3f}/{res["heuristic"][1]:5.1f}/{res["heuristic"][2]:.0f} '
                  f'| S {res["supervised"][0]:.3f}/{res["supervised"][1]:5.1f}/{res["supervised"][2]:.0f} '
                  f'| A {res["adaptive"][0]:.3f}/{res["adaptive"][1]:5.1f}/{res["adaptive"][2]:.0f} | lệch {e:.3f} | thứ tự {ok}', flush=True)
    pd.DataFrame(rows).to_csv(R + 'cachdoc2_hieu_chinh.csv', index=False)
    sys.exit()

c = pd.read_csv(R + 'cachdoc2_hieu_chinh.csv').sort_values('sai_lech').iloc[0]
P = dict(DEFAULT); P.update(theta_min=float(c.theta_min), theta=float(c.theta_max), deferred='operator')
print(f'Chọn θ_min = {P["theta_min"]}, θ_max = {P["theta"]} (sai lệch nhỏ nhất trên lưới hiệu chỉnh)')
rows, tr = [], {}
for m in PAPER:
    for s in range(5):
        r = run(syn, m, s, P); rows.append({k: v for k, v in r.items() if k not in ('trace', 'w')}); tr[f'{m}_{s}'] = r['trace']
        print(f'  {m:<10} seed {s}: MI {r["MI"]:.4f} | critical {r["crit"]:.2f}% | thực hiện {r["executed"]} | chờ duyệt {r["deferred"]}', flush=True)
df = pd.DataFrame(rows); df.to_csv(R + 'cachdoc2_5seed.csv', index=False); json.dump(tr, open(R + 'cachdoc2_trace.json', 'w'))
ini = initial(syn, 0, P)
h0 = pd.read_csv(R + 'ket_qua_5seed.csv')
out = [f'## Bảng R3. Tái hiện de Assis 2026 theo cách đọc 2 (θ_min = {P["theta_min"]}, θ_max = {P["theta"]}, lô chờ duyệt do người cất; 847 ô; TB ± ĐLC 5 seed)\n',
       '| Agent | MI bài | MI cách đọc 2 | MI cách đọc 1 (đúng chữ) | % critical bài | % critical cách đọc 2 | thực hiện bài | thực hiện cách đọc 2 |',
       '|---|---|---|---|---|---|---|---|',
       f'| Ban đầu | 0,918 | {ini["MI_full"]:.4f} | 0,9147 | 100 | {ini["crit_full"]:.1f} | – | – |']
for m, (mi, cr, ex) in PAPER.items():
    d = df[df['mode'] == m]; d1 = h0[h0['mode'] == m]
    out.append(f'| {m} | {mi:.3f} | {d.MI.mean():.4f} ± {d.MI.std():.4f} | {d1.MI.mean():.4f} | {cr:.2f} | {d.crit.mean():.2f} | ≈ {ex} | {d.executed.mean():.0f} |')
order = df.groupby('mode').MI.mean().sort_values()
out.append(f'\nThứ tự MI (thấp → cao): bài: adaptive < heuristic < supervised | cách đọc 2: {" < ".join(order.index)}')
open(R + 'bang_cachdoc2.md', 'w', encoding='utf-8').write('\n'.join(out)); print('\n'.join(out))

fig, ax = plt.subplots(1, 2, figsize=(11, 4)); col = {'heuristic': '#4C78A8', 'supervised': '#E45756', 'adaptive': '#54A24B'}
for m in col:
    T = np.array([[x[1] for x in tr[f'{m}_{s}']] for s in range(5)]); E = np.array([[x[2] for x in tr[f'{m}_{s}']] for s in range(5)])
    t = [x[0] for x in tr[f'{m}_0']]
    ax[0].plot(t, T.mean(0), color=col[m], label=m); ax[1].plot(t, E.mean(0), color=col[m], label=m)
for m, v in [('heuristic', .830), ('supervised', .884), ('adaptive', .822)]:
    ax[0].axhline(v, color=col[m], ls=':', lw=1); ax[0].text(10000, v, f' bài {v}', color=col[m], fontsize=7, va='center')
ax[1].axhline(2500, color='gray', ls=':', lw=1); ax[1].axhline(2000, color='#E45756', ls=':', lw=1)
ax[0].set_title('MI theo số lô – cách đọc 2 (so Hình 3)'); ax[1].set_title('Hành động thực hiện cộng dồn – cách đọc 2 (so Hình 4)')
for a in ax: a.set_xlabel('số lô hàng về'); a.grid(alpha=.3); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(R + 'hinh_cachdoc2.png', dpi=160)
