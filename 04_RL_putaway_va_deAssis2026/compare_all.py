"""
Bài v7b – BẢNG SO SÁNH ĐẦY ĐỦ: LP-RL với MỌI phương pháp hiện có, ở mọi λ (trên 2.913 wave kiểm tra, cùng S0).
Nhóm phương pháp:
  hiện trạng      : Random
  luật cổ điển    : Nearest (closest open location), ABC class-based, Turnover (volume-based động)
  dòng mixed-shelves: Scattered (rải hàng, Weidinger & Boysen 2018)
  dòng tương quan : Correlated (Wanis et al. 2026)
  dòng trộn hàng  : de Assis 2026 heuristic / supervised / adaptive
  tham lam theo J : GreedyJ
  deep RL         : mạng nơ-ron xếp hạng 64 ô rồi chọn, bắt chước + PPO (kiểu Pan et al. 2025; rl_ppo.py, 5 seed)
  của nhóm        : LP-RL (5 seed)
Kiểm định: Wilcoxon cặp theo wave cho từng seed LP-RL với từng phương pháp có dữ liệu từng wave, Holm trên mọi phép thử.
Deep RL chỉ còn trung bình (file từng wave không lưu) -> so sánh trung bình, không kiểm định.
Ghi: results/v7/compare_all.json, results/v7/compare_all.csv
"""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from lp_rl import LAMS, MI0, T0


def cw(h, lam): return lam * h[:, 0] / MI0 + (1 - lam) * h[:, 2] / T0


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


H = {}   # tên -> hàm(lam) -> list hist (mỗi seed)
fixed = {'Random': 'Random', 'Nearest': 'Nearest', 'Turnover': 'Turnover',
         'deAssis heuristic': 'DA_heuristic', 'deAssis supervised': 'DA_supervised', 'deAssis adaptive': 'DA_adaptive'}
for lab, f in fixed.items():
    hs = [np.load(x) for x in sorted(glob.glob(f'results/lp/base_{f}_s*.npy'))]
    H[lab] = lambda lam, hs=hs: hs
for lab in ['ABC', 'Correlated', 'Scattered']:
    hs = [np.load(x)['hist'] for x in sorted(glob.glob(f'results/v7/base/{lab}_0.5_s*.npz'))]
    H[lab] = lambda lam, hs=hs: hs
H['GreedyJ'] = lambda lam: [np.load(f'results/lp/base_GreedyJ_{lam}_s0.npy')]
H['LP-RL'] = lambda lam: [np.load(f'results/v7/base/LP-lp_s{s}_{float(lam)}_s0.npz')['hist'] for s in range(5)]

deep = []
for f in ['results/rl_main2.csv'] + [f'results/rl_main2_s{s}.csv' for s in range(1, 5)]:
    d = pd.read_csv(f); deep.append({round(r.lam, 1): (r.MI_avg, r.T_avg) for r in d.itertuples()})

ORDER = ['Random', 'Nearest', 'ABC', 'Turnover', 'Scattered', 'Correlated', 'deAssis heuristic', 'deAssis supervised',
         'deAssis adaptive', 'GreedyJ', 'Deep RL (rank+assign)', 'LP-RL']
rows, tests = [], []
for lam in LAMS:
    lp = [cw(h, lam) for h in H['LP-RL'](lam)]
    for m in ORDER:
        if m == 'Deep RL (rank+assign)':
            js = [lam * d[lam][0] / MI0 + (1 - lam) * d[lam][1] / T0 for d in deep]
            mi = np.mean([d[lam][0] for d in deep]); T = np.mean([d[lam][1] for d in deep])
        else:
            hs = H[m](lam); js = [cw(h, lam).mean() for h in hs]
            mi = np.mean([h[:, 0].mean() for h in hs]); T = np.mean([h[:, 2].mean() for h in hs])
            if m != 'LP-RL':
                base = np.mean([cw(h, lam) for h in hs], 0)
                for s, c in enumerate(lp):
                    tests.append((lam, m, s, wilcoxon(c, base).pvalue, np.sign(c.mean() - base.mean())))
        rows.append(dict(lam=lam, method=m, J=float(np.mean(js)), J_sd=float(np.std(js, ddof=1)) if len(js) > 1 else 0.0,
                         MI=float(mi), T=float(T), n=len(js)))
df = pd.DataFrame(rows)
adj = holm([t[3] for t in tests])
sig = pd.DataFrame([dict(lam=t[0], method=t[1], seed=t[2], p_adj=a, sign=t[4]) for t, a in zip(tests, adj)])
sig['better'] = (sig.p_adj < 0.05) & (sig.sign < 0); sig['worse'] = (sig.p_adj < 0.05) & (sig.sign > 0)
agg = sig.groupby(['lam', 'method'])[['better', 'worse']].sum().reset_index()
df = df.merge(agg, on=['lam', 'method'], how='left')
# hạng của LP-RL và phương pháp tốt nhất khác ở mỗi λ
summ = []
for lam, g in df.groupby('lam'):
    g = g.sort_values('J'); others = g[g.method != 'LP-RL']
    summ.append(dict(lam=lam, lp_rank=int(list(g.method).index('LP-RL')) + 1, best_other=others.iloc[0].method,
                     best_other_J=others.iloc[0].J, lp_J=float(g[g.method == 'LP-RL'].J.iloc[0]),
                     lp_beats_all_sig=int(((sig.lam == lam).groupby(sig.method).size() > 0).sum())))
os.makedirs('results/v7', exist_ok=True)
df.to_csv('results/v7/compare_all.csv', index=False)
json.dump(dict(rows=df.to_dict('records'), summary=summ, n_tests=len(tests)), open('results/v7/compare_all.json', 'w'), indent=1, default=float)
piv = df.pivot(index='method', columns='lam', values='J').loc[ORDER]
print(piv[[0.0, 0.2, 0.5, 0.8, 1.0]].round(3).to_string())
print(pd.DataFrame(summ).round(3).to_string())
print(df[df.lam.isin([0.0, 0.5, 1.0])][['lam', 'method', 'better', 'worse']].to_string())
