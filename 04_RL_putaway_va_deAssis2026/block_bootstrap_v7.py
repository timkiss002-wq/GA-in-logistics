"""Bài v7c – kiểm định lại bằng moving-block bootstrap (các wave liên tiếp phụ thuộc nhau qua trạng thái kho).
Cho mỗi λ, mỗi seed LP-RL và mỗi phương pháp có dữ liệu từng wave: khoảng tin cậy 95% của chênh lệch trung bình
c_w(LP-RL) − c_w(phương pháp), lấy mẫu lại theo khối B wave liên tiếp (B = 25, 100, 300), 2.000 lần.
"Có ý nghĩa" = khoảng tin cậy không chứa 0, hiệu chỉnh Bonferroni trên mọi phép so sánh. Ghi results/v7/block_bootstrap.json."""
import glob, json, numpy as np
from lp_rl import LAMS, MI0, T0
cw = lambda h, l: l * h[:, 0] / MI0 + (1 - l) * h[:, 2] / T0
fixed = {'Random': 'results/lp/base_Random_s*.npy', 'Nearest': 'results/lp/base_Nearest_s*.npy', 'Turnover': 'results/lp/base_Turnover_s*.npy',
         'deAssis heuristic': 'results/lp/base_DA_heuristic_s*.npy', 'deAssis supervised': 'results/lp/base_DA_supervised_s*.npy',
         'deAssis adaptive': 'results/lp/base_DA_adaptive_s*.npy'}
H = {k: [np.load(f) for f in sorted(glob.glob(p))] for k, p in fixed.items()}
for k in ['Gnone-Class', 'Gnone-Ded', 'Gnone-Hyb', 'WB-Scatter', 'Wanis-Score']:
    H[k] = [np.load(f)['hist'] for f in sorted(glob.glob(f'results/v7/base/{k}_0.5_s*.npz'))]
rng = np.random.default_rng(0); R = 2000
def ci(d, B, alpha):
    n = len(d); nb = int(np.ceil(n / B)); starts = rng.integers(0, n - B + 1, size=(R, nb))
    idx = (starts[:, :, None] + np.arange(B)).reshape(R, -1)[:, :n]
    m = d[idx].mean(1); return np.quantile(m, [alpha / 2, 1 - alpha / 2])
comps = []
for lam in LAMS:
    lp = [cw(np.load(f'results/v7/base/LP-lp_s{s}_{float(lam)}_s0.npz')['hist'], lam) for s in range(5)]
    meths = dict(H); meths['GreedyJ'] = [np.load(f'results/lp/base_GreedyJ_{lam}_s0.npy')]
    for m, hs in meths.items():
        base = np.mean([cw(h, lam) for h in hs], 0)
        for s, c in enumerate(lp): comps.append((lam, m, s, c - base))
alpha = 0.05 / len(comps); out = {}
for B in [25, 100, 300]:
    res = []
    for lam, m, s, d in comps:
        lo, hi = ci(d, B, alpha); res.append(dict(lam=lam, method=m, seed=s, mean=float(d.mean()), lo=float(lo), hi=float(hi),
                                                 better=bool(hi < 0), worse=bool(lo > 0)))
    out[B] = res
    # tóm tắt: với λ = 0,2–0,8, LP-RL có tốt hơn MỌI phương pháp ở cả 5 seed không
    ok = all(r['better'] for r in res if 0.15 < r['lam'] < 0.85)
    nb = sum(r['better'] for r in res); nw = sum(r['worse'] for r in res)
    print(f'B={B}: {len(res)} so sánh; tốt hơn {nb}, kém hơn {nw}; λ=0,2–0,8 tốt hơn mọi PP ở mọi seed: {ok}')
    for lam in [0.0, 0.1, 0.9, 1.0]:
        rr = [r for r in res if r['lam'] == lam]
        print('   λ', lam, 'tốt hơn', sum(r['better'] for r in rr), 'kém hơn', sum(r['worse'] for r in rr), 'không rõ', sum(not r['better'] and not r['worse'] for r in rr), '/', len(rr))
json.dump(dict(n=len(comps), R=R, bonferroni_alpha=alpha, res=out), open('results/v7/block_bootstrap.json', 'w'))
