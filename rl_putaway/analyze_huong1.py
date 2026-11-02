"""
Bảng 9 – Ablation cho hướng 1 (định vị: hàm J + chính sách học được điều kiện theo λ; PPO là tinh chỉnh).
Mỗi biến thể chạy 3 seed huấn luyện, cùng cấu hình với mô hình A (main2: 400.000 bước PPO, 40.000 bước bắt chước),
chỉ đổi đúng một thành phần. So với mô hình A CÙNG seed:
   Δ = J(biến thể) − J(A), trung bình trên các λ được đánh giá; dương = biến thể kém hơn = thành phần đó có ích.
Kiểm định: với từng seed, Wilcoxon cặp theo wave trên chi phí c_w (gộp các λ), hiệu chỉnh Holm trên mọi phép thử.
Chạy:  python analyze_huong1.py   -> results/huong1/bang9_ablation.md, bang7_5seed.md
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats

R = 'results/'; OUT = R + 'huong1/'; os.makedirs(OUT, exist_ok=True)
C = json.load(open('t0.json')); MI0, T0 = C['MI0'], C['T0']
J = lambda mi, t, lam: lam * mi / MI0 + (1 - lam) * t / T0
base = {0: 'main2', 1: 'main2_s1', 2: 'main2_s2'}


def hist(tag, lam):
    p = R + f'picks_RL_{tag}_{lam}.npz'
    return np.load(p)['hist'] if os.path.exists(p) else None


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, ps[j] * (m - i))); adj[j] = run
    return adj


VAR = [  # (tên hiển thị, tiền tố tag, hậu tố theo seed, các λ, câu hỏi)
    ('Bỏ PPO (chỉ bắt chước)', 'ab2_BConly', lambda s: '' if s == 0 else f'_s{s}', [0.2, 0.5, 0.8], 'PPO có cần không?'),
    ('PPO với γ = 0 (không nhìn xa)', 'ab2_gamma0', lambda s: '' if s == 0 else f'_s{s}', [0.2, 0.5, 0.8], 'Nhìn xa có giúp không?'),
    ('Bắt chước chỉ GreedyJ (bỏ danh mục)', 'ab3_greedyT', lambda s: f'_s{s}', [0.2, 0.5, 0.8], 'Danh mục heuristic có cần không?'),
    ('Bỏ đặc trưng mixing (2, 3, 4)', 'ab3_noMIfeat', lambda s: f'_s{s}', [0.2, 0.5, 0.8], 'Đặc trưng mixing có cần không?'),
    ('Mạng riêng cho λ = 0,5 (bỏ điều kiện theo λ)', 'ab3_lam05', lambda s: f'_s{s}', [0.5], 'Một mạng cho mọi λ có tốt hơn mạng chuyên một λ?'),
]

rl = pd.concat([pd.read_csv(f) for f in __import__('glob').glob(R + 'rl_*.csv')], ignore_index=True)


def jcsv(tag, lam):
    x = rl[(rl.tag == tag) & (np.isclose(rl.lam, lam))]
    return None if x.empty else J(x.MI_avg.iloc[0], x.T_avg.iloc[0], lam)


rows, tests = [], []
for name, pre, suf, lams, q in VAR:
    per_seed = []
    for s, bt in base.items():
        tag = pre + suf(s)
        jv = [jcsv(tag, l) for l in lams]; ja = [jcsv(bt, l) for l in lams]
        if None in jv or None in ja: continue
        d = float(np.mean(np.array(jv) - np.array(ja)))          # J tuyến tính theo trung bình -> tính từ csv
        hv = [hist(tag, l) for l in lams]; ha = [hist(bt, l) for l in lams]
        if all(h is not None for h in hv + ha):
            cv = np.concatenate([J(h[:, 0], h[:, 2], l) for h, l in zip(hv, lams)])
            ca = np.concatenate([J(h[:, 0], h[:, 2], l) for h, l in zip(ha, lams)])
            p = stats.wilcoxon(cv, ca).pvalue
        else:
            p = np.nan
        per_seed.append((s, d)); tests.append((name, s, np.sign(d), p))
    rows.append((name, q, lams, per_seed))

_ok = [i for i, t in enumerate(tests) if not np.isnan(t[3])]
_a = holm([tests[i][3] for i in _ok]) if _ok else []
adj = np.full(len(tests), np.nan); adj[_ok] = _a
sig = {(t[0], t[1]): (t[2], a) for t, a in zip(tests, adj)}
out = ['## Bảng 9. Ablation theo hướng 1, mỗi biến thể 3 seed huấn luyện (Δ = J biến thể − J mô hình A cùng seed; dương = biến thể kém hơn)\n',
       '| Câu hỏi | Biến thể | λ đánh giá | Δ seed 0 / 1 / 2 | Δ TB ± ĐLC | số seed kém hơn / tốt hơn có ý nghĩa (Holm) |',
       '|---|---|---|---|---|---|']
for name, q, lams, ps in rows:
    if not ps:
        out.append(f'| {q} | {name} | {", ".join(map(str, lams))} | chưa chạy | – | – |'); continue
    d = np.array([x[1] for x in ps]); worse = better = 0; nan_n = False
    for s, _ in ps:
        sg, a = sig[(name, s)]
        if a < 0.05: worse += sg > 0; better += sg < 0
        if np.isnan(a): nan_n = True
    sd = f'{d.std(ddof=1):.4f}' if len(d) > 1 else '–'
    out.append(f'| {q} | {name} | {", ".join(str(x).replace(".", ",") for x in lams)} | {" / ".join(f"{x:+.4f}" for x in d)} | '
               f'{d.mean():+.4f} ± {sd} | {worse} / {better} (trên {len(d)} seed{"; thiếu dữ liệu từng wave cho một số seed" if nan_n else ""}) |')
txt = '\n'.join(out); open(OUT + 'bang9_ablation.md', 'w', encoding='utf-8').write(txt); print(txt)

# ---------- Bảng 9b: tách theo λ (trung bình trên các seed đã có) ----------
out2 = ['\n## Bảng 9b. Δ tách theo từng λ (TB ± ĐLC trên các seed; dương = biến thể kém hơn)\n',
        '| Biến thể | λ = 0,2 | λ = 0,5 | λ = 0,8 |', '|---|---|---|---|']
for name, pre, suf, lams, q in VAR:
    cells = []
    for l in [0.2, 0.5, 0.8]:
        ds = []
        for s, bt in base.items():
            jv, ja = jcsv(pre + suf(s), l), jcsv(bt, l)
            if jv is not None and ja is not None: ds.append(jv - ja)
        cells.append('–' if not ds else (f'{np.mean(ds):+.4f} ± {np.std(ds, ddof=1):.4f} (n={len(ds)})' if len(ds) > 1 else f'{ds[0]:+.4f} (n=1)'))
    out2.append(f'| {name} | ' + ' | '.join(cells) + ' |')
txt2 = '\n'.join(out2); open(OUT + 'bang9_ablation.md', 'a', encoding='utf-8').write('\n' + txt2); print(txt2)
