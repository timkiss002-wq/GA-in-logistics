"""
Bài v7 – phân tích các thí nghiệm mới (exp_v7.py, train_v7.py). Mọi số trong bài v7 lấy từ results/v7/summary_v7.json.

1. Mô hình THỜI GIAN lấy hàng (trả lời: "MI dùng vì nhanh – vậy thời gian thật thế nào?")
   Thời gian lấy hàng của một wave = đi lại + tìm hàng + phần khác (lấy, chuẩn bị, …).
   Hiệu chỉnh theo cơ cấu điển hình (Tompkins et al. 2010; de Koster et al. 2007): ở hiện trạng (Random)
   đi lại a = 50%, tìm hàng b = 20%, phần khác 30% (không đổi giữa các chính sách vì cùng số dòng).
   Đi lại ∝ độ dài lộ trình T_w. Tìm hàng của một dòng ∝ MI_l của ô được ghé (S1, như de Assis 2026 / Wanis 2026)
   hoặc ∝ số mã trong ô (S2). Thời gian tương đối so với Random:
        τ = a·ΣT/ΣT_R + b·ΣS/ΣS_R + (1 − a − b)
2. Độ nhạy theo giả định A3 (S0), A4 (luật lấy hàng), A5 (nhập hàng).
3. θ riêng theo λ, huấn luyện lại dưới luật lấy hàng khác, bỏ T_ref.
4. T_ref: mức giảm phương sai.
"""
import glob, json, os
import numpy as np, pandas as pd
from lp_rl import LAMS, MI0, T0

R = 'results/v7'; OUT = {}


def load(scen, pol, lam, seed=0):
    f = f'{R}/{scen}/{pol.replace(":", "-")}_{float(lam)}_s{seed}.npz'
    return np.load(f) if os.path.exists(f) else None


def stats(z, lam=0.5):
    h, h2 = z['hist'], z['hist2']
    return dict(MI=h[:, 0].mean(), T=h[:, 2].mean(), J=float(lam * h[:, 0].mean() / MI0 + (1 - lam) * h[:, 2].mean() / T0),
                Tsum=h[:, 2].sum(), Smi=h2[:, 1].sum(), Smod=h2[:, 2].sum(), lines=h2[:, 0].sum(),
                stockout=int(z['stockout']), visMI=h2[:, 1].sum() / h2[:, 0].sum(), dec_ms=float(z['dec_ms']))


def avg(rows):
    keys = rows[0].keys(); return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def method_stats(scen, name, lam=0.5):
    if name == 'LP-RL':
        rows = [stats(z, lam) for s in range(5) if (z := load(scen, f'LP:lp_s{s}', lam)) is not None]
    elif name == 'GreedyJ':
        rows = [stats(z, lam) for z in [load(scen, 'GreedyJ', lam)] if z is not None]
    elif name in ('Random', 'DA_heuristic', 'DA_supervised', 'DA_adaptive'):
        rows = [stats(z, lam) for s in range(5) if (z := load(scen, name, 0.5, s)) is not None]
    else:
        rows = [stats(z, lam) for z in [load(scen, name, 0.5)] if z is not None]
    if not rows: return None
    out = avg(rows); out['n'] = len(rows)
    out['J_sd'] = float(np.std([r['J'] for r in rows], ddof=1)) if len(rows) > 1 else 0.0
    out['J_all'] = [r['J'] for r in rows]
    return out


def tau(st, ref, a=0.5, b=0.2, model='mi'):
    S, SR = (st['Smi'], ref['Smi']) if model == 'mi' else (st['Smod'], ref['Smod'])
    return a * st['Tsum'] / ref['Tsum'] + b * S / SR + (1 - a - b)


METHODS = ['Random', 'Nearest', 'Turnover', 'GreedyJ', 'DA_heuristic', 'DA_supervised', 'DA_adaptive', 'LP-RL']

# ---------------- 1. thời gian lấy hàng ----------------
ref = method_stats('base', 'Random')
time_rows = []
for m in METHODS:
    st = method_stats('base', m, 0.5)
    if st is None: continue
    time_rows.append(dict(method=m, MI=st['MI'], visMI=st['visMI'], T=st['T'], J=st['J'],
                          dT=1 - st['Tsum'] / ref['Tsum'], dS=1 - st['Smi'] / ref['Smi'], dSmod=1 - st['Smod'] / ref['Smod'],
                          tau=tau(st, ref), tau_mod=tau(st, ref, model='mod'), dec_ms=st['dec_ms']))
OUT['time_lam05'] = time_rows
# LP-RL theo từng λ và GreedyJ theo từng λ: λ nào cho thời gian nhỏ nhất
lam_rows = []
for lam in LAMS:
    for m in ['LP-RL', 'GreedyJ']:
        st = method_stats('base', m, lam)
        if st is None: continue
        lam_rows.append(dict(method=m, lam=lam, tau=tau(st, ref), tau_mod=tau(st, ref, model='mod'),
                             dT=1 - st['Tsum'] / ref['Tsum'], dS=1 - st['Smi'] / ref['Smi']))
OUT['time_by_lam'] = lam_rows
# độ nhạy theo cơ cấu thời gian (a, b) và mô hình tìm hàng
grid = []
fixed = {m: method_stats('base', m, 0.5) for m in METHODS if m not in ('LP-RL', 'GreedyJ')}
for a in [0.4, 0.5, 0.6]:
    for b in [0.1, 0.2, 0.3]:
        for model in ['mi', 'mod']:
            best_lp = min((tau(method_stats('base', 'LP-RL', l), ref, a, b, model), l) for l in LAMS
                          if method_stats('base', 'LP-RL', l))
            lp05 = tau(method_stats('base', 'LP-RL', 0.5), ref, a, b, model)
            gj = min(tau(method_stats('base', 'GreedyJ', l), ref, a, b, model) for l in LAMS if method_stats('base', 'GreedyJ', l))
            bb = min([(tau(s, ref, a, b, model), m) for m, s in fixed.items() if s] + [(gj, 'GreedyJ')])
            grid.append(dict(a=a, b=b, model=model, lp_best=best_lp[0], lp_best_lam=best_lp[1], lp05=lp05,
                             base_best=bb[0], base_best_name=bb[1], implied_lam=b / (a + b)))
OUT['time_grid'] = grid

# ---------------- 2. độ nhạy giả định ----------------
SCN = ['base', 'pick_random', 'pick_top2', 'inb_lot', 'inb_delay', 's0_seed1', 's0_seed2', 's0_fill70', 's0_fill90', 's0_class']
rob = []
for sc in SCN:
    row = dict(scen=sc)
    for m in ['Random', 'Nearest', 'Turnover', 'GreedyJ', 'DA_heuristic', 'LP-RL']:
        st = method_stats(sc, m, 0.5)
        if st: row[m] = st['J']; row[m + '_T'] = st['T']; row[m + '_MI'] = st['MI']; row[m + '_so'] = st['stockout']
        if st and m == 'LP-RL': row['LP_sd'] = st['J_sd']; row['LP_all'] = st['J_all']
    bases = {m: row[m] for m in ['Random', 'Nearest', 'Turnover', 'GreedyJ', 'DA_heuristic'] if m in row}
    if bases and 'LP-RL' in row:
        bm = min(bases, key=bases.get); row['best_base'] = bm; row['dJ'] = row['LP-RL'] - bases[bm]
        row['lp_wins'] = int(sum(j < bases[bm] for j in row['LP_all']))
    zr = load(sc, 'LP:lpR_s0', 0.5)
    if sc == 'pick_random':
        rr = [stats(z) for s in range(3) if (z := load(sc, f'LP:lpR_s{s}', 0.5)) is not None]
        if rr: row['LPR'] = float(np.mean([r['J'] for r in rr])); row['LPR_sd'] = float(np.std([r['J'] for r in rr], ddof=1)); row['LPR_all'] = [r['J'] for r in rr]
    rob.append(row)
OUT['robust'] = rob

# ---------------- 3. θ riêng theo λ, bỏ T_ref ----------------
per_lam = []
for lf in [0.0, 0.5, 1.0]:
    rr = [stats(z, lf)['J'] for s in range(2) if (z := load('base', f'LP:lpF{lf}_s{s}', lf)) is not None]
    sh = method_stats('base', 'LP-RL', lf)
    if rr and sh: per_lam.append(dict(lam=lf, fixed=float(np.mean(rr)), fixed_all=rr, shared=sh['J'], shared_sd=sh['J_sd']))
OUT['per_lambda'] = per_lam
nt = []
for lam in [0.2, 0.5, 0.8]:
    for s in range(2):
        z = load('base', f'LP:ab_notref_s{s}', lam); zm = load('base', f'LP:lp_s{s}', lam)
        if z is not None and zm is not None: nt.append(dict(lam=lam, seed=s, dJ=stats(z, lam)['J'] - stats(zm, lam)['J']))
OUT['notref'] = nt

# ---------------- 4. T_ref: giảm phương sai ----------------
tref = np.load('tref.npy')
from lp_rl import W_TEST
vr = []
for s in range(5):
    z = load('base', f'LP:lp_s{s}', 0.5)
    if z is None: continue
    T = z['hist'][:, 2]; tr = tref[W_TEST:W_TEST + len(T)]
    vr.append(dict(var_T=float(T.var()), var_T_minus_ref=float((T - tr).var()), corr=float(np.corrcoef(T, tr)[0, 1])))
OUT['tref_var'] = vr

os.makedirs(R, exist_ok=True)
json.dump(OUT, open(f'{R}/summary_v7.json', 'w'), indent=1, default=float)
pd.DataFrame(time_rows).to_csv(f'{R}/time_lam05.csv', index=False)
pd.DataFrame(grid).to_csv(f'{R}/time_grid.csv', index=False)
pd.DataFrame([{k: v for k, v in r.items() if not k.endswith('_all')} for r in rob]).to_csv(f'{R}/robust.csv', index=False)
print(pd.DataFrame(time_rows).round(4).to_string())
print(pd.DataFrame(lam_rows).round(4).to_string())
print(pd.DataFrame(grid).round(4).to_string())
print(pd.DataFrame([{k: v for k, v in r.items() if not k.endswith('_all') and not k.endswith(('_T', '_MI'))} for r in rob]).round(4).to_string())
print(per_lam); print(nt); print(vr)
