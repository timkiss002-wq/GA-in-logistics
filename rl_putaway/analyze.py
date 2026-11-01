"""
BƯỚC 7 – Tổng hợp: bảng kết quả, kiểm định thống kê, hình.
Chỉ số chính (toàn bộ 2.913 wave kiểm tra, TB theo wave):
   MI     : Mixing Index kho lúc lấy wave
   T      : độ dài lộ trình wave (NN + 2-opt, khớp GRO r = 0,996)
   J(lam) = lam*MI/MI0 + (1-lam)*T/T0      (Random ≈ 1)
Kiểm chứng trên 300 wave cách đều bằng GRO (results/gro_*.csv).
"""
import glob, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

R = 'results/'
C = json.load(open('t0.json')); MI0, T0 = C['MI0'], C['T0']
out = []
J = lambda mi, t, lam: lam * mi / MI0 + (1 - lam) * t / T0


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, ps[j] * (m - i))); adj[j] = run
    return adj


def per_wave(name):
    z = np.load(R + f'picks_{name}.npz'); return z['hist']        # cột: MI, D proxy, T


b = pd.read_csv(R + 'baselines.csv')
b = b[b.policy != 'deAssis']          # bản rút gọn trước đây, đã thay bằng 3 chế độ đầy đủ (DA_*)
if os.path.exists(R + 'baselines_deassis.csv'):
    b = pd.concat([b, pd.read_csv(R + 'baselines_deassis.csv')[b.columns.intersection(pd.read_csv(R + 'baselines_deassis.csv').columns)]], ignore_index=True)
g = b.groupby(['policy', 'lam'], dropna=False)[['MI_avg', 'Dr_avg', 'T_avg']].agg(['mean', 'std']).reset_index()
g.columns = ['policy', 'lam', 'MI', 'MI_sd', 'Dr', 'Dr_sd', 'T', 'T_sd']
rl = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(R + 'rl_*.csv'))], ignore_index=True) \
    if glob.glob(R + 'rl_*.csv') else pd.DataFrame(columns=['tag', 'lam', 'MI_avg', 'Dr_avg', 'T_avg'])

# ---------- Bảng 1 ----------
out.append(f'MI0 = {MI0:.4f} | T0 = {T0:.1f} (Random từ S0, wave huấn luyện)\n')
out.append('## Bảng 1. Toàn bộ 2.913 wave kiểm tra (baseline: TB ± độ lệch chuẩn 5 seed; RL: 1 lần chạy, chọn ô theo xác suất lớn nhất)\n')
out.append('| Chính sách | λ | MI | T (lộ trình/wave) | T/T0 | D proxy/D0 | J(0,5) |\n|---|---|---|---|---|---|---|')
for _, r in g.iterrows():
    lam = '' if np.isnan(r.lam) else f'{r.lam:.1f}'
    out.append(f'| {r.policy} | {lam} | {r.MI:.4f} ± {r.MI_sd:.4f} | {r["T"]:.1f} ± {r.T_sd:.1f} | {r["T"] / T0:.4f} | {r.Dr:.4f} | {J(r.MI, r["T"], .5):.4f} |')
for _, r in rl[rl.tag.isin(['main', 'main2'])].iterrows():
    out.append(f'| RL-{r.tag} | {r.lam:.1f} | {r.MI_avg:.4f} | {r.T_avg:.1f} | {r.T_avg / T0:.4f} | {r.Dr_avg:.4f} | {J(r.MI_avg, r.T_avg, .5):.4f} |')

# ---------- Bảng 2: mỗi lambda, so J của RL với baseline tốt nhất ----------
if len(rl):
    out.append('\n## Bảng 2. Ở mỗi λ: J của RL so với baseline tốt nhất theo đúng J(λ) đó (Wilcoxon cặp theo wave, Holm)\n')
    out.append('| tag | λ | baseline tốt nhất | J baseline | J RL | ΔJ | p Holm |\n|---|---|---|---|---|---|---|')
    rows = []
    for _, r in rl[rl.tag.isin(['main', 'main2'])].iterrows():
        lam = r.lam; best = None
        for _, x in g.iterrows():
            name = x.policy if np.isnan(x.lam) else f'GreedyJ_{x.lam}'
            if x.policy == 'GreedyJ' and abs(x.lam - lam) > 1e-9: continue   # GreedyJ chỉ so ở đúng λ của nó
            j = J(x.MI, x['T'], lam)
            if best is None or j < best[1]: best = (name, j)
        hb = per_wave(best[0]); hr = per_wave(f'RL_{r.tag}_{lam}')
        cb, cr = J(hb[:, 0], hb[:, 2], lam), J(hr[:, 0], hr[:, 2], lam)
        p = stats.wilcoxon(cr, cb).pvalue
        rows.append([r.tag, lam, best[0], cb.mean(), cr.mean(), cr.mean() - cb.mean(), p])
    adj = holm([x[-1] for x in rows])
    for x, pa in zip(rows, adj):
        out.append(f'| {x[0]} | {x[1]:.1f} | {x[2]} | {x[3]:.4f} | {x[4]:.4f} | {x[5]:+.4f} | {pa:.1e} |')

# ---------- Bảng 3: GRO ----------
gro = {os.path.basename(f)[4:-4]: pd.read_csv(f) for f in sorted(glob.glob(R + 'gro_*.csv')) if not f.endswith('gro_deAssis.csv')}
if gro:
    base = gro['Random'].dist.values
    out.append('\n## Bảng 3. Kiểm chứng bằng GRO trên 300 wave kiểm tra cách đều (Wilcoxon so với Random)\n')
    out.append('| Chính sách | TB | Trung vị | ĐLC | Min | Q1 | Q3 | Max | so với Random | p |\n|---|---|---|---|---|---|---|---|---|---|')
    for k, d in gro.items():
        x = d.dist.values; p = stats.wilcoxon(x, base).pvalue if k != 'Random' else np.nan
        out.append(f'| {k} | {x.mean():.1f} | {np.median(x):.1f} | {x.std(ddof=1):.1f} | {x.min():.1f} | {np.percentile(x, 25):.1f} | '
                   f'{np.percentile(x, 75):.1f} | {x.max():.1f} | {x.mean() / base.mean() - 1:+.1%} | {p:.1e} |')
    keys = [k for k in ['Random', 'Nearest', 'Turnover', 'DA_heuristic', 'DA_supervised', 'DA_adaptive', 'GreedyJ_0.5', 'RL_main2_0.5'] if k in gro]
    fr = stats.friedmanchisquare(*[gro[k].dist.values for k in keys])
    out.append(f'\nFriedman ({", ".join(keys)}): χ² = {fr.statistic:.1f}, p = {fr.pvalue:.1e}')

# ---------- Bảng 4: ablation ----------
ab = rl[rl.tag.str.startswith('ab') | rl.tag.isin(['main', 'main2'])]
if len(ab):
    desc = {'main2': 'MÔ HÌNH A (đầy đủ): bắt chước heuristic tốt nhất theo λ + PPO 400k bước',
            'main': 'A nhưng chỉ bắt chước GreedyJ (gốc so sánh cho các dòng dưới) + PPO 600k',
            'ab_BConly': 'chỉ bắt chước GreedyJ, KHÔNG PPO',
            'ab_gamma0': 'γ = 0: không nhìn xa (chỉ reward tức thời)',
            'ab_noBC': 'không bắt chước, PPO từ đầu',
            'ab_noDist': 'bỏ đặc trưng khoảng cách (5, 6, 8, 10)',
            'ab_noMIfeat': 'bỏ đặc trưng mixing (2, 3, 4)',
            'ab_noPrior': 'bỏ đường tắt ΔJ tức thời + không bắt chước',
            'ab_noTref': 'không dùng T_ref giảm nhiễu reward',
            'ab_lam05': 'agent chuyên cho λ = 0,5 (không dùng chung λ)',
            'ab2_BConly': 'A nhưng KHÔNG PPO (chỉ bắt chước heuristic tốt nhất theo λ)',
            'ab2_gamma0': 'A với γ = 0 (PPO nhưng không nhìn xa)'}
    out.append('\n## Bảng 4. Ablation study (J trên toàn bộ 2.913 wave kiểm tra; Δ so với gốc tương ứng; dương = kém hơn)\n')
    out.append('| Biến thể | Thay đổi | J(0,2) | J(0,5) | J(0,8) | so với | Δ TB |\n|---|---|---|---|---|---|---|')
    refs = {t: {r.lam: J(r.MI_avg, r.T_avg, r.lam) for _, r in rl[rl.tag == t].iterrows()} for t in ['main', 'main2']}
    for tag in ['main2'] + sorted(t for t in ab.tag.unique() if t.startswith('ab2_')) + ['main'] + sorted(t for t in ab.tag.unique() if t.startswith('ab_')):
        base_t = 'main2' if tag.startswith('ab2_') or tag == 'main2' else 'main'
        if tag == 'main2': base_t = 'main'
        ref = refs[base_t]
        d = ab[ab.tag == tag].set_index('lam'); cells = []; deltas = []
        for lv in [0.2, 0.5, 0.8]:
            if lv in d.index:
                j = J(d.loc[lv].MI_avg, d.loc[lv].T_avg, lv); cells.append(f'{j:.4f}'); deltas.append(j - ref.get(lv, np.nan))
            else:
                cells.append('–')
        out.append(f'| {tag} | {desc.get(tag, "")} | {" | ".join(cells)} | {base_t if tag != "main" else "–"} | {np.nanmean(deltas):+.4f} |')

# ---------- Bảng 5: theo tháng (độ bền) ----------
import pandas as _pd
co = _pd.read_csv('data/Customer_Order.csv', sep=';', encoding='utf-8-sig'); co.columns = [c.strip() for c in co.columns]
co['dt'] = _pd.to_datetime(co.creationDate, format='%d/%m/%Y %H:%M')
month = co.groupby('waveNumber').dt.max().dt.month
I = dict(np.load('inst.npz', allow_pickle=True)); wid = I['wave_id'][int(I['n_train']):]
mon = np.array([month.get(w, np.nan) for w in wid])
names = ['Random', 'Nearest', 'Turnover', 'DA_heuristic', 'DA_supervised', 'DA_adaptive', 'GreedyJ_0.5', 'RL_main2_0.5']
if all(os.path.exists(R + f'picks_{n}.npz') for n in names):
    out.append('\n## Bảng 5. Độ bền theo tháng: J(0,5) trung bình theo wave trong từng tháng kiểm tra\n')
    ms = [m for m in [7, 8, 9, 10] if (mon == m).sum() > 0]
    out.append('| Chính sách | ' + ' | '.join(f'T{m} ({(mon == m).sum()} wave)' for m in ms) + ' |\n|---|' + '---|' * len(ms))
    for n in names:
        h = per_wave(n); c = J(h[:, 0], h[:, 2], 0.5)
        out.append(f'| {n} | ' + ' | '.join(f'{c[mon == m].mean():.4f}' for m in ms) + ' |')

# ---------- Pareto ----------
pts = [(r.policy if np.isnan(r.lam) else f'GreedyJ_{r.lam}', r.MI, r['T']) for _, r in g.iterrows()]
pts += [(f'RL_{r.tag}_{r.lam}', r.MI_avg, r.T_avg) for _, r in rl[rl.tag.isin(['main2'])].iterrows()]
nd = [p for p in pts if not any(q[1] <= p[1] and q[2] <= p[2] and (q[1] < p[1] or q[2] < p[2]) for q in pts)]
out.append('\nCác điểm KHÔNG bị trội (Pareto) trong (MI, T): ' + ', '.join(f'{p[0]} (MI {p[1]:.3f}, T {p[2]:.0f})' for p in sorted(nd, key=lambda x: x[1])))

# ---------- proxy vs lộ trình ----------
m = g.dropna(subset=['T'])
rho = stats.spearmanr(m.Dr, m['T'])[0]
out.append(f'\nTương quan thứ hạng giữa proxy D/D0 và lộ trình thật T trên {len(m)} baseline: Spearman ρ = {rho:.2f}')

open(R + 'bang_ket_qua.md', 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))

# ---------- Hình 1: đánh đổi MI – T/T0 ----------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
gj = g[g.policy == 'GreedyJ'].sort_values('lam')
ax.plot(gj.MI, gj['T'] / T0, 'o-', color='#4C78A8', label='GreedyJ, λ = 0 … 1')
for _, r in gj.iterrows():
    ax.annotate(f'{r.lam:.1f}', (r.MI, r['T'] / T0), fontsize=7, xytext=(3, 3), textcoords='offset points', color='#4C78A8')
for name, mk, c in [('Random', 's', '#888888'), ('Nearest', '^', '#E45756'), ('Turnover', 'D', '#54A24B'), ('DA_heuristic', 'v', '#B279A2'), ('DA_supervised', '<', '#9D755D'), ('DA_adaptive', '>', '#FF9DA6')]:
    r = g[g.policy == name].iloc[0]; ax.scatter(r.MI, r['T'] / T0, marker=mk, s=70, color=c, label=name, zorder=3)
for tag, col, lab in [('main2', '#F58518', 'Mô hình A (RL), λ = 0 … 1'), ('main', '#FFBF79', 'A – chỉ bắt chước GreedyJ')]:
    d = rl[rl.tag == tag].sort_values('lam')
    if not len(d): continue
    ax.plot(d.MI_avg, d.T_avg / T0, 'o--', label=lab, color=col, zorder=4)
    if tag == 'main2':
        for _, r in d.iterrows():
            ax.annotate(f'{r.lam:.1f}', (r.MI_avg, r.T_avg / T0), fontsize=7, xytext=(3, -9), textcoords='offset points', color=col)
ax.set_xlabel('MI trung bình (thấp hơn = kho gọn hơn)'); ax.set_ylabel('T / T0 (thấp hơn = đi ngắn hơn)')
ax.set_title('Đánh đổi mixing – quãng đường trên 2.913 wave kiểm tra'); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(R + 'fig1_danh_doi.png', dpi=160)

# ---------- Hình 2: GRO boxplot ----------
if gro:
    keys = [k for k in ['Random', 'DA_heuristic', 'DA_adaptive', 'Turnover', 'Nearest', 'GreedyJ_0.5', 'RL_main2_0.0', 'RL_main2_0.5', 'RL_main2_0.8'] if k in gro]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.boxplot([gro[k].dist for k in keys], showmeans=True)
    ax.set_xticks(range(1, len(keys) + 1), keys, rotation=20, fontsize=8)
    ax.set_ylabel('Quãng đường GRO / wave'); ax.set_title('Kiểm chứng bằng GRO – 300 wave kiểm tra'); ax.grid(axis='y', alpha=.3)
    fig.tight_layout(); fig.savefig(R + 'fig2_gro_boxplot.png', dpi=160)

# ---------- Hình 3: theo thời gian ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for name, c in [('Random', '#888888'), ('Nearest', '#E45756'), ('DA_adaptive', '#B279A2'), ('GreedyJ_0.5', '#4C78A8'), ('RL_main2_0.5', '#F58518')]:
    if not os.path.exists(R + f'picks_{name}.npz'): continue
    h = per_wave(name); k = 50
    axs[0].plot(np.convolve(h[:, 0], np.ones(k) / k, 'valid'), color=c, label=name)
    axs[1].plot(np.convolve(h[:, 2], np.ones(k) / k, 'valid') / T0, color=c, label=name)
axs[0].set_title('MI theo thời gian (TB trượt 50 wave)'); axs[1].set_title('T/T0 theo thời gian (TB trượt 50 wave)')
for a in axs: a.set_xlabel('wave kiểm tra thứ'); a.grid(alpha=.3)
axs[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(R + 'fig3_theo_thoi_gian.png', dpi=160)
