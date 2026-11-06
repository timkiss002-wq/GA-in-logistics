"""
Phân tích kết quả LP-RL (bài v6). Sinh mọi bảng/hình/số cho bài từ file kết quả, không chép tay.
Đầu ra: results/lp/summary.json, results/lp/bang_*.md, results/lp/fig_tradeoff.pdf, results/lp/fig_curve.pdf
"""
import glob, json, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib as mpl
import matplotlib.pyplot as plt
from lp_rl import LAMS, MI0, T0, FEAT, D

OUT = 'results/lp'
mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8})


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def cw(h, lam): return lam * h[:, 0] / MI0 + (1 - lam) * h[:, 2] / T0


# ---------------- baselines ----------------
BASE = {}
for name in ['Random', 'Nearest', 'Turnover', 'DA_heuristic', 'DA_supervised', 'DA_adaptive']:
    hs = [np.load(f) for f in sorted(glob.glob(f'{OUT}/base_{name}_s*.npy'))]
    if hs: BASE[name] = hs
GJ = {l: np.load(f'{OUT}/base_GreedyJ_{l}_s0.npy') for l in LAMS if os.path.exists(f'{OUT}/base_GreedyJ_{l}_s0.npy')}
LABEL = {'Random': 'Random (current)', 'Nearest': 'Nearest', 'Turnover': 'Turnover', 'GreedyJ': 'GreedyJ',
         'DA_heuristic': 'de Assis heuristic', 'DA_supervised': 'de Assis supervised', 'DA_adaptive': 'de Assis adaptive'}


def base_cw(name, lam):
    if name == 'GreedyJ': return cw(GJ[lam], lam)
    return np.mean([cw(h, lam) for h in BASE[name]], 0)


def base_stats(name, lam):
    hs = [GJ[lam]] if name == 'GreedyJ' else BASE[name]
    return dict(MI=np.mean([h[:, 0].mean() for h in hs]), T=np.mean([h[:, 2].mean() for h in hs]),
                J=np.mean([cw(h, lam).mean() for h in hs]))


# ---------------- LP-RL ----------------
def lp_hist(tag, lam):
    z = np.load(f'{OUT}/picks_{tag}_{lam}.npz'); return z['hist']


def tags(prefix):
    return sorted(t[len('models_lp/'):-5] for t in glob.glob(f'models_lp/{prefix}_s*.json')
                  if os.path.exists(f'{OUT}/eval_{t[len("models_lp/"):-5]}_test.csv'))


MAIN = tags('lp')
S = {'main_tags': MAIN}


def lp_stats(tagl, lam):
    hs = [lp_hist(t, lam) for t in tagl]
    v = lambda f: (float(np.mean([f(h) for h in hs])), float(np.std([f(h) for h in hs], ddof=1)) if len(hs) > 1 else 0.0)
    return dict(MI=v(lambda h: h[:, 0].mean()), T=v(lambda h: h[:, 2].mean()), J=v(lambda h: cw(h, lam).mean()))


# Bảng 1: kết quả tại λ = 0,5
lam = 0.5; rows = []
for n in ['Random', 'DA_heuristic', 'DA_supervised', 'DA_adaptive', 'Nearest', 'Turnover', 'GreedyJ']:
    st = base_stats(n, lam); rows.append(dict(policy=LABEL[n], MI=st['MI'], T=st['T'], J=st['J']))
st = lp_stats(MAIN, lam)
rows.append(dict(policy='LP-RL (ours)', MI=st['MI'][0], MI_sd=st['MI'][1], T=st['T'][0], T_sd=st['T'][1], J=st['J'][0], J_sd=st['J'][1]))
t1 = pd.DataFrame(rows); S['table_main'] = t1.to_dict('records')
rnd = base_stats('Random', lam); S['vs_random'] = dict(T_pct=100 * (1 - st['T'][0] / rnd['T']), MI_pct=100 * (1 - st['MI'][0] / rnd['MI']))
near = base_stats('Nearest', lam); S['vs_nearest'] = dict(T_pct=100 * (1 - st['T'][0] / near['T']), MI_pct=100 * (1 - st['MI'][0] / near['MI']))
dah = base_stats('DA_heuristic', lam); S['vs_deassis'] = dict(T_pct=100 * (1 - st['T'][0] / dah['T']), MI_pct=100 * (st['MI'][0] / dah['MI'] - 1))

# Bảng 2: theo λ, so với baseline tốt nhất, kiểm định cặp theo wave (Wilcoxon), Holm trên mọi phép thử
cands = ['Nearest', 'Turnover', 'GreedyJ', 'DA_heuristic', 'DA_supervised', 'DA_adaptive', 'Random']
rows, pv, keys = [], [], []
for lam in LAMS:
    bj = {n: base_cw(n, lam).mean() for n in cands}; best = min(bj, key=bj.get); bc = base_cw(best, lam)
    d = []
    for t in MAIN:
        c = cw(lp_hist(t, lam), lam); d.append(c.mean() - bc.mean())
        p = wilcoxon(c, bc).pvalue; pv.append(p); keys.append((lam, t, np.sign(c.mean() - bc.mean())))
    st = lp_stats(MAIN, lam)
    rows.append(dict(lam=lam, best=LABEL[best], J_best=bj[best], J_lp=st['J'][0], J_lp_sd=st['J'][1],
                     dJ=float(np.mean(d)), dJ_sd=float(np.std(d, ddof=1)), MI_lp=st['MI'][0], T_lp=st['T'][0]))
adj = holm(pv)
for r in rows:
    k = [(a < 0.05, s) for (l, t, s), a in zip(keys, adj) if l == r['lam']]
    r['better'] = sum(1 for sig, s in k if sig and s < 0); r['worse'] = sum(1 for sig, s in k if sig and s > 0)
t2 = pd.DataFrame(rows); S['table_lambda'] = t2.to_dict('records')

# Bảng 3: ablation (Δ = biến thể − LP-RL cùng seed)
AB = {'ab_wave': 'Reward only per wave (no split)', 'ab_gamma0': r'Myopic ($\gamma=0$)',
      'ab_nomix': 'No mixing features (2--5)', 'ab_nolam': r'No $\lambda X$ block'}
rows = []
for p, lab in AB.items():
    tl = tags(p)
    if not tl: continue
    per = {l: [] for l in [0.2, 0.5, 0.8]}
    for t in tl:
        s = t.split('_s')[-1]; mt = f'lp_s{s}'
        for l in per:
            per[l].append(cw(lp_hist(t, l), l).mean() - cw(lp_hist(mt, l), l).mean())
    allm = [np.mean([per[l][i] for l in per]) for i in range(len(tl))]
    rows.append(dict(variant=lab, seeds=len(tl), **{f'd{l}': float(np.mean(per[l])) for l in per},
                     mean=float(np.mean(allm)), sd=float(np.std(allm, ddof=1)) if len(allm) > 1 else 0.0))
t3 = pd.DataFrame(rows); S['table_ablation'] = t3.to_dict('records')

# GreedyJ viết đúng dưới dạng θ tuyến tính (θ đặt tay) đã có trong bảng chính -> nêu trong bài
# Bảng 4: học được θ nào (trọng số hiệu dụng w(λ) = θ_x + λ θ_λx, đơn vị: theo đặc trưng đã chuẩn hoá)
TH = np.array([json.load(open(f'models_lp/{t}.json'))['theta'] for t in MAIN])
S['theta_mean'] = TH.mean(0).tolist(); S['theta_sd'] = TH.std(0, ddof=1).tolist()
rows = []
for j, f in enumerate(FEAT):
    w0 = TH[:, j] + 0.0 * TH[:, D + j]; w5 = TH[:, j] + 0.5 * TH[:, D + j]; w1 = TH[:, j] + 1.0 * TH[:, D + j]
    rows.append(dict(feature=f, w0=w0.mean(), w0_sd=w0.std(ddof=1), w05=w5.mean(), w05_sd=w5.std(ddof=1),
                     w1=w1.mean(), w1_sd=w1.std(ddof=1)))
t4 = pd.DataFrame(rows); S['table_theta'] = t4.to_dict('records')

# Bảng 5: overfitting – J ở λ = 0,5 trên tập huấn luyện / kiểm định / kiểm tra
rows = []
for t in MAIN:
    r = {}
    for sp in ['train', 'val', 'test']:
        f = f'{OUT}/eval_{t}_{sp}.csv'
        if os.path.exists(f):
            e = pd.read_csv(f); r[sp] = float(e.loc[np.isclose(e.lam, 0.5), 'J'].iloc[0])
    m = json.load(open(f'models_lp/{t}.json'))
    rows.append(dict(tag=t, **r, train_min=m['train_sec'] / 60, best_step=m['best_step']))
t5 = pd.DataFrame(rows); S['table_fit'] = t5.to_dict('records')
if os.path.exists(f'{OUT}/split_baselines.csv'):
    S['split_baselines'] = pd.read_csv(f'{OUT}/split_baselines.csv').to_dict('records')

json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
for i, t in enumerate([t1, t2, t3, t4, t5], 1):
    open(f'{OUT}/bang_{i}.md', 'w').write(t.round(4).to_markdown(index=False))
    print(f'--- Bảng {i}'); print(t.round(4).to_string(index=False))
print(json.dumps({k: S[k] for k in ['vs_random', 'vs_nearest', 'vs_deassis']}, indent=1, default=float))

# ---------------- Hình đánh đổi ----------------
fig, ax = plt.subplots(figsize=(3.45, 3.1))
mk = {'Random': ('s', '#7f7f7f'), 'Nearest': ('^', '#1f77b4'), 'Turnover': ('v', '#17becf'),
      'DA_heuristic': ('D', '#9467bd'), 'DA_supervised': ('d', '#c5b0d5'), 'DA_adaptive': ('P', '#8c564b')}
for n, (m, c) in mk.items():
    st = base_stats(n, 0.5); ax.plot(st['T'], st['MI'], m, color=c, ms=5, label=LABEL[n], ls='none')
g = np.array([[GJ[l][:, 2].mean(), GJ[l][:, 0].mean()] for l in LAMS])
ax.plot(g[:, 0], g[:, 1], '-o', color='#ff7f0e', ms=2.5, lw=0.8, label=r'GreedyJ ($\lambda$=0..1)')
lp = np.array([[lp_stats(MAIN, l)['T'][0], lp_stats(MAIN, l)['MI'][0], lp_stats(MAIN, l)['T'][1], lp_stats(MAIN, l)['MI'][1]] for l in LAMS])
ax.errorbar(lp[:, 0], lp[:, 1], xerr=lp[:, 2], yerr=lp[:, 3], fmt='-o', color='#d62728', ms=2.5, lw=0.9, elinewidth=0.5,
            capsize=0, ecolor='#d62728', alpha=0.9, label=r'LP-RL ($\lambda$=0..1, $\pm$s.d.)')
ax.annotate(r'$\lambda=0$', lp[0, :2], xytext=(4, 4), textcoords='offset points', fontsize=6, color='#d62728')
ax.annotate(r'$\lambda=1$', lp[-1, :2], xytext=(4, -8), textcoords='offset points', fontsize=6, color='#d62728')
ax.set_xlabel('Route length per wave $T$ (dataset units)'); ax.set_ylabel('Mixing index $MI$')
ax.legend(fontsize=5.8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2); ax.grid(lw=0.3, alpha=0.5)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_tradeoff.pdf'); fig.savefig(f'{OUT}/fig_tradeoff.png', dpi=220)

# ---------------- Đường học (overfitting) ----------------
fig, ax = plt.subplots(figsize=(3.45, 2.3))
for i, t in enumerate(MAIN):
    m = json.load(open(f'models_lp/{t}.json')); col = f'C{i}'
    for r, pc in enumerate(m.get('probe_curves', [])):        # 3 lần khởi động ngắn (0–100k), nét chấm
        pc = pd.DataFrame(pc); ax.plot(pc.step / 1000, pc.val_J, lw=0.8 if r == m['chosen_restart'] else 0.5,
                                       ls='-' if r == m['chosen_restart'] else ':', color=col, alpha=1 if r == m['chosen_restart'] else 0.6)
    c = pd.DataFrame(m['curve']); off = m.get('probe', 0)                  # học tiếp từ θ tốt nhất của lần được chọn
    ax.plot((c.step + off) / 1000, c.val_J, lw=0.9, color=col, label=t.replace('lp_s', 'seed '))
    k = c.val_J.idxmin() if m['best_step'] > 0 else None
for n, c in [('Nearest', '#1f77b4'), ('GreedyJ', '#ff7f0e')]:
    if os.path.exists(f'{OUT}/split_baselines.csv'):
        sb = pd.read_csv(f'{OUT}/split_baselines.csv'); v = sb[(sb.policy == n) & (sb.split == 'val')].J.mean()
        ax.axhline(v, color='k', ls='--', lw=0.6)
ax.text(405, 0.832, 'Nearest,\nGreedyJ', fontsize=5.5, va='center')
ax.axvline(100, color='grey', lw=0.5); ax.text(50, 0.912, 'three starts', ha='center', fontsize=6); ax.text(250, 0.912, 'continuation of the kept start', ha='center', fontsize=6)
ax.set_xlabel('Training decisions (thousands)'); ax.set_ylabel('Validation $J$')
ax.set_ylim(0.80, 0.92); ax.set_xlim(0, 460)
ax.legend(fontsize=5.5, frameon=False, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.07), handlelength=1.2); ax.grid(lw=0.3, alpha=0.5)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_curve.pdf'); fig.savefig(f'{OUT}/fig_curve.png', dpi=220)
