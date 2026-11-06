"""
Hình phương pháp (kiểu hình "MORSE policy"): mặt bằng kho thật + ma trận đầu vào/đầu ra có ghi chiều.
  (a) Kho: mỗi ô tô màu theo điểm s_l mà chính sách gán cho đơn vị đang chờ cất; khung đỏ = ô được chọn.
  (b) X_t ∈ R^{N×d} → Φ_t = [X_t, λX_t] ∈ R^{N×2d} → s_t = Φ_t θ ∈ R^N → π_t = softmax(mask(s_t)) → a_t.
Chạy: python fig_method.py --tag lp_s0 --lam 0.5 --wave 7000
"""
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from env import PutawayEnv
import lp_rl
from fig_layout import draw_layout

mpl.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8})


def heat(ax, M, title, cmap='RdBu_r', vmax=2.5, rows=14):
    """Vẽ một ma trận (một số hàng đầu + '⋮' + vài hàng cuối) dưới dạng ô màu."""
    top = M[:rows - 3]; bot = M[-2:]
    Z = np.vstack([top, np.full((1, M.shape[1]), np.nan), bot])
    ax.imshow(Z, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto', interpolation='nearest')
    ax.text(M.shape[1] / 2 - 0.5, rows - 3, r'$\vdots$', ha='center', va='center', fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.6)
    ax.set_title(title, fontsize=8, pad=3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='lp_s0'); ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--wave', type=int, default=7000); ap.add_argument('--out', default='results/lp/fig_method.pdf')
    a = ap.parse_args()
    meta, pol = lp_rl.load(a.tag)
    env = PutawayEnv(lp_rl.I, lam=a.lam, s0=lp_rl.S['s0']); env.reset(a.wave, 3, None, a.lam)
    for _ in range(5):                                    # vài quyết định để kho "đang chạy"
        _, s = pol.scores(env); env.step(int(np.argmax(s)))
    P, s = pol.scores(env); m = env.mask()
    X = (lp_rl.raw_features(env) - pol.mu) / pol.sd
    a_t = int(np.argmax(s))
    z = np.where(m, s - s[m].max(), -np.inf); pi = np.exp(z); pi /= pi.sum()
    free = np.flatnonzero(m); order = free[np.argsort(-s[free])]   # chỉ các ô còn chỗ, xếp theo điểm giảm dần

    fig = plt.figure(figsize=(7.16, 3.35))
    # (a) kho
    axk = fig.add_axes([0.0, 0.07, 0.25, 0.88])
    sv = np.where(m, s, np.nan); lo, hi = np.nanpercentile(sv, [5, 99.5])
    draw_layout(axk, values=sv, cmap='viridis', vmin=lo, vmax=hi, highlight=a_t)
    axk.set_title('(a) Warehouse: score $s_l$ of each\nlocation for the arriving unit', fontsize=8, pad=2)
    cax = fig.add_axes([0.06, 0.03, 0.12, 0.014])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(lo, hi), cmap='viridis'), cax=cax,
                      orientation='horizontal'); cb.set_ticks([]); cb.outline.set_linewidth(0.4)
    fig.text(0.055, 0.037, 'low', fontsize=6, ha='right', va='center'); fig.text(0.185, 0.037, 'high', fontsize=6, va='center')

    y0, h = 0.22, 0.58
    # (b) X
    ax1 = fig.add_axes([0.33, y0, 0.08, h]); heat(ax1, X[order], r'$X_t\in\mathbb{R}^{N\times d}$')
    ax1.set_xlabel(r'$d=9$ features', fontsize=7, labelpad=2)
    fig.text(0.322, y0 + h / 2, r'locations with a free slot ($\leq N=1{,}692$)', rotation=90, va='center', ha='center', fontsize=6.5)
    # Φ
    ax2 = fig.add_axes([0.47, y0, 0.14, h]); heat(ax2, P[order], r'$\Phi_t=[X_t,\ \lambda X_t]\in\mathbb{R}^{N\times 2d}$')
    ax2.axvline(8.5, color='k', lw=0.6); ax2.set_xlabel(r'$X_t$' + '          ' + r'$\lambda X_t$', fontsize=7, labelpad=2)
    # θ
    ax3 = fig.add_axes([0.655, y0 + 0.08, 0.018, h - 0.16])
    th = pol.theta[:, None]; ax3.imshow(th, cmap='RdBu_r', vmin=-np.abs(th).max(), vmax=np.abs(th).max(), aspect='auto')
    ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_title(r'$\theta\in\mathbb{R}^{2d}$', fontsize=8, pad=3)
    # s
    ax4 = fig.add_axes([0.74, y0, 0.022, h]); sv_o = s[order][:, None]
    Z = np.vstack([sv_o[:11], [[np.nan]], sv_o[-2:]]); Z[~np.isfinite(Z)] = np.nan
    ax4.imshow(Z, cmap='viridis', vmin=lo, vmax=hi, aspect='auto'); ax4.set_xticks([]); ax4.set_yticks([])
    ax4.text(0, 11, r'$\vdots$', ha='center', va='center'); ax4.set_title(r'$s_t\in\mathbb{R}^{N}$', fontsize=8, pad=3)
    # π
    ax5 = fig.add_axes([0.83, y0, 0.07, h]); pv = pi[order]
    ax5.barh(np.arange(11), pv[:11], color='#4c72b0', height=0.8); ax5.set_ylim(13.5, -0.5)
    ax5.barh([12, 13], pv[-2:], color='#4c72b0', height=0.8); ax5.text(0, 11, r'$\vdots$', va='center')
    ax5.barh(0, pv[0], color='#d62728', height=0.8)
    ax5.set_xticks([]); ax5.set_yticks([]); ax5.set_title(r'$\pi_t\in\Delta^{N-1}$', fontsize=8, pad=3)
    for sp in ['top', 'right']: ax5.spines[sp].set_visible(False)
    fig.text(0.915, y0 + h - 0.03, r'$a_t$', color='#d62728', fontsize=9, va='center')

    def arrow(x0, x1, y, txt=None, below=None):
        fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle='-|>', mutation_scale=8, lw=0.8))
        if txt: fig.text((x0 + x1) / 2, y + 0.03, txt, ha='center', fontsize=7)
        if below: fig.text((x0 + x1) / 2, y - 0.05, below, ha='center', fontsize=6.5, color='#555555')
    ym = y0 + h / 2
    arrow(0.235, 0.29, ym, 'features', 'per loc.')
    arrow(0.415, 0.465, ym, r'$\times\lambda$', 'concat')
    arrow(0.615, 0.65, ym)
    arrow(0.676, 0.735, ym, r'$\Phi_t\theta$', 'linear')
    arrow(0.765, 0.825, ym, 'mask,', 'softmax')
    # chú thích dưới
    fig.text(0.33, 0.105, r'Training (RL): sample $a_t\sim\pi_t$; reward = mixing change $\times$ waves left, plus a route term after each wave;', fontsize=6.6)
    fig.text(0.33, 0.055, r'update $\theta\leftarrow\theta+\eta\,\hat A_t\,\nabla_\theta\log\pi_t(a_t)$ with $\nabla_\theta\log\pi_t(a)=\phi_{t,a}-\sum_l\pi_{t,l}\,\phi_{t,l}$.', fontsize=6.6)
    fig.text(0.33, 0.005, r'Test: $a_t=\arg\max_l s_{t,l}$ over free locations. Rows sorted by score (best first); full locations are masked.', fontsize=6.6)
    fig.text(0.33, 0.93, '(b) Linear putaway policy: input and output for one decision $t$ (one arriving unit)', fontsize=8)
    fig.savefig(a.out, bbox_inches='tight'); fig.savefig(a.out.replace('.pdf', '.png'), dpi=220, bbox_inches='tight')
    print('chosen', lp_rl.I['cell_name'][a_t], 'free cells', m.sum())


if __name__ == '__main__':
    main()
