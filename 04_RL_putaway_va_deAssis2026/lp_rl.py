"""
LP-RL: Linear Putaway policy trained by Reinforcement Learning (bài v6, 06/11/2026; thay Model A).

Thay đổi so với Model A (theo góp ý mentor 05/11/2026):
  1. KHÔNG lọc trước 64 ô ứng viên. Chính sách chấm điểm MỌI ô còn chỗ (N = 1.692 ô, ô đầy bị mặt nạ).
  2. Mô hình là HÀM TUYẾN TÍNH, không phải mạng nơ-ron:
        X_t  ∈ R^{N×d}    đặc trưng của các ô (d = 9, danh sách FEAT), chuẩn hoá (trừ TB, chia ĐLC đo khi chạy Random)
        Φ_t  = [X_t, λ·X_t] ∈ R^{N×2d}   (λ nhân vào để MỘT bộ tham số dùng được cho mọi λ)
        θ    ∈ R^{2d}     tham số (18 số);  trọng số hiệu dụng của đặc trưng j: w_j(λ) = θ_j + λ·θ_{d+j}
        s_t  = Φ_t θ ∈ R^N  điểm của các ô;  π = softmax(s_t trên các ô còn chỗ)
     Huấn luyện: chọn ô theo xác suất π (khám phá). Đánh giá: chọn ô có điểm cao nhất.
     θ = 0 -> π đều = Random (hiện trạng). GreedyJ cũng thuộc họ này (θ đặt tay, xem theta_greedyJ).
  3. Học θ bằng RL: policy gradient kiểu actor–critic (REINFORCE + baseline học được + GAE), Adam.
     ∇θ log π(a) = φ_a − Σ_l π_l φ_l   (công thức đóng, không cần torch)
     Critic: V(s) = g(s)·ψ, g = 6 đặc trưng toàn kho, ψ giải bằng bình phương tối thiểu (ridge) mỗi vòng.
  4. TÁCH REWARD (reward='decomp', mặc định): phần trộn của tổng reward được ghi nhận ngay tại quyết định gây ra nó:
        r_t = −λ/MI0 · n_t · ΔMI_t  +  (1−λ)(1 − T_w/T0)·[wave w được lấy sau bước t]
        ΔMI_t = MI(s_{t+1}) − MI(s_t), n_t = số wave của đoạn còn chưa lấy.
     Không chiết khấu thì tổng reward bằng tổng reward theo wave (sai khác một hằng số) -> cùng mục tiêu J.
     reward='wave': reward 1 − c_w chỉ khi wave được lấy (dùng cho ablation).
  5. Chống overfit: chia 3 phần theo thời gian
        huấn luyện  : wave 2000–6493  (đoạn 100 wave, λ bốc ngẫu nhiên mỗi đoạn)
        kiểm định   : wave 6494–6793  (300 wave, KHÔNG dùng để học, chỉ để chọn θ tốt nhất)
        kiểm tra    : wave 6794–9706  (2.913 wave, chạy một lần ở cuối)

Chạy:   python lp_rl.py train --tag lp_s0 --seed 0
        python lp_rl.py eval  --tag lp_s0 [--split train|val|test] [--lams 0.5]
Ablation: --reward wave | --gamma 0 | --drop mix | --nolam
"""
import argparse, json, os, time
import numpy as np
from env import PutawayEnv, TREF

I = dict(np.load('inst.npz', allow_pickle=True)); S = np.load('s0.npz')
W_TRAIN0 = int(S['w_start'])                 # 2000
W_TEST = int(I['n_train'])                   # 6794
N_TEST = len(I['wave_id']) - W_TEST          # 2913
N_VAL = 300
W_VAL = W_TEST - N_VAL                       # 6494
LAMS = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
_C = json.load(open('t0.json')); MI0, T0 = _C['MI0'], _C['T0']

FEAT = ['free', 'sameSKU', 'sameModel', 'MI', 'dMI', 'dist', 'cut', 'cutDemand', 'heat']
GROUPS = {'mix': [1, 2, 3, 4], 'dist': [5, 6, 7]}
D = len(FEAT)


def raw_features(env):
    """X ∈ R^{N×d}: đặc trưng của mọi ô cho đơn vị đang chờ cất (lấy từ env.obs())."""
    F = env.obs(); k = env.queue[0]
    pk = env.p[k] * env.K                                     # nhu cầu tương đối của SKU (TB = 1)
    return np.stack([F[:, 0], F[:, 1], F[:, 2], F[:, 3], F[:, 4] / 10,
                     F[:, 5], F[:, 6], F[:, 6] * pk, F[:, 7]], 1)


class Policy:
    def __init__(self, theta, mu, sd, drop=(), use_lam=True):
        self.theta = np.asarray(theta, float); self.mu, self.sd = mu, sd
        self.keep = np.ones(D); self.keep[list(drop)] = 0.0; self.use_lam = use_lam

    def phi(self, env):
        X = (raw_features(env) - self.mu) / self.sd * self.keep        # N×d
        lam = env.lam if self.use_lam else 0.0
        return np.concatenate([X, lam * X], 1)                          # N×2d

    def scores(self, env):
        P = self.phi(env); s = P @ self.theta
        return P, np.where(env.mask(), s, -np.inf)


def standardise(seed=0, n_waves=300):
    """TB và ĐLC của từng đặc trưng trên các ô còn chỗ, đo khi chạy Random trên wave huấn luyện."""
    env = PutawayEnv(I, s0=S['s0']); rng = np.random.default_rng(seed)
    env.reset(W_TRAIN0, n_waves, None, 0.5); xs = []
    while not env.done():
        m = env.mask()
        if rng.random() < 0.1: xs.append(raw_features(env)[m])
        env.step(int(rng.choice(np.flatnonzero(m))))
    X = np.concatenate(xs); return X.mean(0), X.std(0) + 1e-8


def global_feat(env):
    near = env.d_rank < 0.2
    return np.array([1.0, env.mi_mean() / env.MI0, env.D / env.D0, env.lam, env.lam ** 2,
                     (env.cap - env.tot[near]).sum() / (env.cap * near.sum())])


def evaluate(pol, lam, w_start, n_waves, seed=0, save=None):
    """Chạy greedy (argmax) và trả về lịch sử từng wave: (MI_w, D_w, T_w)."""
    env = PutawayEnv(I, lam=lam, s0=S['s0']); env.reset(w_start, n_waves, None, lam)
    while not env.done():
        _, s = pol.scores(env); env.step(int(np.argmax(s)))
    h = np.array(env.hist)
    if save:
        np.savez_compressed(save, hist=h,
                            cells=np.array([c for w in env.picks for c, _ in w]),
                            ptr=np.cumsum([0] + [len(w) for w in env.picks]))
    return h


def J_of(h, lam):
    return float(lam * h[:, 0].mean() / MI0 + (1 - lam) * h[:, 2].mean() / T0)


def theta_greedyJ(sd, scale=3.0):
    """θ cho đúng luật GreedyJ (chọn ô có ΔĴ nhỏ nhất) trên đặc trưng đã chuẩn hoá:
    điểm = −ΔĴ = −λ·c5·x5 + (1−λ)·c8·x8  ->  θ[x8] = c8·σ8, θ[λx8] = −c8·σ8, θ[λx5] = −c5·σ5 (rồi nhân hệ số)."""
    env = PutawayEnv(I, s0=S['s0'])
    c5 = 1 / (env.L * env.MI0); c8 = 2 * env.dmax / (env.D0 * env.K)
    th = np.zeros(2 * D); th[7] = c8 * sd[7]; th[D + 7] = -c8 * sd[7]; th[D + 4] = -c5 * sd[4]
    return th / np.abs(th).max() * scale


def put_dJ(env, a):
    """Thay đổi tức thời của J ước lượng (MI trung bình và quãng đường kỳ vọng D) do riêng việc cất vào ô a."""
    F = env.obs(); k = env.queue[0]; lam = env.lam
    dmi = (F[a, 4] / 10) / env.L / env.MI0
    dd = -env.p[k] * 2 * F[a, 6] * env.dmax / env.D0
    return lam * dmi + (1 - lam) * dd


def train(tag, seed=0, steps=400_000, T=4096, n_env=8, gamma=0.99, gae_lam=0.95, lr=0.05, lr_decay=0.5, drop=(),
          use_lam=True, lam_fixed=None, seg=100, p_reset=0.2, val_every=5, reward='decomp', init='zero',
          init_scale=2.0, init_theta=None, save=True, log=print):
    """Policy gradient (actor–critic). Mỗi lần cập nhật gom T quyết định từ n_env kho mô phỏng chạy song song,
    mỗi kho một đoạn wave và một λ riêng -> một lô luôn có nhiều λ khác nhau (gradient ổn định hơn)."""
    rng = np.random.default_rng(seed)
    mu, sd = standardise(seed)
    th0 = np.array(init_theta) if init_theta is not None else (theta_greedyJ(sd, init_scale) if init == 'greedy' else np.zeros(2 * D))
    pol = Policy(th0, mu, sd, drop, use_lam)
    m_adam = np.zeros(2 * D); v_adam = np.zeros(2 * D); b1, b2 = 0.9, 0.999
    psi = np.zeros(6)
    workers = [dict(env=PutawayEnv(I, s0=S['s0'], use_tref=True), state=None) for _ in range(n_env)]

    def new_segment(wk):
        lam = lam_fixed if lam_fixed is not None else float(rng.choice(LAMS))
        w0 = int(rng.integers(W_TRAIN0, W_VAL - seg))
        st = None if (wk['state'] is None or rng.random() < p_reset) else wk['state']
        wk['env'].reset(w0, seg, st, lam)

    def val_J(theta):
        p = Policy(theta, mu, sd, drop, use_lam)
        lams = [lam_fixed] if lam_fixed is not None else [0.2, 0.5, 0.8]
        return float(np.mean([J_of(evaluate(p, l, W_VAL, N_VAL), l) for l in lams]))

    def step(env, a):
        """Thực hiện quyết định a; trả về reward và cờ hết đoạn."""
        if reward == 'wave':                     # reward 1 − c_w chỉ khi wave được lấy
            return env.step(a)
        # Tách reward (tổng không đổi nếu không chiết khấu, chỉ dời thời điểm ghi nhận phần MI):
        #   phần trộn: −λ/MI0 · ΔMI_t · n_t,  ΔMI_t = MI(s_{t+1}) − MI(s_t), n_t = số wave còn lại của đoạn;
        #   phần quãng đường: (1−λ)(1 − T_w/T0) khi wave w được lấy.
        n_rem = env.w_end - env.w; mi0 = env.mi_mean(); nh = len(env.hist)
        _, d = env.step(a)
        r = -env.lam / env.MI0 * (env.mi_mean() - mi0) * n_rem
        if len(env.hist) > nh:
            Tw = env.hist[-1][2]; w = env.w - 1
            Tc = Tw - TREF[w] + T0 if TREF is not None else Tw    # bỏ phần nhiễu do chính wave (chỉ khi học)
            r += (1 - env.lam) * (1 - Tc / T0)
        return r, d

    for wk in workers: new_segment(wk)
    best = (val_J(pol.theta), pol.theta.copy(), 0)
    hist = [dict(step=0, val_J=best[0], sec=0.0)]
    log(f'[{tag}] bước 0: J kiểm định = {best[0]:.4f} (θ ban đầu: {init})')
    t0 = time.time(); done_steps = 0; it = 0; n_w = T // n_env
    while done_steps < steps:
        Gs, ADV, GFs, RETs = [], [], [], []
        for wk in workers:
            env = wk['env']
            G = np.zeros((n_w, 2 * D)); R = np.zeros(n_w); DN = np.zeros(n_w); GF = np.zeros((n_w, 6))
            for t in range(n_w):
                P, s = pol.scores(env)
                ok = np.isfinite(s); z = np.where(ok, s - s[ok].max(), -np.inf); p = np.exp(z); p /= p.sum()
                a = int(rng.choice(len(p), p=p))
                G[t] = P[a] - p[ok] @ P[ok]                      # ∇θ log π(a) = φ_a − Σ π_l φ_l
                GF[t] = global_feat(env)
                R[t], DN[t] = step(env, a)
                if DN[t]:
                    wk['state'] = env.cnt.copy(); new_segment(wk)
            V = GF @ psi; last_v = global_feat(env) @ psi
            adv = np.zeros(n_w); g = 0.0
            for t in reversed(range(n_w)):                       # GAE
                nv = last_v if t == n_w - 1 else V[t + 1]
                delta = R[t] + gamma * nv * (1 - DN[t]) - V[t]
                g = delta + gamma * gae_lam * (1 - DN[t]) * g; adv[t] = g
            Gs.append(G); ADV.append(adv); GFs.append(GF); RETs.append(adv + V)
        G, adv, GF, ret = map(np.concatenate, (Gs, ADV, GFs, RETs))
        psi = np.linalg.solve(GF.T @ GF + 1e-3 * np.eye(6), GF.T @ ret)       # critic tuyến tính (ridge)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        grad = (adv[:, None] * G).mean(0)                                   # ước lượng ∇θ E[tổng reward]
        it += 1; done_steps += n_w * n_env
        lr_t = lr * (1 - lr_decay * min(1.0, done_steps / steps))
        m_adam = b1 * m_adam + (1 - b1) * grad; v_adam = b2 * v_adam + (1 - b2) * grad ** 2
        pol.theta = pol.theta + lr_t * (m_adam / (1 - b1 ** it)) / (np.sqrt(v_adam / (1 - b2 ** it)) + 1e-8)
        if it % val_every == 0 or done_steps >= steps:
            vj = val_J(pol.theta)
            hist.append(dict(step=done_steps, val_J=vj, sec=time.time() - t0))
            flag = ''
            if vj < best[0]: best = (vj, pol.theta.copy(), done_steps); flag = '  *'
            log(f'[{tag}] bước {done_steps:7d}: J kiểm định = {vj:.4f} | {time.time() - t0:5.0f}s{flag}')
    meta = dict(tag=tag, seed=seed, steps=steps, T=T, n_env=n_env, gamma=gamma, gae_lam=gae_lam, lr=lr, lr_decay=lr_decay,
                drop=list(drop), use_lam=use_lam, lam_fixed=lam_fixed, reward=reward, init=init, init_scale=init_scale,
                theta=best[1].tolist(), theta_last=pol.theta.tolist(), best_step=best[2], best_val_J=best[0], mu=mu.tolist(), sd=sd.tolist(),
                feat=FEAT, train_sec=time.time() - t0, curve=hist)
    if save:
        os.makedirs('models_lp', exist_ok=True); json.dump(meta, open(f'models_lp/{tag}.json', 'w'), indent=1)
    return meta


def train_restarts(tag, seed=0, steps=400_000, restarts=3, probe=100_000, max_rounds=3, log=print, **kw):
    """Khởi động lại nhiều lần (mọi lựa chọn dựa trên tập KIỂM ĐỊNH, không nhìn tập kiểm tra):
    mỗi vòng chạy `restarts` lần ngắn (`probe` quyết định, hạt ngẫu nhiên khác nhau). Nếu không lần nào tốt hơn θ ban đầu
    (θ = 0) trên tập kiểm định thì chạy thêm một vòng (tối đa `max_rounds` vòng). Giữ lần tốt nhất rồi học tiếp từ θ của
    lần đó cho đủ `steps` quyết định."""
    runs = []
    for rnd in range(max_rounds):
        runs += [train(f'{tag}_r{r}', 1000 * seed + r, probe, save=False, log=log, **kw)
                 for r in range(rnd * restarts, (rnd + 1) * restarts)]
        j0 = runs[0]['curve'][0]['val_J']
        if min(m['best_val_J'] for m in runs) < j0 - 1e-9: break
        log(f'[{tag}] vòng {rnd}: không lần nào tốt hơn θ ban đầu -> thêm một vòng khởi động')
    k = int(np.argmin([m['best_val_J'] for m in runs])); log(f'[{tag}] chọn lần khởi động {k} (J kiểm định {runs[k]["best_val_J"]:.4f})')
    m = train(tag, 1000 * seed + k, steps - probe, init_theta=runs[k]['theta'], save=False, log=log, **kw)
    if m['best_val_J'] > runs[k]['best_val_J']:                     # học tiếp không tốt hơn -> giữ θ của lần chọn
        m.update(theta=runs[k]['theta'], best_val_J=runs[k]['best_val_J'], best_step=0)
    m.update(seed=seed, restarts=len(runs), probe=probe, chosen_restart=k, steps=steps,
             probe_curves=[r['curve'] for r in runs], probe_val=[r['best_val_J'] for r in runs],
             train_sec=m['train_sec'] + sum(r['train_sec'] for r in runs),
             decisions_total=len(runs) * probe + (steps - probe))
    os.makedirs('models_lp', exist_ok=True); json.dump(m, open(f'models_lp/{tag}.json', 'w'), indent=1)
    return m


def load(tag):
    m = json.load(open(f'models_lp/{tag}.json'))
    return m, Policy(m['theta'], np.array(m['mu']), np.array(m['sd']), m['drop'], m['use_lam'])


def eval_tag(tag, lams=LAMS, split='test'):
    m, pol = load(tag)
    w0, n = {'test': (W_TEST, N_TEST), 'val': (W_VAL, N_VAL), 'train': (W_TRAIN0, W_VAL - W_TRAIN0)}[split]
    os.makedirs('results/lp', exist_ok=True); rows = []
    for lam in lams:
        save = f'results/lp/picks_{tag}_{lam}.npz' if split == 'test' else None
        h = evaluate(pol, lam, w0, n, save=save)
        rows.append(dict(tag=tag, split=split, lam=lam, MI_avg=h[:, 0].mean(), T_avg=h[:, 2].mean(), J=J_of(h, lam)))
        if split == 'test': np.save(f'results/lp/cw_{tag}_{lam}.npy', lam * h[:, 0] / MI0 + (1 - lam) * h[:, 2] / T0)
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode'); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=400_000); ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--lr', type=float, default=0.05); ap.add_argument('--drop', default='')
    ap.add_argument('--nolam', action='store_true'); ap.add_argument('--lam', type=float, default=None)
    ap.add_argument('--split', default='test'); ap.add_argument('--reward', default='decomp'); ap.add_argument('--H', type=float, default=0.0); ap.add_argument('--lams', default='all'); ap.add_argument('--restarts', type=int, default=3)
    a = ap.parse_args()
    drop = tuple(i for gname in a.drop.split(',') if gname for i in GROUPS[gname])
    if a.mode == 'train':
        train_restarts(a.tag, a.seed, a.steps, restarts=a.restarts, gamma=a.gamma, lr=a.lr, drop=drop,
                       use_lam=not a.nolam, lam_fixed=a.lam, reward=a.reward)
    else:
        import pandas as pd
        lams = LAMS if a.lams == 'all' else [float(x) for x in a.lams.split(',')]
        rows = eval_tag(a.tag, lams, a.split)
        for r in rows: print(f"λ {r['lam']}: MI {r['MI_avg']:.4f} | T {r['T_avg']:.1f} | J {r['J']:.4f}")
        os.makedirs('results/lp', exist_ok=True)
        pd.DataFrame(rows).to_csv(f'results/lp/eval_{a.tag}_{a.split}.csv', index=False)
