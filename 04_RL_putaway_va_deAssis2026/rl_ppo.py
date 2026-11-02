"""
BƯỚC 5 – Agent RL: PPO có mặt nạ, chấm điểm từng ô ứng viên (rank-then-assign).

Ý tưởng (giống kho tí hon, chỉ to hơn):
  - Mỗi đơn vị hàng về là 1 quyết định. Không đưa cả kho 1.692 ô × 2.454 SKU vào mạng (quá lớn),
    mà mô tả TỪNG Ô bằng 13 đặc trưng (chỗ trống, cùng SKU/cùng mã, MI ô, ΔMI, khoảng cách, ...).
  - Lọc trước 64 ô ứng viên (giống bước "rank" của Pan et al. 2025): 16 ô làm J giảm nhiều nhất ngay,
    16 ô trống gần depot nhất, 16 ô đã có cùng mã giày (ΔMI nhỏ nhất), 16 ô trống ngẫu nhiên.
  - Mạng chấm điểm DÙNG CHUNG cho mọi ô: điểm_l = MLP(đặc trưng_l). Xác suất chọn = softmax(điểm) trên ô hợp lệ.
  - Mạng giá trị V(s) = MLP(trung bình & max của các ô ứng viên + đặc trưng toàn kho).
  - Một mạng cho mọi lambda: lambda là một đặc trưng đầu vào, mỗi đoạn huấn luyện bốc lambda ngẫu nhiên.
  - Khởi động bằng bắt chước (behaviour cloning). teacher='greedy': luôn bắt chước GreedyJ.
    teacher='portfolio': bắt chước heuristic tốt nhất theo λ trên dữ liệu kiểm tra nội bộ (300 wave cuối giai đoạn
    huấn luyện): Nearest khi λ < 0,5, GreedyJ khi λ ≥ 0,5. Sau đó PPO tinh chỉnh rồi PPO tinh chỉnh để học nhìn xa (gamma = 0,998).
  - Reward (env.py, chế độ 'tour'): mỗi khi một wave được lấy, agent nhận -(c_w - 1),
    c_w = lam*MI_w/MI0 + (1-lam)*T_w/T0, T_w = độ dài lộ trình thật của wave. Reward đến TRỄ so với quyết định
    -> đây chính là lý do cần RL (một quy tắc tham lam không nhìn thấy hậu quả ở các wave sau).

Huấn luyện chỉ dùng wave 2000–6793 (trước 17/07/2023). Kiểm tra trên wave 6794–9706.
Chạy:  python rl_ppo.py train --tag main --steps 400000
       python rl_ppo.py eval  --tag main
"""
import argparse, os, time, json
import numpy as np, torch, torch.nn as nn
from env import PutawayEnv, pol_greedyJ

torch.set_num_threads(int(os.environ.get('TORCH_THREADS', 1)))
I = dict(np.load('inst.npz', allow_pickle=True)); S = np.load('s0.npz')
W_TRAIN0, W_TEST = int(S['w_start']), int(I['n_train'])
N_TEST = len(I['wave_id']) - W_TEST
LAMS = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
NC, NF, NG = 64, 13, 4
_C = json.load(open('t0.json')); MI0_, T0_ = _C['MI0'], _C['T0']


# ---------------- đặc trưng + ô ứng viên ----------------
def immediate_dJ(env, F):
    k = env.queue[0]; lam = env.lam
    dMI = (F[:, 4] / 10) / env.L / env.MI0
    dD = -env.p[k] * 2 * F[:, 6] * env.dmax / env.D0
    return lam * dMI + (1 - lam) * dD


def candidates(env, rng, drop=()):
    F = env.obs(); m = env.mask(); free = np.flatnonzero(m)
    dj = immediate_dJ(env, F)
    sets = [free[np.argsort(dj[free])[:16]],
            free[np.argsort(env.d[free])[:16]]]
    same = free[F[free, 2] > 0]
    sets.append(same[np.argsort(F[same, 4])[:16]] if len(same) else free[:0])
    sets.append(rng.choice(free, min(16, len(free)), replace=False))
    c = np.unique(np.concatenate(sets))
    if len(c) < NC:
        extra = np.setdiff1d(free, c)
        c = np.concatenate([c, rng.choice(extra, min(NC - len(c), len(extra)), replace=False)])
    c = c[:NC]
    X = np.zeros((NC, NF), np.float32); valid = np.zeros(NC, bool)
    X[:len(c), :12] = F[c]; X[:len(c), 12] = dj[c] * 1000; valid[:len(c)] = True
    for j in drop: X[:, j] = 0.0                       # dùng cho ablation: bỏ một nhóm đặc trưng
    near20 = env.d_rank < 0.2
    g = np.array([env.mi_mean() / env.MI0, env.D / env.D0, env.lam,
                  (env.cap - env.tot[near20]).sum() / (env.cap * near20.sum())], np.float32)
    cells = np.full(NC, -1); cells[:len(c)] = c
    return X, valid, g, cells


# ---------------- mạng ----------------
class Net(nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(NF + NG, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.pi = nn.Linear(h, 1)
        # "đường tắt" tới ΔJ tức thời: lúc mới khởi tạo agent đã gần giống GreedyJ (mềm), PPO học phần nhìn xa
        self.alpha = nn.Parameter(torch.tensor(8.0))
        self.v = nn.Sequential(nn.Linear(2 * h + NG, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, X, valid, g):           # X: B×NC×NF, valid: B×NC, g: B×NG
        gg = g[:, None, :].expand(-1, X.shape[1], -1)
        H = self.enc(torch.cat([X, gg], -1))
        logit = (self.pi(H).squeeze(-1) - self.alpha * X[..., 12]).masked_fill(~valid, -1e9)
        Hm = H.masked_fill(~valid[..., None], 0)
        mean = Hm.sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        mx = H.masked_fill(~valid[..., None], -1e9).max(1).values
        v = self.v(torch.cat([mean, mx, g], -1)).squeeze(-1)
        return logit, v


def act(net, X, valid, g, greedy=False):
    if greedy:                                      # đánh giá: chỉ cần ô có điểm cao nhất
        with torch.inference_mode():
            logit, _ = net(torch.from_numpy(X)[None], torch.from_numpy(valid)[None], torch.from_numpy(g)[None])
            return int(logit[0].argmax()), 0.0, 0.0
    with torch.no_grad():
        logit, v = net(torch.from_numpy(X)[None], torch.from_numpy(valid)[None], torch.from_numpy(g)[None])
        dist = torch.distributions.Categorical(logits=logit[0])
        a = int(logit[0].argmax()) if greedy else int(dist.sample())
        return a, float(dist.log_prob(torch.tensor(a))), float(v[0])


# ---------------- vòng mô phỏng liên tục (không reset mỗi đoạn) ----------------
class Runner:
    def __init__(self, seed, lam_fixed=None, drop=(), seg_waves=100, p_reset=0.2, use_tref=True, teacher='greedy'):
        self.teacher = teacher
        self.env = PutawayEnv(I, s0=S['s0'], use_tref=use_tref); self.rng = np.random.default_rng(seed)
        self.lam_fixed, self.drop, self.seg, self.p_reset = lam_fixed, drop, seg_waves, p_reset
        self.state = None; self._new_segment()

    def _new_segment(self):
        lam = self.lam_fixed if self.lam_fixed is not None else float(self.rng.choice(LAMS))
        w0 = int(self.rng.integers(W_TRAIN0, W_TEST - self.seg))
        st = None if (self.state is None or self.rng.random() < self.p_reset) else self.state
        self.env.reset(w0, self.seg, st, lam)

    def collect(self, net, T, expert=False):
        buf = {k: [] for k in ['X', 'valid', 'g', 'a', 'logp', 'v', 'r', 'done']}
        for _ in range(T):
            X, valid, g, cells = candidates(self.env, self.rng, self.drop)
            if expert:                       # nhãn bắt chước: ô GreedyJ sẽ chọn
                if self.teacher == 'portfolio' and self.env.lam < 0.5:
                    a = int(np.argmin(np.where(valid, self.env.d[cells.clip(0)], np.inf)))      # ô Nearest chọn
                else:                            # ô GreedyJ chọn (ΔJ nhỏ nhất)
                    a = int(np.argmin(np.where(valid, X[:, 12], np.inf)))
                lp = v = 0.0
            else:
                a, lp, v = act(net, X, valid, g)
            r, done = self.env.step(int(cells[a]))
            for k, x in zip(buf, [X, valid, g, a, lp, v, r, done]): buf[k].append(x)
            if done:
                self.state = self.env.cnt.copy(); self._new_segment()
        return {k: np.array(x) for k, x in buf.items()}


def gae(b, last_v, gamma, lam_gae):
    T = len(b['r']); adv = np.zeros(T, np.float32); g = 0.0
    for t in reversed(range(T)):
        nv = last_v if t == T - 1 else b['v'][t + 1]
        nd = 1.0 - b['done'][t]
        delta = b['r'][t] + gamma * nv * nd - b['v'][t]
        g = delta + gamma * lam_gae * nd * g; adv[t] = g
    return adv, adv + b['v']


def train(tag, steps, lam_fixed=None, drop=(), gamma=0.998, bc_steps=40000, seed=0, T=4096, use_tref=True, teacher='greedy'):
    torch.manual_seed(seed); net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    run = Runner(seed, lam_fixed, drop, use_tref=use_tref, teacher=teacher); log = []; t0 = time.time()
    # (a) bắt chước GreedyJ
    if bc_steps:
        B = run.collect(net, bc_steps, expert=True)
        X, V, G, A = map(torch.from_numpy, [B['X'], B['valid'], B['g'], B['a']])
        for ep in range(8):
            perm = torch.randperm(len(A))
            for i in range(0, len(A), 512):
                j = perm[i:i + 512]; logit, _ = net(X[j], V[j], G[j])
                loss = nn.functional.cross_entropy(logit, A[j]); opt.zero_grad(); loss.backward(); opt.step()
            acc = (net(X[:5000], V[:5000], G[:5000])[0].argmax(1) == A[:5000]).float().mean().item()
            print(f'[BC] epoch {ep}: loss {loss.item():.3f}, trùng GreedyJ {acc:.1%}', flush=True)
    # (b) PPO
    done_steps = 0; best_val = np.inf
    while done_steps < steps:
        B = run.collect(net, T); done_steps += T
        X, valid, g, _ = candidates(run.env, run.rng, drop)
        _, _, last_v = act(net, X, valid, g)
        adv, ret = gae(B, last_v, gamma, 0.95)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        tX, tV, tG, tA, tLP = map(torch.from_numpy, [B['X'], B['valid'], B['g'], B['a'], B['logp'].astype(np.float32)])
        tAdv, tRet = torch.from_numpy(adv), torch.from_numpy(ret.astype(np.float32))
        for ep in range(4):
            perm = torch.randperm(T)
            for i in range(0, T, 512):
                j = perm[i:i + 512]
                logit, v = net(tX[j], tV[j], tG[j])
                d = torch.distributions.Categorical(logits=logit)
                ratio = torch.exp(d.log_prob(tA[j]) - tLP[j])
                pl = -torch.min(ratio * tAdv[j], ratio.clamp(0.9, 1.1) * tAdv[j]).mean()
                vl = ((v - tRet[j]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pl + 0.5 * vl
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        if (done_steps // T) % 12 == 0 or done_steps >= steps:     # kiểm tra trên 300 wave cuối giai đoạn huấn luyện
            vals = []
            for lv in ([lam_fixed] if lam_fixed is not None else [0.2, 0.5, 0.8]):
                r = evaluate(net, lv, drop=drop, w_start=W_TEST - 300, n_waves=300)
                vals.append(lv * r['MI_avg'] / MI0_ + (1 - lv) * r['T_avg'] / T0_)
            vj = float(np.mean(vals)); print(f'[VAL] {done_steps:7d} bước | J kiểm tra (TB các λ) = {vj:.4f}', flush=True)
            if vj < best_val:
                best_val = vj; os.makedirs('models', exist_ok=True); torch.save(net.state_dict(), f'models/{tag}.pt')
        rec = dict(step=done_steps, r_mean=float(B['r'].mean()), ent=float(ent.detach()), vloss=float(vl.detach()), sec=time.time() - t0)
        log.append(rec)
        if (done_steps // T) % 5 == 0:
            print(f'[PPO] {done_steps:7d} bước | reward TB {rec["r_mean"]:+.4f} | entropy {rec["ent"]:.2f} | '
                  f'{rec["sec"]:.0f}s', flush=True)
    if steps == 0:
        os.makedirs('models', exist_ok=True); torch.save(net.state_dict(), f'models/{tag}.pt')
    json.dump(dict(log=log, lam_fixed=lam_fixed, drop=list(drop), gamma=gamma, bc_steps=bc_steps, steps=steps, use_tref=use_tref, teacher=teacher),
              open(f'models/{tag}.json', 'w'))
    return net


def evaluate(net, lam, seed=0, drop=(), n_waves=N_TEST, save=None, w_start=W_TEST):
    env = PutawayEnv(I, lam=lam, s0=S['s0']); rng = np.random.default_rng(seed)
    env.reset(w_start, n_waves, None, lam)
    while not env.done():
        X, valid, g, cells = candidates(env, rng, drop)
        a, _, _ = act(net, X, valid, g, greedy=True)
        env.step(int(cells[a]))
    h = np.array(env.hist)
    if save:
        np.savez_compressed(save, cells=np.array([c for w in env.picks for c, _ in w]),
                            skus=np.array([k for w in env.picks for _, k in w]),
                            ptr=np.cumsum([0] + [len(w) for w in env.picks]), hist=h)
    return dict(MI_avg=h[:, 0].mean(), Dr_avg=h[:, 1].mean() / env.D0, T_avg=h[:, 2].mean(), MI_end=env.mi_mean(), Dr_end=env.D / env.D0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode'); ap.add_argument('--tag', default='main'); ap.add_argument('--steps', type=int, default=400000)
    ap.add_argument('--lam', type=float, default=None); ap.add_argument('--drop', default='')
    ap.add_argument('--gamma', type=float, default=0.998); ap.add_argument('--bc', type=int, default=40000)
    ap.add_argument('--lams', default='all'); ap.add_argument('--no_tref', action='store_true'); ap.add_argument('--teacher', default='greedy'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    drop = tuple(int(x) for x in a.drop.split(',') if x)
    if a.mode == 'train':
        train(a.tag, a.steps, a.lam, drop, a.gamma, a.bc, a.seed, use_tref=not a.no_tref, teacher=a.teacher)
    else:
        net = Net(); net.load_state_dict(torch.load(f'models/{a.tag}.pt')); net.eval()
        meta = json.load(open(f'models/{a.tag}.json')); drop = tuple(meta['drop'])
        lams = LAMS if a.lams == 'all' else [float(x) for x in a.lams.split(',')]
        rows = []
        os.makedirs('results', exist_ok=True)
        for lam in lams:
            t = time.time()
            r = evaluate(net, lam, drop=drop, save=f'results/picks_RL_{a.tag}_{lam}.npz'); r.update(lam=lam, tag=a.tag)
            rows.append(r); print(f'lam {lam}: MI {r["MI_avg"]:.4f} | D/D0 {r["Dr_avg"]:.4f} | T {r["T_avg"]:.1f} | {time.time() - t:.0f}s', flush=True)
        import pandas as pd
        pd.DataFrame(rows).to_csv(f'results/rl_{a.tag}.csv', index=False)
