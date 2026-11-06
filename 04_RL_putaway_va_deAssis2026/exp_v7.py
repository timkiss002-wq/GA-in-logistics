"""
Bài v7 – chạy đánh giá trên 2.913 wave kiểm tra dưới các kịch bản (trả lời phản biện), lưu từng wave.

Kịch bản (SCEN):
  base        : như bài v6 (A3–A5 gốc); dùng cho mô hình THỜI GIAN lấy hàng (cần thêm MI của ô được ghé)
  pick_random, pick_top2 : đổi giả định A4 (luật chọn ô khi lấy hàng)
  inb_lot, inb_delay     : đổi giả định A5 (hàng về theo lô / về trễ một wave)
  s0_seed1, s0_seed2, s0_fill70, s0_fill90, s0_class : đổi giả định A3 (trạng thái đầu)
Chính sách: 'LP:<tag>' (θ đã học), Random, Nearest, Turnover, GreedyJ, DA_heuristic/DA_supervised/DA_adaptive.
Đầu ra: results/v7/<scen>/<policy>_<lam>_s<seed>.npz  (hist: MI_w, D_w, T_w; hist2: dòng, ΣMI_l ô ghé, Σ số mã ô ghé,
        số ô khác nhau, số dòng hết hàng)

Chạy:  python exp_v7.py jobs <nhóm>  -> in danh sách việc;  python exp_v7.py run "<scen> <policy> <lam> <seed>"
"""
import os, sys, time
import numpy as np
from robust_env import RobustEnv
from env import BASELINES, DeAssis2026, DA_MODES
import lp_rl
from lp_rl import I, S, W_TEST, N_TEST, LAMS

OUT = 'results/v7'
SCEN = {'base': {}, 'pick_random': dict(pick='random'), 'pick_top2': dict(pick='top2'),
        'inb_lot': dict(inbound='lot'), 'inb_delay': dict(inbound='delay')}
for n in ['seed1', 'seed2', 'fill70', 'fill90', 'class']:
    SCEN[f's0_{n}'] = dict(s0file=f's0_v7/{n}.npz')


def run_job(scen, policy, lam, seed):
    lam = float(lam); seed = int(seed)
    f = f'{OUT}/{scen}/{policy.replace(":", "-")}_{lam}_s{seed}.npz'
    if os.path.exists(f): return f
    os.makedirs(os.path.dirname(f), exist_ok=True)
    kw = dict(SCEN[scen]); s0 = np.load(kw.pop('s0file'))['s0'] if 's0file' in kw else S['s0']
    env = RobustEnv(I, lam=lam, s0=s0, seed=seed, **kw)
    rng = np.random.default_rng(seed)
    if policy.startswith('LP:'):
        _, pol = lp_rl.load(policy[3:])
        act = lambda e, r: int(np.argmax(pol.scores(e)[1]))
    elif policy in BASELINES:
        act = BASELINES[policy]
    else:
        act = DeAssis2026(DA_MODES[policy]); act.reset()
    t = time.time(); env.reset(W_TEST, N_TEST, None, lam)
    nd = 0; td = 0.0
    while not env.done():
        t1 = time.perf_counter(); a = act(env, rng); td += time.perf_counter() - t1; nd += 1
        env.step(a)
    np.savez_compressed(f, hist=np.array(env.hist), hist2=np.array(env.hist2), stockout=env.stockout,
                        sec=time.time() - t, dec_ms=1000 * td / max(nd, 1), n_dec=nd)
    print(f'{scen} {policy} λ={lam} s{seed}: {time.time() - t:.0f}s, {1000 * td / max(nd, 1):.2f} ms/quyết định', flush=True)
    return f


def jobs(group):
    J = []
    lp = [f'LP:lp_s{s}' for s in range(5)]
    if group == 'base':
        J += [('base', p, l, 0) for p in lp for l in LAMS]
        J += [('base', 'GreedyJ', l, 0) for l in LAMS]
        J += [('base', 'Random', 0.5, s) for s in range(5)] + [('base', p, 0.5, 0) for p in ['Nearest', 'Turnover']]
        J += [('base', m, 0.5, s) for m in DA_MODES for s in range(3)]
    if group == 'robust':
        for sc in ['pick_random', 'pick_top2', 'inb_lot', 'inb_delay', 's0_seed1', 's0_seed2', 's0_fill70', 's0_fill90', 's0_class']:
            J += [(sc, p, 0.5, 0) for p in lp]
            J += [(sc, 'Random', 0.5, s) for s in range(2)] + [(sc, p, 0.5, 0) for p in ['Nearest', 'Turnover', 'GreedyJ']]
            J += [(sc, 'DA_heuristic', 0.5, s) for s in range(2)]
    if group == 'retrained':          # chính sách học lại trong kịch bản tương ứng / θ riêng theo λ / không Tref
        J += [('pick_random', f'LP:lpR_s{s}', 0.5, 0) for s in range(3)]
        for lf in ['0.0', '0.5', '1.0']:
            J += [('base', f'LP:lpF{lf}_s{s}', float(lf), 0) for s in range(2)]
        J += [('base', f'LP:ab_notref_s{s}', l, 0) for s in range(2) for l in [0.2, 0.5, 0.8]]
    return J


if __name__ == '__main__':
    if sys.argv[1] == 'jobs':
        for j in jobs(sys.argv[2]): print(*j)
    else:
        run_job(*sys.argv[2].split())
