"""
Chạy lại các baseline trên 2.913 wave kiểm tra và LƯU chi phí từng wave, để so sánh cặp với LP-RL.

Chính sách không phụ thuộc λ (Random, Nearest, Turnover, 3 chế độ de Assis R1): chạy một lần,
J(λ) cho mọi λ tính từ cùng lịch sử (MI_w, T_w). GreedyJ phụ thuộc λ: chạy riêng 11 giá trị.
Random và de Assis có ngẫu nhiên: 5 seed. Nearest, Turnover, GreedyJ gần như tất định: 1 seed.
Ghi: results/lp/base_<tên>_s<seed>.npy  (mảng hist: MI_w, D_w, T_w)
"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
from env import PutawayEnv, BASELINES, DeAssis2026, DA_MODES, run
from lp_rl import I, S, W_TEST, N_TEST, LAMS

OUT = 'results/lp'


def job(args):
    name, lam, seed = args
    tag = f'{name}_{lam}' if name == 'GreedyJ' else name
    f = f'{OUT}/base_{tag}_s{seed}.npy'
    if os.path.exists(f): return f
    env = PutawayEnv(I, lam=0.5 if lam is None else lam, s0=S['s0'])
    pol = BASELINES[name] if name in BASELINES else DeAssis2026(DA_MODES[name])
    if hasattr(pol, 'reset'): pol.reset()
    t = time.time(); r = run(env, pol, W_TEST, N_TEST, seed=seed, lam=lam)
    np.save(f, r['hist']); print(f'{tag} s{seed}: {time.time() - t:.0f}s', flush=True)
    return f


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    jobs = [('Random', None, s) for s in range(5)] + [('Nearest', None, 0), ('Turnover', None, 0)]
    jobs += [('GreedyJ', l, 0) for l in LAMS]
    jobs += [(m, None, s) for m in DA_MODES for s in range(5)]
    with Pool(int(os.environ.get('NPROC', 1))) as p:
        list(p.imap_unordered(job, jobs))
