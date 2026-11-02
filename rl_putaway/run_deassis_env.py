"""
Chạy 3 chế độ de Assis 2026 (bản đầy đủ theo Algorithm 1) trong môi trường của nhóm trên toàn bộ wave kiểm tra,
5 seed, rồi đo GRO trên 300 wave. Thay cho baseline 'deAssis' rút gọn trước đây (đổi tên thành MinDMI).
"""
import os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import sys
from env import PutawayEnv, DeAssis2026, DA_MODES, DA2_MODES, DA2_THETA, run
READING2 = '--cachdoc2' in sys.argv
MODES = DA2_MODES if READING2 else DA_MODES

I = dict(np.load('inst.npz', allow_pickle=True)); S = dict(np.load('s0.npz'))
W_TEST = int(I['n_train']); N_TEST = len(I['wave_id']) - W_TEST


def job(args):
    name, seed = args
    env = PutawayEnv(I, lam=0.5, s0=S['s0'])
    pol = DeAssis2026(MODES[name], theta_min=DA2_THETA[0], theta_max=DA2_THETA[1]) if READING2 else DeAssis2026(MODES[name])
    t = time.time(); r = run(env, pol, W_TEST, N_TEST, seed=seed)
    if seed == 0:
        np.savez_compressed(f'results/picks_{name}.npz', cells=np.array([c for w in r['picks'] for c, _ in w]),
                            skus=np.array([k for w in r['picks'] for _, k in w]),
                            ptr=np.cumsum([0] + [len(w) for w in r['picks']]), hist=r['hist'])
    return dict(policy=name, lam=np.nan, seed=seed, MI_avg=r['MI_avg'], Dr_avg=r['Dr_avg'], T_avg=r['T_avg'],
                MI_end=r['MI'], Dr_end=r['D_ratio'], sec=time.time() - t, deferred=pol.deferred, no_cell=pol.no_cell,
                w_end=str(np.round(pol.w, 3).tolist()))


if __name__ == '__main__':
    jobs = [(n, s) for n in MODES for s in range(5)]
    with Pool(int(os.environ.get('NPROC', 1))) as p:
        rows = list(p.imap_unordered(job, jobs))
    df = pd.DataFrame(rows).sort_values(['policy', 'seed'])
    df.to_csv('results/baselines_deassis2.csv' if READING2 else 'results/baselines_deassis.csv', index=False)
    print(df.groupby('policy')[['MI_avg', 'T_avg', 'deferred', 'no_cell']].agg(['mean', 'std']).round(4).to_string())
