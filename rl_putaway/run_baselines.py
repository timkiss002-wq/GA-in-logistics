"""
BƯỚC 4 – Chạy các baseline + grid search lambda cho GreedyJ.

Giai đoạn đánh giá: toàn bộ wave kiểm tra (chỉ số 6794 – 9706, 18/07 – 19/10/2023), xuất phát từ S0.
Mỗi chính sách chạy 5 seed (seed chỉ ảnh hưởng chọn ngẫu nhiên / phá thế hoà).
Chỉ số báo cáo = TRUNG BÌNH THEO WAVE trong giai đoạn kiểm tra:
   MI_avg      : Mixing Index trung bình kho lúc lấy mỗi wave
   Dr_avg      : D/D0 trung bình (quãng đường kỳ vọng, chuẩn hoá theo S0)
   J(lam)      : lam*MI_avg/MI0 + (1-lam)*Dr_avg
Ghi: results/baselines.csv, results/picks_<tên>.npz (ô đã ghé từng wave, để đo GRO ở bước sau)
"""
import os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from env import PutawayEnv, BASELINES, run

I = dict(np.load('inst.npz', allow_pickle=True)); S = dict(np.load('s0.npz'))
W_TEST = int(I['n_train']); N_TEST = len(I['wave_id']) - W_TEST
SEEDS = range(5)
LAMS = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
os.makedirs('results', exist_ok=True)


def job(args):
    name, lam, seed = args
    env = PutawayEnv(I, lam=lam, s0=S['s0'])
    t = time.time(); r = run(env, BASELINES[name], W_TEST, N_TEST, seed=seed)
    if seed == 0:
        tag = name if name != 'GreedyJ' else f'GreedyJ_{lam}'
        np.savez_compressed(f'results/picks_{tag}.npz',
                            cells=np.array([c for w in r['picks'] for c, _ in w]),
                            skus=np.array([k for w in r['picks'] for _, k in w]),
                            ptr=np.cumsum([0] + [len(w) for w in r['picks']]), hist=r['hist'])
    return dict(policy=name, lam=lam if name == 'GreedyJ' else np.nan, seed=seed, MI_avg=r['MI_avg'],
                Dr_avg=r['Dr_avg'], T_avg=r['T_avg'], MI_end=r['MI'], Dr_end=r['D_ratio'], sec=time.time() - t)


if __name__ == '__main__':
    jobs = [(n, 0.5, s) for n in ['Random', 'Nearest', 'Turnover', 'deAssis'] for s in SEEDS]
    jobs += [('GreedyJ', l, s) for l in LAMS for s in SEEDS]
    t0 = time.time()
    with Pool(int(os.environ.get('NPROC', 2))) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            rows.append(r)
            if (i + 1) % 10 == 0: print(f'{i + 1}/{len(jobs)}  {time.time() - t0:.0f}s', flush=True)
    df = pd.DataFrame(rows).sort_values(['policy', 'lam', 'seed'])
    env = PutawayEnv(I, s0=S['s0'])
    df['MI0'] = env.MI0
    df.to_csv('results/baselines.csv', index=False)
    g = df.groupby(['policy', 'lam'], dropna=False)[['MI_avg', 'Dr_avg', 'T_avg']].agg(['mean', 'std'])
    print(g.round(4).to_string())
