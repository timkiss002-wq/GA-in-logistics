"""J của Random/Nearest/GreedyJ trên từng phần dữ liệu (huấn luyện/kiểm định/kiểm tra) để so với LP-RL (kiểm tra overfit)."""
import numpy as np, pandas as pd
from env import PutawayEnv, BASELINES, run
from lp_rl import I, S, W_TRAIN0, W_VAL, N_VAL, W_TEST, N_TEST, J_of
SPL = {'train': (W_TRAIN0, W_VAL - W_TRAIN0), 'val': (W_VAL, N_VAL), 'test': (W_TEST, N_TEST)}
rows = []
for n in ['Random', 'Nearest', 'GreedyJ']:
    for lam in [0.2, 0.5, 0.8]:
        for sp, (w0, nw) in SPL.items():
            env = PutawayEnv(I, lam=lam, s0=S['s0']); r = run(env, BASELINES[n], w0, nw, seed=0, lam=lam)
            rows.append(dict(policy=n, lam=lam, split=sp, J=J_of(r['hist'], lam)))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('results/lp/split_baselines.csv', index=False)
