"""T0 = độ dài lộ trình trung bình/wave của chính sách Random xuất phát từ S0, trên 1.000 wave huấn luyện (2000–2999)."""
import json, numpy as np
from env import PutawayEnv, BASELINES, run
I = dict(np.load('inst.npz', allow_pickle=True)); S = np.load('s0.npz')
env = PutawayEnv(I, s0=S['s0']); r = run(env, BASELINES['Random'], int(S['w_start']), 1000, seed=7)
T0 = float(r['hist'][:, 2].mean()); json.dump(dict(T0=T0, MI0=env.MI0, D0=env.D0), open('t0.json', 'w'))
print(f'T0 = {T0:.1f} | MI0 = {env.MI0:.4f} | D0 = {env.D0:.1f}')
