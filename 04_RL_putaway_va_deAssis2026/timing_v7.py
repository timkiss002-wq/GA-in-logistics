"""
Bài v7 – đo thời gian TÍNH TOÁN của từng thước đo / chính sách (trả lời: "MI dùng vì nhanh, bài mình thì sao?").
Chạy khi máy rảnh (không chạy song song việc khác). Ghi results/v7/timing.json.
  - MI kho (công thức 1) cho một trạng thái
  - lộ trình nhanh (NN + 2-opt, giả định A6) cho một wave
  - GRO (GA của de Assis 2024, như eval_gro.py) cho một wave
  - một quyết định cất hàng của mỗi chính sách
"""
import json, time
import numpy as np
from env import PutawayEnv, BASELINES, DeAssis2026
from tour_fast import tour_len
import lp_rl
from lp_rl import I, S, W_TEST
from eval_gro import tour as gro_tour

env = PutawayEnv(I, lam=0.5, s0=S['s0']); env.reset(W_TEST, 200, None, 0.5)
out = {}
t = time.perf_counter(); n = 2000
for _ in range(n): env.mi_mean()
out['MI_state_ms'] = 1000 * (time.perf_counter() - t) / n
ptr, sk = I['wave_ptr'], I['wave_sku']
cells = [[env.near[k] for k in sk[ptr[w]:ptr[w + 1]]] for w in range(W_TEST, W_TEST + 200)]
tour_len(I['cell_xy'][cells[0]])                       # biên dịch numba
t = time.perf_counter()
for c in cells: tour_len(I['cell_xy'][c])
out['tour_fast_wave_ms'] = 1000 * (time.perf_counter() - t) / len(cells)
t = time.perf_counter(); m = 10
for c in cells[:m]: gro_tour(c, seed=0)
out['GRO_wave_s'] = (time.perf_counter() - t) / m
_, pol = lp_rl.load('lp_s0'); rng = np.random.default_rng(0)
acts = {'LP-RL': lambda e, r: int(np.argmax(pol.scores(e)[1])), **BASELINES, 'DA_adaptive': DeAssis2026('adaptive')}
for name, act in acts.items():
    e = PutawayEnv(I, lam=0.5, s0=S['s0']); e.reset(W_TEST, 30, None, 0.5); ts = []
    while not e.done():
        t = time.perf_counter(); a = act(e, rng); ts.append(time.perf_counter() - t); e.step(a)
    out[f'decision_ms_{name}'] = 1000 * float(np.mean(ts))
out['train_min_lp'] = [json.load(open(f'models_lp/lp_s{s}.json'))['train_sec'] / 60 for s in range(5)]
json.dump(out, open('results/v7/timing.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
