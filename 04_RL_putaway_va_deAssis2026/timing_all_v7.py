"""Bài v7c – thời gian ra MỘT quyết định cất hàng của MỌI phương pháp trong Bảng VI, đo tuần tự (không chạy song song),
cùng trạng thái S0, 30 wave kiểm tra đầu tiên. Ghi results/v7/timing_all.json."""
import json, time, numpy as np
from env import PutawayEnv, BASELINES, DeAssis2026, DA_MODES
from lit_methods import LIT2
import lp_rl
from lp_rl import I, S, W_TEST
_, pol = lp_rl.load('lp_s0')
acts = {'Random': BASELINES['Random'], 'Nearest': BASELINES['Nearest'], 'Turnover': BASELINES['Turnover'], **LIT2,
        **{f'DA_{m}': DeAssis2026(DA_MODES[f'DA_{m}']) for m in ['heuristic', 'supervised', 'adaptive']},
        'GreedyJ': BASELINES['GreedyJ'], 'LP-RL': lambda e, r: int(np.argmax(pol.scores(e)[1]))}
out = {}
for name, act in acts.items():
    if hasattr(act, 'reset'): act.reset()
    e = PutawayEnv(I, lam=0.5, s0=S['s0']); e.reset(W_TEST, 30, None, 0.5); rng = np.random.default_rng(0)
    act(e, rng)                                       # khởi tạo bộ nhớ đệm (ma trận khoảng cách, tương quan…) không tính
    e.reset(W_TEST, 30, None, 0.5); ts = []
    while not e.done():
        t = time.perf_counter(); a = act(e, rng); ts.append(time.perf_counter() - t); e.step(a)
    out[name] = dict(ms=1000 * float(np.mean(ts)), p95=1000 * float(np.percentile(ts, 95)), n=len(ts))
    print(f"{name:16s} {out[name]['ms']:8.3f} ms  (p95 {out[name]['p95']:.3f})", flush=True)
json.dump(out, open('results/v7/timing_all.json', 'w'), indent=1)
