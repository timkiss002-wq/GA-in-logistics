"""
Bài v7 – huấn luyện LP-RL với đúng quy trình của bài v6 (lp_rl.train_restarts), thêm 3 biến thể để trả lời phản biện:
  --pick random : huấn luyện và kiểm định dưới luật lấy hàng 'ô bất kỳ đang chứa SKU' (đổi giả định A4)
  --lam X       : θ riêng cho một λ cố định (kiểm tra ràng buộc 'một θ cho mọi λ')
  --notref      : bỏ T_ref (không trừ nhiễu của chính wave) khi huấn luyện
  --s0 class    : huấn luyện từ trạng thái đầu S0 khác (đổi giả định A3)
Ví dụ: python train_v7.py --tag lpR_s0 --seed 0 --pick random
"""
import argparse
from functools import partial
import lp_rl, robust_env

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--pick', default='nearest'); ap.add_argument('--lam', type=float, default=None)
ap.add_argument('--notref', action='store_true'); ap.add_argument('--s0', default=None)
a = ap.parse_args()
if a.pick != 'nearest':
    lp_rl.PutawayEnv = partial(robust_env.RobustEnv, pick=a.pick, seed=a.seed)
if a.s0:                                   # huấn luyện từ trạng thái đầu khác (s0_v7/<tên>.npz)
    import numpy as np
    lp_rl.S = {'s0': np.load(f's0_v7/{a.s0}.npz')['s0'], 'w_start': lp_rl.S['w_start']}
if a.notref:
    lp_rl.TREF = None
lp_rl.train_restarts(a.tag, a.seed, 400_000, restarts=3, lam_fixed=a.lam)
