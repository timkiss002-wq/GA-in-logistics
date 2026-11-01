"""
BƯỚC 3 – Trạng thái đầu chuẩn S0 = kho ngẫu nhiên ở TRẠNG THÁI ỔN ĐỊNH.

Kho vừa đổ ngẫu nhiên chưa phải hiện trạng thật: khi người lấy hàng luôn lấy ở ô gần nhất còn hàng
được cất lại ngẫu nhiên, hàng bán chạy dần bị đẩy ra xa và quãng đường tăng (304 -> ~555).
Vì vậy chạy chính sách Random của công ty qua 2.000 wave đầu (05/01 – ~03/2023) để kho ổn định,
rồi lấy trạng thái đó làm S0 chung cho mọi phương pháp. MI0, D0 đo tại S0.
"""
import numpy as np
from env import PutawayEnv, BASELINES, run
I = dict(np.load('inst.npz', allow_pickle=True))
W0 = 2000
env = PutawayEnv(I)
res = run(env, BASELINES['Random'], 0, W0, seed=123, trace_every=2000)
for t, mi, dr in res['trace'][::4]:
    print(f'  sau {t:6d} quyết định: MI = {mi:.4f} | D = {dr*env.D0:6.1f}')
np.savez_compressed('s0.npz', s0=env.cnt, w_start=W0)
e2 = PutawayEnv(I, s0=env.cnt)
print(f'S0: MI0 = {e2.MI0:.4f} | D0 = {e2.D0:.1f} | wave dùng tiếp từ chỉ số {W0}')
