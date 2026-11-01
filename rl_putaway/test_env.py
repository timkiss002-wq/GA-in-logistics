"""Kiểm thử môi trường: 5 điều kiện phải đúng trước khi dùng cho RL."""
import numpy as np, time
from env import PutawayEnv, BASELINES, run

I = dict(np.load('inst.npz', allow_pickle=True))
env = PutawayEnv(I, lam=0.5, reward='delta')
print(f'[1] MI ban đầu (kho ngẫu nhiên dựng lại) = {env.MI0:.4f}   (de Assis 2026: 0,918)')
print(f'    D0 (quãng đường kỳ vọng / dòng, khứ hồi) = {env.D0:.1f} đơn vị toạ độ')
stock0 = env.cnt.sum(0).copy()
t = time.time(); res = run(env, BASELINES['Random'], 0, 50, seed=1)
n = sum(len(c) for c in res['picks']); dt = time.time() - t
print(f'[2] chạy 50 wave bằng agent ngẫu nhiên: {n} quyết định, {dt:.1f} s ({1000*dt/n:.2f} ms/quyết định)')
D_inc = env.D
env2 = PutawayEnv(I, reward='delta'); env2.set_state(env.cnt)
print(f'[3] D cộng dồn = {D_inc:.4f} | D tính lại từ đầu = {env2.D:.4f} | MI cộng dồn = {env.mi_mean():.6f} | tính lại = {env2.mi_mean():.6f}')
print(f"[4] tổng reward = {res['sum_r']:.6f} | -1000*(J cuối - J đầu) = {-1000*(res['J']-res['J_start']):.6f}")
print(f'[5] tồn kho từng SKU không đổi: {np.array_equal(stock0, env.cnt.sum(0))} | không ô nào quá 18: {env.tot.max() <= 18}')

e3 = PutawayEnv(I, lam=0.3, s0=np.load('s0.npz')['s0'])
r3 = run(e3, BASELINES['Nearest'], 3000, 40)
h = r3['hist'][1:]; T0 = __import__('env').T0
c = 0.3 * h[:, 0] / e3.MI0 + 0.7 * h[:, 2] / T0
print(f'[6] chế độ tour: tổng reward = {r3["sum_r"]:.6f} | -sum(c_w - 1) = {-(c - 1).sum():.6f}')
