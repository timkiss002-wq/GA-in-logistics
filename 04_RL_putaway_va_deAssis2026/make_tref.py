"""
T_ref(w): độ dài lộ trình của wave w khi kho vận hành theo Random từ S0 (chạy một lần cho mọi wave >= 2000).
Chỉ dùng để GIẢM NHIỄU reward khi huấn luyện: reward dùng (T_w - T_ref(w)) thay cho T_w.
Vì T_ref(w) không phụ thuộc chính sách của agent nên chính sách tối ưu không đổi (control variate);
mọi kết quả báo cáo vẫn dùng T_w gốc.
"""
import numpy as np
from env import PutawayEnv, BASELINES, run
I = dict(np.load('inst.npz', allow_pickle=True)); S = np.load('s0.npz')
w0 = int(S['w_start']); n = len(I['wave_id']) - w0
env = PutawayEnv(I, s0=S['s0']); r = run(env, BASELINES['Random'], w0, n, seed=11)
tref = np.full(len(I['wave_id']), np.nan); tref[w0:] = r['hist'][:, 2]
np.save('tref.npy', tref); print('T_ref TB =', np.nanmean(tref))
