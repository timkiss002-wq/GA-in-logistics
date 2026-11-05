"""
T3.2 – Baseline LUẬT TUYẾN TÍNH (không mạng nơ-ron, không PPO).

Câu hỏi: Model A thắng GreedyJ là nhờ MẠNG + PPO, hay chỉ nhờ dùng thêm đặc trưng?
Giả thuyết (viết TRƯỚC khi chạy, 05/11/2026):
  - Nếu T3.2 ≈ Model A  -> lợi ích đến từ đặc trưng; phần "học" (mạng + PPO) không cần thiết.
  - Nếu Model A thắng rõ -> phần học có giá trị riêng (bằng chứng vừa phải: T3.2 có thể tìm w chưa đủ).

Luật:  điểm(ô) = w · x(ô),  x = các đặc trưng của ô ứng viên (giống hệt Model A), chọn ô điểm CAO NHẤT.
  - Dùng đúng 64 ô ứng viên của Model A (hàm candidates trong rl_ppo.py) và chọn argmax như lúc đánh giá Model A.
  - Chỉ giữ các đặc trưng KHÁC NHAU giữa các ô: 0–8 và 12 (ΔJ tức thời).
    Đặc trưng 9 (độ bán chạy của SKU), 10 (ô gần nhất có SKU), 11 (λ) có cùng giá trị cho mọi ô
    -> cộng vào điểm mọi ô cùng một lượng -> không đổi được lựa chọn. Luật tuyến tính KHÔNG dùng được chúng;
       mạng nơ-ron thì dùng được (kết hợp, ví dụ "xa chỉ tệ khi SKU bán chạy"). Đây là khác biệt thật giữa hai bên.
  - Chỉ hướng của w quan trọng (nhân w với số dương thì argmax không đổi) -> luôn chuẩn hoá |w| = 1.

Tìm w (thử ngẫu nhiên quanh bộ tốt nhất), RIÊNG cho từng λ (lợi thế cho T3.2: Model A chỉ có 1 mạng cho mọi λ):
  lần 0 : w = GreedyJ (chỉ dùng ΔJ: điểm = −ΔJ)  -> điểm xuất phát, T3.2 không thể tệ hơn GreedyJ trên dữ liệu chấm.
  lần i : w_thử = w_tốt_nhất + σ·nhiễu ngẫu nhiên, chuẩn hoá; nếu J tốt hơn thì giữ.
  Mỗi lần thử được chấm bằng J (cùng công thức với bài) trên 2 đoạn × 150 wave HUẤN LUYỆN cố định theo seed,
  cùng hạt ngẫu nhiên cho mọi lần thử (so sánh công bằng giữa các w).
  Cuối cùng: 10 bộ tốt nhất được chấm lại trên 300 wave cuối giai đoạn huấn luyện (giống cách Model A chọn checkpoint),
  lấy bộ tốt nhất, rồi đánh giá MỘT lần trên 2.913 wave test.

Chạy:  python t32_linear.py --lam 0.4 --seed 0            (in từng lần thử)
Kết quả: results/rl_T32_s{seed}.csv (cùng định dạng rl_*.csv), results/picks_RL_T32_s{seed}_{lam}.npz,
         results/t32/trace_lam{lam}_s{seed}.csv  (J tốt nhất theo số lần thử -> kiểm tra đường có đi ngang không)
"""
import argparse, json, os, sys, time, types
import numpy as np

# rl_ppo.py import torch ở đầu file; luật tuyến tính không cần torch. Nếu máy không có torch thì giả lập để
# vẫn dùng ĐÚNG hàm candidates() của Model A (không chép lại code -> chắc chắn giống hệt).
try:
    import torch  # noqa
except ImportError:
    _t = types.ModuleType('torch'); _nn = types.ModuleType('torch.nn'); _nn.Module = object
    _t.nn = _nn; _t.set_num_threads = lambda n: None; sys.modules['torch'] = _t; sys.modules['torch.nn'] = _nn
from rl_ppo import candidates, I, S, W_TRAIN0, W_TEST, N_TEST, MI0_, T0_
from env import PutawayEnv

FEAT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12]
NAME = ['cho_trong', 'cung_SKU', 'cung_ma', 'MI_o', 'dMI', 'khoang_cach', 'gain', 'do_nong', 'hang_kc', 'dJ']


def J_of(h, lam):                                   # cùng công thức với bài / analyze_huong1.py
    return lam * h[:, 0].mean() / MI0_ + (1 - lam) * h[:, 2].mean() / T0_


def run_rule(w, lam, w_start, n_waves, seed=0):
    env = PutawayEnv(I, lam=lam, s0=S['s0']); rng = np.random.default_rng(seed)
    env.reset(w_start, n_waves, None, lam)
    while not env.done():
        X, valid, g, cells = candidates(env, rng)
        sc = np.where(valid, X[:, FEAT] @ w, -np.inf)
        env.step(int(cells[int(np.argmax(sc))]))
    return env, np.array(env.hist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_try', type=int, default=400); ap.add_argument('--sigma', type=float, default=0.3)
    ap.add_argument('--seg', type=int, default=150); ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args(); lam, seed = a.lam, a.seed
    rng = np.random.default_rng(1000 + seed)
    segs = sorted(int(x) for x in rng.integers(W_TRAIN0, W_TEST - 300 - a.seg, 2))   # 2 đoạn huấn luyện cố định
    score = lambda w: np.mean([J_of(run_rule(w, lam, s0, a.seg, seed)[1], lam) for s0 in segs])

    w_best = np.zeros(len(FEAT)); w_best[-1] = -1.0      # lần 0 = GreedyJ
    j_best = score(w_best); tried = [(j_best, w_best.copy())]; trace = [(0, j_best, j_best)]
    t0 = time.time()
    print(f'λ={lam} seed={seed} | đoạn chấm: wave {segs} × {a.seg} | lần 0 (GreedyJ): J={j_best:.4f}', flush=True)
    for i in range(1, a.n_try):
        w = w_best + a.sigma * rng.normal(size=len(FEAT)); w /= np.linalg.norm(w)
        j = score(w); tried.append((j, w.copy()))
        better = j < j_best
        if better: j_best, w_best = j, w
        trace.append((i, j, j_best))
        if not a.quiet and (better or i % 25 == 0):
            print(f'lần {i:3d}: J={j:.4f} | tốt nhất={j_best:.4f}{"  <- MỚI" if better else ""} | {time.time() - t0:.0f}s', flush=True)

    # chọn lại trên 300 wave giữ riêng (giống Model A chọn checkpoint)
    top = sorted(tried, key=lambda x: x[0])[:10]
    val = [(J_of(run_rule(w, lam, W_TEST - 300, 300, seed)[1], lam), w) for _, w in top]
    jv, w_fin = min(val, key=lambda x: x[0])
    print('w cuối:', ', '.join(f'{n}={v:+.2f}' for n, v in zip(NAME, w_fin)), f'| J kiểm tra nội bộ={jv:.4f}', flush=True)

    # đánh giá một lần trên test
    env, h = run_rule(w_fin, lam, W_TEST, N_TEST, seed)
    r = dict(MI_avg=h[:, 0].mean(), Dr_avg=h[:, 1].mean() / env.D0, T_avg=h[:, 2].mean(), MI_end=env.mi_mean(),
             Dr_end=env.D / env.D0, lam=lam, tag=f'T32_s{seed}', J_test=J_of(h, lam), sec=time.time() - t0,
             **{f'w_{n}': v for n, v in zip(NAME, w_fin)})
    print(f'TEST: J={r["J_test"]:.4f} | MI {r["MI_avg"]:.4f} | T {r["T_avg"]:.1f} | tổng {r["sec"]:.0f}s', flush=True)
    os.makedirs('results/t32', exist_ok=True)
    import pandas as pd
    pd.DataFrame(trace, columns=['lan', 'J', 'J_tot_nhat']).to_csv(f'results/t32/trace_lam{lam}_s{seed}.csv', index=False)
    pd.DataFrame([r]).to_csv(f'results/t32/row_lam{lam}_s{seed}.csv', index=False)
    np.savez_compressed(f'results/picks_RL_T32_s{seed}_{lam}.npz', hist=h)


if __name__ == '__main__':
    main()
