# slap_rl – RL cất hàng (putaway) tối ưu đồng thời mixing và quãng đường lấy hàng

Dữ liệu: kho giày dép (de Assis et al., 2025, *Data in Brief*). Nền tảng: de Assis et al. (2026), dùng cho MI và bài toán cất hàng tuần tự; de Assis et al. (2024) và Gnone et al. (2025), dùng cho GRO và mô hình khoảng cách.

## Chạy lại từ đầu (đúng thứ tự)

| Bước | Lệnh | Làm gì | Kiểm tra |
|---|---|---|---|
| 1 | `python data_prep.py` | Dựng kho (1.692 ô tầng 1–2), 2.454 SKU, tồn kho, kho ngẫu nhiên ban đầu, chia wave theo thời gian (70/30) | in ra số ô, SKU, wave |
| 2 | `python make_s0.py` | Chạy Random qua 2.000 wave đầu để kho ổn định → S0 | MI0 = 0,8645 |
| 3 | `python make_t0.py`, `python make_tref.py` | T0 = lộ trình TB/wave của Random; T_ref(w) để giảm nhiễu reward | T0 = 4.866,9 |
| 4 | `python test_env.py` | 6 kiểm thử môi trường | [4] và [6]: tổng reward = −J |
| 5 | `python run_baselines.py` | 5 baseline × 5 seed + GreedyJ với 11 mức λ | `results/baselines.csv` |
| 6 | `python rl_ppo.py train --tag main --steps 600000 --bc 20000` | Huấn luyện RL (khoảng 20 phút trên CPU 2 lõi) | dòng `[VAL]` |
| 7 | `python rl_ppo.py eval --tag main` | Đánh giá RL trên 2.913 wave kiểm tra, λ = 0…1 | `results/rl_main.csv` |
| 8 | `python eval_gro.py results/picks_*.npz` | Đo bằng GRO trên 300 wave kiểm tra | `results/gro_*.csv` |
| 9 | `./run_ablation.sh` | Ablation study | `results/rl_ab_*.csv` |
| 10 | `python analyze.py` | Bảng, kiểm định, hình | `results/bang_ket_qua.md`, `fig*.png` |

Phụ thuộc: numpy, pandas, scipy, matplotlib, numba, torch (bản CPU là đủ). `data/` trỏ tới thư mục `02_data/order_picking`.

## File

| File | Vai trò |
|---|---|
| `env.py` | Môi trường: trạng thái kho, lấy hàng theo wave thật, nhập bù, MI, proxy D, reward; 5 baseline |
| `tour_fast.py` | Độ dài lộ trình nhanh (NN + 2-opt), khớp GRO với r = 0,996 |
| `gro.py` | GRO của nhóm, giữ nguyên từ `slap_gro_v2` |
| `rl_ppo.py` | Agent PPO: chấm điểm 64 ô ứng viên, mặt nạ, dùng chung cho mọi λ |
| `check_tour.py` | Kiểm định `tour_fast` so với GRO |
