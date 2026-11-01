# rl_putaway – Agent RL cất hàng tối ưu đồng thời mixing và quãng đường lấy hàng

Bộ dữ liệu: kho giày dép (de Assis et al., 2025, *Data in Brief*).

Thư mục này nối hai bài gốc:
- **de Assis et al. (2026):** đóng góp chỉ số Mixing Index (MI) và cách mô hình hoá việc cất hàng thành chuỗi quyết định.
- **de Assis et al. (2024) và Gnone et al. (2025):** đóng góp GRO và mô hình khoảng cách lối đi.

Báo cáo đầy đủ nằm ở **[BAO_CAO_NGHIEN_CUU.md](BAO_CAO_NGHIEN_CUU.md)**. Các câu hỏi cho mentor nằm ở **[CAU_HOI_CHO_MENTOR.md](CAU_HOI_CHO_MENTOR.md)**.

## Tóm tắt kết quả (2.913 wave kiểm tra, 18/07–19/10/2023)

**Hàm mục tiêu:** J = λ·MI/MI0 + (1−λ)·T/T0, lấy trung bình theo wave.
- MI là Mixing Index của kho; T là độ dài lộ trình của wave.
- Kho Random ổn định (S0) có J = 1 với mọi λ. J < 1 nghĩa là tốt hơn hiện trạng.

| Chính sách (λ = 0,5) | MI | GRO / wave (300 wave) | J(0,5) |
|---|---|---|---|
| Random (hiện trạng) | 0,861 | 4.713 | 0,989 |
| de Assis 2026, heuristic / supervised / adaptive | 0,634 / 0,641 / 0,660 | 4.139 / 4.139 / 4.172 | 0,800 / 0,805 / 0,819 |
| Nearest (ô trống gần depot nhất) | 0,821 | 3.068 | 0,793 |
| Turnover (hàng bán chạy vào ô gần) | 0,753 | 3.230 | 0,773 |
| GreedyJ (giảm J tức thời) | 0,676 | 3.608 | 0,769 |
| **Mô hình A (RL)** | 0,684 | 3.256 | **0,737** |

**A tốt ở vùng cân bằng.** Với λ = 0,3–0,7, A có J thấp nhất (Wilcoxon cặp, hiệu chỉnh Holm, p < 10⁻²⁷). So với hiện trạng, A đi ngắn hơn 31% và giảm MI 21%.

**Hạn chế ở hai đầu.**
- Khi λ ≤ 0,1, Nearest thắng.
- Khi λ ≥ 0,9, heuristic của de Assis thắng.
- Ở λ = 0,2 và λ = 0,8, A không khác biệt có ý nghĩa với chính sách tốt nhất.

**Ablation.**
- Bỏ PPO thì J kém thêm 0,011.
- PPO với γ = 0 (không nhìn xa) cũng kém đúng bằng mức đó. Vậy lợi ích đến từ việc học hậu quả trễ của quyết định cất hàng.

**Tái hiện de Assis 2026 chỉ được một phần.** MI ban đầu khớp và hướng thay đổi đúng, nhưng mức giảm MI lớn hơn bài. Đã thử độ nhạy với mọi tham số bài không công bố. Chi tiết ở `results/deassis2026/bang_tai_hien.md`.

## Chuẩn bị dữ liệu

Giải nén bộ dữ liệu rồi đặt các file CSV (Picking_Wave, Customer_Order, Random_Storage, Storage_Location, …) vào `rl_putaway/data/`. Cũng có thể tạo symlink `rl_putaway/data` trỏ tới thư mục dữ liệu ở `../data/`.

```
pip install -r requirements.txt
```

## Chạy lại

Các file trung gian `inst.npz`, `s0.npz`, `t0.json`, `tref.npy` và mô hình đã huấn luyện trong `models/` đều có sẵn. Muốn kiểm tra nhanh thì có thể bỏ qua bước 1–3 và bước 6.

| Bước | Lệnh | Kết quả / kiểm tra |
|---|---|---|
| 1 | `python data_prep.py` | 1.692 ô, 2.454 SKU, 9.707 wave (huấn luyện 6.794 / kiểm tra 2.913) |
| 2 | `python make_s0.py` | Trạng thái đầu S0 = kho Random ổn định sau 2.000 wave, MI0 = 0,8645 |
| 3 | `python make_t0.py && python make_tref.py` | T0 = 4.866,9 |
| 4 | `python test_env.py` | 6 kiểm thử. [4] và [6]: tổng reward = −J |
| 5 | `python run_baselines.py && python run_deassis_env.py` | `results/baselines*.csv` |
| 6 | `python rl_ppo.py train --tag main2 --steps 400000 --bc 40000 --teacher portfolio` | Khoảng 20 phút trên CPU 2 lõi |
| 7 | `python rl_ppo.py eval --tag main2` | `results/rl_main2.csv` |
| 8 | `python eval_gro.py results/picks_*.npz` | `results/gro_*.csv` (GRO trên 300 wave) |
| 9 | `./run_ablation.sh` | `results/rl_ab*.csv` |
| 10 | `python analyze.py` | `results/bang_ket_qua.md`, `results/fig*.png` |
| R | `python deassis2026_repl.py && python deassis2026_repl.py --sens && python deassis2026_bang.py` | Tái hiện de Assis 2026 trong thiết lập gốc |

## Cấu trúc

| File | Vai trò |
|---|---|
| `env.py` | Môi trường. Mỗi wave thật được lấy (ô gần depot nhất), rồi từng đơn vị được nhập bù; mỗi đơn vị là 1 quyết định cất. Tính MI, lộ trình, reward. Chứa các baseline và 3 chế độ de Assis 2026 (`DeAssis2026`) |
| `rl_ppo.py` | Mô hình A: PPO có mặt nạ, chấm điểm 64 ô ứng viên bằng một mạng dùng chung, một mạng cho mọi λ, khởi động bằng bắt chước |
| `tour_fast.py` | Độ dài lộ trình nhanh (NN + 2-opt) dùng làm reward. Khớp GRO với r = 0,996 (`check_tour.py`) |
| `gro.py` | GRO của nhóm, giữ nguyên từ phần tái hiện Gnone 2025 |
| `deassis2026_repl.py` | Tái hiện Algorithm 1 của de Assis 2026, kèm phân tích độ nhạy |
| `models/` | `main2.pt` là mô hình A. Các file `ab*.pt` là các biến thể ablation |
| `results/` | Bảng (`bang_ket_qua.md`), hình, CSV từng wave cho từng chính sách |
| `hoc_rl/` | Q-learning trên kho 4 ô, in từng bước, để hiểu RL. Kèm script kiểm tra MI = 0,918 |
