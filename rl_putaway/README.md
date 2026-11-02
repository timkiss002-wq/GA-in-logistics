# rl_putaway – Agent RL cất hàng tối ưu đồng thời mixing và quãng đường lấy hàng

Bộ dữ liệu: kho giày dép (de Assis et al., 2025, *Data in Brief*).

Thư mục này nối hai bài gốc:
- **de Assis et al. (2026):** đóng góp chỉ số Mixing Index (MI) và cách mô hình hoá việc cất hàng thành chuỗi quyết định.
- **de Assis et al. (2024) và Gnone et al. (2025):** đóng góp GRO và mô hình khoảng cách lối đi.

Báo cáo đầy đủ nằm ở **[BAO_CAO_NGHIEN_CUU.md](BAO_CAO_NGHIEN_CUU.md)**. Các câu hỏi cho mentor nằm ở **[CAU_HOI_CHO_MENTOR.md](CAU_HOI_CHO_MENTOR.md)**.

## Tóm tắt kết quả (2.913 wave kiểm tra, 18/07–19/10/2023)

> **Cập nhật 02/11/2026.** Bản trước của README này kết luận "A tốt nhất ở λ = 0,3–0,7" và "lợi ích đến từ nhìn xa". Cả hai kết luận chỉ dựa trên 1 seed huấn luyện và **đã bị kết quả nhiều seed bác bỏ** (3 seed, rồi 5 seed). Phần dưới đây thay thế hoàn toàn bản cũ. Chi tiết: [BAO_CAO_NGHIEN_CUU.md](BAO_CAO_NGHIEN_CUU.md), mục 5.2–5.3; Bảng 7 và 8 trong `results/bang_ket_qua.md`.

**Hàm mục tiêu:** J = λ·MI/MI0 + (1−λ)·T/T0, lấy trung bình theo wave.
- MI là Mixing Index của kho; T là độ dài lộ trình của wave.
- Kho Random ổn định (S0) có J ≈ 1 với mọi λ. J < 1 nghĩa là tốt hơn hiện trạng.
- λ = 0 chỉ còn quãng đường (phía Gnone 2025); λ = 1 chỉ còn MI (phía de Assis 2026).

**Cách nhóm định vị đóng góp (hướng 1).** Đóng góp chính là (1) hàm mục tiêu J nối hai bài gốc và (2) một chính sách cất hàng học được, **điều kiện theo λ**: một mạng dùng chung cho mọi λ, khởi động bằng bắt chước danh mục heuristic (Nearest khi λ < 0,5, GreedyJ khi λ ≥ 0,5). PPO chỉ là bước tinh chỉnh; ablation cho thấy phần tinh chỉnh này đóng góp ít và không ổn định giữa các seed.

| Chính sách (λ = 0,5) | MI | Lộ trình T / wave | J(0,5) |
|---|---|---|---|
| Random (hiện trạng) | 0,861 | 4.782 | 0,989 |
| de Assis 2026, heuristic / supervised / adaptive | 0,634 / 0,641 / 0,660 | 4.220 / 4.224 / 4.259 | 0,800 / 0,805 / 0,819 |
| Nearest (ô trống gần depot nhất) | 0,821 | 3.099 | 0,793 |
| Turnover (hàng bán chạy vào ô gần) | 0,753 | 3.291 | 0,773 |
| GreedyJ (giảm J tức thời) | 0,676 | 3.677 | 0,769 |
| **Mô hình A, TB ± ĐLC 5 seed** | 0,718 ± 0,021 | 3.225 ± 77 | **0,746 ± 0,005** |

**A tốt ở đâu (5 seed, `results/huong1/bang7_5seed.md`).** So với heuristic tốt nhất tại từng λ:
- λ = 0,3 / 0,4 / 0,5: J thấp hơn 0,016 / 0,022 / 0,022; **cả 5/5 seed** tốt hơn có ý nghĩa (Wilcoxon cặp theo wave, Holm).
- λ = 0,6: chỉ 2/5 seed, không kết luận được. λ = 0,2: −0,004 ± 0,008, chưa rõ.
- So với hiện trạng ở λ = 0,5: lộ trình ngắn hơn khoảng 33%, MI thấp hơn khoảng 17%.

**Hạn chế ở hai đầu.** Ở λ = 0, Nearest tốt hơn A (3/5 seed có ý nghĩa); λ = 0,1 không rõ. Khi λ ≥ 0,7, GreedyJ hoặc heuristic de Assis tốt hơn A. MI của A chỉ đổi từ 0,738 (λ = 0) xuống 0,687 (λ = 1), tức điều kiện theo λ còn yếu.

**Ablation (mỗi biến thể 3 seed, `results/huong1/bang9_ablation.md`).** Δ = J biến thể − J của A cùng seed, TB trên λ = 0,2/0,5/0,8; dương là kém hơn.

| Biến thể | Δ TB ± ĐLC | Kết luận |
|---|---|---|
| Bỏ đặc trưng mixing | +0,046 ± 0,001 (3/3 seed có ý nghĩa) | Thành phần quan trọng nhất |
| Mạng riêng cho λ = 0,5 (so ở λ = 0,5) | +0,006 ± 0,003 (2/3) | Mạng dùng chung cho mọi λ tốt hơn |
| Bỏ PPO, chỉ bắt chước | +0,005 ± 0,005 | PPO giúp ít |
| PPO với γ = 0 | +0,000 ± 0,012 | Không có bằng chứng "nhìn xa" giúp ích |
| Bắt chước chỉ GreedyJ (bỏ danh mục) | +0,001 ± 0,002; tách theo λ: +0,030 / +0,001 / −0,027 | Là đánh đổi: danh mục giúp ở λ thấp, làm kém ở λ cao |

Tóm tắt đầy đủ: `results/huong1/KET_QUA_HUONG1.md`.

**Tái hiện de Assis 2026 theo hai cách đọc.**
- Cách đọc 1 (đúng chữ của bài) chỉ khớp một phần: mức giảm MI lớn hơn bài nhiều.
- Cách đọc 2 (ngưỡng θ là sàn ở mức critical, lô chờ duyệt do người cất) khớp hình dạng kết quả của bài: MI lệch 0,016–0,028, tỷ lệ ô critical 99,4–99,8%.
- Ở λ = 0,2–0,5, A tốt hơn de Assis theo **cả hai** cách đọc. Ở λ = 0,8, kết luận phụ thuộc cách đọc. Nhóm dùng cách đọc 1 (baseline mạnh hơn) cho so sánh chính.
- Chi tiết: `results/deassis2026/bang_tai_hien.md`, `bang_cachdoc2.md`, `nguyen_nhan_lech.md`.

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
| R | `python deassis2026_repl.py && python deassis2026_repl.py --sens && python deassis2026_bang.py` | Tái hiện de Assis 2026 trong thiết lập gốc (cách đọc 1) |
| R2 | `python deassis2026_cachdoc2.py --calib && python deassis2026_cachdoc2.py` | Tái hiện theo cách đọc 2 |
| S | `python rl_ppo.py train --tag main2_s1 --steps 400000 --bc 40000 --teacher portfolio --seed 1` (tương tự seed 2) | Các seed bổ sung của mô hình A |

## Cấu trúc

| File | Vai trò |
|---|---|
| `env.py` | Môi trường. Mỗi wave thật được lấy (ô gần depot nhất), rồi từng đơn vị được nhập bù; mỗi đơn vị là 1 quyết định cất. Tính MI, lộ trình, reward. Chứa các baseline và 3 chế độ de Assis 2026 (`DeAssis2026`) |
| `rl_ppo.py` | Mô hình A: PPO có mặt nạ, chấm điểm 64 ô ứng viên bằng một mạng dùng chung, một mạng cho mọi λ, khởi động bằng bắt chước |
| `tour_fast.py` | Độ dài lộ trình nhanh (NN + 2-opt) dùng làm reward. Khớp GRO với r = 0,996 (`check_tour.py`) |
| `gro.py` | GRO của nhóm, giữ nguyên từ phần tái hiện Gnone 2025 |
| `deassis2026_repl.py` | Tái hiện Algorithm 1 của de Assis 2026, kèm phân tích độ nhạy |
| `models/` | `main2*.pt` là mô hình A (seed 0–4). Các file `ab*.pt` là các biến thể ablation (`ab3_*`: ablation hướng 1, 3 seed) |
| `results/` | Bảng (`bang_ket_qua.md`), hình, CSV từng wave cho từng chính sách |
| `hoc_rl/` | Q-learning trên kho 4 ô, in từng bước, để hiểu RL. Kèm script kiểm tra MI = 0,918 |
