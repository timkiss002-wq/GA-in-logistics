# Kết quả củng cố hướng 1 (02/11/2026)

Định vị đã chốt: đóng góp là **hàm mục tiêu J** và **một chính sách cất hàng học được, điều kiện theo λ**; PPO là bước tinh chỉnh.
Mọi lần chạy mới dùng đúng cấu hình của mô hình A (`main2`: 40.000 bước bắt chước, 400.000 bước PPO), chỉ đổi seed hoặc đúng một thành phần.
Trước khi chạy, môi trường trên máy chủ đã được kiểm tra: đánh giá lại `main2` ở λ = 0,5 cho MI 0,6835 và T 3.328,0, khớp từng chữ số với `results/rl_main2.csv`.

## 1. Mô hình A: 5 seed huấn luyện (Bảng 7, `bang7_5seed.md`)

| λ | ΔJ của A so với heuristic tốt nhất (TB ± ĐLC, 5 seed) | Số seed A tốt hơn / kém hơn có ý nghĩa |
|---|---|---|
| 0,0 | +0,015 ± 0,015 | 0 / 3 |
| 0,1 | +0,007 ± 0,011 | 2 / 2 |
| 0,2 | −0,004 ± 0,008 | 3 / 1 |
| **0,3** | **−0,016 ± 0,006** | **5 / 0** |
| **0,4** | **−0,022 ± 0,005** | **5 / 0** |
| **0,5** | **−0,022 ± 0,005** | **5 / 0** |
| 0,6 | −0,006 ± 0,007 | 2 / 0 |
| 0,7 | +0,009 ± 0,012 | 1 / 4 |
| 0,8–1,0 | +0,025 đến +0,059 | 0 / 4–5 |

- **Kết luận đứng vững:** A tốt hơn mọi heuristic ở λ = 0,3–0,5, cả 5/5 seed có ý nghĩa (Wilcoxon cặp theo wave, Holm).
- **Thu hẹp so với bản 3 seed:** λ = 0,6 chỉ còn 2/5 seed, nên không kết luận được. Bản 3 seed ghi "có thể tới 0,6"; giờ nên bỏ.
- Ở λ = 0,5, trung bình 5 seed: MI 0,718 ± 0,021; T 3.225 ± 77; J 0,746 ± 0,005. So với hiện trạng Random (MI 0,861; T 4.782): lộ trình ngắn hơn khoảng 33%, MI thấp hơn khoảng 17%.

## 2. Ablation theo hướng 1, mỗi biến thể 3 seed (Bảng 9 và 9b, `bang9_ablation.md`)

Δ = J biến thể − J của A cùng seed; dương nghĩa là biến thể kém hơn, tức thành phần bị bỏ có ích.

| Thành phần | Δ TB ± ĐLC (λ 0,2/0,5/0,8) | Tách theo λ | Kết luận |
|---|---|---|---|
| Đặc trưng mixing | **+0,046 ± 0,001**, 3/3 seed kém hơn có ý nghĩa | +0,009 / +0,049 / +0,080 | **Thành phần quan trọng nhất**; càng quan trọng khi λ lớn |
| Một mạng cho mọi λ (so với mạng riêng cho λ = 0,5) | **+0,006 ± 0,003** (chỉ ở λ = 0,5), 2/3 seed có ý nghĩa | – | Mạng dùng chung tốt hơn mạng chuyên một λ, hiệu ứng nhỏ nhưng nhất quán |
| PPO (so với chỉ bắt chước) | +0,005 ± 0,005 | +0,013 / +0,007 / −0,005 | Giúp ít, chủ yếu ở λ thấp |
| Nhìn xa (γ = 0,998 so với γ = 0) | +0,000 ± 0,012 | – | Không có bằng chứng |
| Danh mục heuristic (so với chỉ bắt chước GreedyJ) | +0,001 ± 0,002 | **+0,030 / +0,001 / −0,027** | **Không phải lợi ích thuần, mà là đánh đổi** (xem dưới) |

### Phát hiện cần lưu ý: chọn "thầy" để bắt chước quyết định A mạnh ở đâu

- Bắt chước danh mục (Nearest khi λ < 0,5) giúp A mạnh ở λ thấp, nhưng làm A **kém hơn** ở λ = 0,8 khoảng 0,027.
- Biến thể chỉ bắt chước GreedyJ có J ở λ = 0,8 là 0,746 / 0,750 / 0,757 (3 seed). Cả ba đều **thấp hơn** heuristic de Assis (0,760), là baseline tốt nhất ở λ đó. Đổi lại, ở λ = 0,2 nó kém Nearest rõ (0,730 so với 0,699).
- Hệ quả cho bài: không nên viết "danh mục giúp A tốt hơn nói chung". Viết đúng là lựa chọn "thầy" dịch vùng λ mà chính sách học được thắng. Đây cũng là một hướng cải tiến có cơ sở số liệu (ví dụ chọn "thầy" theo từng λ tốt hơn).
- Các số ở λ = 0,8 trên đây chưa kiểm định từng wave so với de Assis; cần chạy kiểm định trước khi đưa vào bài.

## 3. Hạn chế mới thấy rõ (Hình `fig1_danh_doi_5seed.png`)

- **Điều kiện theo λ còn yếu.** Khi λ đi từ 0 đến 1, MI của A chỉ đổi từ 0,738 xuống 0,687. Để so sánh: Nearest có MI 0,821, heuristic de Assis có MI 0,634.
- Nghĩa là A gần như là **một** chính sách tốt ở vùng giữa, chứ chưa thật sự thay đổi hành vi theo λ. Đây là lý do A thua ở hai đầu.
- Nếu viết "chính sách điều kiện theo λ" là đóng góp, reviewer sẽ chỉ ra điểm này từ chính Hình 1. Nên nêu thẳng trong phần hạn chế, hoặc mô tả đóng góp là "một chính sách học được tốt nhất ở vùng cân bằng λ = 0,3–0,5".

## 4. File

- Mô hình mới: `models/main2_s3`, `main2_s4`, `ab3_greedyT_s0..2`, `ab3_noMIfeat_s0..2`, `ab3_lam05_s0..2` (.pt + .json).
- Kết quả: `results/rl_<tag>.csv`, `results/picks_RL_<tag>_<λ>.npz` của các tag trên.
- Script: `analyze_huong1.py` (Bảng 9, 9b). Bảng 7 với 5 seed sinh bằng `analyze.py` sẵn có; bản chạy trên máy chủ bỏ Bảng 5 vì không có `data/Customer_Order.csv`, Bảng 5 không phụ thuộc seed mới.
- Lệnh huấn luyện: `python rl_ppo.py train --tag <tag> --steps 400000 --bc 40000 --seed <s>` cộng một trong các tuỳ chọn `--teacher portfolio|greedy`, `--drop 2,3,4`, `--lam 0.5`.
