# Báo cáo nghiên cứu nội bộ: RL cất hàng tối ưu đồng thời mixing và quãng đường (01/11/2026)

> Đây là ghi chép kỹ thuật để cả nhóm hiểu và kiểm tra lại. Phần viết bài gửi thầy nhóm tự viết, theo dặn của mentor.
> Mọi con số dưới đây lấy từ `results/bang_ket_qua.md`, được sinh tự động bằng `analyze.py`.

## 1. Câu hỏi nghiên cứu

Hai bài gốc mỗi bài tối ưu một đầu:
- de Assis 2026 chỉ giảm mixing (MI) khi cất hàng, không quan tâm người lấy hàng đi bao xa.
- Gnone 2025 chỉ đo quãng đường trên kho tĩnh.

Nhóm hỏi: **một agent RL quyết định cất hàng theo hàm mục tiêu J (gộp MI và quãng đường) có cho kết quả tốt hơn các quy tắc hiện có không? Tốt hơn ở vùng nào, và kém hơn ở đâu?**

## 2. Thiết lập (các quyết định cần ghi trong mục Method)

| # | Quyết định | Lý do |
|---|---|---|
| D1 | Kho gồm 1.692 ô ở tầng 1–2 (tầng lấy hàng), mỗi ô 18 ngăn | Chỉ tầng 1–2 có dòng lấy hàng |
| D2 | SKU = mã × cỡ (2.454 SKU). MI tính theo mã giày | Đúng công thức (2) của de Assis 2026; người lấy hàng phải lấy đúng cỡ |
| D3 | Trạng thái đầu được **dựng lại** thay vì lấy nguyên `Random_Storage.csv` | Vùng trộn trong file có cỡ 4–16 phân bố đều và **0 dòng lấy hàng** rơi vào đó. Nhóm dựng kho ngẫu nhiên, tồn kho tỷ lệ với nhu cầu, lấp 80% (de Assis giải phóng ~20%) |
| D4 | Chạy Random qua 2.000 wave đầu để kho **ổn định**, lấy trạng thái đó làm S0 | Kho vừa đổ ngẫu nhiên chưa phải hiện trạng: quãng đường kỳ vọng tăng từ 304 lên ~544 rồi mới đứng yên |
| D5 | Wave sắp theo thời gian (số wave khớp 100% với ngày tạo đơn). Wave 2000–6793 (đến 17/07/2023) để huấn luyện, 6794–9706 (18/07–19/10, 2.913 wave) để kiểm tra | Không dùng dữ liệu tương lai |
| D6 | Mỗi wave: lấy hàng ở ô gần depot nhất còn SKU đó (quy tắc B), rồi các đơn vị vừa lấy được nhập bù; mỗi đơn vị là 1 quyết định cất hàng | Tồn kho không đổi, chỉ vị trí thay đổi. Vòng nhập → xuất được nối lại, đúng gap của 2 bài gốc |
| D7 | Đo quãng đường bằng độ dài lộ trình thật của từng wave (NN + 2-opt); kiểm chứng bằng GRO trên 300 wave | NN + 2-opt khớp GRO với r = 0,996, lệch TB 0,8%. GRO (bản có 2-opt) lệch lời giải tối ưu 0,03% |

## 3. Hàm mục tiêu và reward

**J = λ · MI/MI0 + (1 − λ) · T/T0**, trung bình theo wave. Trong đó:
- MI0 = 0,8645 và T0 = 4.866,9 là giá trị của kho Random ổn định.
- Vì vậy kho Random có J ≈ 1 với mọi λ. J < 1 nghĩa là tốt hơn hiện trạng.

**Thay đổi so với bản Intro cũ:** MI cũng được chuẩn hoá (MI/MI0). Nhờ vậy hai số hạng cùng thang đo và λ là tỷ trọng của mức cải thiện tương đối.

**Reward:** mỗi khi một wave được lấy, agent nhận −(c_w − 1), với c_w = λ·MI_w/MI0 + (1 − λ)·T_w/T0.
- Tổng reward = −N·(J − 1), nên tối đa tổng reward tương đương tối thiểu J.
- Đã kiểm tra bằng số: kiểm thử [6] trong `test_env.py` cho tổng reward = −Σ(c_w − 1) chính xác đến 6 chữ số.
- Reward đến **trễ**: quyết định cất hôm nay ảnh hưởng đến các wave sau. Đây là lý do cần RL.

**Phát hiện phụ (đáng đưa vào bài).** Proxy quen thuộc "khứ hồi tới ô gần nhất của từng SKU" chỉ tương quan thứ hạng ρ = 0,52 với lộ trình thật trên 15 baseline. Khi tính trên 17 baseline (gồm cả 3 chế độ de Assis đầy đủ), ρ = 0,66. Ví dụ: theo proxy, GreedyJ (0,52) tốt hơn Nearest (0,60); theo lộ trình thật thì ngược lại (3.677 so với 3.099). Vì vậy nhóm bỏ proxy, dùng lộ trình thật làm reward.

## 4. Mô hình A

| Thành phần | Nội dung |
|---|---|
| Quyết định | Mỗi đơn vị hàng về chọn 1 ô còn chỗ |
| Ô ứng viên | 64 ô: 16 ô có ΔJ tức thời tốt nhất, 16 ô trống gần depot nhất, 16 ô đã có cùng mã, 16 ô ngẫu nhiên (ý tưởng "rank rồi assign" của Pan et al. 2025) |
| Đặc trưng từng ô | 13 đặc trưng: chỗ trống, cùng SKU, cùng mã, MI ô, ΔMI, khoảng cách, mức rút ngắn tới SKU, độ nóng ô, thứ hạng khoảng cách, độ bán chạy, ô gần nhất hiện có, λ, ΔJ tức thời |
| Mạng | Điểm của ô = MLP dùng chung cho mọi ô − α·ΔJ tức thời; xác suất = softmax trên các ô hợp lệ; V(s) từ trung bình và max các ô |
| Một mạng cho mọi λ | λ là đầu vào; mỗi đoạn huấn luyện bốc λ ngẫu nhiên |
| Khởi động | Bắt chước heuristic tốt nhất theo λ trên 300 wave cuối giai đoạn huấn luyện: Nearest khi λ < 0,5, GreedyJ khi λ ≥ 0,5 |
| Học | PPO 400.000 bước, γ = 0,998, mặt nạ ô đầy; reward trừ đi T_ref(w) để giảm nhiễu |
| Chọn mô hình | Theo J trên 300 wave kiểm tra nội bộ. Bản "portfolio" (0,8147) tốt hơn bản chỉ bắt chước GreedyJ (0,8205), nên chọn bản portfolio |

**Baseline:**
- Random: hiện trạng công ty.
- Nearest: ô trống gần depot nhất.
- Turnover: hàng bán chạy vào ô gần, dạng động của class-based.
- de Assis 2026: đủ 3 chế độ heuristic, supervised, adaptive theo Algorithm 1 (mục 5.4). Lô supervised chuyển cho người duyệt sẽ do người vận hành cất ngẫu nhiên, vì trong dòng nhập – xuất hàng bắt buộc phải được cất.
- GreedyJ: chọn ô làm J giảm nhiều nhất ngay lập tức, 11 mức λ.

## 5. Kết quả chính (2.913 wave kiểm tra)

### 5.1 So sánh tại λ = 0,5

| Chính sách | MI | Lộ trình T / wave | GRO (300 wave) | J(0,5) |
|---|---|---|---|---|
| Random (hiện trạng) | 0,861 | 4.782 | 4.713 | 0,989 |
| de Assis 2026 – heuristic | **0,634** | 4.220 | 4.139 | 0,800 |
| de Assis 2026 – supervised | 0,641 | 4.224 | 4.139 | 0,805 |
| de Assis 2026 – adaptive | 0,660 | 4.259 | 4.172 | 0,819 |
| Nearest (chỉ quãng đường) | 0,821 | **3.099** | **3.068** | 0,793 |
| Turnover | 0,753 | 3.291 | 3.230 | 0,773 |
| GreedyJ λ = 0,5 | 0,676 | 3.677 | 3.608 | 0,769 |
| **Mô hình A, λ = 0,5** | 0,684 | 3.328 | 3.256 | **0,737** |

Đọc bảng:
- **So với hiện trạng:** A giảm quãng đường GRO 30,9% và MI 21%.
- **So với de Assis:** cả 3 chế độ đều có MI thấp hơn A, nhưng đi xa hơn A 27–28% (GRO 4.139–4.172 so với 3.256).
- **So với Nearest:** Nearest đi ngắn hơn A 6%, nhưng MI cao hơn hẳn (0,82).
- **Thống kê:** Friedman trên 8 chính sách cho χ² = 881,3, p ≈ 10⁻¹⁸⁶.

### 5.2 A tốt ở đâu, hạn chế ở đâu (Bảng 2 trong `bang_ket_qua.md`)

| λ | Heuristic tốt nhất ở λ đó | So với A |
|---|---|---|
| 0–0,1 | Nearest | **A kém hơn** (J +0,038 / +0,022) |
| 0,2 | Nearest | Không khác biệt có ý nghĩa (p = 0,57) |
| 0,3–0,7 | Nearest, Turnover, GreedyJ | **A tốt hơn**: J −0,010 đến −0,031, p < 10⁻²⁷ |
| 0,8 | de Assis heuristic | Không khác biệt (p = 0,81) |
| 0,9–1,0 | de Assis heuristic | **A kém hơn** (J +0,018 / +0,031) |

Rút ra:
- **A tốt ở vùng cân bằng.** Khi kho coi trọng cả hai mục tiêu (λ ≈ 0,3–0,7), A là chính sách tốt nhất.
- A nằm trên đường Pareto, ở đoạn giữa Nearest và de Assis heuristic.
- A tốt nhất trong cả 4 tháng kiểm tra (Bảng 5).
- **Hạn chế:** ở hai đầu, quy tắc chuyên một mục tiêu vẫn thắng. Agent dùng chung λ chưa "uốn" đủ mạnh: MI của A chỉ chạy trong khoảng 0,66–0,70 khi λ đi từ 1 về 0.

### 5.3 Ablation study: vì sao chọn từng thành phần (Bảng 4)

Δ là trung bình J trên ba mức λ = 0,2 / 0,5 / 0,8. Dương nghĩa là kém hơn.

| Câu hỏi | Thí nghiệm | Kết quả | Kết luận |
|---|---|---|---|
| PPO có giúp gì, hay chỉ cần bắt chước? | A so với A không PPO | không PPO kém hơn +0,011 | **PPO giúp** |
| Cái giúp là "nhìn xa" hay chỉ là học thêm? | A so với A có PPO nhưng γ = 0 | γ = 0 kém hơn +0,012, bằng mức không PPO | **Toàn bộ lợi ích đến từ nhìn xa (reward trễ)**. Đây là lý do dùng RL |
| Bắt chước ai? | Portfolio so với chỉ GreedyJ | Portfolio tốt hơn −0,006 (rõ nhất ở λ = 0,2: 0,708 so với 0,733) | Chọn portfolio |
| Có cần đặc trưng mixing? | Bỏ đặc trưng 2–4 | Kém hơn +0,031 (λ = 0,8: +0,08) | Cần |
| Có cần đường tắt ΔJ + khởi động? | Bỏ cả hai | Kém hơn +0,096 | Thành phần quan trọng nhất |
| Một mạng cho mọi λ hay mỗi λ một mạng? | Agent riêng cho λ = 0,5 | Kém hơn +0,0095 ở λ = 0,5 | Dùng chung λ tốt hơn (học được từ mọi λ) |
| Đặc trưng khoảng cách? | Bỏ đặc trưng 5, 6, 8, 10 | Không kém hơn (−0,008) | Thông tin khoảng cách đã có trong ΔJ và ô ứng viên. Có thể rút gọn |
| Giảm nhiễu bằng T_ref? | Bỏ T_ref | Gần như không đổi (−0,002) | Không cần thiết |

**Lưu ý trung thực.** Ở cấu hình chỉ bắt chước GreedyJ, PPO không cải thiện: main ≈ không PPO ≈ γ = 0. Vậy lợi ích của PPO phụ thuộc vào điểm khởi động. Mỗi biến thể chỉ có 1 seed; các ablation phụ huấn luyện 200k bước, ít hơn main (600k).

### 5.4 Tái hiện de Assis et al. (2026) trong thiết lập gốc của bài

Script `deassis2026_repl.py` cài đủ Algorithm 1 cho 3 chế độ: heuristic (công thức 5), supervised (chỉ tự làm khi ΔMI ≤ 0), adaptive (6 tiêu chí, cập nhật trọng số theo công thức 9). Thiết lập giống mục 3.4 của bài: kho ngẫu nhiên thật, giải phóng 20% sức chứa, 10.000 lô chỉ nhập, cả 3 chế độ dùng chung một chuỗi lô. Chạy 5 seed. Chi tiết trong `results/deassis2026/bang_tai_hien.md` và `hinh_tai_hien.png`.

| Agent | MI bài | MI nhóm | % critical bài | % critical nhóm |
|---|---|---|---|---|
| Ban đầu | 0,918 | 0,9147 | 100 | 100 |
| Heuristic | 0,830 | 0,760 ± 0,001 | 99,66 | 33,8 |
| Supervised | 0,884 | 0,724 ± 0,000 | 99,71 | 19,3 |
| Adaptive | 0,822 | 0,748 ± 0,001 | 99,48 | 30,0 |

**Những gì tái hiện được:**
- MI ban đầu: 0,9147 so với 0,918.
- Cả 3 chế độ đều giảm MI.
- Adaptive tốt hơn heuristic.
- Số hành động được thực hiện tăng nhanh rồi đứng yên khi kho hết chỗ (Hình 4).

**Những gì chưa tái hiện được:**
- Mức giảm MI của nhóm lớn hơn bài nhiều.
- Tỷ lệ ô critical của nhóm thấp hơn bài nhiều.
- Supervised của nhóm tốt nhất, trong khi trong bài nó kém nhất.

**Đã thử độ nhạy với mọi tham số bài không công bố** (Bảng R2): θ, ε, top-k, ω, trọng số ban đầu, cách tính r_t, phạm vi ô, định nghĩa SKU, cỡ bắt buộc, cách xử lý lô chờ duyệt, số lô. Không biến thể nào cho lại đúng con số của bài. Hai biến thể làm kết quả nhích về phía bài:
- **Tính MI theo mã × cỡ:** heuristic 0,871, critical 87,7%.
- **Lô chờ duyệt do người vận hành cất ngẫu nhiên:** supervised còn 0,761, ngang heuristic.

Như vậy bài thiếu thông tin để tái hiện chính xác. Nên ghi điều này trong bài và hỏi mentor. Nếu có thể, liên hệ tác giả để xin θ, cách tính r_t và cách xử lý lô chờ duyệt.

## 6. Hạn chế cần nêu trong bài

1. Trạng thái đầu là dựng lại (D3), nên MI0 = 0,8645 khác 0,918 của de Assis.
2. Giả định nhập bù tức thì đúng các đơn vị vừa lấy, và lấy ở ô gần depot nhất (quy tắc B).
3. Mỗi cấu hình RL chỉ huấn luyện 1 seed vì giới hạn tính toán (CPU 2 lõi). Baseline có 5 seed.
4. Kết quả kiểm tra bao gồm giai đoạn chuyển tiếp từ kho ngẫu nhiên: mọi chính sách đều cải thiện dần theo tháng.
5. Toạ độ tính theo đơn vị của dataset, chưa đổi ra mét.

## 7. Việc tiếp theo đề xuất

- Chạy thêm 2–4 seed huấn luyện cho A để có khoảng tin cậy.
- Thử λ lệch về hai đầu, hoặc huấn luyện riêng cho λ ≤ 0,2, để khắc phục hạn chế ở 5.2.
- Phần chi phí (đồng) để sau, đúng như nhóm đã thống nhất.
