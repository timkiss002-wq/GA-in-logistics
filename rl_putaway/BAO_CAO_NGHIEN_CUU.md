# Báo cáo nghiên cứu nội bộ: RL cất hàng tối ưu đồng thời mixing và quãng đường (01/11/2026)

> Đây là ghi chép kỹ thuật để cả nhóm hiểu và kiểm tra lại. Phần viết bài gửi thầy nhóm tự viết, theo dặn của mentor.
> **Định vị đã chốt (02/11/2026, hướng 1):** đóng góp là hàm J và chính sách học được điều kiện theo λ; PPO là bước tinh chỉnh.
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
- Reward đến **trễ**: quyết định cất hôm nay ảnh hưởng đến các wave sau. Đây là lý do nhóm **thử** RL. Kết quả 3 seed (mục 5.3) cho thấy phần học thêm bằng PPO chỉ cải thiện ít, nên bài không lấy "nhìn xa" làm luận điểm.

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
| **Mô hình A, λ = 0,5, seed 0** | 0,684 | 3.328 | 3.256 | **0,737** |
| **Mô hình A, λ = 0,5, TB ± ĐLC 3 seed** | 0,707 ± 0,021 | 3.271 ± 61 | (chỉ đo seed 0) | **0,745 ± 0,007** |

Đọc bảng:
- **So với hiện trạng:** trung bình 3 seed, A giảm lộ trình 32% (3.271 so với 4.782) và MI 18% (0,707 so với 0,861). Riêng seed 0, GRO giảm 30,9%.
- **So với de Assis:** cả 3 chế độ đều có MI thấp hơn A, nhưng lộ trình dài hơn A 29–30% (T 4.220–4.259 so với 3.271).
- **So với Nearest:** Nearest đi ngắn hơn A 5% (3.099 so với 3.271), nhưng MI cao hơn hẳn (0,82).
- **Thống kê:** Friedman trên 8 chính sách cho χ² = 881,3, p ≈ 10⁻¹⁸⁶.

### 5.2 A tốt ở đâu, hạn chế ở đâu (Bảng 2 trong `bang_ket_qua.md`)

**Cập nhật: kết quả qua 3 seed huấn luyện (Bảng 7).** Bảng bên dưới thay cho kết luận cũ chỉ dựa trên 1 seed.

| λ | Heuristic tốt nhất ở λ đó | ΔJ của A, TB ± ĐLC 3 seed | Số seed A tốt hơn / kém hơn có ý nghĩa |
|---|---|---|---|
| 0 | Nearest | +0,023 ± 0,013 | 0 / 1 |
| 0,1 | Nearest | +0,014 ± 0,008 | 0 / 2 |
| 0,2 | Nearest | +0,001 ± 0,007 | 1 / 1 |
| **0,3** | Nearest | **−0,013 ± 0,005** | **3 / 0** |
| **0,4** | Turnover | **−0,021 ± 0,006** | **3 / 0** |
| **0,5** | GreedyJ | **−0,024 ± 0,007** | **3 / 0** |
| 0,6 | GreedyJ | −0,009 ± 0,009 | 2 / 0 |
| 0,7 | GreedyJ | +0,004 ± 0,013 | 1 / 2 |
| 0,8–1,0 | de Assis heuristic | +0,018 đến +0,046 | 0 / 2–3 |

Rút ra:
- **A tốt nhất một cách chắc chắn ở λ = 0,3–0,5**: cả 3 seed đều tốt hơn mọi baseline, có ý nghĩa thống kê. Ở λ = 0,6, A tốt hơn ở 2/3 seed.
- Kết luận 1 seed trước đây ("tốt nhất ở λ 0,3–0,7") là **quá lạc quan**. Phạm vi đúng là λ ≈ 0,3–0,6.
- Kết quả thay đổi khá nhiều giữa các seed: MI của A ở λ = 0,5 là 0,707 ± 0,021, lộ trình là 3.271 ± 61. Bài cần báo cáo TB ± ĐLC, không lấy riêng seed tốt nhất.
- A nằm trên đường Pareto, ở đoạn giữa Nearest và de Assis heuristic (Hình 1, có thanh sai số).
- **Hạn chế:** ở hai đầu λ, quy tắc chuyên một mục tiêu vẫn thắng.

### 5.3 Ablation study: vì sao chọn từng thành phần (Bảng 4 và Bảng 8)

**Cập nhật quan trọng: ablation then chốt qua 3 seed (Bảng 8).** Δ là J của biến thể trừ J của A cùng seed, trung bình trên λ = 0,2 / 0,5 / 0,8. Dương nghĩa là biến thể kém hơn.

| Biến thể | Δ từng seed | Δ TB ± ĐLC | Kết luận |
|---|---|---|---|
| A không PPO (chỉ bắt chước) | +0,011 / +0,001 / +0,003 | +0,005 ± 0,005 | PPO giúp **ít và không ổn định** |
| A với γ = 0 (không nhìn xa) | +0,012 / +0,000 / −0,012 | +0,000 ± 0,012 | **Không có bằng chứng** "nhìn xa" giúp ích |

**Điều chỉnh so với kết luận 1 seed.** Với seed 0, γ = 0 kém hơn 0,012 và nhóm đã kết luận "lợi ích đến từ nhìn xa". Qua 3 seed, kết luận này **không đứng vững**. Phần lớn ưu thế của A so với các heuristic đến từ:
- **mạng dùng chung cho mọi λ, bắt chước "danh mục" heuristic** (Nearest khi λ thấp, GreedyJ khi λ cao): bản chỉ bắt chước đã tốt hơn từng heuristic ở vùng λ giữa, vì mạng nội suy được giữa hai kiểu hành vi;
- **đường tắt ΔJ tức thời và đặc trưng mixing** (xem dưới).

PPO chỉ tinh chỉnh thêm một chút.

Các ablation khác (1 seed, cấu hình bắt chước GreedyJ):

| Câu hỏi | Thí nghiệm | Kết quả | Kết luận |
|---|---|---|---|
| Bắt chước ai? | Portfolio so với chỉ GreedyJ | Portfolio tốt hơn −0,006 (rõ nhất ở λ = 0,2: 0,708 so với 0,733) | Chọn portfolio |
| Có cần đặc trưng mixing? | Bỏ đặc trưng 2–4 | Kém hơn +0,031 (λ = 0,8: +0,08) | Cần |
| Có cần đường tắt ΔJ + khởi động? | Bỏ cả hai | Kém hơn +0,096 | Thành phần quan trọng nhất |
| Một mạng cho mọi λ hay mỗi λ một mạng? | Agent riêng cho λ = 0,5 | Kém hơn +0,0095 ở λ = 0,5 | Dùng chung λ tốt hơn |
| Đặc trưng khoảng cách? | Bỏ đặc trưng 5, 6, 8, 10 | Không kém hơn (−0,008) | Có thể rút gọn |
| Giảm nhiễu bằng T_ref? | Bỏ T_ref | Gần như không đổi (−0,002) | Không cần thiết |

Các ablation này mới có 1 seed. Với độ biến động giữa seed khoảng 0,005–0,012, chỉ hai kết luận có hiệu ứng lớn hơn hẳn độ nhiễu là đáng tin: "cần đặc trưng mixing" (+0,031) và "cần đường tắt + khởi động" (+0,096).

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

Như vậy, với thông tin bài công bố, không thể tái hiện chính xác. Chẩn đoán chi tiết nằm trong `results/deassis2026/nguyen_nhan_lech.md`.

### 5.5 Cách đọc 2: tái hiện gần với bài hơn

Manh mối chính: bài có MI 0,830 mà 99,66% ô vẫn trên 0,8. Điều đó chỉ xảy ra khi gần như mọi ô dừng đúng ở m = 3 đơn vị của mã chiếm ưu thế. Vậy ngưỡng θ phải đang chặn việc gom hàng ở mức critical.

**Cách đọc 2** gồm hai điểm:
- Chỉ cất vào ô có θ_min ≤ MI_l sau khi cất ≤ θ_max. θ_min và θ_max được hiệu chỉnh theo Bảng 2 của bài, vì bài không công bố (lưới 21 tổ hợp; chọn θ ∈ [0,82; 0,88]).
- Lô supervised chờ duyệt do người vận hành cất ngẫu nhiên.

Thiết lập chạy: 847 ô, 5 seed (`deassis2026_cachdoc2.py`).

| Chế độ | MI bài | MI cách đọc 2 | MI cách đọc 1 | % critical bài | % critical cách đọc 2 | Hành động bài | Hành động cách đọc 2 |
|---|---|---|---|---|---|---|---|
| Heuristic | 0,830 | 0,846 ± 0,001 | 0,760 | 99,66 | 99,36 | ≈ 2.500 | 2.742 |
| Supervised | 0,884 | 0,856 ± 0,001 | 0,724 | 99,71 | 99,79 | ≈ 2.000 | 1.003 |
| Adaptive | 0,822 | 0,846 ± 0,001 | 0,748 | 99,48 | 99,36 | ≈ 2.500 | 2.740 |

**Khớp với bài:**
- MI lệch bài 0,016–0,028; với cách đọc 1, độ lệch là 0,07–0,16.
- % ô critical khoảng 99,4–99,8%, sát bài.
- Supervised kém nhất, đúng như bài.
- Heuristic đứng yên quanh 0,85 sau khoảng 3.000 lô; số hành động dừng ở khoảng 2.700 (Hình 3 và 4 của bài: 0,83 và 2.500).

**Chưa khớp:**
- Trong bài, adaptive tốt hơn heuristic 0,008; ở cách đọc 2, hai chế độ ngang nhau.
- Supervised thực hiện khoảng 1.000 hành động, trong khi bài là khoảng 2.000.

Kết luận: **cách đọc 2 tái hiện được hình dạng kết quả của bài**, chỉ lệch nhỏ về độ lớn. Đây là bản tái hiện tốt nhất có thể khi không có code gốc.

### 5.6 Mô hình A có còn hơn de Assis không, theo cả hai cách đọc? (Bảng 6, seed 0)

| λ | So với de Assis, cách đọc 1 (đúng chữ, baseline mạnh hơn) | So với de Assis, cách đọc 2 (gần bài hơn) |
|---|---|---|
| 0,2 | A tốt hơn cả 3 chế độ, ΔJ −0,13 đến −0,14 | A tốt hơn, ΔJ −0,26 đến −0,27 |
| 0,5 | A tốt hơn, ΔJ −0,06 đến −0,08 | A tốt hơn, ΔJ −0,21 đến −0,24 |
| 0,8 | Seed 0: ngang heuristic (p = 0,81), tốt hơn supervised và adaptive. **Qua 3 seed: kém heuristic** (ΔJ +0,018 ± 0,016, 2/3 seed kém có ý nghĩa) | A tốt hơn, ΔJ −0,17 đến −0,21 |

Trong môi trường của nhóm, de Assis theo cách đọc 2 chỉ còn MI 0,80–0,83, còn lộ trình gần như bằng Random (GRO 4.640–4.722 so với 4.713). Lý do: ngưỡng chặn phần lớn việc gom hàng, nên hàng bị cất ngẫu nhiên.

**Ở λ = 0,2–0,5, kết luận của nhóm không phụ thuộc cách đọc.** Ở λ = 0,8, kết luận phụ thuộc cách đọc: theo cách đọc 1, heuristic de Assis tốt hơn A. Cách đọc 1 là baseline mạnh hơn, nên nhóm dùng nó làm so sánh chính để kết quả đứng vững hơn.

## 6. Hạn chế cần nêu trong bài

1. Trạng thái đầu là dựng lại (D3), nên MI0 = 0,8645 khác 0,918 của de Assis.
1b. de Assis 2026 không tái hiện chính xác được do bài thiếu tham số. Nhóm báo cáo hai cách đọc (mục 5.4–5.6); θ của cách đọc 2 được hiệu chỉnh theo số của bài.
2. Giả định nhập bù tức thì đúng các đơn vị vừa lấy, và lấy ở ô gần depot nhất (quy tắc B).
3. Mô hình A và 2 ablation then chốt có 3 seed huấn luyện; các ablation khác chỉ có 1 seed (CPU 2 lõi). Baseline có 5 seed.
4. Kết quả kiểm tra bao gồm giai đoạn chuyển tiếp từ kho ngẫu nhiên: mọi chính sách đều cải thiện dần theo tháng.
5. Toạ độ tính theo đơn vị của dataset, chưa đổi ra mét.

## 7. Việc tiếp theo đề xuất

- Đã chạy 3 seed cho A (mục 5.2, 5.3). Nếu có máy mạnh hơn, nên lên 5 seed và chạy lại các ablation phụ với nhiều seed.
- Phần RL (PPO) hiện chỉ cải thiện ít so với bắt chước. Hướng thử: huấn luyện lâu hơn, reward theo hiệu chi phí từng wave, hoặc tăng tầm nhìn (γ) cùng giảm nhiễu reward.
- Thử λ lệch về hai đầu, hoặc huấn luyện riêng cho λ ≤ 0,2, để khắc phục hạn chế ở 5.2.
- Phần chi phí (đồng) để sau, đúng như nhóm đã thống nhất.
