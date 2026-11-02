# Vì sao chưa tái hiện được de Assis et al. (2026)? Chẩn đoán nguyên nhân (01/11/2026)

Script: `deassis2026_giathuyet.py`. Kết quả thô: `results/deassis2026/gia_thuyet.csv`.

## 1. Bài báo cho ra số nào

Bảng 2 của bài, sau 10.000 lô hàng về:

| Chế độ | MI trung bình | % ô có MI_l > 0,8 |
|---|---|---|
| Ban đầu | 0,918 | 100 |
| Heuristic | 0,830 | 99,66 |
| Supervised | 0,884 | 99,71 |
| Adaptive | 0,822 | 99,48 |

## 2. Manh mối mạnh nhất: con số của bài gần như chỉ có một trạng thái kho cuối khớp được

Với ô đầy 18 đơn vị, MI_l = 1 − m/18, trong đó m là số đơn vị của mã chiếm ưu thế trong ô. MI_l chỉ nhận vài giá trị rời rạc:

| m | 2 | 3 | 4 |
|---|---|---|---|
| MI_l | 0,889 | **0,833** | 0,778 |

Heuristic của bài có MI trung bình 0,830, đồng thời 99,66% ô vẫn có MI_l > 0,8. Hai điều này chỉ cùng đúng khi gần như mọi ô dừng đúng ở m = 3, và gần như không ô nào đạt m ≥ 4.

Nghĩa là trong bài có một cơ chế chặn việc gom hàng ngay trước ngưỡng 0,8. Đây cũng chính là ngưỡng "critical" của bài.

**Khi đọc đúng chữ của bài, không có cơ chế chặn nào.** Lúc đó heuristic gom tiếp. Kết quả trên 847 ô: 64% số ô có m ≥ 4, MI = 0,767, chỉ 37% ô critical. Đó là lý do số của nhóm "tốt hơn" bài nhiều.

## 3. Kiểm định từng giả thuyết

Mỗi giả thuyết chạy 3 seed, cả 3 chế độ. Mỗi ô ghi MI / % critical / số lô được cất.

| Giả thuyết | Heuristic | Supervised | Adaptive | Khớp % critical ~100? | Đúng thứ tự A < H < S? |
|---|---|---|---|---|---|
| Bài báo | 0,830 / 99,7 / ~2.500 | 0,884 / 99,7 / ~2.000 | 0,822 / 99,5 / ~2.500 | | |
| H0 Đọc đúng chữ (1.093 ô thật) | 0,760 / 34 / 3.920 | 0,724 / 20 / 3.919 | 0,747 / 30 / 3.920 | Không | Không |
| H1 847 ô đổ ngẫu nhiên (đúng quy mô bài) | 0,766 / 37 / 3.036 | 0,723 / 20 / 3.035 | 0,752 / 32 / 3.036 | Không | Không |
| H2 SKU = mã × cỡ | 0,871 / 88 / 3.920 | 0,776 / 40 / 3.059 | 0,852 / 87 / 3.920 | Gần | Không |
| **H3 θ là SÀN: không cất nếu MI ô sau < 0,8** | 0,861 / **100** / 3.035 | 0,824 / 77 / 1.179 | 0,838 / **100** / 3.035 | **Có** | Không |
| H4 Lô chờ duyệt do người vận hành cất ngẫu nhiên | 0,766 / 37 / 3.036 | 0,768 / 34 / 2.362 | 0,752 / 32 / 3.036 | Không | **Có** |
| H5 Lô hàng về theo nhu cầu lấy hàng | 0,780 / 44 / 3.036 | 0,739 / 26 / 2.745 | 0,761 / 34 / 3.036 | Không | Không |
| H6 Số lượng mỗi lô theo đơn hàng (TB 1,86) | 0,766 / 35 / 1.852 | 0,735 / 23 / 1.892 | 0,755 / 31 / 1.933 | Không | Không |
| H7 Giải phóng chỗ bằng cách làm rỗng 20% số ô | 0,845 / 80 / 3.029 | 0,733 / 80 / 3.022 | 0,840 / 81 / 3.029 | Không | Không |
| **H3 + H4** | 0,861 / 100 / 3.035 | 0,840 / 99 / 1.250 | 0,838 / 100 / 3.035 | **Có** | Gần (A < S < H) |

Ngoài ra, các tham số đã thử ở Bảng R2 đều làm MI đổi dưới 0,01: ε, top-k, ω, trọng số ban đầu, cách tính r_t, θ trần, phạm vi ô. Vậy chúng **không phải** nguyên nhân chính.

## 4. Kết luận: nguyên nhân có khả năng nhất, xếp theo mức độ

1. **Ngưỡng θ, rất có thể.** Bài chỉ viết "MI_l sau ≤ θ" và không cho giá trị. Nếu ràng buộc này là trần, như cách đọc chữ, thì nó gần như không bao giờ chặn gì. Chỉ khi ngưỡng tác động như một **sàn ở 0,8** thì mới tái hiện được dấu hiệu "gần như mọi ô dừng ở m = 3, ~100% ô critical". Có thể code gốc dùng ngưỡng theo chiều khác với công thức in trong bài, hoặc có một điều kiện dừng không được mô tả.
2. **Cách xử lý lô supervised chờ người duyệt, khả năng cao.** Bài nói "chuyển cho người duyệt" nhưng không nói sau đó lô đó ra sao.
   - Nếu lô bị bỏ, supervised thành chế độ tốt nhất (ngược với bài).
   - Nếu người vận hành cất ngẫu nhiên, supervised kém đi và thứ tự gần với bài hơn.
3. **Định nghĩa SKU và trạng thái đầu, khả năng trung bình.** MI tính theo mã × cỡ đẩy kết quả về gần bài (critical 88%). Trạng thái đầu là "kho đầy ngẫu nhiên dựng từ layout thật" với 847 ô, không phải file có sẵn trong dataset. Trong `Random_Storage.csv`, tầng 1 chỉ có 249 ô trộn.
4. **Các điểm bài tự mâu thuẫn hoặc thiếu**, cần ghi chú khi trích dẫn:
   - "Tỷ lệ thực hiện 5,85%" của supervised không khớp với ~2.000 hành động trên 10.000 lô (20%).
   - Không có giá trị cho θ, ς, ε, k và trọng số ban đầu w.
   - Không định nghĩa r_t của chế độ adaptive.
   - Look-ahead chia lô gần như không bao giờ được kích hoạt, vì 99,6% lô trong dữ liệu chỉ có 1 đơn vị.

**Tổng kết:** không có tổ hợp giả thuyết nào khớp đồng thời cả MI, % critical và thứ tự của bài. Với thông tin hiện có, **không thể tái hiện chính xác** bài này.

Ngoài nhóm, còn ít nhất một repo công khai khác cũng tái hiện bài này (github.com/baocutewaa/slap-agentic-architecture). Repo đó dùng 600 ô ở tầng 3–4, MI ban đầu 0,9136, và chưa công bố kết quả.

## 5. Đề xuất

- **Hỏi tác giả** (rodrigo@iti.sdu.dk, địa chỉ có trong bài) 4 câu:
  1. Giá trị θ và ς, và ngưỡng được áp dụng theo chiều nào?
  2. Lô supervised chờ duyệt được xử lý thế nào trong mô phỏng?
  3. r_t của chế độ adaptive là gì, và w được khởi tạo bằng bao nhiêu?
  4. Trạng thái đầu 847 ô được dựng ra sao, và SKU là mã hay mã × cỡ?
- **Khi viết bài:** báo cáo tái hiện theo hai cách đọc, làm cận trên và cận dưới:
  - H0, đọc đúng chữ: θ không ràng buộc.
  - H3 + H4: θ là sàn 0,8, lô chờ duyệt do người cất ngẫu nhiên.

  Nêu rõ rằng kết luận của nhóm (A tốt nhất ở vùng λ cân bằng) được kiểm tra với 3 chế độ theo cách đọc H0.
- **Kiểm tra thêm nếu cần:** đánh giá lại mô hình A với baseline de Assis theo cách đọc H3 + H4, để chắc kết luận không phụ thuộc cách đọc.

## 6. Kết quả chạy theo cách đọc 2 (đã thực hiện)

Chi tiết: `bang_cachdoc2.md`, `hinh_cachdoc2.png`, `cachdoc2_hieu_chinh.csv`.

- **Hiệu chỉnh.** Lưới θ_min ∈ {0,78…0,83} × θ_max ∈ {0,86…1,0}, cộng với lô chờ duyệt do người cất. Giá trị tốt nhất là θ ∈ [0,82; 0,88].
  - Kết quả rất nhạy với θ_max, vì MI_l chỉ nhận giá trị rời rạc: 0,88 ≈ 1 − 2/17.
- **5 seed trên 847 ô:**
  - Heuristic 0,846 / 99,4% critical / 2.742 hành động.
  - Supervised 0,856 / 99,8% / 1.003.
  - Adaptive 0,846 / 99,4% / 2.740.
- **So với bài** (0,830 / 0,884 / 0,822, critical ~99,5–99,7%):
  - Khớp: dạng đường cong, % critical, supervised kém nhất.
  - Lệch: MI lệch 0,016–0,028; adaptive ngang heuristic (bài: tốt hơn 0,008); supervised thực hiện ít hành động hơn bài.
- **Hệ quả cho bài của nhóm:** mô hình A tốt hơn cả 3 chế độ de Assis theo cả hai cách đọc ở λ = 0,2 và 0,5. Ở λ = 0,8, A ngang heuristic theo cách đọc 1 và tốt hơn hẳn theo cách đọc 2 (Bảng 6 trong `bang_ket_qua.md`).
