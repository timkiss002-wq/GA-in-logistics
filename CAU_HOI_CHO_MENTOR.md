# Câu hỏi cần mentor góp ý

Mỗi câu hỏi gồm: bối cảnh, lựa chọn hiện tại của nhóm, và các phương án. Số liệu chi tiết xem `README.md` và `results/`.

---

## A. Quyết định phương pháp (ảnh hưởng trực tiếp tới kết luận)

### 1. Khi một SKU nằm ở nhiều ô, người lấy hàng lấy ở ô nào?
- **Bối cảnh:** bài Gnone không nói. Câu trả lời đảo ngược hoàn toàn thứ hạng:
  - C (WMS chỉ định ô): Dedicated tốt nhất, giống bài.
  - A (GA tự chọn ô) và B (ô gần depot): Hybrid tốt nhất, Dedicated tệ nhất.
- **Nhóm đề xuất:** C làm thiết lập chính, A và B làm phân tích độ nhạy. Lý do:
  - `Picking_Wave.csv` ghi sẵn một ô cho mỗi dòng lấy hàng;
  - bài 2024 viết "mỗi vị trí lấy từ WMS phải được ghé";
  - C khớp kết quả của bài.
- **Hỏi:** anh có đồng ý cách chọn này không? Việc thứ hạng đảo ngược theo cách chọn ô có đủ để thành một giả thuyết/đóng góp riêng của bài không?

### 2. "Original" được tính như thế nào?
- **Bối cảnh:** bài chỉ ghi "company's baseline distance".
  - Nhóm đang dùng: bố trí thực tế + người lấy hàng đi theo lối kiểu S-shape. Đây là một mốc "khó", vì lộ trình thật của công ty dài hơn mốc này 9,9%.
  - Nếu quy Original về lộ trình thật (×1,099), % giảm của C sẽ là Random −17,2%, Dedicated −33,5%, Hybrid −29,0%, Class-Based −12,2%. Ba trên bốn chính sách khớp bài trong khoảng 2,3 điểm %.
- **Hỏi:** nên giữ mốc S-shape (thận trọng), hay quy về lộ trình thật của công ty? Hệ số 1,099 đo trên pick list thật (TB 13,6 sản phẩm), áp sang wave sinh có được không?

### 3. Mỗi wave có bao nhiêu món?
- **Bối cảnh:** bài ghi lúc thì "25 SKU", lúc thì "up to 25 items".
  - Cố định 25 SKU: CV khoảng 10%, F = 76.
  - Bốc theo phân phối wave thật: F = 5,1 (bài: 8,2), nhưng có quá nhiều wave 1–5 món (CV 51% so với 35% của bài), và Shapiro–Wilk không đạt.
- **Hỏi:** nhóm nên chọn cách nào làm bản chính? Có nên chặn dưới (ví dụ ≥ 5 món), hay chỉ cần báo cáo cả hai?

### 4. Vị trí depot
- **Bối cảnh:** bài chỉ nói có "một điểm vào/ra duy nhất", không cho toạ độ.
  - Nhóm đặt depot ở CC-08 (giữa kho), vì đó là tâm vùng hàng bán chạy của file Dedicated/Hybrid.
  - Lớp A của file Class-Based lại nằm quanh CC-11/12. Có thể vì vậy mà Class-Based chỉ giảm −3,5% (bài −24,3%), và ở wave 1–5 món còn dài hơn Original 43%.
- **Hỏi:** nhóm định chạy thêm phân tích độ nhạy theo vị trí depot (CC-08, CC-11/12, một điểm ở mép kho). Có cần không, hay chấp nhận Class-Based là điểm lệch đã biết?

### 5. Dùng bản GRO "hợp lý" thay vì bản đúng từng chữ pseudo-code
- **Bối cảnh:** làm đúng từng chữ pseudo-code của bài 2024 thì nửa đầu lộ trình bị đóng băng, và kết quả tệ hơn cả công ty (42/51 wave). Bản hợp lý (mỗi cá thể một thứ tự riêng, lai OX) tái hiện được mức giảm 21,0% (bài: 21,5%).
- **Hỏi:** nhóm dùng bản hợp lý và ghi rõ điểm sai lệch so với pseudo-code trong bài. Như vậy có được không?

## B. Trình bày và thống kê

### 6. Đơn vị đo
- **Bối cảnh:** README của dataset ghi toạ độ là mét, nhưng hai ô liền kề cách nhau 16 đơn vị và kho dài 1.440 đơn vị. Suy ngược từ hai bài báo ra hai hệ số khác nhau khoảng 7 lần: ≈ 0,023 m/đơn vị (bài 2024) và ≈ 0,160 m/đơn vị (bài 2025).
- **Hỏi:** chỉ so % và thứ hạng (cùng boxplot chuẩn hoá theo % Original) có chấp nhận được không? Hay nhóm nên hiệu chỉnh đơn vị bằng bản vẽ `Full layout.dwg`?

### 7. Số lần chạy GA
- **Bối cảnh:** nhóm chạy 10 seed mỗi wave và lấy trung bình. Một số tài liệu về metaheuristic khuyến nghị 30 lần chạy. Dao động giữa các seed của GRO hợp lý nhỏ (SD trung bình khoảng 31 đơn vị trên mỗi wave ở bài 2024).
- **Hỏi:** 10 seed có đủ không?

### 8. Kiểm định thống kê
- **Bối cảnh:** bài dùng ANOVA một chiều, dù thiết kế là ghép cặp (cùng 50 wave cho mọi chính sách). Nhóm báo cả ANOVA + Tukey HSD (để so với bài) lẫn Friedman + Wilcoxon ghép cặp có hiệu chỉnh Holm.
- **Hỏi:** khi viết, nên lấy bộ kiểm định nào làm chính?

## C. Dữ liệu

### 9. "2.000 SKU" của bài so với dữ liệu
- **Bối cảnh:** dữ liệu có 208 mã giày. Tính theo mã × cỡ thì chỉ có 969 SKU có ô ở cả 4 file chính sách, chiếm 57,7% nhu cầu. Nhóm chỉ sinh wave từ 969 SKU này, để 4 chính sách được so trên cùng một bộ wave.
- **Hỏi:** cách lọc này có hợp lý không?

## D. Hướng đi tiếp

### 10. Có nên email hỏi tác giả không?
- **Bối cảnh:** các chi tiết còn thiếu gồm: cách chọn ô, toạ độ depot, phân phối số món mỗi wave, cách tính Original, đơn vị, và code GRO.
- **Hỏi:** anh có muốn nhóm gửi email cho nhóm tác giả (ÉTS Montréal) không? Nếu có, anh có góp ý gì về cách viết?

### 11. Khi nào dừng phần tái hiện?
- **Bối cảnh:** ba trên bốn chính sách đã gần khớp bài; thứ hạng và kết luận hậu kiểm đã khớp ở phương án C.
- **Hỏi:** phần tái hiện như hiện tại đã đủ để chuyển sang phần của nhóm chưa? Phần của nhóm gồm: tính lại MI và D/D₀ bằng GRO hợp lý theo phương án đã chốt, quét λ trong max R = λ(1 − MI) + (1 − λ)(1 − D/D₀), rồi tới GA/RL cho xếp kho. Hay nhóm cần khớp thêm Class-Based trước?
