# Câu hỏi cho mentor – phần RL cất hàng (01/11/2026)

Các câu được xếp theo mức độ ảnh hưởng tới kết quả.

1. **Chuẩn hoá hàm mục tiêu.** Nhóm đổi J từ λ·MI + (1−λ)·D/D0 sang λ·MI/MI0 + (1−λ)·T/T0. Lý do:
   - Hai số hạng có cùng thang đo.
   - Kho hiện trạng có J = 1 với mọi λ.
   - λ trở thành tỷ trọng của mức cải thiện tương đối.

   Anh/chị thấy cách chuẩn hoá này có ổn không?

2. **Đo quãng đường.** Proxy "khứ hồi tới ô gần nhất của từng SKU" chỉ tương quan thứ hạng ρ = 0,66 với lộ trình thật. Nhóm chuyển sang dùng độ dài lộ trình thật của từng wave (NN + 2-opt, khớp GRO với r = 0,996). Hệ quả là reward đến trễ, sau mỗi wave. Cách này có hợp lý để làm reward cho RL không?

3. **Không tái hiện chính xác được de Assis 2026.** Nhóm đã cài đủ Algorithm 1 và chạy độ nhạy cho các tham số bài không công bố: θ, ε, top-k, trọng số ban đầu, r_t, cách xử lý lô chờ duyệt. Kết quả: mức giảm MI vẫn lớn hơn bài (0,76 so với 0,83), và supervised đứng ở thứ tự ngược với bài. Nhóm nên ghi đây là hạn chế, hay nên email tác giả để hỏi?

4. **Trạng thái đầu dựng lại.** Vùng trộn trong `Random_Storage.csv` (1.093 ô) có cỡ 4–16 phân bố đều và không có dòng lấy hàng nào rơi vào đó. Vì vậy nhóm dựng lại kho ngẫu nhiên theo nhu cầu thật. Như vậy có chấp nhận được không?

5. **Hạn chế ở hai đầu λ.** Khi λ ≤ 0,1 hoặc λ ≥ 0,9, các quy tắc chuyên một mục tiêu thắng mô hình A. Nhóm nên trình bày A là "tốt nhất ở vùng cân bằng", hay nên cố cải thiện thêm, ví dụ huấn luyện riêng cho từng đầu?

6. **Số seed.** Mỗi cấu hình RL hiện mới có 1 seed, do chạy trên CPU 2 lõi. Với hội nghị thì cần tối thiểu bao nhiêu seed?
