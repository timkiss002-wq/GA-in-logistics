# GA in logistics – Tái hiện GRO (de Assis et al., 2024) và Gnone et al. (2025)

Repo này chứa code và kết quả của nhóm khi tái hiện hai bài báo trên cùng một bộ dữ liệu kho giày (de Assis et al., 2025, Data in Brief):

1. **de Assis et al. (2024), IEEE Access** – thuật toán Genetic Route Optimisation (GRO) cho lộ trình lấy hàng.
2. **Gnone et al. (2025), IFAC-PapersOnLine 59(10)** – so sánh 4 chính sách lưu kho (Random, Dedicated, Class-Based, Hybrid) bằng GRO.

Các câu hỏi cần mentor góp ý nằm trong **[CAU_HOI_CHO_MENTOR.md](CAU_HOI_CHO_MENTOR.md)**.

---

## 1. Xem nhanh kết quả, không cần chạy lại

| Muốn xem | Mở file |
|---|---|
| Đủ các bảng như bài Gnone (Bảng 2, Bảng 3, Shapiro–Wilk, ANOVA, hậu kiểm, thời gian) cho 3 phương án chọn ô | `results/bang_nhu_bai_bao.xlsx` (mỗi bảng một sheet) hoặc `results/bang_nhu_bai_bao.txt` |
| Hình 3a – boxplot (đơn vị gốc) | `results/hinh3a_boxplot.png` |
| Hình 3a – boxplot chuẩn hoá theo % Original trung bình | `results/hinh3a_boxplot_phan_tram.png` |
| Hình 3b – scatter Original vs GRO | `results/hinh3b_scatter.png` |
| Ảnh hưởng của số món mỗi wave (25 cố định so với thay đổi) | `results/C_bien_doi_ket_qua.txt`, `results/hinh_C_bien_doi.png` |
| Tái hiện GRO 2024 (51 pick list thật) | `results/tai_hien_gro2024_bang5.csv` |
| Vì sao cách chọn ô đảo ngược thứ hạng (minh hoạ 1 wave) | `results/minh_hoa_wave0.png` |

## 2. Cài đặt và chạy

```bash
pip install -r requirements.txt
# Đặt bộ dữ liệu vào data/ (xem data/README.md)
```

| Lệnh | Làm gì | Cần dữ liệu? | Thời gian (2 nhân CPU) |
|---|---|---|---|
| `python bang_nhu_bai_bao.py` | Dựng lại mọi bảng và hình 3a, 3b từ `results/` | Không | vài giây |
| `python ve_boxplot_phan_tram.py` | Boxplot chuẩn hoá theo % Original | Không | vài giây |
| `python tai_hien_gnone2025.py kiemtra` | So mô hình khoảng cách với Dijkstra trên 3.000 cặp điểm (mong đợi: sai số 0) | Có | vài giây |
| `python tai_hien_gnone2025.py demo` | 1 wave: in quá trình tiến hoá của GRO sau mỗi 100 thế hệ | Có | ~10 giây |
| `python tai_hien_gro2024.py` | Tái hiện bài 2024: 51 pick list thật × 10 seed | Có | ~10 phút |
| `python tai_hien_gnone2025.py` | Phương án A và B: 50 wave × 4 chính sách × 10 seed | Có | ~1 giờ |
| `python chay_C_10seed.py` | Phương án C: 50 wave × 4 chính sách × 10 seed, kèm Original cố định | Có | ~15 phút |
| `python chay_C_wave_bien_doi.py` | Phương án C, số món mỗi wave thay đổi theo phân phối wave thật | Có | ~8 phút |
| `python chan_doan_gnone2025.py` | Thí nghiệm chẩn đoán: tầng 1–2 so với 4 tầng, × A/B/C (1 seed) | Có | ~10 phút |

Kết quả ghi vào `results/` và ghi đè lên file cũ cùng tên.

## 3. Cấu trúc code

| File | Vai trò |
|---|---|
| `data.py` | Đọc dữ liệu: ô kệ, 4 file chính sách, wave; chuẩn hoá cỡ giày |
| `gro.py` | Mô hình khoảng cách theo lối đi (rectilinear qua 3 lối dọc), ma trận khoảng cách, bản tham chiếu GA + quy hoạch động + 2-opt |
| `tai_hien_gro2024.py` | Tái hiện bài 2024: GRO "đúng chữ pseudo-code" và GRO "hợp lý" |
| `tai_hien_gnone2025.py` | Sinh 50 wave, router GRO hợp lý (6 bước GA), Original, phương án A/B, thống kê |
| `chay_C_10seed.py` | Phương án C (ô chỉ định sẵn) và Original cố định |
| `chay_C_wave_bien_doi.py` | Phương án C với số món mỗi wave thay đổi |
| `chan_doan_gnone2025.py` | Thay từng giả định một để tìm nguyên nhân lệch với bài báo |
| `bang_nhu_bai_bao.py` | Dựng bảng và hình giống bài Gnone |
| `ve_boxplot_phan_tram.py` | Boxplot chuẩn hoá |

## 4. Thiết lập và giả định của nhóm

Các điểm được đánh dấu **(*)** là chỗ bài báo không nói rõ, nhóm phải tự đặt giả định.

- **Router (GRO hợp lý):**
  - Np = 1000, G = 1000, tournament T = 3, Pc = 90%, Pm = 8% (Bảng 2 của Gnone).
  - Mỗi cá thể có một thứ tự đi riêng, lai OX, đột biến đổi chỗ 2 SKU, giữ lại cá thể tốt nhất (elitism).
  - Bản đúng từng chữ pseudo-code của bài 2024 cho kết quả tệ hơn công ty: nửa đầu lộ trình bị "đóng băng". Vì vậy nhóm dùng bản hợp lý. Bản này tái hiện được mức giảm 21,0% (bài: 21,5%).
- **Tour khép kín:** depot → các ô → depot. **Depot = CC-08 (403, 631) (*)**.
- **Chỉ lấy hàng ở tầng 1–2.** Tầng 3–4 là kho dự trữ (Gnone, mục 3).
- **Wave:** 50 wave, bốc SKU (mã × cỡ) không lặp, xác suất tỉ lệ với nhu cầu trong `Customer_Order`, seed 2025. Chỉ dùng SKU có ô ở cả 4 chính sách (969 SKU, chiếm 57,7% nhu cầu).
  - Bản chính: 25 SKU mỗi wave **(*)**.
  - Bản thử: số SKU thay đổi theo phân phối của wave thật.
- **Cách chọn ô khi một SKU nằm ở nhiều ô (*):**
  - **A** – GA tự chọn ô (Family-TSP);
  - **B** – luôn lấy ở ô gần depot nhất;
  - **C** – WMS chỉ định sẵn một ô (ngẫu nhiên, cố định theo wave). GA chỉ tìm thứ tự đi.
- **Original (*):** bố trí thực tế của công ty (`Random_Storage.csv`) + ô chỉ định + đi theo lối kiểu S-shape, không dùng GA. Dùng chung một Original cho cả A, B, C. Trên 51 pick list thật, lộ trình thật của công ty dài hơn cách đi S-shape này 9,9%.
- **Số lần chạy:** 10 seed mỗi wave; giá trị của mỗi wave là trung bình 10 seed.
- **Đơn vị:** đơn vị toạ độ của dataset. Chưa quy đổi được sang mét, nên chỉ so % và thứ hạng.
- **Kiểm chứng:**
  - Mô hình khoảng cách khớp Dijkstra trên đồ thị lối đi, sai số 0.
  - GRO cách lời giải tốt nhất biết được 1,7–3,3% (phương án C).

## 5. Kết quả chính (% giảm so với Original, trung bình 50 wave)

| | Random | Dedicated | Class-Based | Hybrid | Thứ hạng |
|---|---|---|---|---|---|
| **Bài báo** | −16,0 | −31,2 | −24,3 | −29,5 | **D < H** < C < R |
| A – GA tự chọn ô | −47,8 | −26,9 | −48,5 | −66,1 | H < C < R < D |
| B – ô gần depot | −29,6 | −26,9 | −37,9 | −52,6 | H < C < R < D |
| **C – ô chỉ định** | −9,0 | −26,9 | −3,5 | −21,9 | **D < H** < R < C |
| C, số món thay đổi | −7,1 | −31,3 | −7,1 | −26,4 | D < H < R ≈ C (Median: D < H < C < R) |

- Chỉ C tái hiện được thứ hạng D < H và kết luận hậu kiểm của bài ("Dedicated và Hybrid thấp hơn có ý nghĩa so với Random và Class-Based").
- Khi cho số món mỗi wave thay đổi, F của ANOVA giảm từ 76,3 xuống 5,1 (bài: 8,2), và độ phân tán gần với bài hơn.
- Class-Based vẫn lệch. Nghi do depot: lớp A trong file Class-Based nằm quanh CC-11/12, không quanh CC-08.
- **Phát hiện:** thứ hạng chính sách lưu kho **đảo ngược** tuỳ theo người lấy hàng có được chọn ô hay không (A/B so với C). Bài Gnone không nêu giả định này.
