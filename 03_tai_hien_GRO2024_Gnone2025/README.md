# slap_gro – tái hiện GRO (de Assis et al., 2024) và Gnone et al. (2025)

Bộ dữ liệu: de Assis et al. (2025), *Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company*, nằm trong thư mục `data\`.

## Cài đặt (một lần)
```
pip install -r requirements.txt
```
Cần các thư viện numpy, pandas, scipy và numba.

## Các file

| File | Vai trò |
|---|---|
| `data.py` | Đọc dữ liệu (ô kệ, 4 file chính sách, wave), chuẩn hoá cỡ giày, tính Mixing Index |
| `gro.py` | Mô hình khoảng cách theo lối đi; bản tham chiếu GA + quy hoạch động + 2-opt (`gro_dp`) |
| `instances.py` | Hàm phụ trợ: ghép wave với bố trí kho |
| `ga_6_buoc.py`, `ga_tu_hoc.py` | GA nhỏ để học và trình bày: 4–8 ô, in từng bước B1–B6 |
| `tai_hien_gro2024.py` | Tái hiện de Assis et al. (2024): 51 pick list thật, GRO đúng-chữ và GRO hợp-lý |
| `tai_hien_gnone2025.py` | Tái hiện Gnone et al. (2025): 50 wave sinh × 4 chính sách × 2 giả định chọn ô × 10 seed |

## Chạy `tai_hien_gnone2025.py`

| Lệnh | Làm gì | Thời gian |
|---|---|---|
| `python tai_hien_gnone2025.py kiemtra` | So mô hình khoảng cách với Dijkstra (kết quả mong đợi: sai số 0) | vài giây |
| `python tai_hien_gnone2025.py demo` | 1 wave: in quá trình tiến hoá của GRO cho 4 chính sách × 2 giả định | ~10 giây |
| `python tai_hien_gnone2025.py` | Thí nghiệm đầy đủ: 4.000 lần chạy GRO + thống kê | ~1 giờ (2 nhân) |
| `python tai_hien_gnone2025.py thongke` | In lại bảng thống kê từ `results\gnone2025_chi_tiet.csv` | vài giây |

Kết quả nằm trong `results\`:
- `gnone2025_chi_tiet.csv`: mỗi wave × chính sách × giả định × seed.
- `gnone2025_bang3_<giả định>.csv`: bảng giống Bảng 3 của bài.
- `gnone2025_theo_wave_<giả định>.csv`: mỗi wave là một dòng, lấy trung bình 10 seed.
- `gnone2025_thong_ke.txt`: toàn bộ kiểm định.
- `gnone2025_waves.csv`: danh sách SKU của 50 wave.

## Giả định của nhóm (bài báo không nói rõ)

1. **Chỉ lấy hàng ở tầng 1–2.** Tầng 3–4 là kho dự trữ (Gnone, mục 3).
2. **Wave:** 25 SKU khác nhau, bốc không lặp, xác suất tỉ lệ với tổng số lượng đặt trong Customer_Order. Chỉ dùng SKU có ô ở cả 4 chính sách, để 4 chính sách được so trên cùng một bộ wave. Seed của wave = 2025.
3. **Router = GRO hợp-lý:**
   - mỗi cá thể có một thứ tự đi riêng và một ô được chọn cho từng SKU;
   - tournament T = 3;
   - lai OX cho thứ tự, trộn đều cho ô;
   - đột biến: đổi chỗ 2 SKU và đổi 1 ô;
   - giữ lại cá thể giỏi nhất (elitism);
   - Np = 1000, G = 1000, Pc = 90%, Pm = 8% (Bảng 2 của Gnone).
4. **Hai giả định chọn ô:**
   - A (`tu_chon`): GA tự chọn ô cho từng SKU;
   - B (`gan_depot`): mỗi SKU chỉ lấy ở ô gần depot nhất.
5. **Original:**
   - bố trí thực tế của công ty (Random_Storage.csv), không dùng GA;
   - người lấy hàng đi hết lối này sang lối khác, đổi chiều mỗi lối, chọn đi từ đầu hay từ cuối kho tuỳ cách nào ngắn hơn;
   - ô: chọn ngẫu nhiên trong họ (giả định A) hoặc ô gần depot (giả định B).

   Trên 51 pick list thật, cách đi này ngắn hơn lộ trình thật của công ty khoảng 7%.
6. **Depot** = CC-08 (403, 631). **Đơn vị** = đơn vị toạ độ của dataset; chưa quy đổi được sang mét của bài báo.
7. **Lời giải tốt nhất biết được (BKS)** = min(GRO qua 10 seed, GA + quy hoạch động + 2-opt). Dùng để báo khoảng cách tới tối ưu.
