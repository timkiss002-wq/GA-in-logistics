# Bộ dữ liệu (tải riêng, không đưa lên repo)

de Assis, R. F., et al. (2025). *Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company*. Data in Brief. Mendeley Data, mã `pf2w725pw3`: https://data.mendeley.com/datasets/pf2w725pw3

Cách đặt dữ liệu:
1. Tải file zip từ Mendeley.
2. Giải nén vào thư mục `data/` này. Đường dẫn có thể có thư mục con, ví dụ `data/Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company/Picking_Wave.csv`.

`data.py` sẽ tự tìm thư mục chứa `Picking_Wave.csv` trong `data/`. Nếu dữ liệu nằm ở nơi khác, đặt biến môi trường `SLAP_DATA` trỏ tới thư mục đó.

Các file được dùng:
- `Picking_Wave.csv`
- `Customer_Order.csv`
- `Storage_Location.csv`
- `Support_Points_Navigation.csv`
- `Random_Storage.csv`
- `Dedicated_Storage.csv`
- `Class_Based_Storage.csv`
- `Hybrid_Storage.csv`

**Không cần dữ liệu** nếu chỉ muốn xem lại bảng và hình: `bang_nhu_bai_bao.py` và `ve_boxplot_phan_tram.py` đọc trực tiếp từ các file kết quả có sẵn trong `results/`.
