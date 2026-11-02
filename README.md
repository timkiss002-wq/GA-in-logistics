# 08_pitching – toàn bộ code, kết quả và tài liệu cho buổi pitching

**Đề tài:** Cất hàng (putaway) trong kho giày dép, tối ưu đồng thời mức trộn hàng trong ô (Mixing Index, MI) và quãng đường lấy hàng (đo bằng GRO).
Hướng tiếp cận: kết hợp hai bài gốc, de Assis et al. (2026) cho phía nhập hàng và Gnone et al. (2025) cho phía lấy hàng.

**Dữ liệu:** de Assis et al. (2025), *Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company*, Data in Brief. Mendeley Data `pf2w725pw3`.

Ngày gom: 02/11/2026. Đây là bản **sao**: mọi file gốc vẫn nằm nguyên ở `03_code`, `04_outputs`, `07_impliment`, `06_vietbao`.

---

## Cấu trúc thư mục

| Thư mục | Nội dung | Lên GitHub? |
|---|---|---|
| `01_bai_bao_goc/` | PDF các bài tham khảo (Gnone 2025, de Assis 2024/2026, dataset paper, Wanis 2026…) | Không (bản quyền) |
| `02_data/` | Dữ liệu gốc `order_picking/` và 2 file Excel giải thích dữ liệu | Không (tải từ Mendeley) |
| `03_tai_hien_GRO2024_Gnone2025/` | **Implement bài gốc 1:** GRO của de Assis et al. (2024) và thí nghiệm 4 chính sách lưu kho của Gnone et al. (2025), phương án chọn ô A/B/C, GA minh hoạ từng bước | Có (trừ `data/`) |
| `04_RL_putaway_va_deAssis2026/` | **Implement bài gốc 2:** tái hiện de Assis et al. (2026), 2 cách đọc. **Mô hình chính của nhóm:** môi trường kho, hàm J, agent PPO (mô hình A), baseline, ablation, đánh giá lại bằng GRO | Có (trừ `data/` và `results/picks_*.npz`) |
| `05_hoc_RL_kho_ti_hon/` | RL từ con số 0: Q-learning trên kho 4 ô, kiểm tra lại công thức MI | Có |
| `06_du_lieu_va_wave_minh_hoa/` | Làm sạch dữ liệu, minh hoạ wave 33361 từng bước, hình nhu cầu và sản lượng | Có |
| `07_ket_qua_tong_hop/` | Bảng Excel, hình và báo cáo đã tổng hợp (RL 01/11, tái hiện Gnone) | Có |
| `08_bai_bao_nhap/` | Bản nháp bài báo IEEE (docx, pdf, zip Overleaf), Introduction cũ | Không (bài chưa công bố) |
| `09_tai_lieu_tham_khao/` | File RIS để nhập Zotero | Có |

---

## Mạch trình bày gợi ý và file tương ứng

| # | Nội dung pitching | Mở file nào | Chạy demo |
|---|---|---|---|
| 1 | Bài toán và dữ liệu: 1 wave chứa gì, quãng đường tính thế nào | `06_du_lieu_va_wave_minh_hoa/wave_33361_tung_buoc.xlsx` | `python wave_33361_replay.py` |
| 2 | GA/GRO hoạt động ra sao (quần thể, chọn lọc, lai, đột biến) | `03_.../ga_6_buoc.py`, `ga_tu_hoc.py` | `python ga_6_buoc.py` |
| 3 | Tái hiện GRO 2024 | `03_.../results/tai_hien_gro2024_bang5.csv` | `python tai_hien_gro2024.py` |
| 4 | Tái hiện Gnone 2025: thứ hạng chính sách đảo theo cách chọn ô (A/B/C) | `03_.../results/bang_nhu_bai_bao.xlsx`, `hinh3a_boxplot.png` | `python tai_hien_gnone2025.py demo` |
| 5 | Tái hiện de Assis 2026 (MI), 2 cách đọc và nguyên nhân lệch | `04_.../results/deassis2026/bang_tai_hien.md`, `bang_cachdoc2.md`, `nguyen_nhan_lech.md` | `python deassis2026_repl.py` |
| 6 | Gap: mỗi bài chỉ nhìn một phía (nhập hoặc lấy) → hàm mục tiêu J = λ·MI + (1−λ)·D/D0 | `04_.../BAO_CAO_NGHIEN_CUU.md` | – |
| 7 | RL hiểu từ kho tí hon (tổng reward = −J) | `05_hoc_RL_kho_ti_hon/README.md` | `python rl_ti_hon.py --xem 0.5` |
| 8 | Mô hình A: kết quả 5 seed, đánh đổi MI–quãng đường | `04_.../results/huong1/KET_QUA_HUONG1.md`, `fig1_danh_doi_5seed.png` | `python rl_ppo.py eval --tag main2` |
| 9 | Ablation và hạn chế | `04_.../results/huong1/bang9_ablation.md` | – |

## Các con số chính (đã đối chiếu với file kết quả)

**Tái hiện GRO 2024 (51 pick list thật).** Quãng đường trung bình của lộ trình thực tế là 3.446,8. Với GRO hợp lý là 2.724,1, tức giảm 20,97%. Bài báo công bố giảm 21,48%. GRO đúng theo pseudo-code của bài cho 5.021,4, dài hơn cả lộ trình thực tế.

**Tái hiện Gnone 2025.** Thứ hạng các chính sách lưu kho đảo ngược tuỳ cách chọn ô khi một SKU nằm ở nhiều ô. Chỉ phương án C (WMS gán sẵn ô) cho lại thứ hạng của bài: Dedicated < Hybrid < Class-Based < Random. Class-Based vẫn lệch, nghi do vị trí depot.

**Tái hiện de Assis 2026 (MI ban đầu 0,918).**

| Agent | MI bài | Cách đọc 1 | Cách đọc 2 |
|---|---|---|---|
| heuristic | 0,830 | 0,760 | 0,846 |
| supervised | 0,884 | 0,724 | 0,856 |
| adaptive | 0,822 | 0,748 | 0,846 |

Cách đọc 2 gần số của bài hơn.

**Mô hình A (5 seed).** A tốt hơn mọi heuristic ở λ = 0,3–0,5, có ý nghĩa ở cả 5/5 seed (Wilcoxon theo wave, Holm). Ở λ = 0,5 so với hiện trạng Random:
- lộ trình ngắn hơn khoảng 33% (3.225 so với 4.782);
- MI thấp hơn khoảng 17% (0,718 so với 0,861).

**Ablation.** Đặc trưng mixing là thành phần quan trọng nhất. PPO và "nhìn xa" (γ) đóng góp nhỏ hoặc không có bằng chứng. Vì vậy đóng góp của bài là **J cộng chính sách học được điều kiện theo λ**, còn PPO chỉ là bước tinh chỉnh.

**Hạn chế cần nói trước.**
- Điều kiện theo λ còn yếu: MI của A chỉ đổi từ 0,738 xuống 0,687 khi λ đi từ 0 đến 1.
- Khoảng cách tính theo đơn vị toạ độ của dataset, chưa đổi ra mét.
- Bài gốc không công bố code, nên một số tham số phải giả định. Các giả định ghi trong README của từng thư mục.

---

## Cài đặt và chạy

```
pip install numpy pandas scipy matplotlib numba networkx openpyxl torch
```
Bản torch CPU là đủ.

- `03_...`: đọc README trong thư mục. Dữ liệu nằm sẵn trong `03_.../data/` (bản trên máy).
- `04_...`: chạy theo bảng 10 bước trong README. Dữ liệu nằm sẵn trong `04_.../data/`. File `inst.npz`, `s0.npz`, `tref.npy` và mô hình trong `models/` đã có sẵn, không cần huấn luyện lại để đánh giá.
- `06_...`: hai script đọc dữ liệu từ `../02_data/order_picking`.

Nếu lấy code từ GitHub (không có dữ liệu): tải zip từ Mendeley, giải nén `order_picking` vào `03_.../data/` và `04_.../data/`. Các file `results/picks_*.npz` (khoảng 37 MB) không đưa lên GitHub; chạy lại bước 7 trong README của `04_...` để sinh lại.
