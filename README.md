# 08_pitching – toàn bộ code, kết quả và tài liệu cho buổi pitching

**Đề tài:** Cất hàng (putaway) trong kho giày dép, tối ưu đồng thời mức trộn hàng trong ô (Mixing Index, MI) và quãng đường lấy hàng (đo bằng GRO).
Hướng tiếp cận: kết hợp hai bài gốc, de Assis et al. (2026) cho phía nhập hàng và Gnone et al. (2025) cho phía lấy hàng.

**Dữ liệu:** de Assis et al. (2025), *Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company*, Data in Brief. Mendeley Data `pf2w725pw3`.

Ngày gom: 02/11/2026. Đây là bản **sao**: mọi file gốc vẫn nằm nguyên ở `03_code`, `04_outputs`, `07_impliment`, `06_vietbao`.

---

## Cấu trúc thư mục

Đọc nhanh: **03** = bài gốc 1 (GRO, Gnone 2025) · **04** = bài gốc 2 (de Assis 2026) + mô hình chính của nhóm (RL putaway) · **05–06** = tài liệu học và minh hoạ · **07** = kết quả đã tổng hợp để trình bày.

```
GA-in-logistics/
├── README.md                          # File này: tổng quan, cấu trúc, mạch pitching, số liệu chính
├── .gitignore                         # Loại dữ liệu gốc, bài báo, bản nháp, picks_*.npz
│
├── 03_tai_hien_GRO2024_Gnone2025/     # BÀI GỐC 1: GRO (de Assis 2024) + 4 chính sách lưu kho (Gnone 2025)
│   ├── README.md                      #   cách chạy + giả định của nhóm
│   ├── requirements.txt
│   ├── data/                          #   (trống trên GitHub) đặt order_picking/ từ Mendeley vào đây
│   ├── data.py                        #   đọc ô kệ, 4 file chính sách, wave; tính MI
│   ├── gro.py                         #   mô hình khoảng cách theo lối đi; GA + DP + 2-opt
│   ├── instances.py                   #   ghép wave với bố trí kho
│   ├── ga_6_buoc.py, ga_tu_hoc.py     #   GA nhỏ 4–8 ô, in từng bước B1–B6 (để học/trình bày)
│   ├── tai_hien_gro2024.py            #   tái hiện GRO 2024 trên 51 pick list thật
│   ├── tai_hien_gnone2025.py          #   tái hiện Gnone 2025 (kiemtra | demo | full | thongke)
│   ├── bang_nhu_bai_bao.py            #   xuất bảng giống bài báo (txt, xlsx)
│   ├── chan_doan_gnone2025.py         #   chẩn đoán vì sao lệch so với bài
│   ├── chay_C_10seed.py, chay_C_wave_bien_doi.py   # phương án C (WMS gán sẵn ô)
│   ├── ve_boxplot_phan_tram.py
│   ├── CAU_HOI_CHO_MENTOR.md
│   └── results/                       #   csv/txt/log/png: bảng 3, boxplot, scatter, bảng 5 GRO 2024
│
├── 04_RL_putaway_va_deAssis2026/      # BÀI GỐC 2 + MÔ HÌNH CHÍNH CỦA NHÓM
│   ├── README.md                      #   quy trình 10 bước chạy lại từ đầu
│   ├── BAO_CAO_NGHIEN_CUU.md          #   báo cáo: gap, hàm J, kết quả, hạn chế
│   ├── data/                          #   (trống trên GitHub) đặt order_picking/ từ Mendeley vào đây
│   │
│   │   # --- Chuẩn bị dữ liệu và trạng thái ban đầu (bước 1–3) ---
│   ├── data_prep.py                   #   dựng kho 1.692 ô, 2.454 SKU, chia wave 70/30  → inst.npz
│   ├── make_s0.py                     #   chạy Random 2.000 wave → kho ổn định         → s0.npz
│   ├── make_t0.py, make_tref.py       #   T0 và T_ref(w) để chuẩn hoá reward           → t0.json, tref.npy
│   │
│   │   # --- Môi trường, khoảng cách, agent (bước 4–7) ---
│   ├── env.py                         #   môi trường kho: lấy hàng, nhập bù, MI, D, reward; 5 baseline
│   ├── test_env.py                    #   6 kiểm thử (tổng reward = −J)
│   ├── tour_fast.py, check_tour.py    #   lộ trình nhanh NN + 2-opt, kiểm định so với GRO (r = 0,996)
│   ├── gro.py, eval_gro.py            #   GRO và đo lại quãng đường trên 300 wave kiểm tra
│   ├── run_baselines.py               #   Random, Nearest, Turnover, GreedyJ(λ)…
│   ├── rl_ppo.py                      #   agent PPO (train / eval), điều kiện theo λ
│   ├── run_ablation.sh                #   chạy ablation
│   ├── analyze.py, analyze_huong1.py  #   bảng, kiểm định Wilcoxon–Holm, hình
│   │
│   │   # --- Tái hiện de Assis 2026 ---
│   ├── run_deassis_env.py, deassis2026_repl.py      # cách đọc 1
│   ├── deassis2026_cachdoc2.py                      # cách đọc 2
│   ├── deassis2026_giathuyet.py, deassis2026_bang.py
│   │
│   ├── models/                        #   mô hình đã huấn luyện (.pt + .json): main, main2_s0..s4, ab_*, ab2_*, ab3_*
│   └── results/
│       ├── huong1/                    #   ★ KẾT QUẢ CHÍNH: 5 seed, bảng 7, ablation bảng 9, hình đánh đổi
│       ├── deassis2026/               #   bảng tái hiện 2 cách đọc, nguyên nhân lệch, hình
│       ├── bang_ket_qua.md, fig1–3_*.png
│       ├── baselines*.csv
│       ├── rl_*.csv                   #   kết quả RL theo λ (proxy D)
│       └── gro_*.csv                  #   kết quả đo lại bằng GRO
│                                      #   (picks_*.npz ~37 MB không lên GitHub, sinh lại bằng bước 7)
│
├── 05_hoc_RL_kho_ti_hon/              # Học RL từ số 0: Q-learning kho 4 ô, kiểm tra công thức MI
│   ├── README.md                      #   kèm 5 bài tập tự làm
│   ├── rl_ti_hon.py, kiem_tra_MI.py
│   └── ket_qua_*.txt, xem_tung_buoc_lambda_0.5.txt
│
├── 06_du_lieu_va_wave_minh_hoa/       # Làm sạch dữ liệu + minh hoạ wave 33361 từng bước
│   ├── clean_data.py, wave_33361_replay.py
│   ├── wave_33361_tung_buoc.xlsx
│   └── pareto_nhu_cau.png, san_luong_theo_thang.png
│
├── 07_ket_qua_tong_hop/               # Bản tổng hợp để trình bày (bản sao từ results/)
│   ├── RL_putaway_2026-11-01/         #   báo cáo, Bang_ket_qua_RL.xlsx, Hình 1–5
│   └── tai_hien_Gnone2025/            #   bang_nhu_bai_bao.xlsx, boxplot, scatter
│
└── 09_tai_lieu_tham_khao/             # File .ris để nhập Zotero (bài gốc, dữ liệu, SLAP nền tảng, khác)
```

**Chỉ có trên máy, không đưa lên GitHub:** `01_bai_bao_goc/` (PDF bài tham khảo, bản quyền), `02_data/` (dữ liệu gốc, tải từ Mendeley `pf2w725pw3`), `08_bai_bao_nhap/` (bản nháp IEEE, chưa công bố).

**Nên đọc theo thứ tự:** `README.md` này → `05_.../README.md` (hiểu RL và J trên kho tí hon) → `03_.../README.md` → `04_.../README.md` → `04_.../results/huong1/KET_QUA_HUONG1.md`.

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
