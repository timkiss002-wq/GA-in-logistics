# LP-RL: chính sách cất hàng tuyến tính, học bằng RL (bài v6, 06/11/2026)

Thay cho "Model A" (mạng nơ-ron trên 64 ô ứng viên). Theo góp ý mentor ngày 05/11/2026:

| Góp ý | Thay đổi |
|---|---|
| Con số 64 ô ứng viên không hợp lý | Bỏ lọc. Chính sách chấm điểm **mọi** ô còn chỗ (N = 1.692) |
| Mô hình là hàm tuyến tính, cách huấn luyện là RL | Điểm s = Φθ, θ có 18 số, học bằng policy gradient (actor–critic) |
| Dữ liệu nhỏ mà huấn luyện lâu, sợ overfit | 18 tham số; chia 3 phần theo thời gian (huấn luyện / kiểm định / kiểm tra); báo cáo J trên cả 3 phần |
| Tên "Model A" khó hiểu | Đổi thành **LP-RL** (Linear Putaway policy trained by RL) |
| Viết rõ giả định, ma trận có chiều, vẽ kho | Bài có Assumption 1–7, ma trận X ∈ R^{N×d}, Φ ∈ R^{N×2d}, θ ∈ R^{2d}, s ∈ R^N; hình mặt bằng kho thật |

## Ma trận đầu vào / đầu ra cho một quyết định t (một đơn vị hàng về)

```
X_t ∈ R^{N×d}      N = 1.692 ô, d = 9 đặc trưng mỗi ô (bảng FEAT trong lp_rl.py)
Φ_t = [X_t, λX_t] ∈ R^{N×18}
θ   ∈ R^{18}
s_t = Φ_t θ ∈ R^N        điểm của các ô
π_t = softmax(s_t trên các ô còn chỗ)     -> huấn luyện: bốc ô theo π_t; kiểm tra: lấy ô điểm cao nhất
```

## Cách học θ (RL)

1. **Tách reward.** Phần trộn hàng được ghi nhận ngay ở quyết định gây ra nó: r_t = −λ/MI0 · ΔMI_t · (số wave còn lại) + (1−λ)(1 − T_w/T0) khi wave w được lấy. Không chiết khấu thì tổng reward y hệt reward theo wave (chỉ dời thời điểm). Trong các lần chạy thử ban đầu (chưa có 8 kho song song và khởi động lại), thiếu bước này thì trọng số đặc trưng trộn hàng đổi dấu giữa các lần chạy. Ở thiết lập cuối, ablation cho thấy reward theo wave cho J tương đương (tốt hơn ở λ thấp, kém hơn ở λ = 0,8).
2. **8 kho mô phỏng chạy song song.** Mỗi lô 4.096 quyết định = 512 quyết định × 8 kho, mỗi kho một đoạn wave và một λ riêng → mỗi lần cập nhật luôn thấy nhiều λ.
3. **Khởi động lại, chọn theo tập kiểm định.** Mỗi lần huấn luyện chạy 3 lần ngắn (100.000 quyết định) từ θ = 0. Nếu không lần nào tốt hơn θ = 0 trên tập kiểm định thì chạy thêm 3 lần (tối đa 3 vòng). Lần tốt nhất được học tiếp 300.000 quyết định. Phần lớn các lần ngắn trôi về vùng xấu (ưu tiên gom hàng, bỏ quên quãng đường) trước khi học được tín hiệu quãng đường, nên bước này là cần thiết.

## Chạy lại từ đầu

```bash
python run_baselines_lp.py      # baseline trên 2.913 wave kiểm tra, lưu chi phí từng wave (~5 phút)
python split_baselines.py       # J của Random/Nearest/GreedyJ trên từng phần dữ liệu (kiểm tra overfit)
./run_lp_all.sh                 # LP-RL 5 seed + 3 ablation × 2 seed (huấn luyện + đánh giá), ~2–3 giờ trên CPU 2 lõi
for s in 0 1 2 3 4; do python lp_rl.py eval --tag lp_s$s --split train --lams 0.5; python lp_rl.py eval --tag lp_s$s --split val --lams 0.5; done
python analyze_lp.py            # mọi bảng + hình của bài -> results/lp/
python make_tex_tables.py       # bảng LaTeX -> results/lp/tex_tables.json
python fig_method.py --tag lp_s0   # hình phương pháp (mặt bằng kho + ma trận)
python eval_gro.py results/lp/picks_lp_s*_0.5.npz   # kiểm tra lại bằng GRO trên 300 wave (results/gro_lp_s*_0.5.csv)
```

## File

- `lp_rl.py` – đặc trưng, chính sách tuyến tính, huấn luyện policy gradient, đánh giá.
- `run_baselines_lp.py`, `split_baselines.py` – baseline.
- `analyze_lp.py` – bảng (`results/lp/bang_*.md`), số liệu (`results/lp/summary.json`), hình đánh đổi và đường học.
- `fig_layout.py`, `fig_method.py` – vẽ kho và hình phương pháp.
- `models_lp/<tag>.json` – θ đã học, chuẩn hoá (mu, sd), đường học, thời gian huấn luyện.
