# hoc_rl/ – ví dụ tí hon để hiểu RL và kiểm tra MI

| File | Làm gì | Chạy |
|---|---|---|
| `kiem_tra_MI.py` | Tính lại Mixing Index (công thức 2, de Assis 2026) trên `Random_Storage.csv` thật, đối chiếu MI = 0,918 của bài | `python kiem_tra_MI.py` |
| `rl_ti_hon.py` | Q-learning trên kho 4 ô, 2 mã hàng, reward r = −ΔJ. So kết quả agent với đáp án đúng (liệt kê hết 3.480 cách cất) | `python rl_ti_hon.py` |
| | Xem agent học từng bước (Q trước/sau, khám phá hay khai thác, ΔMI, ΔD/D₀, reward) | `python rl_ti_hon.py --xem 0.5` |
| `ket_qua_*.txt`, `xem_tung_buoc_*.txt` | Kết quả đã chạy sẵn | — |

## Bài tập tự làm để hiểu RL (sửa trong `rl_ti_hon.py`)
1. Chạy `--xem 0` rồi `--xem 1`: agent cất khác nhau thế nào? Vì sao?
2. Đổi `TY_LE_LAY` thành A 0,5 / B 0,5: mức λ nào agent bắt đầu tách A và B?
3. Đặt `so_episode=50`: agent còn tìm ra đáp án đúng không? Vì sao cần nhiều episode?
4. Đặt `eps_dau=0` (không khám phá): chuyện gì xảy ra?
5. Kiểm tra trong output: "tổng reward" luôn bằng "−J cuối". Đây là lý do tối đa reward ⇔ min J.

## Từ kho tí hon lên kho thật
| Kho tí hon | Kho thật |
|---|---|
| 4 ô, sức chứa 3 | ~1.093 ô vùng random, 18 ngăn |
| 2 mã, 6 lô về | 208 mã × cỡ, hàng nghìn lô về |
| Bảng Q (liệt kê từng trạng thái) | Mạng nơ-ron (PPO) vì số trạng thái quá lớn |
| D tính trực tiếp | D đo bằng GRO trên wave thật; trong lúc học dùng proxy nhanh |
