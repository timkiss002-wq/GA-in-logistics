## Bảng R1. Tái hiện Bảng 2 của de Assis et al. (2026) sau 10.000 lô (nhóm: TB ± ĐLC 5 seed, 1.093 ô vùng ngẫu nhiên)

| Agent | MI (bài) | MI (nhóm) | % critical MI_l > 0,8 (bài) | % critical (nhóm) | số hành động thực hiện (bài, đọc từ Hình 4) | thực hiện (nhóm) | chờ duyệt (nhóm) |
|---|---|---|---|---|---|---|---|
| Baseline (Random, kho đầy) | 0,918 | 0,9147 | 100,0 | 100,0 | – | – | – |
| Sau khi giải phóng 20% | – | 0,9036 | – | 99,6 | – | – | – |
| heuristic | 0.830 | 0.7601 ± 0.0014 | 99.66 | 33.76 ± 0.59 | ≈ 2.500 | 3920 | 0 |
| supervised | 0.884 | 0.7235 ± 0.0003 | 99.71 | 19.34 ± 1.34 | ≈ 2.000 | 3919 | 5520 |
| adaptive | 0.822 | 0.7477 ± 0.0009 | 99.48 | 29.99 ± 1.01 | ≈ 2.500 | 3920 | 0 |

Thứ tự MI trong bài: adaptive < heuristic < supervised. Của nhóm: supervised < adaptive < heuristic.

## Bảng R2. Độ nhạy theo các tham số bài không công bố (TB 3 seed)

| Biến thể | heuristic MI | supervised MI | adaptive MI | heuristic % critical |
|---|---|---|---|---|
| chuẩn | 0.7598 | 0.7236 | 0.7473 | 33.9433 |
| θ = 0.95 | 0.7598 | 0.7236 | 0.7473 | 33.9433 |
| θ = 0.9 | 0.7604 | 0.7236 | 0.7473 | 34.2482 |
| ε = 0.0 | 0.7587 | 0.7236 | 0.7473 | 33.5468 |
| ε = 0.05 | 0.7615 | 0.7236 | 0.7473 | 34.6142 |
| top-k = 1 | 0.7582 | 0.7236 | 0.7448 | 34.2177 |
| top-k = 10 | 0.7633 | 0.7237 | 0.7497 | 35.8036 |
| ω = 0 (không khám phá) | 0.7582 | 0.7236 | 0.7448 | 34.2177 |
| w0 = [1,0,0,0,0,0] | – | – | 0.7540 | – |
| w0 = [1,.1,.1,.1,.1,-.1] | – | – | 0.7494 | – |
| r_t = ΔMI toàn kho | – | – | 0.7473 | – |
| chỉ tầng 1 (249 ô) | 0.8150 | 0.7242 | 0.8006 | 59.1700 |
| SKU = mã × cỡ | 0.8709 | 0.7763 | 0.8523 | 87.6487 |
| cỡ bắt buộc tương thích | 0.7677 | 0.7237 | 0.7601 | 36.3830 |
| lô chờ duyệt: người cất ngẫu nhiên | – | 0.7609 | – | – |
| 3.000 lô thay vì 10.000 | 0.7689 | 0.7665 | 0.7664 | 34.5227 |