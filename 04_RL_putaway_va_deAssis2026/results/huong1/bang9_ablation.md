## Bảng 9. Ablation theo hướng 1, mỗi biến thể 3 seed huấn luyện (Δ = J biến thể − J mô hình A cùng seed; dương = biến thể kém hơn)

| Câu hỏi | Biến thể | λ đánh giá | Δ seed 0 / 1 / 2 | Δ TB ± ĐLC | số seed kém hơn / tốt hơn có ý nghĩa (Holm) |
|---|---|---|---|---|---|
| PPO có cần không? | Bỏ PPO (chỉ bắt chước) | 0,2, 0,5, 0,8 | +0.0110 / +0.0013 / +0.0034 | +0.0052 ± 0.0051 | 2 / 0 (trên 3 seed) |
| Nhìn xa có giúp không? | PPO với γ = 0 (không nhìn xa) | 0,2, 0,5, 0,8 | +0.0116 / +0.0003 / -0.0118 | +0.0000 ± 0.0117 | 2 / 1 (trên 3 seed) |
| Danh mục heuristic có cần không? | Bắt chước chỉ GreedyJ (bỏ danh mục) | 0,2, 0,5, 0,8 | +0.0038 / -0.0005 / +0.0005 | +0.0013 ± 0.0023 | 1 / 1 (trên 3 seed) |
| Đặc trưng mixing có cần không? | Bỏ đặc trưng mixing (2, 3, 4) | 0,2, 0,5, 0,8 | +0.0475 / +0.0448 / +0.0454 | +0.0459 ± 0.0014 | 3 / 0 (trên 3 seed) |
| Một mạng cho mọi λ có tốt hơn mạng chuyên một λ? | Mạng riêng cho λ = 0,5 (bỏ điều kiện theo λ) | 0,5 | +0.0089 / +0.0059 / +0.0026 | +0.0058 ± 0.0031 | 2 / 0 (trên 3 seed) |

## Bảng 9b. Δ tách theo từng λ (TB ± ĐLC trên các seed; dương = biến thể kém hơn)

| Biến thể | λ = 0,2 | λ = 0,5 | λ = 0,8 |
|---|---|---|---|
| Bỏ PPO (chỉ bắt chước) | +0.0134 ± 0.0102 (n=3) | +0.0069 ± 0.0047 (n=3) | -0.0047 ± 0.0048 (n=3) |
| PPO với γ = 0 (không nhìn xa) | +0.0070 ± 0.0142 (n=3) | +0.0003 ± 0.0124 (n=3) | -0.0073 ± 0.0148 (n=3) |
| Bắt chước chỉ GreedyJ (bỏ danh mục) | +0.0296 ± 0.0092 (n=3) | +0.0014 ± 0.0041 (n=3) | -0.0272 ± 0.0112 (n=3) |
| Bỏ đặc trưng mixing (2, 3, 4) | +0.0085 ± 0.0124 (n=3) | +0.0494 ± 0.0027 (n=3) | +0.0798 ± 0.0131 (n=3) |
| Mạng riêng cho λ = 0,5 (bỏ điều kiện theo λ) | – | +0.0058 ± 0.0031 (n=3) | – |