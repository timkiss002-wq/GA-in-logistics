# v9 – LP-RL bản gọn (11/11/2026)

LP-RL **không chia thưởng (reward split), không T_ref**: thưởng chỉ trao sau mỗi wave. Tag model: `lpS`.

## Huấn luyện
```bash
# 5 seed bản chính
for s in 0 1 2 3 4; do python3 train_v7.py --tag lpS_s$s --seed $s --notref --reward wave; done
# ablation (cấu hình gọn)
python3 train_v7.py --tag abS_gamma0_s0 --seed 0 --notref --reward wave --gamma 0      # cận thị, 5 seed
python3 train_v7.py --tag abS_nomix_s0  --seed 0 --notref --reward wave --drop 1,2,3,4  # bỏ đặc trưng trộn, 2 seed
python3 train_v7.py --tag abS_nolam_s0  --seed 0 --notref --reward wave --nolam         # bỏ khối λX, 2 seed
python3 train_v7.py --tag lpSF1.0_s0    --seed 0 --notref --reward wave --lam 1.0       # θ riêng cho 1 λ
python3 train_v7.py --tag ab_notref_s2  --seed 2 --notref                               # giữ chia thưởng, bỏ T_ref
```
Hàng đợi đã chạy: `queue_v9*.txt`; log: `logs_v9/`.

## Đánh giá và bảng
```bash
./eval_main_v9.sh          # 11 λ + 9 kịch bản độ bền + train/val/test + GRO cho lpS
./eval_notref_v9.sh        # ab_notref ở 11 λ
./eval_ab_v9.sh            # ablation ở 11 λ
LPTAG=lpS python3 compare_all.py        # -> results/v7/compare_all_lpS.csv  (Bảng V)
LPTAG=lpS python3 analyze_v7.py         # -> results/v7/summary_v7_lpS.json, robust_lpS.csv, time_*_lpS.csv
LPTAG=lpS python3 fig_tradeoff_v7.py; LPTAG=lpS python3 fig_time_v7.py; python3 fig_curve_v9.py
python3 ablation_v9.py                  # -> results/v7/ablation_v9.csv (Bảng VI)
python3 fig_method_v8.py --tag lpS_s0 --out fig_method.pdf
```
Không đặt `LPTAG` thì các script chạy như bản v7 (tag `lp`). Ghi chú kết quả: tài liệu project `results/LP_RL_v9_huan_luyen_lai_2026-11-11.md`.
