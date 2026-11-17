#!/bin/bash
# v9: đánh giá ablation (cấu hình gọn) ở 11 λ; đợi model có rồi mới chạy
for t in abS_gamma0_s0 abS_nomix_s0 abS_nomix_s1 abS_gamma0_s1 abS_nolam_s0 abS_nolam_s1; do
  while [ ! -f models_lp/$t.json ]; do sleep 20; done
  for l in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do python3 exp_v7.py run "base LP:$t $l 0"; done
done
for lf in 0.0 1.0; do for s in 0 1; do t=lpSF${lf}_s$s
  while [ ! -f models_lp/$t.json ]; do sleep 20; done
  python3 exp_v7.py run "base LP:$t $lf 0"
done; done
echo DONE
