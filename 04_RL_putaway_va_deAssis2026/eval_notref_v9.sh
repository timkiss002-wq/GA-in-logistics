#!/bin/bash
# v9: đánh giá cấu hình "giữ chia thưởng, bỏ T_ref" (tag ab_notref) đủ 5 seed
for s in 0 1 2 3 4; do
  while [ ! -f models_lp/ab_notref_s$s.json ]; do sleep 30; done
  for l in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do python3 exp_v7.py run "base LP:ab_notref_s$s $l 0"; done
done
echo DONE
