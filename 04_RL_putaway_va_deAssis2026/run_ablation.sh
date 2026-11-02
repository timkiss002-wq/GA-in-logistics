#!/bin/bash
# Ablation: mỗi biến thể huấn luyện 300.000 bước rồi đánh giá trên toàn bộ wave kiểm tra ở λ = 0,2 / 0,5 / 0,8
set -x
python3 rl_ppo.py train --tag ab_BConly  --steps 0 --bc 20000
python3 rl_ppo.py train --tag ab_gamma0   --steps 200000 --bc 20000 --gamma 0
python3 rl_ppo.py train --tag ab_noBC     --steps 200000 --bc 0
python3 rl_ppo.py train --tag ab_noDist   --steps 200000 --bc 20000 --drop 5,6,8,10
python3 rl_ppo.py train --tag ab_noMIfeat --steps 200000 --bc 20000 --drop 2,3,4
python3 rl_ppo.py train --tag ab_noPrior  --steps 200000 --bc 0 --drop 12
python3 rl_ppo.py train --tag ab_lam05    --steps 200000 --bc 20000 --lam 0.5
python3 rl_ppo.py train --tag ab_noTref   --steps 200000 --bc 20000 --no_tref
for t in ab_BConly ab_gamma0 ab_noBC ab_noDist ab_noMIfeat ab_noPrior ab_noTref; do python3 rl_ppo.py eval --tag $t --lams 0.2,0.5,0.8; done
python3 rl_ppo.py eval --tag ab_lam05 --lams 0.5
