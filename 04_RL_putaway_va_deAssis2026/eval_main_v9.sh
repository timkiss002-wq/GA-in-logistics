#!/bin/bash
# v9: đánh giá 5 seed LP-RL gọn (lpS): 11 λ trên wave kiểm tra, 9 kịch bản độ bền, train/val/test ở λ=0,5, GRO
for s in 0 1 2 3 4; do
  for l in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do python3 exp_v7.py run "base LP:lpS_s$s $l 0"; done
done
for sc in pick_random pick_top2 inb_lot inb_delay s0_seed1 s0_seed2 s0_fill70 s0_fill90 s0_class; do
  for s in 0 1 2 3 4; do python3 exp_v7.py run "$sc LP:lpS_s$s 0.5 0"; done
done
for s in 0 1 2 3 4; do for sp in test val train; do python3 lp_rl.py eval --tag lpS_s$s --lams 0.5 --split $sp; done; done
python3 eval_gro.py results/lp/picks_lpS_s{0,1,2,3,4}_0.5.npz
echo DONE
