#!/bin/bash
cd "$(dirname "$0")"
for s in 0 1; do (python train_v7.py --tag lpC_s$s --seed $s --s0 class > logs_v7/lpC_s$s.log 2>&1; python exp_v7.py run "s0_class LP:lpC_s$s 0.5 0" >> logs_v7/eval.log 2>&1) & done; wait; echo DONE_B
