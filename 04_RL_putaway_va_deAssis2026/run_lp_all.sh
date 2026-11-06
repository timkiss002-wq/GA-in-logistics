#!/bin/bash
# Huấn luyện + đánh giá LP-RL: 5 seed cấu hình chính, 3 seed cho mỗi ablation. Chạy 2 việc song song.
cd "$(dirname "$0")"
jobs=()
for s in 0 1 2 3 4; do jobs+=("lp_s$s --seed $s"); done
for s in 0 1; do
  jobs+=("ab_wave_s$s --seed $s --reward wave")   # reward chỉ theo wave (không tách phần MI)
  jobs+=("ab_gamma0_s$s --seed $s --gamma 0")     # không nhìn xa
  jobs+=("ab_nomix_s$s --seed $s --drop mix")     # bỏ đặc trưng trộn hàng
done
run() { set -- $1; tag=$1; shift; [ -f models_lp/$tag.json ] || python lp_rl.py train --tag $tag "$@" > logs_lp/$tag.log 2>&1; \
        [ -f results/lp/eval_${tag}_test.csv ] || python lp_rl.py eval --tag $tag >> logs_lp/$tag.log 2>&1; echo "done $tag"; }
export -f run; mkdir -p logs_lp
printf '%s\n' "${jobs[@]}" | xargs -P 2 -I{} bash -c 'run "{}"'
