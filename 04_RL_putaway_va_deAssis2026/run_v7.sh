#!/bin/bash
# Bài v7: mọi thí nghiệm mới. 2 việc song song. ~1,5–2 giờ trên CPU 2 lõi.
cd "$(dirname "$0")"; mkdir -p logs_v7
python make_s0_v7.py > logs_v7/make_s0.log 2>&1
{
  for s in 0 1 2; do echo "T --tag lpR_s$s --seed $s --pick random"; done
  for lf in 0.0 0.5 1.0; do for s in 0 1; do echo "T --tag lpF${lf}_s$s --seed $s --lam $lf"; done; done
  for s in 0 1; do echo "T --tag ab_notref_s$s --seed $s --notref"; done
  python exp_v7.py jobs base | sed 's/^/E /'
  python exp_v7.py jobs robust | sed 's/^/E /'
} > logs_v7/jobs1.txt
task() { kind=$1; shift
  if [ "$kind" = T ]; then tag=$2; [ -f models_lp/$tag.json ] || python train_v7.py "$@" > logs_v7/$tag.log 2>&1; echo "trained $tag"
  else python exp_v7.py run "$*" >> logs_v7/eval.log 2>&1; fi; }
export -f task
xargs -P 2 -L 1 bash -c 'task "$@"' _ < logs_v7/jobs1.txt
python exp_v7.py jobs retrained | sed 's/^/E /' > logs_v7/jobs2.txt
xargs -P 2 -L 1 bash -c 'task "$@"' _ < logs_v7/jobs2.txt
echo ALL_DONE
