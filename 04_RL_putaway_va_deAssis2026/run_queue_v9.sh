#!/bin/bash
# chạy hàng đợi huấn luyện v9, 2 việc song song (máy 2 nhân); đợi 2 việc đầu (lpS_s0, lpS_s1) xong trước
while pgrep -f "tag lpS_s[01] " >/dev/null; do sleep 20; done
cat queue_v9.txt | xargs -P 2 -I{} bash -c 'set -- {}; python3 train_v7.py {} > logs_v9/$2.log 2>&1'
