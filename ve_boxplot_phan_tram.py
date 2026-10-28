"""
HÌNH 3a CHUẨN HOÁ: boxplot theo % của Original trung bình.

Mỗi giá trị quãng đường được chia cho Mean của Original (cùng nguồn) rồi nhân 100:
  bài báo : chia cho 1056,90 m
  nhóm    : chia cho 6615,7 đơn vị toạ độ (Original cố định, chung cho A, B, C)
=> Original trung bình = 100 ở mọi khung, đơn vị (m hay đơn vị toạ độ) triệt tiêu, so được hình dạng hộp.
Cả 2 bên vẽ râu = Min–Max (bài báo chỉ cho Min/Max nên phải dùng cùng quy ước).

Chạy:  python ve_boxplot_phan_tram.py   -> results/hinh3a_boxplot_phan_tram.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bang_nhu_bai_bao import doc, mo_ta, BAI, COT, PA, KQ

R, org, M = doc()
mau = {'Original': '#9a9a9a', 'Random': '#4c78a8', 'Dedicated': '#e45756', 'Class-Based': '#f2a93b', 'Hybrid': '#54a24b'}


def hop(t, goc):
    k = 100 / goc
    return [dict(label=c, whislo=t.loc['Min', c] * k, q1=t.loc['Q1 (25%)', c] * k, med=t.loc['Median', c] * k,
                 q3=t.loc['Q3 (75%)', c] * k, whishi=t.loc['Max', c] * k, mean=t.loc['Mean', c] * k, fliers=[]) for c in COT]


khung = [('Bài báo', BAI, BAI.loc['Mean', 'Original'])] + [(f'Nhóm – {PA[k][1]}', mo_ta(M[k]), mo_ta(M[k]).loc['Mean', 'Original']) for k in PA]
fig, ax = plt.subplots(1, 4, figsize=(17, 5), sharey=True)
for a, (ten, t, goc) in zip(ax, khung):
    b = a.bxp(hop(t, goc), showmeans=True, patch_artist=True)
    for patch, c in zip(b['boxes'], COT):
        patch.set_facecolor(mau[c]); patch.set_alpha(.75)
    a.axhline(100, ls='--', color='k', lw=.8)
    a.set_title(ten, fontsize=10); a.grid(axis='y', alpha=.3)
    a.tick_params(axis='x', rotation=35, labelsize=8)
ax[0].set_ylabel('% của Original trung bình (Original TB = 100)')
fig.suptitle('Hình 3a chuẩn hoá – quãng đường theo % Original trung bình (râu = Min–Max, ▲ = Mean, gạch cam = Median)', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(KQ, 'hinh3a_boxplot_phan_tram.png'), dpi=160)

# in số để đọc
for ten, t, goc in khung:
    print(f'\n{ten}:')
    print((t.loc[['Min', 'Q1 (25%)', 'Median', 'Q3 (75%)', 'Max']] / goc * 100).round(0).to_string())
