"""
DỰNG LẠI ĐỦ CÁC BẢNG / HÌNH KẾT QUẢ CỦA Gnone et al. (2025) CHO 3 PHƯƠNG ÁN CHỌN Ô A, B, C.

  Bảng 2  – tham số GRO
  Bảng 3  – thống kê mô tả quãng đường (Mean, Std, Min, Q1, Median, Q3, Max) + % so với Original
  Shapiro – Wilk (W, p) từng cột
  ANOVA   – một chiều 5 nhóm (như bài) + hậu kiểm Tukey HSD (từng cặp) + Wilcoxon ghép cặp/Holm (kiểm tra thêm)
  Hình 3a – boxplot (thêm 1 khung vẽ lại hộp của bài từ số liệu Bảng 3)
  Hình 3b – scatter Original vs GRO theo wave
  Thời gian chạy GRO mỗi wave

Đầu vào (results/):
  gnone2025_chi_tiet.csv     – A ('tu_chon') và B ('gan_depot'), 10 seed
  gnone2025_C_chi_tiet.csv   – C ('chi_dinh'), 10 seed
  original_co_dinh.csv       – Original cố định, dùng chung cho cả A, B, C
Mỗi wave = trung bình 10 seed (bài báo không nói số lần chạy; chọn cách này để giảm nhiễu của GA).

Chạy:  python bang_nhu_bai_bao.py   -> results/bang_nhu_bai_bao.xlsx, results/bang_nhu_bai_bao.txt, results/hinh3a_boxplot.png, results/hinh3b_scatter.png
"""
import os
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CS = ['Random', 'Dedicated', 'Class-Based', 'Hybrid']
COT = ['Original'] + CS
PA = {'A': ('tu_chon', 'A – GA tự chọn ô'), 'B': ('gan_depot', 'B – ô gần depot nhất'), 'C': ('chi_dinh', 'C – ô chỉ định sẵn')}
CHI_SO = ['Mean', 'Std', 'Min', 'Q1 (25%)', 'Median', 'Q3 (75%)', 'Max']

# ---------------- Số liệu của bài báo (Bảng 3, đơn vị m) ----------------
BAI = pd.DataFrame({
    'Original':    [1056.90, 366.91, 246.00, 837.75, 1103.00, 1295.00, 1750.00],
    'Random':      [887.32, 294.41, 246.43, 715.30, 923.75, 1121.75, 1380.00],
    'Dedicated':   [726.77, 354.94, 33.80, 512.01, 749.11, 913.25, 1529.00],
    'Class-Based': [800.55, 327.06, 109.34, 615.02, 839.84, 986.55, 1424.00],
    'Hybrid':      [744.96, 310.15, 105.90, 536.73, 781.48, 996.00, 1428.00]}, index=CHI_SO)
BAI_SHAPIRO = {'Original': (0.9723, 0.2864), 'Random': (0.9651, 0.1458), 'Dedicated': (0.9903, 0.9537),
               'Class-Based': (0.9755, 0.3797), 'Hybrid': (0.9838, 0.7210)}
BAI_ANOVA_F = 8.248


def doc():
    R = pd.concat([pd.read_csv(os.path.join(KQ, 'gnone2025_chi_tiet.csv')),
                   pd.read_csv(os.path.join(KQ, 'gnone2025_C_chi_tiet.csv'))], ignore_index=True)
    org = pd.read_csv(os.path.join(KQ, 'original_co_dinh.csv')).set_index('wave').original
    M = {}
    for p, (gd, _) in PA.items():
        X = R[R.gia_dinh == gd]
        m = X.groupby(['wave', 'chinh_sach']).gro.mean().unstack()
        m['Original'] = org
        M[p] = m[COT]
    return R, org, M


def mo_ta(m):
    d = m.describe().T
    t = pd.DataFrame({'Mean': d['mean'], 'Std': d['std'], 'Min': d['min'], 'Q1 (25%)': d['25%'],
                      'Median': d['50%'], 'Q3 (75%)': d['75%'], 'Max': d['max']}).T
    return t[COT]


def thu_hang(s):
    return ' < '.join(s.drop('Original').sort_values().index)


def main():
    R, org, M = doc()
    out, sheets = [], {}
    p = lambda s='': out.append(s)

    # ================= BẢNG 2 =================
    b2 = pd.DataFrame({
        'Tham số': ['Kích thước quần thể (Np)', 'Số thế hệ (G)', 'Tournament (T)', 'Xác suất lai (Pc)', 'Xác suất đột biến (Pm)',
                    'Số wave', 'Số SKU mỗi wave', 'Số lần chạy (seed) mỗi wave', 'Tầng lấy hàng', 'Depot', 'Đơn vị quãng đường'],
        'Bài báo': ['1000', '1000', '3', '90 %', '8 %', '50', '25 (tối đa)', 'không nêu', 'tầng 1–2 (mục 3)', 'không nêu toạ độ', 'm'],
        'Nhóm': ['1000', '1000', '3', '90 %', '8 %', '50', '25', '10', 'tầng 1–2', 'CC-08 (403, 631)', 'đơn vị toạ độ dataset']})
    sheets['Bang2_tham_so'] = b2
    p('BẢNG 2 – THAM SỐ GRO'); p(b2.to_string(index=False))

    # ================= BẢNG 3 =================
    p('\n\nBẢNG 3 – THỐNG KÊ MÔ TẢ QUÃNG ĐƯỜNG (50 wave; mỗi wave = TB 10 seed; Original cố định chung)')
    bai = BAI.copy(); bai.loc['% so với Original (Mean)'] = (bai.loc['Mean'] / bai.loc['Mean', 'Original'] - 1) * 100
    bai.loc['% so với Original (Median)'] = (bai.loc['Median'] / bai.loc['Median', 'Original'] - 1) * 100
    bai.loc['Thứ hạng theo Mean'] = ''
    bai.loc['Thứ hạng theo Mean', 'Original'] = thu_hang(BAI.loc['Mean'])
    gop = [bai.assign(**{'Nguồn': 'Bài báo (m)'})]
    p('\nBài báo (m):'); p(BAI.round(2).to_string())
    p('   % so với Original theo Mean  : ' + ', '.join(f'{c} {bai.loc["% so với Original (Mean)", c]:+.1f}%' for c in CS))
    p('   % so với Original theo Median: ' + ', '.join(f'{c} {bai.loc["% so với Original (Median)", c]:+.1f}%' for c in CS))
    p('   Thứ hạng (Mean)  : ' + thu_hang(BAI.loc['Mean']))
    p('   Thứ hạng (Median): ' + thu_hang(BAI.loc['Median']))
    for k, (gd, ten) in PA.items():
        t = mo_ta(M[k])
        t2 = t.copy().astype(object)
        t2.loc['% so với Original (Mean)'] = (t.loc['Mean'] / t.loc['Mean', 'Original'] - 1) * 100
        t2.loc['% so với Original (Median)'] = (t.loc['Median'] / t.loc['Median', 'Original'] - 1) * 100
        t2.loc['Thứ hạng theo Mean'] = ''
        t2.loc['Thứ hạng theo Mean', 'Original'] = thu_hang(t.loc['Mean'])
        gop.append(t2.assign(**{'Nguồn': f'Nhóm – {ten} (đv toạ độ)'}))
        p(f'\nNhóm – {ten} (đơn vị toạ độ):'); p(t.round(1).to_string())
        p('   % so với Original theo Mean  : ' + ', '.join(f'{c} {t2.loc["% so với Original (Mean)", c]:+.1f}%' for c in CS))
        p('   % so với Original theo Median: ' + ', '.join(f'{c} {t2.loc["% so với Original (Median)", c]:+.1f}%' for c in CS))
        p('   Thứ hạng (Mean)  : ' + thu_hang(t.loc['Mean']))
        p('   Thứ hạng (Median): ' + thu_hang(t.loc['Median']))
    b3 = pd.concat(gop); b3.index.name = 'Chỉ số'
    sheets['Bang3_mo_ta'] = b3.reset_index()[['Nguồn', 'Chỉ số'] + COT]

    # Bảng 3 gọn: chỉ Mean và Median, tất cả nguồn cạnh nhau
    gon = []
    for nguon, t in [('Bài báo (m)', BAI)] + [(PA[k][1], mo_ta(M[k])) for k in PA]:
        for cs in ['Mean', 'Median']:
            r = {'Nguồn': nguon, 'Chỉ số': cs}
            for c in COT:
                r[c] = round(t.loc[cs, c], 1)
                if c != 'Original':
                    r[f'{c} %'] = round((t.loc[cs, c] / t.loc[cs, 'Original'] - 1) * 100, 1)
            r['Thứ hạng'] = thu_hang(t.loc[cs])
            gon.append(r)
    sheets['Bang3_gon_Mean_Median'] = pd.DataFrame(gon)

    # ================= SHAPIRO–WILK =================
    p('\n\nKIỂM ĐỊNH SHAPIRO–WILK (H0: phân phối chuẩn; p > 0,05 -> không bác bỏ)')
    sw = []
    for c in COT:
        r = {'Cột': c, 'Bài báo W': BAI_SHAPIRO[c][0], 'Bài báo p': BAI_SHAPIRO[c][1]}
        for k in PA:
            s = stats.shapiro(M[k][c]); r[f'{k} W'] = round(s.statistic, 4); r[f'{k} p'] = round(s.pvalue, 4)
        sw.append(r)
    sw = pd.DataFrame(sw); sheets['Shapiro_Wilk'] = sw; p(sw.to_string(index=False))

    # ================= ANOVA + HẬU KIỂM =================
    p('\n\nANOVA MỘT CHIỀU (5 nhóm: Original + 4 chính sách, như bài báo)')
    av, tk, wx = [], [], []
    for k, (gd, ten) in PA.items():
        m = M[k]
        f = stats.f_oneway(*[m[c] for c in COT])
        fr = stats.friedmanchisquare(*[m[c] for c in COT])
        av.append({'Phương án': ten, 'F': round(f.statistic, 3), 'p': f'{f.pvalue:.2e}',
                   'Friedman χ² (ghép theo wave)': round(fr.statistic, 2), 'Friedman p': f'{fr.pvalue:.2e}'})
        th = stats.tukey_hsd(*[m[c] for c in COT])
        cap = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        pw = [stats.wilcoxon(m[COT[i]], m[COT[j]]).pvalue for i, j in cap]
        o = np.argsort(pw); adj = np.empty(len(pw)); run = 0
        for r_, i in enumerate(o):
            run = max(run, min(1, (len(pw) - r_) * pw[i])); adj[i] = run
        for (i, j), pa in zip(cap, adj):
            ci = th.confidence_interval()
            tk.append({'Phương án': ten, 'Cặp': f'{COT[i]} – {COT[j]}',
                       'Chênh TB (cột 1 − cột 2)': round(m[COT[i]].mean() - m[COT[j]].mean(), 1),
                       'Tukey p': round(th.pvalue[i, j], 4),
                       'Tukey 95% CI': f'[{ci.low[i, j]:.0f}; {ci.high[i, j]:.0f}]',
                       'Wilcoxon p (Holm)': f'{pa:.2e}',
                       'Có ý nghĩa (Tukey, α=0,05)': 'có' if th.pvalue[i, j] < 0.05 else 'không'})
    av.insert(0, {'Phương án': 'Bài báo', 'F': BAI_ANOVA_F, 'p': '≈ 0', 'Friedman χ² (ghép theo wave)': '–', 'Friedman p': '–'})
    av = pd.DataFrame(av); tk = pd.DataFrame(tk)
    sheets['ANOVA'] = av; sheets['Hau_kiem_tung_cap'] = tk
    p(av.to_string(index=False))
    p('\nBài báo chỉ ghi: "Dedicated và Hybrid thấp hơn có ý nghĩa so với Random và Class-Based" (không nêu tên phép hậu kiểm).')
    p('\nHẬU KIỂM TỪNG CẶP: Tukey HSD (đi kèm ANOVA) và Wilcoxon ghép cặp theo wave, hiệu chỉnh Holm')
    p(tk.to_string(index=False))

    # ================= THỜI GIAN =================
    p('\n\nTHỜI GIAN CHẠY GRO MỖI WAVE (giây, 1 lần chạy = 1 wave × 1 seed, 1 nhân CPU, Python + numba)')
    tg = R.groupby(['gia_dinh', 'chinh_sach']).giay.mean().unstack()[CS]
    tg.index = [PA[k][1] for k in PA for gd in [PA[k][0]] if gd in tg.index]
    tg['TB 4 chính sách'] = tg.mean(axis=1)
    so_o = R[R.gia_dinh != 'chi_dinh'].groupby(['gia_dinh', 'chinh_sach']).so_o_tb.mean().unstack()[CS]
    tg = tg.round(2)
    tg.loc['Bài báo'] = ['–', 'thấp nhất', '–', '–', 128]
    tg.index.name = 'Phương án'
    sheets['Thoi_gian'] = tg.reset_index()
    p(tg.to_string())
    p('Số ô ứng viên trung bình mỗi SKU (A): ' + ', '.join(f'{c} {so_o.loc["tu_chon", c]:.1f}' for c in CS) + '; B và C: 1 ô.')

    # ================= Ghi file =================
    with pd.ExcelWriter(os.path.join(KQ, 'bang_nhu_bai_bao.xlsx')) as xw:
        for n, d in sheets.items():
            d.to_excel(xw, sheet_name=n, index=False)
        for k in PA:
            M[k].round(1).to_excel(xw, sheet_name=f'Theo_wave_{k}')
    open(os.path.join(KQ, 'bang_nhu_bai_bao.txt'), 'w', encoding='utf-8').write('\n'.join(out))

    # ================= HÌNH 3a – BOXPLOT =================
    mau = {'Original': '#9a9a9a', 'Random': '#4c78a8', 'Dedicated': '#e45756', 'Class-Based': '#f2a93b', 'Hybrid': '#54a24b'}
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.8))
    hop = [dict(label=c, whislo=BAI.loc['Min', c], q1=BAI.loc['Q1 (25%)', c], med=BAI.loc['Median', c],
                q3=BAI.loc['Q3 (75%)', c], whishi=BAI.loc['Max', c], mean=BAI.loc['Mean', c], fliers=[]) for c in COT]
    b = ax[0].bxp(hop, showmeans=True, patch_artist=True)
    ax[0].set_title('Bài báo (vẽ lại từ Bảng 3, m)\nrâu = Min–Max', fontsize=10); ax[0].set_ylabel('m')
    for a, k in zip(ax[1:], PA):
        b2_ = a.boxplot([M[k][c] for c in COT], tick_labels=COT, showmeans=True, patch_artist=True)
        a.set_title(f'Nhóm – {PA[k][1]}\n(đơn vị toạ độ; râu = 1,5 IQR)', fontsize=10)
        for patch, c in zip(b2_['boxes'], COT): patch.set_facecolor(mau[c]); patch.set_alpha(.75)
    for patch, c in zip(b['boxes'], COT): patch.set_facecolor(mau[c]); patch.set_alpha(.75)
    for a in ax:
        a.tick_params(axis='x', rotation=35, labelsize=8); a.grid(axis='y', alpha=.3)
    fig.suptitle('Hình 3a – Quãng đường lấy hàng theo chính sách (50 wave)', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(KQ, 'hinh3a_boxplot.png'), dpi=160); plt.close(fig)

    # ================= HÌNH 3b – SCATTER =================
    fig, ax = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    lim = [min(M[k][CS].min().min() for k in PA) * 0.9, max(org.max(), max(M[k][CS].max().max() for k in PA)) * 1.03]
    for a, k in zip(ax, PA):
        for c in CS:
            a.scatter(org, M[k][c], s=16, alpha=.75, color=mau[c], label=c)
        a.plot(lim, lim, '--', color='k', lw=1, label='y = x (bằng Original)')
        a.set_xlim(lim); a.set_ylim(lim); a.grid(alpha=.3)
        a.set_title(f'{PA[k][1]}', fontsize=10); a.set_xlabel('Original của wave (đv toạ độ)')
    ax[0].set_ylabel('Quãng đường sau GRO (đv toạ độ)'); ax[0].legend(fontsize=8)
    fig.suptitle('Hình 3b – Original vs GRO theo từng wave (điểm dưới đường chéo = GRO ngắn hơn)', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(KQ, 'hinh3b_scatter.png'), dpi=160); plt.close(fig)
    print('\n'.join(out))


if __name__ == '__main__':
    main()
