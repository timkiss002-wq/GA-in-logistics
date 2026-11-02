"""
PHƯƠNG ÁN C VỚI WAVE CÓ SỐ MÓN THAY ĐỔI ("up to 25 items" – Gnone et al., 2025, mục 5).

Khác thí nghiệm trước ở ĐÚNG MỘT điểm: số SKU mỗi wave không cố định 25 nữa.
  Bước 1: lấy phân phối số SKU khác nhau trong 9.707 wave thật (Picking_Wave.csv), chỉ giữ wave ≤ 25 SKU.
  Bước 2: với mỗi wave sinh mới, bốc số món n từ phân phối đó (seed 2025).
  Bước 3: bốc n SKU không lặp, xác suất tỉ lệ nhu cầu lịch sử (giống hệt cách cũ).
Còn lại giữ nguyên: tầng 1–2, depot CC-08, ô chỉ định sẵn (rng 1000 + wave), GRO hợp-lý Np=G=1000, T=3, Pc=90%, Pm=8%,
10 seed mỗi wave; Original = Random_Storage + ô chỉ định + đi theo lối S-shape.

Chạy:  python chay_C_wave_bien_doi.py   -> results/C_bien_doi_chi_tiet.csv, results/C_bien_doi_waves.csv,
                                            results/C_bien_doi_ket_qua.txt, results/hinh_C_bien_doi.png
"""
import os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
import tai_hien_gnone2025 as T
from data import load_waves, P, size_key
from gro import build_instance
from chay_C_10seed import o_chi_dinh

SEED = 2025
CS = list(T.CHINH_SACH)
COT = ['Original'] + CS
KQ = T.KQ


def phan_phoi_so_mon():
    w = load_waves()
    n = w.assign(sku=list(zip(w.ref, w['size']))).groupby('wave').sku.nunique()
    return n[n <= 25]


def sinh_wave(n_that):
    co = pd.read_csv(P('Customer_Order.csv'), sep=';', encoding='utf-8-sig')
    co['sku'] = list(zip(co.Reference.str.strip(), co['Size (US)'].map(size_key)))
    nhu_cau = co.groupby('sku')['quantity (units)'].sum()
    chung = set.intersection(*[set(h) for h in T.HO.values()])
    nhu_cau = nhu_cau[[s in chung for s in nhu_cau.index]].sort_index()
    p = nhu_cau.to_numpy(float) / nhu_cau.sum()
    sku = list(nhu_cau.index)
    rng = np.random.default_rng(SEED)
    so_mon = rng.choice(n_that.to_numpy(), T.SO_WAVE)
    return [[sku[i] for i in rng.choice(len(sku), n, replace=False, p=p)] for n in so_mon]


N_THAT = phan_phoi_so_mon()
WAVES = sinh_wave(N_THAT)


def mot_viec(args):
    w, ten, sd = args
    D, start, k = build_instance(o_chi_dinh(WAVES[w], T.HO[ten], w))
    t0 = time.time()
    d, _ = T.gro_hop_ly(D, start, k, T.NP, T.G, T.T, T.PC, T.PM, sd, False)
    return dict(wave=w, so_mon=len(WAVES[w]), chinh_sach=ten, seed=sd, gro=d, giay=time.time() - t0)


def phan_tich(R, org):
    from bang_nhu_bai_bao import BAI, BAI_ANOVA_F
    out = []; p = out.append
    M = R.groupby(['wave', 'chinh_sach']).gro.mean().unstack()
    M['Original'] = org; M = M[COT]
    cu = pd.read_csv(os.path.join(KQ, 'gnone2025_C_chi_tiet.csv')).groupby(['wave', 'chinh_sach']).gro.mean().unstack()
    cu['Original'] = pd.read_csv(os.path.join(KQ, 'original_co_dinh.csv')).set_index('wave').original; cu = cu[COT]
    sm = pd.Series([len(x) for x in WAVES])
    p(f'Số món mỗi wave (50 wave sinh): TB {sm.mean():.1f}, min {sm.min()}, median {sm.median():.0f}, max {sm.max()} '
      f'| phân phối thật (≤25): TB {N_THAT.mean():.1f}, median {N_THAT.median():.0f}')

    def tom_tat(ten, X, don_vi):
        mean = X.mean(); cv = X.std() / mean * 100
        p(f'\n{ten} ({don_vi}):')
        d = X.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        p(d.round(1).to_string())
        p('   CV (%)                  : ' + ', '.join(f'{c} {cv[c]:.0f}' for c in COT))
        p('   Min / Mean (%)          : ' + ', '.join(f'{c} {X[c].min() / mean[c] * 100:.0f}' for c in COT))
        p('   % so với Original (Mean): ' + ', '.join(f'{c} {(mean[c] / mean["Original"] - 1) * 100:+.1f}' for c in CS))
        p('   Thứ hạng (Mean)  : ' + ' < '.join(mean[CS].sort_values().index))
        p('   Thứ hạng (Median): ' + ' < '.join(X[CS].median().sort_values().index))
        return cv
    cvb = BAI.loc['Std'] / BAI.loc['Mean'] * 100
    p('\nBài báo (m): CV (%) ' + ', '.join(f'{c} {cvb[c]:.0f}' for c in COT) + ' | Min/Mean (%) ' +
      ', '.join(f'{c} {BAI.loc["Min", c] / BAI.loc["Mean", c] * 100:.0f}' for c in COT))
    p('   % so với Original: ' + ', '.join(f'{c} {(BAI.loc["Mean", c] / BAI.loc["Mean", "Original"] - 1) * 100:+.1f}' for c in CS)
      + ' | Thứ hạng: Dedicated < Hybrid < Class-Based < Random')
    tom_tat('C cũ – cố định 25 SKU', cu, 'đơn vị toạ độ')
    tom_tat('C mới – số món thay đổi', M, 'đơn vị toạ độ')

    p('\nShapiro–Wilk (C mới):  ' + ', '.join(f'{c} W={stats.shapiro(M[c]).statistic:.3f} p={stats.shapiro(M[c]).pvalue:.3f}' for c in COT))
    for ten, X in [('C cũ (25 SKU)', cu), ('C mới (số món thay đổi)', M)]:
        f = stats.f_oneway(*[X[c] for c in COT])
        p(f'ANOVA 5 nhóm, {ten}: F = {f.statistic:.3f}, p = {f.pvalue:.2e}')
    p(f'ANOVA 5 nhóm, bài báo: F = {BAI_ANOVA_F}')
    fr = stats.friedmanchisquare(*[M[c] for c in COT])
    p(f'Friedman (C mới, ghép theo wave): chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.2e}')
    th = stats.tukey_hsd(*[M[c] for c in COT])
    p('\nHậu kiểm Tukey HSD và Wilcoxon ghép cặp/Holm (C mới):')
    cap = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    pw = [stats.wilcoxon(M[COT[i]], M[COT[j]]).pvalue for i, j in cap]
    o = np.argsort(pw); adj = np.empty(len(pw)); run = 0
    for r_, i in enumerate(o):
        run = max(run, min(1, (len(pw) - r_) * pw[i])); adj[i] = run
    for (i, j), pa in zip(cap, adj):
        p(f'   {COT[i]:11s} – {COT[j]:11s}: chênh TB {M[COT[i]].mean() - M[COT[j]].mean():8.1f} | Tukey p = {th.pvalue[i, j]:.4f}'
          f' | Wilcoxon p_Holm = {pa:.2e}')
    p('\nThời gian GRO TB mỗi wave (s): ' + ', '.join(f'{c} {R[R.chinh_sach == c].giay.mean():.2f}' for c in CS))
    # % giảm theo nhóm số món
    grp = pd.cut(sm, [0, 5, 12, 18, 25], labels=['1–5', '6–12', '13–18', '19–25'])
    G_ = (M[CS].div(M['Original'], axis=0) - 1) * 100
    p('\n% giảm so với Original của từng wave, trung bình theo nhóm số món:')
    p(G_.groupby(grp.values, observed=True).mean().round(1).assign(so_wave=grp.value_counts().sort_index().values).to_string())
    return M, cu, '\n'.join(out)


def ve(M, cu):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from bang_nhu_bai_bao import BAI
    mau = {'Original': '#9a9a9a', 'Random': '#4c78a8', 'Dedicated': '#e45756', 'Class-Based': '#f2a93b', 'Hybrid': '#54a24b'}

    def hop_tu_bang(t, goc):
        k = 100 / goc
        return [dict(label=c, whislo=t.loc['Min', c] * k, q1=t.loc['Q1 (25%)', c] * k, med=t.loc['Median', c] * k,
                     q3=t.loc['Q3 (75%)', c] * k, whishi=t.loc['Max', c] * k, mean=t.loc['Mean', c] * k, fliers=[]) for c in COT]

    def hop_tu_du_lieu(X):
        k = 100 / X['Original'].mean()
        return [dict(label=c, whislo=X[c].min() * k, q1=X[c].quantile(.25) * k, med=X[c].median() * k,
                     q3=X[c].quantile(.75) * k, whishi=X[c].max() * k, mean=X[c].mean() * k, fliers=[]) for c in COT]
    khung = [('Bài báo', hop_tu_bang(BAI, BAI.loc['Mean', 'Original'])),
             ('Nhóm – C, cố định 25 SKU', hop_tu_du_lieu(cu)),
             ('Nhóm – C, số món thay đổi (≤ 25)', hop_tu_du_lieu(M))]
    fig, ax = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    for a, (ten, h) in zip(ax, khung):
        b = a.bxp(h, showmeans=True, patch_artist=True)
        for patch, c in zip(b['boxes'], COT):
            patch.set_facecolor(mau[c]); patch.set_alpha(.75)
        a.axhline(100, ls='--', color='k', lw=.8); a.set_title(ten, fontsize=10); a.grid(axis='y', alpha=.3)
        a.tick_params(axis='x', rotation=35, labelsize=8)
    ax[0].set_ylabel('% của Original trung bình (Original TB = 100)')
    fig.suptitle('Ảnh hưởng của số món mỗi wave (râu = Min–Max, ▲ = Mean)', fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(KQ, 'hinh_C_bien_doi.png'), dpi=160)


if __name__ == '__main__':
    org = pd.Series([T.lo_trinh_theo_loi([c[0] for c in o_chi_dinh(WAVES[w], T.HO['Random'], w)]) for w in range(T.SO_WAVE)])
    pd.DataFrame({'wave': range(T.SO_WAVE), 'so_mon': [len(x) for x in WAVES], 'original': org,
                  'sku': [' '.join(f'{r}/{s}' for r, s in x) for x in WAVES]}).to_csv(os.path.join(KQ, 'C_bien_doi_waves.csv'), index=False)
    T.gro_hop_ly(*build_instance([[(100., 61.)], [(200., 151.)]]), 10, 2, 3, .9, .08, 0, False)
    viec = [(w, ten, sd) for w in range(T.SO_WAVE) for ten in CS for sd in range(T.SO_SEED)]
    t0 = time.time(); rows = []
    with Pool(int(os.environ.get('SLAP_WORKERS', os.cpu_count()))) as pool:
        for j, r in enumerate(pool.imap_unordered(mot_viec, viec, chunksize=4)):
            rows.append(r)
            if (j + 1) % 200 == 0:
                print(f'   đã chạy {j + 1}/{len(viec)} ({time.time() - t0:.0f} s)', flush=True)
    R = pd.DataFrame(rows).sort_values(['chinh_sach', 'wave', 'seed'])
    R.to_csv(os.path.join(KQ, 'C_bien_doi_chi_tiet.csv'), index=False)
    M, cu, txt = phan_tich(R, org)
    open(os.path.join(KQ, 'C_bien_doi_ket_qua.txt'), 'w', encoding='utf-8').write(txt)
    ve(M, cu)
    print(txt)
    print(f'Xong sau {time.time() - t0:.0f} s')
