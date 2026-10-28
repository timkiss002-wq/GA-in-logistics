"""
CHẨN ĐOÁN: vì sao kết quả của nhóm khác Gnone et al. (2025)?
Giữ nguyên 50 wave và router GRO hợp-lý, mỗi lần chỉ thay MỘT giả định:
  1) Tầng lấy hàng : chỉ tầng 1-2  |  cả 4 tầng
  2) Cách chọn ô   : A = GA tự chọn ô  |  B = ô gần depot nhất  |  C = một ô bất kỳ do hệ thống chỉ định (cố định, ngẫu nhiên)
Mỗi biến thể chạy seed 0 (đủ để thấy thứ hạng; dao động giữa seed đã đo ở thí nghiệm chính).

Chạy:  python chan_doan_gnone2025.py
"""
import os, collections, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import tai_hien_gnone2025 as T
from data import load_layout
from gro import build_instance

TANG = {'tang_1_2': T.TANG_LAY_HANG, 'ca_4_tang': set(T.LOC.index)}
HO = {}
for tang, cho_phep in TANG.items():
    for ten, f in T.CHINH_SACH.items():
        h = collections.defaultdict(set)
        for l, slots in load_layout(f).items():
            if l in cho_phep:
                for s in slots:
                    h[s].add(T.PP[l])
        HO[tang, ten] = {s: sorted(v) for s, v in h.items()}


def ung_vien(wave, ho, cach, rng):
    out = []
    for s in wave:
        c = ho[s]
        if cach == 'B_gan_depot':
            d = build_instance([c])[0][0, 1:]; c = [c[int(np.argmin(d))]]
        elif cach == 'C_o_chi_dinh':
            c = [c[rng.integers(len(c))]]
        out.append(c)
    return out


def viec(args):
    tang, cach, ten, w = args
    rng = np.random.default_rng(1000 + w)
    cands = ung_vien(T.WAVES[w], HO[tang, ten], cach, rng)
    D, start, k = build_instance(cands)
    d, _ = T.gro_hop_ly(D, start, k, T.NP, T.G, T.T, T.PC, T.PM, 0, False)
    org = T.original(cands, w) if ten == 'Random' else np.nan
    return dict(tang=tang, cach=cach, chinh_sach=ten, wave=w, gro=d, original=org, so_o=float(np.mean([len(c) for c in HO[tang, ten].values()])))


if __name__ == '__main__':
    t0 = time.time()
    jobs = [(tg, c, ten, w) for tg in TANG for c in ['A_tu_chon', 'B_gan_depot', 'C_o_chi_dinh']
            for ten in T.CHINH_SACH for w in range(T.SO_WAVE)]
    with Pool() as p:
        R = pd.DataFrame(p.map(viec, jobs, chunksize=4))
    R.to_csv(os.path.join(T.KQ, 'chan_doan_chi_tiet.csv'), index=False)
    bai_bao = pd.Series({'Original': 1056.90, 'Random': 887.32, 'Dedicated': 726.77, 'Class-Based': 800.55, 'Hybrid': 744.96})
    rows = []
    for (tg, c), X in R.groupby(['tang', 'cach'], sort=False):
        m = X.groupby('chinh_sach').gro.mean()
        m['Original'] = X.original.mean()
        r = (m / m['Original'] - 1) * 100
        xh = ' < '.join(m.drop('Original').sort_values().index)
        rows.append(dict(tang=tg, cach_chon_o=c, **{f'{k}_%': round(r[k], 1) for k in ['Random', 'Dedicated', 'Class-Based', 'Hybrid']},
                         thu_hang=xh))
    rows.append(dict(tang='BÀI BÁO', cach_chon_o='?', **{f'{k}_%': round((bai_bao[k] / bai_bao['Original'] - 1) * 100, 1)
                                                         for k in ['Random', 'Dedicated', 'Class-Based', 'Hybrid']},
                     thu_hang='Dedicated < Hybrid < Class-Based < Random'))
    B = pd.DataFrame(rows)
    so_o = R.groupby(['tang', 'chinh_sach']).so_o.first().unstack().round(1)
    txt = ('Số ô trung bình mỗi SKU:\n' + so_o.to_string() + '\n\nQuãng đường so với Original (%), và thứ hạng:\n' + B.to_string(index=False))
    print(txt); open(os.path.join(T.KQ, 'chan_doan_ket_qua.txt'), 'w', encoding='utf-8').write(txt)
    print(f'\n({time.time() - t0:.0f} s)')
