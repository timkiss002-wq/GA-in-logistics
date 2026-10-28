"""
PHƯƠNG ÁN C – ô chỉ định sẵn, tầng 1–2, đủ 10 seed (bổ sung cho A và B đã có trong gnone2025_chi_tiet.csv).

C: mỗi SKU của wave được "WMS" chỉ định sẵn MỘT ô (bốc ngẫu nhiên trong các ô chứa SKU đó, cố định theo wave,
   rng = 1000 + số wave – giống hệt thí nghiệm chẩn đoán ngày 26/10). GA chỉ tìm thứ tự đi.
Tour luôn khép kín: depot -> các ô -> depot (depot = CC-08).

Đồng thời tính ORIGINAL CỐ ĐỊNH, dùng chung cho cả A, B, C:
   bố trí thực tế của công ty (Random_Storage.csv) + ô do WMS chỉ định (cùng quy tắc với C) + đi theo lối kiểu S-shape,
   không dùng GA. Original chỉ phụ thuộc wave, không phụ thuộc phương án chọn ô.

Chạy:  python chay_C_10seed.py        -> ghi results/gnone2025_C_chi_tiet.csv và results/original_co_dinh.csv
"""
import os, time
import numpy as np, pandas as pd
from multiprocessing import Pool
import tai_hien_gnone2025 as T
from gro import build_instance


def o_chi_dinh(wave, ho, w):
    """Quy tắc C: mỗi SKU một ô, bốc ngẫu nhiên trong họ ô, cố định theo wave."""
    rng = np.random.default_rng(1000 + w)
    return [[ho[s][rng.integers(len(ho[s]))]] for s in wave]


def mot_viec(args):
    w, ten, sd = args
    cands = o_chi_dinh(T.WAVES[w], T.HO[ten], w)
    D, start, k = build_instance(cands)
    t0 = time.time()
    d, _ = T.gro_hop_ly(D, start, k, T.NP, T.G, T.T, T.PC, T.PM, sd, False)
    return dict(wave=w, chinh_sach=ten, gia_dinh='chi_dinh', seed=sd, gro=d, giay=time.time() - t0)


if __name__ == '__main__':
    # 1) Original cố định
    org = [T.lo_trinh_theo_loi([c[0] for c in o_chi_dinh(T.WAVES[w], T.HO['Random'], w)]) for w in range(T.SO_WAVE)]
    pd.DataFrame({'wave': range(T.SO_WAVE), 'original': org}).to_csv(os.path.join(T.KQ, 'original_co_dinh.csv'), index=False)
    print(f'Original cố định: TB = {np.mean(org):.1f}')
    # 2) C x 4 chính sách x 50 wave x 10 seed
    viec = [(w, ten, sd) for w in range(T.SO_WAVE) for ten in T.CHINH_SACH for sd in range(T.SO_SEED)]
    T.gro_hop_ly(*build_instance([[(100., 61.)], [(200., 151.)]]), 10, 2, 3, .9, .08, 0, False)   # biên dịch numba trước
    t0 = time.time(); rows = []
    with Pool(int(os.environ.get('SLAP_WORKERS', os.cpu_count()))) as pool:
        for j, r in enumerate(pool.imap_unordered(mot_viec, viec, chunksize=4)):
            rows.append(r)
            if (j + 1) % 100 == 0:
                print(f'   đã chạy {j + 1}/{len(viec)} ({time.time() - t0:.0f} s)', flush=True)
    pd.DataFrame(rows).sort_values(['chinh_sach', 'wave', 'seed']).to_csv(
        os.path.join(T.KQ, 'gnone2025_C_chi_tiet.csv'), index=False)
    print(f'Xong sau {time.time() - t0:.0f} s')
