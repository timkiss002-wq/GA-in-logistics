"""
BƯỚC 6 – Đo quãng đường THẬT bằng GRO trên wave thật của giai đoạn kiểm tra.

Proxy D trong lúc huấn luyện chỉ là "khứ hồi tới ô gần nhất của từng SKU". Ở đây, với mỗi wave kiểm tra
được chọn, lấy đúng các ô mà người lấy hàng đã ghé trong mô phỏng của từng phương pháp, rồi tìm lộ trình
ngắn nhất depot -> các ô -> depot bằng GRO (de Assis 2024; Gnone 2025). Wave <= 12 ô khác nhau: giải đúng
(Held–Karp) để kiểm tra GRO.

Chạy: python eval_gro.py results/picks_Random.npz results/picks_GreedyJ_0.5.npz ...
Ghi : results/gro_<tên>.csv (mỗi dòng 1 wave)
"""
import sys, os, time
import numpy as np, pandas as pd
from gro import build_instance, gro_dp, exact

I = dict(np.load('inst.npz', allow_pickle=True))
XY = I['cell_xy']
N_EVAL = 300                                        # 300 wave cách đều trong giai đoạn kiểm tra
NP, G, PAT = 300, 400, 80


def tour(cells, seed=0):
    pts = sorted({tuple(XY[c]) for c in cells})
    D, start, k = build_instance([[p] for p in pts])
    if len(pts) <= 12:
        return exact(D, start, k), len(pts)
    best, _ = gro_dp(D, start, k, NP, G, 3, 0.9, 0.08, seed, PAT, True, True)   # GA + 2-opt/or-opt
    return best, len(pts)


def main(files):
    for f in files:
        name = os.path.basename(f)[6:-4]
        out = f'results/gro_{name}.csv'
        if os.path.exists(out): continue
        z = np.load(f); ptr = z['ptr']; n = len(ptr) - 1
        idx = np.linspace(0, n - 1, N_EVAL).astype(int)
        t = time.time(); rows = []
        for w in idx:
            d, m = tour(z['cells'][ptr[w]:ptr[w + 1]], seed=int(w))
            rows.append(dict(wave=int(w), n_cells=m, dist=d))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'{name:<22} GRO TB = {np.mean([r["dist"] for r in rows]):7.1f} | {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == '--check':        # GRO so với lời giải đúng trên wave nhỏ
        z = np.load('results/picks_Random.npz'); ptr = z['ptr']; errs = []
        for w in range(0, 2900, 3):
            pts = sorted({tuple(XY[c]) for c in z['cells'][ptr[w]:ptr[w + 1]]})
            if 9 <= len(pts) <= 13:
                D, s, k = build_instance([[p] for p in pts])
                e = exact(D, s, k); g, _ = gro_dp(D, s, k, NP, G, 3, 0.9, 0.08, 0, PAT, True, True); errs.append(g / e - 1)
        print(f'GRO vs lời giải đúng trên {len(errs)} wave (9–13 ô): sai số TB {np.mean(errs):.3%}, max {np.max(errs):.3%}, '
              f'trúng tối ưu {np.mean(np.array(errs) < 1e-9):.0%}')
    else:
        main(sys.argv[1:])
