"""
KHÔNG DÙNG TRONG BÀI (bản v7c thay bằng lit_methods.py – phương pháp đã công bố). Giữ lại để truy vết.
Bài v7b – các luật cất hàng ĐẠI DIỆN cho các dòng nghiên cứu hiện có (để so sánh LP-RL với "phương pháp hiện tại").
Các luật này là phiên bản trực tuyến (từng đơn vị hàng về) dựng theo ý tưởng của bài gốc, không phải tái hiện y nguyên.

  ABC        – class-based 3 lớp (Petersen & Aase 2004; Gnone et al. 2025): SKU chia A/B/C theo nhu cầu cộng dồn
               50/30/20%; ô chia 3 vùng theo khoảng cách tới depot, cỡ vùng tỷ lệ tồn kho của lớp. Đơn vị về
               -> ô trống ngẫu nhiên trong vùng của lớp; vùng đầy -> vùng gần nhất còn chỗ.
  Correlated – lưu kho theo tương quan (ý tưởng của Wanis et al. 2026): cất cạnh hàng hay được lấy cùng wave.
               Điểm ô = Σ_mã (số đơn vị mã r trong ô) × lift(mã của SKU, r); lift ước lượng trên wave trước giai đoạn
               kiểm tra. Điểm bằng nhau -> ô gần depot hơn.
  Scattered  – rải hàng kiểu mixed-shelves (Weidinger & Boysen 2018): cất vào ô CHƯA có SKU này và xa nhất tính từ
               các ô đang có SKU (theo lối đi), để luôn có một đơn vị gần mọi lộ trình. Hoà -> gần depot hơn.
"""
import numpy as np
from gro import dist_matrix, CROSS_X, CROSS_YMIN

_CACHE = {}


def _abc(env):
    if 'abc' in _CACHE: return _CACHE['abc']
    p = env.p; o = np.argsort(-p); cum = np.cumsum(p[o])
    cls = np.empty(env.K, int); cls[o] = np.where(cum <= 0.5, 0, np.where(cum <= 0.8, 1, 2))
    stock = env.cnt.sum(0); share = np.array([stock[cls == c].sum() for c in range(3)], float); share /= share.sum()
    rank = env.d_rank                                         # 0 = gần nhất
    bounds = np.cumsum(share); zone = np.searchsorted(bounds, rank, side='right').clip(0, 2)
    _CACHE['abc'] = (cls, zone); return cls, zone


def pol_abc(env, rng):
    cls, zone = _abc(env); k = env.queue[0]; m = env.mask(); c = cls[k]
    for z in sorted(range(3), key=lambda z: abs(z - c)):        # vùng của lớp, rồi vùng gần nhất
        f = np.flatnonzero(m & (zone == z))
        if len(f): return int(rng.choice(f))
    return int(rng.choice(np.flatnonzero(m)))


def _lift(env):
    if 'lift' in _CACHE: return _CACHE['lift']
    I = env.I; R = env.R; ptr, sk = I['wave_ptr'], I['wave_sku']; n = int(I['n_train'])
    C = np.zeros((R, R)); f = np.zeros(R)
    for w in range(n):
        r = np.unique(env.sku_ref[sk[ptr[w]:ptr[w + 1]]]); f[r] += 1; C[np.ix_(r, r)] += 1
    f = f / n; P = C / n
    L = P / np.maximum(np.outer(f, f), 1e-12); np.fill_diagonal(L, 0.0)
    L = np.log(np.maximum(L, 1e-6)).clip(0, None)              # chỉ giữ tương quan dương (lift > 1)
    _CACHE['lift'] = L; return L


def pol_correlated(env, rng):
    L = _lift(env); k = env.queue[0]; r = env.sku_ref[k]; m = env.mask()
    sc = env.cref @ L[r] - 1e-6 * env.d                         # hoà -> gần depot
    sc = np.where(m, sc, -np.inf); return int(np.argmax(sc))


def _dcell(env):
    if 'D' in _CACHE: return _CACHE['D']
    xy = env.XY.astype(float)
    D = dist_matrix(xy[:, 0].copy(), xy[:, 1].copy(), CROSS_X, CROSS_YMIN)
    _CACHE['D'] = D; return D


def pol_scattered(env, rng):
    D = _dcell(env); k = env.queue[0]; m = env.mask()
    h = np.flatnonzero(env.cnt[:, k] > 0)
    if len(h) == 0:
        sc = -env.d
    else:
        sc = D[:, h].min(1) - 1e-6 * env.d                      # xa nhất tính từ các ô đang có SKU
    sc = np.where(m & (env.cnt[:, k] == 0), sc, -np.inf)
    if not np.isfinite(sc).any(): sc = np.where(m, -env.d, -np.inf)
    return int(np.argmax(sc))


LIT = {'ABC': pol_abc, 'Correlated': pol_correlated, 'Scattered': pol_scattered}
