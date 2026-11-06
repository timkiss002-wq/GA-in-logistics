"""
Bài v7c – PHƯƠNG PHÁP ĐÃ CÔNG BỐ dùng làm baseline (thay cho các luật tự dựng ABC/Scattered/Correlated của bản v7b).
Mỗi phương pháp theo đúng định nghĩa/công thức của bài gốc; chỗ bài gốc không nói (vì bài gốc là bài toán tĩnh,
còn ở đây hàng về từng đơn vị) được ghi rõ ở cột "điều chỉnh tối thiểu".

  Gnone-Class  (Gnone et al. 2025): lớp SKU theo đúng định nghĩa của bài – A = 20% SKU bán chạy nhất, B = 30% tiếp theo,
               C = 50% còn lại (nhu cầu giai đoạn huấn luyện). Vùng của lớp = các ô có ABCCOD tương ứng trong FILE
               Class_Based_Storage.csv mà bài công bố. Đơn vị về -> ô trống ngẫu nhiên trong vùng của lớp
               (điều chỉnh: vùng đầy -> vùng lớp kế tiếp A->B->C).
  Gnone-Ded    (Gnone et al. 2025): ô dành sẵn cho SKU trong FILE Dedicated_Storage.csv. Đơn vị về -> ô dành sẵn còn chỗ,
               gần depot nhất (điều chỉnh: SKU không có ô dành sẵn hoặc ô dành sẵn đã đầy -> ô trống gần depot nhất).
  Gnone-Hyb    (Gnone et al. 2025): FILE Hybrid_Storage.csv. SKU có ô cố định ở vùng X -> như Dedicated; SKU khác -> ô trống
               ngẫu nhiên trong vùng (XYZCOD) mà file đặt SKU đó (điều chỉnh: không có -> ô trống ngẫu nhiên).
  WB-Scatter   (Weidinger & Boysen 2018): mục tiêu của bài – tối thiểu khoảng cách LỚN NHẤT từ các điểm đo tới đơn vị gần
               nhất của SKU (dạng p-center). Điểm đo = 44 điểm giao lối đi của bộ dữ liệu (Support_Points_Navigation).
               Một đơn vị về -> ô trống làm max_m min_h d(m, h) nhỏ nhất (hoà: tổng khoảng cách nhỏ hơn, rồi gần depot).
               (Bài gốc giải theo lô bằng tìm kiếm nhị phân; với một đơn vị, quyết định tối ưu chính là luật này.)
  Wanis-Score  (Wanis et al. 2026, Stage 2, công thức 10–14, trọng số tốt nhất Bảng 7: w_c = w_m = w_d = 0,4,
               M_max = 50): Score(s,l) = w_c·Φcorr + w_m·Φmix + w_d·Φdist, chọn ô có Score nhỏ nhất.
               Φcorr = −Σ_{u∈U_l} c_su·x̂_s·x̂_u, c = lift × joint support từ ĐƠN HÀNG (Customer_Order) giai đoạn huấn luyện,
               chuẩn hoá theo max, giữ 50 liên kết mạnh nhất mỗi SKU; Φmix = entropy tỷ lệ SKU trong ô sau khi đặt;
               Φdist = x̂_s·δ_l + (1−x̂_s)(1−δ_l). (Điều chỉnh: bỏ Stage 1 NSGA-II gom họ vì đó là bước bố trí lại cả kho;
               dùng x̂ = x/X_max trong Φcorr – bài không nói thang đo của x; ứng viên = mọi ô còn chỗ.)
"""
import numpy as np, pandas as pd
from gro import dist_matrix, CROSS_X, CROSS_YMIN, AISLES

_C = {}


def _sku_index(env):
    if 'sid' not in _C:
        _C['sid'] = {s: i for i, s in enumerate(env.I['sku_name'])}
        _C['cells'] = [c.strip() for c in env.I['cell_name']]
    return _C['sid'], _C['cells']


def _plan(env, name):
    key = 'plan_' + name
    if key in _C: return _C[key]
    sid, cells = _sku_index(env)
    d = pd.read_csv(f'data/{name}.csv', sep=';', encoding='utf-8-sig', dtype=str); d['Location'] = d.Location.str.strip()
    d = d.set_index('Location'); code = d.columns[0]
    zone = np.array([d.loc[c, code].strip() for c in cells])
    planned = {}
    for li, c in enumerate(cells):
        for v in d.loc[c].iloc[1:]:
            if isinstance(v, str) and ';' in v:
                r, s = v.split(';'); k = sid.get(f'{r.strip()}|{float(s):.1f}')
                if k is not None: planned.setdefault(k, set()).add(li)
    planned = {k: np.array(sorted(v)) for k, v in planned.items()}
    _C[key] = (zone, planned); return _C[key]


def _demand_train(env):
    if 'x' in _C: return _C['x']
    ptr, sk = env.I['wave_ptr'], env.I['wave_sku']; n = int(env.I['n_train'])
    x = np.bincount(sk[:ptr[n]], minlength=env.K).astype(float)
    _C['x'] = x; return x


# ---------------- Gnone et al. (2025) ----------------
def pol_gnone_class(env, rng):
    if 'gcls' not in _C:
        x = _demand_train(env); o = np.argsort(-x); K = env.K
        cls = np.empty(K, '<U1'); cls[o[:int(0.2 * K)]] = 'A'; cls[o[int(0.2 * K):int(0.5 * K)]] = 'B'; cls[o[int(0.5 * K):]] = 'C'
        _C['gcls'] = cls
    zone, _ = _plan(env, 'Class_Based_Storage'); k = env.queue[0]; m = env.mask(); c = _C['gcls'][k]
    for z in [c] + [q for q in 'ABC' if q != c]:
        f = np.flatnonzero(m & (zone == z))
        if len(f): return int(rng.choice(f))
    return int(rng.choice(np.flatnonzero(m)))


def _nearest_free(env, m):
    return int(np.argmin(np.where(m, env.d, np.inf)))


def pol_gnone_ded(env, rng):
    _, planned = _plan(env, 'Dedicated_Storage'); k = env.queue[0]; m = env.mask()
    p = planned.get(k)
    if p is not None:
        f = p[m[p]]
        if len(f): return int(f[np.argmin(env.d[f])])
    _C['ded_fallback'] = _C.get('ded_fallback', 0) + 1
    return _nearest_free(env, m)


def pol_gnone_hyb(env, rng):
    zone, planned = _plan(env, 'Hybrid_Storage'); k = env.queue[0]; m = env.mask()
    p = planned.get(k)
    if p is not None:
        if (zone[p] == 'X').any():                                   # SKU cố định (vùng X)
            f = p[m[p] & (zone[p] == 'X')]
            if len(f): return int(f[np.argmin(env.d[f])])
        zs = set(zone[p])
        f = np.flatnonzero(m & np.isin(zone, list(zs)))
        if len(f): return int(rng.choice(f))
    return int(rng.choice(np.flatnonzero(m)))


# ---------------- Weidinger & Boysen (2018) ----------------
def _measure(env):
    if 'DM' in _C: return _C['DM']
    pts = [(cx, a) for cx, cy in zip(CROSS_X, CROSS_YMIN) for a in AISLES if a >= cy]       # 44 điểm giao lối đi
    pts = np.array(pts, float); xy = env.XY.astype(float)
    allp = np.vstack([pts, xy]); D = dist_matrix(allp[:, 0].copy(), allp[:, 1].copy(), CROSS_X, CROSS_YMIN)
    _C['DM'] = D[:len(pts), len(pts):]                                                        # 44 × L
    return _C['DM']


def pol_wb_scatter(env, rng):
    DM = _measure(env); k = env.queue[0]; m = env.mask()
    h = np.flatnonzero(env.cnt[:, k] > 0)
    cur = DM[:, h].min(1) if len(h) else np.full(DM.shape[0], np.inf)              # khoảng cách hiện tại mỗi điểm đo
    new = np.minimum(cur[:, None], DM)                                              # 44 × L nếu đặt vào ô l
    mx = new.max(0); sm = np.where(np.isfinite(new), new, 1e9).sum(0)
    key = np.where(m, mx * 1e6 + sm + 1e-6 * env.d, np.inf)
    return int(np.argmin(key))


# ---------------- Wanis et al. (2026), Stage 2 ----------------
W_C, W_M, W_D, M_MAX = 0.4, 0.4, 0.4, 50


def _wanis(env):
    if 'wan' in _C: return _C['wan']
    sid, _ = _sku_index(env)
    co = pd.read_csv('data/Customer_Order.csv', sep=';', encoding='utf-8-sig', dtype=str)
    co.columns = ['cust', 'order', 'otc', 'ref', 'size', 'qty', 'date', 'wave', 'op']
    wid = env.I['wave_id']; train_w = set(wid[:int(env.I['n_train'])].astype(str))
    co = co[co.wave.str.strip().isin(train_w)]
    sz = co['size'].astype(float); sz = np.where(sz >= 20, sz / 10, sz)              # 105 -> 10,5 (mã hoá cỡ của file)
    co['k'] = [sid.get(f'{r.strip()}|{s:.1f}') for r, s in zip(co.ref, sz)]
    co = co.dropna(subset=['k']); co['k'] = co.k.astype(int)
    orders = co.groupby('order').k.apply(lambda v: np.unique(v.values))
    N = len(orders); K = env.K
    a = np.zeros(K); pairs = {}
    for ks in orders:
        a[ks] += 1
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                pr = (ks[i], ks[j]); pairs[pr] = pairs.get(pr, 0) + 1
    a /= N
    C = np.zeros((K, K))
    for (i, j), n in pairs.items():
        ass = n / N; lift = ass / (a[i] * a[j]); C[i, j] = C[j, i] = lift * ass
    if C.max() > 0: C /= C.max()
    for i in range(K):                                                                # giữ M_max liên kết mạnh nhất
        row = C[i]; nz = np.flatnonzero(row)
        if len(nz) > M_MAX: row[nz[np.argsort(row[nz])[:-M_MAX]]] = 0.0
    C = np.maximum(C, C.T)
    x = _demand_train(env); xh = x / x.max()
    _C['wan'] = (C, xh, len(orders)); return _C['wan']


def pol_wanis(env, rng):
    C, xh, _ = _wanis(env); k = env.queue[0]; m = env.mask()
    cnt = env.cnt.astype(float)                                                        # L × K
    corr = -((cnt > 0).astype(float) @ (C[k] * xh)) * xh[k]                                            # Φcorr
    T = cnt.sum(1) + 1                                                                 # số đơn vị trong ô sau khi đặt
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.where(cnt > 0, cnt * np.log(np.maximum(cnt, 1)), 0).sum(1)              # Σ_u q_u log q_u (trước)
    ck = cnt[:, k]
    S = S - np.where(ck > 0, ck * np.log(np.maximum(ck, 1)), 0) + (ck + 1) * np.log(ck + 1)
    ent = np.log(T) - S / T                                                            # Φmix = −Σ p log p sau khi đặt
    delta = env.d / env.dmax
    dist = xh[k] * delta + (1 - xh[k]) * (1 - delta)                                  # Φdist
    sc = W_C * corr + W_M * ent + W_D * dist
    return int(np.argmin(np.where(m, sc, np.inf)))


LIT2 = {'Gnone-Class': pol_gnone_class, 'Gnone-Ded': pol_gnone_ded, 'Gnone-Hyb': pol_gnone_hyb,
        'WB-Scatter': pol_wb_scatter, 'Wanis-Score': pol_wanis}
