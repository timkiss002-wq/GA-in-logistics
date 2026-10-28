"""
TÁI HIỆN de Assis et al. (2024) – GRO (Genetic Route Optimisation)
Bài: "Optimising Warehouse Order Picking: Real Case Application in the Shoe Manufacturing Industry"

Hai phiên bản GRO, cùng dữ liệu, cùng tham số (Bảng 4 của bài: alpha=1000, beta=1000, lambda=3, gamma=80%, omega=10%):
  * DUNG_CHU : làm đúng pseudo-code Algorithm 1-6 của bài
  * HOP_LY   : sửa tối thiểu để GA tối ưu được cả thứ tự đi (thứ tự riêng từng cá thể, lai OX, giữ con giỏi nhất)

Chạy:
  python tai_hien_gro2024.py demo        -> 1 wave, in quá trình tiến hoá của cả 2 phiên bản
  python tai_hien_gro2024.py             -> 51 wave x 2 phiên bản x 10 seed, in Bảng 5 của nhóm
"""
import sys, time
import numpy as np, pandas as pd
from scipy import stats
from data import load_waves, load_locations, P

# ---------------- THAM SỐ (Bảng 4 của bài) ----------------
ALPHA, BETA, LAMBDA, GAMMA, OMEGA = 1000, 1000, 3, 0.80, 0.10
SO_WAVE, SO_SEED = 51, 10
DEPOT = (403, 631)
LOI_NGANG = np.array([-29, 61, 151, 241, 331, 421, 511, 631, 751, 841, 931, 1021, 1111, 1201, 1291, 1381, 1471])
LOI_DOC = [(66, -29), (403, -29), (686, 631)]          # (x của lối dọc, y thấp nhất lối đó tới được)


# ============ CHUẨN BỊ: kho, khoảng cách, pick list ============
def diem_dung(x, y):
    return (int(x), int(LOI_NGANG[np.argmin(np.abs(LOI_NGANG - y))]))


def khoang_cach(p, q):
    """Cùng lối ngang: đi thẳng. Khác lối: ra một lối dọc, đi dọc, rẽ vào (chọn lối dọc ngắn nhất)."""
    (x1, y1), (x2, y2) = p, q
    if y1 == y2:
        return abs(x1 - x2)
    return min(abs(x1 - xd) + abs(y1 - y2) + abs(x2 - xd) for xd, ymin in LOI_DOC if min(y1, y2) >= ymin)


def chuan_bi_du_lieu(seed=2024):
    """Chọn ngẫu nhiên 51 pick list thật có 15-27 món (giống bài), tạo 'họ' ô cho từng sản phẩm."""
    loc = load_locations()
    w = load_waves()
    co = pd.read_csv(P('Customer_Order.csv'), sep=';', encoding='utf-8-sig')
    co.columns = [c.strip() for c in co.columns]
    ngay = pd.to_datetime(co['creationDate'], dayfirst=True).groupby(co['waveNumber']).min()
    w['tuan'] = w.wave.map(ngay).dt.isocalendar().week
    w['co_toa_do'] = w['loc'].isin(loc.index)
    w['sku'] = list(zip(w.ref, w['size']))
    # Họ ô của một sản phẩm = các ô mà sản phẩm đó được lấy ra trong cùng tuần (gần đúng với tồn kho lúc đó)
    ho = w[w.co_toa_do].groupby(['sku', 'tuan'])['loc'].agg(lambda s: sorted(set(s))).to_dict()
    hop_le = w.groupby('wave').agg(n=('sku', 'size'), ok=('co_toa_do', 'all'))
    ung_vien = hop_le[(hop_le.n >= 15) & (hop_le.n <= 27) & hop_le.ok].index.to_numpy()
    chon = np.sort(np.random.default_rng(seed).choice(ung_vien, SO_WAVE, replace=False))
    pick_lists = []
    for wid in chon:
        d = w[w.wave == wid]
        san_pham = list(dict.fromkeys(d.sku))                      # sản phẩm khác nhau, giữ thứ tự xuất hiện
        tuan = d.tuan.iloc[0]
        cac_ho = [[diem_dung(*loc.loc[l, ['x', 'y']]) for l in ho[(s, tuan)]] for s in san_pham]
        thuc_te = [diem_dung(*loc.loc[l, ['x', 'y']]) for l in d['loc']]     # đúng thứ tự dòng trong file
        pick_lists.append(dict(wave=int(wid), so_mon=len(d), san_pham=san_pham, ho=cac_ho, thuc_te=thuc_te))
    return pick_lists


def quang_duong_thuc_te(diem):
    """'Distance Actual': đi theo đúng thứ tự trong pick list, từ depot rồi về depot."""
    tong, truoc = 0, DEPOT
    for p in diem:
        tong += khoang_cach(truoc, p); truoc = p
    return tong + khoang_cach(truoc, DEPOT)


def ma_tran(pl):
    """Đánh số mọi ô ứng viên; trả về ma trận khoảng cách (chỉ số 0 = depot)."""
    diem = [DEPOT]; ho_idx = []
    for h in pl['ho']:
        ho_idx.append(np.arange(len(diem), len(diem) + len(h))); diem += h
    D = np.array([[khoang_cach(p, q) for q in diem] for p in diem], dtype=float)
    return D, ho_idx


# ============ B2. FITNESS (Algorithm 3), dùng chung cho cả đàn ============
def b2_fitness(diem_cua_dan, D):
    """diem_cua_dan[i, k] = ô robot i ghé ở bước k. Cộng depot -> ... -> depot."""
    d = D[0, diem_cua_dan[:, 0]] + D[diem_cua_dan[:, -1], 0]
    return d + D[diem_cua_dan[:, :-1], diem_cua_dan[:, 1:]].sum(axis=1)


def b3_selection(f, rng, so_luong):
    """Algorithm 4 – tournament: mỗi lần bốc LAMBDA cá thể, giữ con ngắn nhất."""
    boc = rng.integers(0, len(f), size=(so_luong, LAMBDA))
    return boc[np.arange(so_luong), np.argmin(f[boc], axis=1)]


# ============ PHIÊN BẢN 1: ĐÚNG CHỮ theo pseudo-code ============
def gro_dung_chu(pl, D, ho_idx, seed, in_tien_trinh=False):
    rng = np.random.default_rng(seed)
    n = len(ho_idx)
    # B1 (Algorithm 2): xáo thứ tự sản phẩm MỘT LẦN cho cả đàn; mỗi cá thể chọn ngẫu nhiên một ô trong họ
    v_sao = rng.permutation(n)
    sp = np.tile(v_sao, (ALPHA, 1))                                      # sản phẩm ở mỗi vị trí
    o = np.stack([[rng.choice(ho_idx[s]) for s in v_sao] for _ in range(ALPHA)])   # ô ở mỗi vị trí
    h = n // 2
    loi = 0
    for the_he in range(BETA):
        f = b2_fitness(o, D)                                             # B2
        if in_tien_trinh and the_he % 100 == 0:
            print(f'   [đúng chữ] thế hệ {the_he:4d}: giỏi nhất {f.min():6.0f} | trung bình {f.mean():7.1f}')
        chon = b3_selection(f, rng, ALPHA)                               # B3
        sp, o = sp[chon].copy(), o[chon].copy()
        # B4 (Algorithm 5): cắt đôi ở giữa, con1 = nửa đầu a + nửa sau b, con2 = ngược lại
        cap = rng.permutation(ALPHA).reshape(-1, 2)
        lai = rng.random(len(cap)) < GAMMA
        a, b = cap[lai, 0], cap[lai, 1]
        c1_sp = np.c_[sp[a, :h], sp[b, h:]]; c2_sp = np.c_[sp[b, :h], sp[a, h:]]
        c1_o = np.c_[o[a, :h], o[b, h:]];   c2_o = np.c_[o[b, :h], o[a, h:]]
        # con hợp lệ phải ghé đủ mỗi sản phẩm đúng 1 lần; nếu không, giữ nguyên bố mẹ (bài không nói cách xử lý)
        dung = (np.sort(c1_sp, 1) == np.arange(n)).all(1) & (np.sort(c2_sp, 1) == np.arange(n)).all(1)
        loi += int((~dung).sum())
        a, b = a[dung], b[dung]
        sp[a], sp[b], o[a], o[b] = c1_sp[dung], c2_sp[dung], c1_o[dung], c2_o[dung]
        # B5 (Algorithm 6): đổi chỗ 2 gen, chỉ trong nửa sau [h, n)
        for i in np.where(rng.random(ALPHA) < OMEGA)[0]:
            if n - h >= 2:
                x, y = rng.choice(np.arange(h, n), 2, replace=False)
                sp[i, [x, y]] = sp[i, [y, x]]; o[i, [x, y]] = o[i, [y, x]]
        # B6: đàn con thay hẳn đàn cũ (bài không có elitism)
    f = b2_fitness(o, D)
    return f.min(), loi


# ============ PHIÊN BẢN 2: HỢP LÝ ============
from numba import njit


@njit(cache=True)
def ox_ca_dan(tt, cap, cat):
    """Lai OX cho mọi cặp (a, b): con a giữ đoạn cat[k] của a, phần còn lại theo thứ tự của b; con b ngược lại."""
    n = tt.shape[1]
    for k in range(cap.shape[0]):
        a, b = cap[k, 0], cap[k, 1]
        lo, hi = cat[k, 0], cat[k, 1]
        pa, pb = tt[a].copy(), tt[b].copy()
        for cha, me, dich in ((pa, pb, a), (pb, pa, b)):
            dung = np.zeros(n, np.bool_)
            con = np.empty(n, np.int64)
            for i in range(lo, hi + 1):
                con[i] = cha[i]; dung[cha[i]] = True
            j = 0
            for g in me:
                if not dung[g]:
                    while lo <= j <= hi:
                        j += 1
                    con[j] = g; j += 1
            tt[dich] = con


def ox(cha, me, rng):
    n = len(cha); a, b = np.sort(rng.choice(n, 2, replace=False))
    con = -np.ones(n, int); con[a:b + 1] = cha[a:b + 1]
    con[con < 0] = [g for g in me if g not in set(cha[a:b + 1])]
    return con


def gro_hop_ly(pl, D, ho_idx, seed, in_tien_trinh=False):
    rng = np.random.default_rng(seed)
    n = len(ho_idx)
    # B1: MỖI cá thể một thứ tự riêng + một ô ngẫu nhiên cho mỗi sản phẩm
    thu_tu = np.stack([rng.permutation(n) for _ in range(ALPHA)])
    chon_o = np.stack([[rng.integers(len(ho_idx[s])) for s in range(n)] for _ in range(ALPHA)])
    bang = np.full((n, max(len(x) for x in ho_idx)), -1)
    for s, x in enumerate(ho_idx):
        bang[s, :len(x)] = x
    def diem(tt, co):                                   # (thứ tự, ô đã chọn) -> dãy ô đi qua
        return bang[tt, np.take_along_axis(co, tt, axis=1)]
    for the_he in range(BETA):
        f = b2_fitness(diem(thu_tu, chon_o), D)                          # B2
        if in_tien_trinh and the_he % 100 == 0:
            print(f'   [hợp lý]  thế hệ {the_he:4d}: giỏi nhất {f.min():6.0f} | trung bình {f.mean():7.1f}')
        tot = np.argmin(f)
        chon = b3_selection(f, rng, ALPHA)                               # B3
        tt, co = thu_tu[chon].copy(), chon_o[chon].copy()
        cap = rng.permutation(ALPHA).reshape(-1, 2)                      # B4: OX cho thứ tự, trộn đều cho ô
        cap = cap[rng.random(len(cap)) < GAMMA]
        cat = np.sort(rng.integers(0, n, size=(len(cap), 2)), axis=1)
        ox_ca_dan(tt, cap, cat)
        m = rng.random((len(cap), n)) < 0.5
        ca, cb = co[cap[:, 0]], co[cap[:, 1]]
        co[cap[:, 0]], co[cap[:, 1]] = np.where(m, cb, ca), np.where(m, ca, cb)
        for i in np.where(rng.random(ALPHA) < OMEGA)[0]:                 # B5: đổi chỗ 2 vị trí bất kỳ + đổi 1 ô
            if n >= 2:
                x, y = rng.choice(n, 2, replace=False); tt[i, [x, y]] = tt[i, [y, x]]
            s = rng.integers(n); co[i, s] = rng.integers(len(ho_idx[s]))
        tt[0], co[0] = thu_tu[tot], chon_o[tot]                          # B6: giữ con giỏi nhất (elitism)
        thu_tu, chon_o = tt, co
    return b2_fitness(diem(thu_tu, chon_o), D).min(), 0


# ============ THỐNG KÊ (giống mục 5.2 của bài) ============
def mot_viec(args):
    k, sd = args
    pl = PLS[k]; D, ho_idx = ma_tran(pl)
    a, loi = gro_dung_chu(pl, D, ho_idx, seed=sd)
    b, _ = gro_hop_ly(pl, D, ho_idx, seed=sd)
    # Tham chiếu: GA lai + quy hoạch động + 2-opt (bản của nhóm), để biết lời giải tốt cỡ nào
    from gro import gro_dp, build_instance
    ref = gro_dp(*build_instance(pl['ho']), patience=150, polish=True, seed=sd)[0]
    return dict(wave=pl['wave'], seed=sd, so_mon=pl['so_mon'], so_san_pham=len(pl['san_pham']),
                thuc_te=quang_duong_thuc_te(pl['thuc_te']), dung_chu=a, hop_ly=b, tham_chieu=ref, lai_loi=loi)


def in_bang_ket_qua(R):
    # Mỗi wave lấy trung bình 10 seed (giống cách báo 1 con số/list của bài), rồi thống kê trên 51 wave
    W = R.groupby('wave')[['thuc_te', 'dung_chu', 'hop_ly', 'tham_chieu']].mean()
    def mo_ta(x):
        return pd.Series({'Trung bình': x.mean(), 'Độ lệch chuẩn': x.std(), 'Min': x.min(), '25%': x.quantile(.25),
                          '50%': x.median(), '75%': x.quantile(.75), 'Max': x.max(), 'CV (%)': x.std() / x.mean() * 100})
    B = pd.DataFrame({'Thực tế': mo_ta(W.thuc_te), 'GRO đúng chữ': mo_ta(W.dung_chu),
                      'GRO hợp lý': mo_ta(W.hop_ly), 'Tham chiếu (DP+2opt)': mo_ta(W.tham_chieu)})
    print('\n=== BẢNG 5 CỦA NHÓM (đơn vị toạ độ của dataset; mỗi wave = trung bình 10 seed) ===')
    print(B.round(1).to_string())
    for c in ['dung_chu', 'hop_ly', 'tham_chieu']:
        giam = (1 - W[c].mean() / W.thuc_te.mean()) * 100
        t = stats.ttest_rel(W.thuc_te, W[c]); sw = stats.shapiro(W[c])
        print(f'{c:11s}: giảm {giam:6.2f}% so với thực tế | t-test ghép cặp t = {t.statistic:7.3f}, p = {t.pvalue:.2e} '
              f'| Shapiro-Wilk W = {sw.statistic:.4f}, p = {sw.pvalue:.4f}')
    sw = stats.shapiro(W.thuc_te); print(f'thực tế    : Shapiro-Wilk W = {sw.statistic:.4f}, p = {sw.pvalue:.4f}')
    S = R.groupby('wave')[['dung_chu', 'hop_ly']].std().mean()
    print(f'Dao động giữa 10 seed (độ lệch chuẩn trung bình mỗi wave): đúng chữ {S.dung_chu:.1f}, hợp lý {S.hop_ly:.1f}')
    print(f'Số wave GRO đúng chữ TỆ HƠN thực tế: {(W.dung_chu > W.thuc_te).sum()}/{len(W)}; '
          f'hợp lý tệ hơn thực tế: {(W.hop_ly > W.thuc_te).sum()}/{len(W)}')
    print(f'Tổng số lần lai sinh lộ trình lỗi (đúng chữ): {int(R.lai_loi.sum())}')
    B.round(2).to_csv('results/tai_hien_gro2024_bang5.csv')


PLS = chuan_bi_du_lieu()

# ============ CHẠY ============
if __name__ == '__main__':
    t0 = time.time()
    pls = PLS
    print(f'Đã chọn {len(pls)} pick list thật (15-27 món). Ví dụ wave {pls[0]["wave"]}: '
          f'{pls[0]["so_mon"]} món, {len(pls[0]["san_pham"])} sản phẩm khác nhau, '
          f'số ô ứng viên mỗi sản phẩm = {[len(h) for h in pls[0]["ho"]]}')

    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        pl = pls[0]; D, ho_idx = ma_tran(pl)
        print(f'\nWave {pl["wave"]} – quãng đường thực tế (đi theo thứ tự pick list): {quang_duong_thuc_te(pl["thuc_te"])}')
        print('\nGRO ĐÚNG CHỮ:'); kq1, loi = gro_dung_chu(pl, D, ho_idx, seed=0, in_tien_trinh=True)
        print(f'   kết quả cuối: {kq1:.0f}  (số lần lai sinh lộ trình lỗi bị bỏ: {loi})')
        print('\nGRO HỢP LÝ:'); kq2, _ = gro_hop_ly(pl, D, ho_idx, seed=0, in_tien_trinh=True)
        print(f'   kết quả cuối: {kq2:.0f}')
        sys.exit()

    # Mỗi (wave, seed) là một việc độc lập -> chạy song song trên các nhân CPU
    from multiprocessing import Pool
    viec = [(k, sd) for k in range(len(pls)) for sd in range(SO_SEED)]
    with Pool() as pool:
        rows = []
        for j, r in enumerate(pool.imap_unordered(mot_viec, viec)):
            rows.append(r)
            if (j + 1) % 50 == 0:
                print(f'   đã chạy {j + 1}/{len(viec)} ({time.time() - t0:.0f} s)', flush=True)
    R = pd.DataFrame(rows).sort_values(['wave', 'seed'])
    R.to_csv('results/tai_hien_gro2024_chi_tiet.csv', index=False)
    in_bang_ket_qua(R)
