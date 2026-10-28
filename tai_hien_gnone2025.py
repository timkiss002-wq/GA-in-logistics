"""
TÁI HIỆN Gnone et al. (2025) – "Enhancing Warehouse Efficiency: A Study on Storage Allocation
Assignment in a Footwear Manufacturing Company" (IFAC-PapersOnLine 59-10, 2485–2490).

Thí nghiệm của bài: 50 wave sinh ngẫu nhiên (25 SKU/wave, bốc theo nhu cầu lịch sử),
5 trường hợp: Original, Random, Dedicated, Class-Based, Hybrid. Router = GRO.

Router của nhóm = GRO HỢP LÝ (bản đã tái hiện được mức giảm 21% của de Assis et al., 2024),
tham số theo Bảng 2 của Gnone: Np = 1000, G = 1000, T = 3, Pc = 90%, Pm = 8%.

Hai giả định chọn ô khi một SKU nằm ở nhiều ô:
  A  "tu_chon"  : GA tự chọn ô (đúng mô hình Family-TSP / SPRP-SS của tài liệu)
  B  "gan_depot": mỗi SKU chỉ lấy ở ô gần depot nhất (GA chỉ còn tìm thứ tự đi)

Chạy:
  python tai_hien_gnone2025.py kiemtra   -> kiểm tra mô hình khoảng cách so với Dijkstra
  python tai_hien_gnone2025.py demo      -> 1 wave, in quá trình tiến hoá (mỗi 100 thế hệ) cho 4 chính sách
  python tai_hien_gnone2025.py           -> thí nghiệm đầy đủ: 50 wave x 4 chính sách x 2 giả định x 10 seed
  python tai_hien_gnone2025.py thongke   -> chỉ in lại bảng thống kê từ file kết quả đã có
"""
import os, sys, time, collections
import numpy as np, pandas as pd
from numba import njit
from data import load_layout, load_locations, P, size_key
from gro import (pick_point, build_instance, dist_matrix, gro_dp,
                 AISLES, CROSS_X, CROSS_YMIN, DEPOT)

# ---------------- THAM SỐ ----------------
NP, G, T, PC, PM = 1000, 1000, 3, 0.90, 0.08      # Bảng 2 của Gnone et al. (2025)
SO_WAVE, SO_SKU, SO_SEED = 50, 25, 10
SEED_WAVE = 2025
CHINH_SACH = {'Random': 'random', 'Dedicated': 'dedicated', 'Class-Based': 'class', 'Hybrid': 'hybrid'}
GIA_DINH = ['tu_chon', 'gan_depot']
KQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

LOC = load_locations()
PP = {l: pick_point(x, y) for l, x, y in zip(LOC.index, LOC.x, LOC.y)}
TANG_LAY_HANG = set(LOC.index[LOC.z <= 2])          # tầng 1-2 để lấy hàng; tầng 3-4 là kho dự trữ (Gnone, mục 3)


# =====================================================================
# CHUẨN BỊ 1. Họ ô: SKU -> các điểm lấy hàng (tầng 1-2) trong từng chính sách
# =====================================================================
def ho_o(ten_file):
    ho = collections.defaultdict(set)
    for l, slots in load_layout(ten_file).items():
        if l in TANG_LAY_HANG:
            for s in slots:
                ho[s].add(PP[l])
    return {s: sorted(v) for s, v in ho.items()}


# =====================================================================
# CHUẨN BỊ 2. Sinh 50 wave giống tác giả: 25 SKU, xác suất tỉ lệ nhu cầu lịch sử
# =====================================================================
def sinh_wave(HO):
    co = pd.read_csv(P('Customer_Order.csv'), sep=';', encoding='utf-8-sig')
    co['sku'] = list(zip(co.Reference.str.strip(), co['Size (US)'].map(size_key)))
    nhu_cau = co.groupby('sku')['quantity (units)'].sum()
    chung = set.intersection(*[set(h) for h in HO.values()])     # SKU lấy được ở cả 4 chính sách
    nhu_cau = nhu_cau[[s in chung for s in nhu_cau.index]].sort_index()
    p = nhu_cau.to_numpy(float) / nhu_cau.sum()
    rng = np.random.default_rng(SEED_WAVE)
    sku = list(nhu_cau.index)
    waves = [[sku[i] for i in rng.choice(len(sku), SO_SKU, replace=False, p=p)] for _ in range(SO_WAVE)]
    info = dict(sku_hop_le=len(nhu_cau), ti_le_nhu_cau=float(nhu_cau.sum() / co['quantity (units)'].sum()))
    return waves, info


def ung_vien(wave, ho, gia_dinh):
    """Danh sách điểm lấy hàng có thể dùng cho từng SKU. Giả định B: chỉ giữ ô gần depot nhất."""
    cands = []
    for s in wave:
        c = ho[s]
        if gia_dinh == 'gan_depot':
            D1 = build_instance([c])[0][0, 1:]
            c = [c[int(np.argmin(D1))]]
        cands.append(c)
    return cands


def bai_toan(wave, ho, gia_dinh):
    """Trả về (D, start, k) cho router: D là ma trận khoảng cách, chỉ số 0 = depot."""
    return build_instance(ung_vien(wave, ho, gia_dinh))


# =====================================================================
# ROUTER: GRO HỢP LÝ, viết theo đúng 6 bước của GA
# =====================================================================
@njit(cache=True)
def _do_dai(thu_tu, o_chon, D, start):
    """B2 Fitness: depot -> các ô theo thứ tự -> depot."""
    truoc = 0; tong = 0.0
    for t in range(thu_tu.shape[0]):
        s = thu_tu[t]; nut = start[s] + o_chon[s]
        tong += D[truoc, nut]; truoc = nut
    return tong + D[truoc, 0]


@njit(cache=True)
def _ox(bo, me, con, a, b):
    """B4 Crossover OX: giữ đoạn a..b của bố, các SKU còn lại xếp theo thứ tự trong mẹ."""
    n = bo.shape[0]
    da_co = np.zeros(n, np.bool_)
    for i in range(a, b + 1):
        con[i] = bo[i]; da_co[bo[i]] = True
    j = 0
    for t in range(n):
        g = me[t]
        if not da_co[g]:
            while a <= j <= b:
                j += 1
            con[j] = g; j += 1


@njit(cache=True)
def _chon_bo_me(fit, T):
    """B3 Selection: tournament, bốc T cá thể, lấy con ngắn nhất."""
    w = np.random.randint(fit.shape[0])
    for _ in range(T - 1):
        r = np.random.randint(fit.shape[0])
        if fit[r] < fit[w]:
            w = r
    return w


@njit(cache=True)
def gro_hop_ly(D, start, k, Np, G, T, Pc, Pm, seed, ghi_lich_su):
    np.random.seed(seed)
    n = start.shape[0]
    lich_su = np.zeros((G + 1, 2))
    # ---- B1 New generation: mỗi cá thể một thứ tự riêng + một ô ngẫu nhiên cho mỗi SKU
    tt = np.empty((Np, n), np.int64); oc = np.empty((Np, n), np.int64)
    for i in range(Np):
        tt[i] = np.random.permutation(n)
        for s in range(n):
            oc[i, s] = np.random.randint(k[s])
    fit = np.empty(Np)
    tt2 = np.empty_like(tt); oc2 = np.empty_like(oc)
    for g in range(G + 1):
        # ---- B2 Fitness evaluation
        for i in range(Np):
            fit[i] = _do_dai(tt[i], oc[i], D, start)
        bi = np.argmin(fit)
        if ghi_lich_su:
            lich_su[g, 0] = fit[bi]; lich_su[g, 1] = fit.mean()
        if g == G:
            break
        # ---- B6 Offspring: con giỏi nhất chép thẳng sang đời sau (elitism)
        tt2[0] = tt[bi]; oc2[0] = oc[bi]
        for i in range(1, Np):
            p0 = _chon_bo_me(fit, T); p1 = _chon_bo_me(fit, T)          # B3
            if n > 1 and np.random.random() < Pc:                         # B4
                a = np.random.randint(n); b = np.random.randint(n)
                if a > b:
                    a, b = b, a
                _ox(tt[p0], tt[p1], tt2[i], a, b)
                for s in range(n):                                         # ô: trộn đều từ bố và mẹ
                    oc2[i, s] = oc[p0, s] if np.random.random() < 0.5 else oc[p1, s]
            else:
                tt2[i] = tt[p0]; oc2[i] = oc[p0]
            if n > 1 and np.random.random() < Pm:                         # B5: đổi chỗ 2 SKU + đổi 1 ô
                a = np.random.randint(n); b = np.random.randint(n)
                x = tt2[i, a]; tt2[i, a] = tt2[i, b]; tt2[i, b] = x
                s = np.random.randint(n); oc2[i, s] = np.random.randint(k[s])
        tt, tt2 = tt2, tt                                                  # đàn con thay đàn cũ -> quay lại B2
        oc, oc2 = oc2, oc
    return fit[bi], lich_su


def lo_trinh_theo_loi(diem):
    """Cách đi 'theo kinh nghiệm': đi hết lối ngang này rồi sang lối kế tiếp, đổi chiều x mỗi lối (kiểu S-shape).
    Thử đi từ đầu kho và từ cuối kho, lấy cách ngắn hơn. Trả về quãng đường depot -> ... -> depot."""
    def dai(seq):
        p = np.array([DEPOT] + seq + [DEPOT])
        return sum(dist_matrix(p[i:i + 2, 0].copy(), p[i:i + 2, 1].copy(), CROSS_X, CROSS_YMIN)[0, 1] for i in range(len(p) - 1))
    kq = []
    for nguoc in (False, True):
        ys = sorted({q[1] for q in diem}, reverse=nguoc); seq = []
        for i, y in enumerate(ys):
            seq += sorted([q for q in diem if q[1] == y], reverse=(i % 2 == 1))
        kq.append(dai(seq))
    return min(kq)


def original(cands, seed):
    """'Original' (bài không nói cách tính; đây là giả định của nhóm):
    bố trí thực tế của công ty (Random_Storage.csv), không có GA. Mỗi SKU lấy ở một ô bất kỳ trong họ
    (giả định A, trung bình theo seed) hoặc ô gần depot (giả định B), đi theo lối (lo_trinh_theo_loi).
    Trên 51 pick list thật, cách đi theo lối này ngắn hơn lộ trình thật của công ty khoảng 7%, nên Original ở đây
    là ước lượng hơi lạc quan (thận trọng) cho cách công ty đang làm."""
    rng = np.random.default_rng(seed)
    return lo_trinh_theo_loi([c[rng.integers(len(c))] for c in cands])


# =====================================================================
# KIỂM TRA mô hình khoảng cách: so với Dijkstra trên đồ thị lối đi thật
# =====================================================================
def kiem_tra_khoang_cach(so_cap=3000, seed=0):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    pts = sorted(set(PP.values()) | {DEPOT})
    nut = {}
    def id_(p):
        if p not in nut: nut[p] = len(nut)
        return nut[p]
    canh = []
    for a in AISLES:                                   # mỗi lối ngang: nối các điểm liên tiếp trên lối
        xmax = 403. if a <= 511 else 686.
        xs = sorted({p[0] for p in pts if p[1] == a} | {cx for cx, cy in zip(CROSS_X, CROSS_YMIN) if a >= cy and cx <= xmax})
        for x1, x2 in zip(xs, xs[1:]):
            canh.append((id_((x1, a)), id_((x2, a)), x2 - x1))
    for cx, cy in zip(CROSS_X, CROSS_YMIN):            # mỗi lối dọc: nối các giao điểm liên tiếp
        ys = sorted(a for a in AISLES if a >= cy)
        for y1, y2 in zip(ys, ys[1:]):
            canh.append((id_((cx, y1)), id_((cx, y2)), y2 - y1))
    i, j, w = map(np.array, zip(*canh))
    G_ = coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(len(nut), len(nut))).tocsr()
    rng = np.random.default_rng(seed)
    ds = [p for p in pts if p in nut]
    sai = []
    src = [ds[q] for q in rng.choice(len(ds), 60, replace=False)]
    dist = dijkstra(G_, indices=[nut[p] for p in src])
    for r, p in enumerate(src):
        for q in [ds[t] for t in rng.choice(len(ds), so_cap // 60, replace=False)]:
            mh = dist_matrix(np.array([p[0], q[0]]), np.array([p[1], q[1]]), CROSS_X, CROSS_YMIN)[0, 1]
            sai.append(abs(mh - dist[r, nut[q]]))
    print(f'Kiểm tra khoảng cách: {len(sai)} cặp điểm, sai số lớn nhất so với Dijkstra = {max(sai):.6f}')
    return max(sai)


# =====================================================================
# THÍ NGHIỆM
# =====================================================================
def chuan_bi():
    HO = {ten: ho_o(f) for ten, f in CHINH_SACH.items()}
    waves, info = sinh_wave(HO)
    return HO, waves, info


def mot_viec(args):
    w, ten, gd, sd = args
    cands = ung_vien(WAVES[w], HO[ten], gd)
    D, start, k = build_instance(cands)
    t0 = time.time()
    gro, _ = gro_hop_ly(D, start, k, NP, G, T, PC, PM, sd, False)
    t_gro = time.time() - t0
    ref = gro_dp(D, start, k, Np=1000, G=1000, seed=sd, patience=300, polish=True)[0] if sd == 0 else np.nan
    org = original(cands, sd) if ten == 'Random' else np.nan
    return dict(wave=w, chinh_sach=ten, gia_dinh=gd, seed=sd, gro=gro, tham_chieu=ref, original=org,
                so_o_tb=float(np.mean(k)), giay=t_gro)


def thong_ke(R):
    from scipy import stats
    out = []
    p = lambda s='': (print(s), out.append(s))
    thu_tu = ['Original', 'Random', 'Dedicated', 'Class-Based', 'Hybrid']
    bai_bao = {'Original': 1056.90, 'Random': 887.32, 'Dedicated': 726.77, 'Class-Based': 800.55, 'Hybrid': 744.96}
    for gd in GIA_DINH:
        X = R[R.gia_dinh == gd]
        M = X.groupby(['wave', 'chinh_sach']).gro.mean().unstack()           # mỗi wave = trung bình 10 seed
        M['Original'] = X[X.chinh_sach == 'Random'].groupby('wave').original.mean()
        M = M[thu_tu]
        p(f'\n================ GIẢ ĐỊNH {"A: GA tự chọn ô" if gd == "tu_chon" else "B: lấy ở ô gần depot nhất"} ================')
        mt = M.describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
        mt['so_voi_Original_%'] = (mt['mean'] / mt.loc['Original', 'mean'] - 1) * 100
        mt['bai_bao_mean'] = [bai_bao[c] for c in mt.index]
        mt['bai_bao_so_voi_Original_%'] = [(bai_bao[c] / bai_bao['Original'] - 1) * 100 for c in mt.index]
        p('Bảng 3 của nhóm (đơn vị toạ độ dataset; mỗi wave = trung bình 10 seed):')
        p(mt.round(1).to_string())
        p('Thứ hạng (ngắn -> dài), nhóm : ' + ' < '.join(mt.drop('Original')['mean'].sort_values().index))
        p('Thứ hạng (ngắn -> dài), bài báo: Dedicated < Hybrid < Class-Based < Random')
        sd = X.groupby(['wave', 'chinh_sach']).gro.std().groupby('chinh_sach').mean()
        p('Dao động giữa 10 seed (SD trung bình mỗi wave): ' + ', '.join(f'{c} {sd[c]:.1f}' for c in thu_tu[1:]))
        g_ = X.groupby(['wave', 'chinh_sach']).agg(tb=('gro', 'mean'), tot=('gro', 'min'), ref=('tham_chieu', 'min'))
        bks = np.minimum(g_.tot, g_.ref)                                   # lời giải tốt nhất biết được
        gap_tb = ((g_.tb / bks - 1) * 100).groupby('chinh_sach').mean()
        gap_tot = ((g_.tot / bks - 1) * 100).groupby('chinh_sach').mean()
        p('Khoảng cách tới lời giải tốt nhất biết được (BKS = min(GRO 10 seed, DP+2opt)), %:')
        p('   trung bình 10 seed: ' + ', '.join(f'{c} {gap_tb[c]:.2f}' for c in thu_tu[1:]))
        p('   tốt nhất 10 seed  : ' + ', '.join(f'{c} {gap_tot[c]:.2f}' for c in thu_tu[1:]))
        p('Shapiro-Wilk: ' + ', '.join(f'{c} W={stats.shapiro(M[c]).statistic:.3f} p={stats.shapiro(M[c]).pvalue:.3f}' for c in thu_tu))
        f = stats.f_oneway(*[M[c] for c in thu_tu])
        p(f'ANOVA một chiều (như bài báo): F = {f.statistic:.3f}, p = {f.pvalue:.2e}')
        fr = stats.friedmanchisquare(*[M[c] for c in thu_tu])
        p(f'Friedman (ghép cặp theo wave): chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.2e}')
        cap = [(a, b) for i, a in enumerate(thu_tu) for b in thu_tu[i + 1:]]
        pv, rows = [], []
        for a, b in cap:
            d = M[a] - M[b]
            wt = stats.wilcoxon(M[a], M[b])
            r = np.sign(d)[d != 0]; rk = stats.rankdata(np.abs(d[d != 0]))
            rb = (rk[r > 0].sum() - rk[r < 0].sum()) / rk.sum()             # rank-biserial (effect size)
            pv.append(wt.pvalue); rows.append([a, b, d.mean(), wt.pvalue, rb])
        o = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0
        for r_, i in enumerate(o):
            run = max(run, min(1, (m - r_) * pv[i])); adj[i] = run
        p('Wilcoxon ghép cặp + hiệu chỉnh Holm (chênh = A - B; r = rank-biserial):')
        for (a, b, dm, pw, rb), pa in zip(rows, adj):
            p(f'   {a:11s} vs {b:11s}: chênh TB {dm:8.1f} | p_Holm = {pa:.2e} | r = {rb:+.2f}' + ('' if pa < 0.05 else '  (không có ý nghĩa)'))
        M.to_csv(os.path.join(KQ, f'gnone2025_theo_wave_{gd}.csv'))
        mt.to_csv(os.path.join(KQ, f'gnone2025_bang3_{gd}.csv'))
    open(os.path.join(KQ, 'gnone2025_thong_ke.txt'), 'w', encoding='utf-8').write('\n'.join(out))


HO, WAVES, INFO = chuan_bi()

if __name__ == '__main__':
    os.makedirs(KQ, exist_ok=True)
    che_do = sys.argv[1] if len(sys.argv) > 1 else 'chay'
    print(f'{len(WAVES)} wave x {SO_SKU} SKU, bốc theo nhu cầu từ {INFO["sku_hop_le"]} SKU lấy được ở cả 4 chính sách '
          f'(chiếm {INFO["ti_le_nhu_cau"] * 100:.1f}% tổng nhu cầu).')
    if che_do == 'kiemtra':
        kiem_tra_khoang_cach()
    elif che_do == 'demo':
        w = 0
        print('Wave 0:', ', '.join(f'{r}/{s}' for r, s in WAVES[w][:6]), '...')
        for ten in CHINH_SACH:
            for gd in GIA_DINH:
                cands = ung_vien(WAVES[w], HO[ten], gd)
                D, start, k = build_instance(cands)
                kq, ls = gro_hop_ly(D, start, k, NP, G, T, PC, PM, 0, True)
                print(f'\n{ten} | giả định {gd} | số ô ứng viên TB/SKU = {np.mean(k):.1f} | đi theo lối, không GA = {original(cands, 0):.0f}')
                for g in range(0, G + 1, 100):
                    print(f'   thế hệ {g:4d}: giỏi nhất {ls[g, 0]:7.0f} | trung bình đàn {ls[g, 1]:8.1f}')
    elif che_do == 'thongke':
        thong_ke(pd.read_csv(os.path.join(KQ, 'gnone2025_chi_tiet.csv')))
    else:
        from multiprocessing import Pool
        kiem_tra_khoang_cach()
        viec = [(w, ten, gd, sd) for w in range(SO_WAVE) for ten in CHINH_SACH for gd in GIA_DINH for sd in range(SO_SEED)]
        t0 = time.time(); rows = []
        with Pool(int(os.environ.get('SLAP_WORKERS', os.cpu_count()))) as pool:
            for j, r in enumerate(pool.imap_unordered(mot_viec, viec, chunksize=4)):
                rows.append(r)
                if (j + 1) % 200 == 0:
                    print(f'   đã chạy {j + 1}/{len(viec)} ({time.time() - t0:.0f} s)', flush=True)
        R = pd.DataFrame(rows).sort_values(['gia_dinh', 'chinh_sach', 'wave', 'seed'])
        R.to_csv(os.path.join(KQ, 'gnone2025_chi_tiet.csv'), index=False)
        pd.DataFrame({'wave': range(SO_WAVE), 'sku': [' '.join(f'{r}/{s}' for r, s in w) for w in WAVES]}).to_csv(
            os.path.join(KQ, 'gnone2025_waves.csv'), index=False)
        thong_ke(R)
