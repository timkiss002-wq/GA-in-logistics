"""
GA THUẦN – phiên bản để HỌC, không tối ưu tốc độ.
Ý tưởng giống Flappy Bird GA: mỗi "con chim" là một LỘ TRÌNH lấy hàng.
Con nào đi quãng đường ngắn hơn thì "khỏe" hơn, được chọn làm bố mẹ nhiều hơn.

Chạy:   python ga_tu_hoc.py            (4 điểm, in chi tiết từng bước)
        python ga_tu_hoc.py 8          (8 điểm, in tóm tắt từng thế hệ + vẽ hình)
"""
import sys, random, itertools
import pandas as pd

# ---------------------------------------------------------------
# PHẦN 1. KHO HÀNG: lấy toạ độ thật từ dữ liệu
# ---------------------------------------------------------------
from data import load_locations
LOC = load_locations()                       # bảng: mã ô -> (x, y, tầng z)

LOI_NGANG = [-29, 61, 151, 241, 331, 421, 511, 631, 751, 841, 931, 1021,
             1111, 1201, 1291, 1381, 1471]    # các lối đi ngang (toạ độ y)
LOI_DOC = {'LC': 66, 'CC': 403, 'RC': 686}   # các lối đi dọc (toạ độ x); RC chỉ có ở y >= 631
DEPOT = (403, 631)                           # cửa kho: người lấy hàng xuất phát và quay về đây


def diem_dung(o):
    """Người lấy hàng đứng ở lối ngang gần ô nhất, ngay tại toạ độ x của ô."""
    x, y = LOC.loc[o, 'x'], LOC.loc[o, 'y']
    y_loi = min(LOI_NGANG, key=lambda a: abs(a - y))
    return (int(x), int(y_loi))


def khoang_cach(p, q):
    """Đi trong lối, không xuyên kệ.
    Cùng lối ngang -> đi thẳng. Khác lối -> ra một lối dọc, đi dọc, rẽ vào."""
    (x1, y1), (x2, y2) = p, q
    if y1 == y2:
        return abs(x1 - x2)
    cach = []
    for ten, xd in LOI_DOC.items():
        if ten == 'RC' and min(y1, y2) < 631:      # lối dọc RC không xuống tới nửa dưới kho
            continue
        cach.append(abs(x1 - xd) + abs(y1 - y2) + abs(x2 - xd))
    return min(cach)


def do_dai(lo_trinh, diem):
    """Tổng quãng đường: depot -> các ô theo thứ tự -> depot."""
    tong, truoc = 0, DEPOT
    for o in lo_trinh:
        tong += khoang_cach(truoc, diem[o])
        truoc = diem[o]
    return tong + khoang_cach(truoc, DEPOT)


# ---------------------------------------------------------------
# PHẦN 2. BA PHÉP TOÁN CỦA GA
# ---------------------------------------------------------------
def chon_bo_me(dan, diem, k=2):
    """Tournament: bốc ngẫu nhiên k con, giữ con có quãng đường ngắn nhất."""
    ung_vien = random.sample(dan, k)
    return min(ung_vien, key=lambda r: do_dai(r, diem))


def lai_ghep(bo, me):
    """Lai OX: giữ nguyên một đoạn của bố, phần còn lại điền theo thứ tự xuất hiện trong mẹ."""
    n = len(bo)
    a, b = sorted(random.sample(range(n), 2))
    con = [None] * n
    con[a:b + 1] = bo[a:b + 1]                       # đoạn "gen" lấy từ bố
    con_lai = [g for g in me if g not in con]         # các điểm còn thiếu, theo thứ tự của mẹ
    for i in range(n):
        if con[i] is None:
            con[i] = con_lai.pop(0)
    return con, (a, b)


def dot_bien(con, ti_le):
    """Với xác suất ti_le: đổi chỗ 2 điểm ngẫu nhiên."""
    if random.random() < ti_le:
        i, j = random.sample(range(len(con)), 2)
        con = con[:]
        con[i], con[j] = con[j], con[i]
        return con, (i, j)
    return con, None


# ---------------------------------------------------------------
# PHẦN 3. VÒNG ĐỜI CỦA ĐÀN
# ---------------------------------------------------------------
def chay_ga(cac_o, so_ca_the, so_the_he, ti_le_dot_bien, seed, in_chi_tiet):
    random.seed(seed)
    diem = {o: diem_dung(o) for o in cac_o}
    # Thế hệ 0: các lộ trình xáo trộn ngẫu nhiên (đàn chim "ngu ngốc" ban đầu)
    dan = [random.sample(cac_o, len(cac_o)) for _ in range(so_ca_the)]
    lich_su = []
    for g in range(so_the_he + 1):
        dan.sort(key=lambda r: do_dai(r, diem))
        tot_nhat = dan[0]
        lich_su.append(do_dai(tot_nhat, diem))
        if in_chi_tiet:
            print(f'\n===== THẾ HỆ {g} =====')
            for r in dan:
                print('   ', ' -> '.join(r), '=', do_dai(r, diem))
        else:
            tb = sum(do_dai(r, diem) for r in dan) / len(dan)
            print(f'Thế hệ {g:3d}: tốt nhất = {lich_su[-1]:6.0f} | trung bình cả đàn = {tb:7.1f}')
        if g == so_the_he:
            break
        # Tạo đàn mới. Con tốt nhất được giữ nguyên (elitism) để kết quả không bao giờ tệ đi.
        dan_moi = [tot_nhat]
        while len(dan_moi) < so_ca_the:
            bo = chon_bo_me(dan, diem)
            me = chon_bo_me(dan, diem)
            con, (a, b) = lai_ghep(bo, me)
            con2, doi = dot_bien(con, ti_le_dot_bien)
            if in_chi_tiet:
                print(f'  bố {bo} ({do_dai(bo, diem)}) x mẹ {me} ({do_dai(me, diem)})')
                print(f'     giữ đoạn vị trí {a}..{b} của bố -> con {con} ({do_dai(con, diem)})')
                if doi:
                    print(f'     ĐỘT BIẾN đổi chỗ vị trí {doi} -> {con2} ({do_dai(con2, diem)})')
            dan_moi.append(con2)
        dan = dan_moi
    return dan[0], lich_su, diem


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    # Lấy n ô ở tầng 1, rải khắp kho (cố định để lần nào chạy cũng cùng bài toán)
    tang1 = LOC[LOC.z == 1].sort_values(['y', 'x'])
    cac_o = list(tang1.index[::max(1, len(tang1) // n)][:n])
    print('Các ô cần ghé:', {o: diem_dung(o) for o in cac_o}, '| depot', DEPOT)

    if n <= 8:
        # Bài nhỏ: thử HẾT mọi thứ tự để biết đáp án đúng, rồi xem GA có tìm ra không
        diem = {o: diem_dung(o) for o in cac_o}
        tat_ca = sorted((do_dai(list(p), diem), p) for p in itertools.permutations(cac_o))
        print(f'Thử hết {len(tat_ca)} thứ tự: ngắn nhất = {tat_ca[0][0]}, dài nhất = {tat_ca[-1][0]}')

    if n <= 4:
        tot, ls, _ = chay_ga(cac_o, so_ca_the=6, so_the_he=5, ti_le_dot_bien=0.3, seed=1, in_chi_tiet=True)
    else:
        tot, ls, diem = chay_ga(cac_o, so_ca_the=50, so_the_he=60, ti_le_dot_bien=0.1, seed=1, in_chi_tiet=False)
        print('\nChạy lại với 10 seed khác nhau (mỗi seed = một "lứa chim" khác):')
        import io, contextlib
        for s in range(10):
            with contextlib.redirect_stdout(io.StringIO()):
                _, l, _ = chay_ga(cac_o, 50, 60, 0.1, seed=s, in_chi_tiet=False)
            print(f'  seed {s}: thế hệ 0 = {l[0]:.0f} -> thế hệ 60 = {l[-1]:.0f}')
        try:
            import matplotlib.pyplot as plt
            plt.plot(ls, marker='.')
            plt.xlabel('Thế hệ'); plt.ylabel('Quãng đường tốt nhất'); plt.title(f'GA trên {n} ô (seed 1)')
            plt.savefig('ga_hoi_tu.png', dpi=120, bbox_inches='tight'); print('Đã lưu hình ga_hoi_tu.png')
        except ImportError:
            pass
    print('\nLộ trình tốt nhất GA tìm được:', ' -> '.join(tot))
