"""
GA tìm lộ trình lấy hàng, viết đúng theo 6 bước:
  B1 New generation   -> B2 Fitness evaluation -> B3 Selection
  B4 Crossover        -> B5 Mutation           -> B6 Offspring (đàn con) -> quay lại B2
Chạy:  python ga_6_buoc.py
"""
import random
from data import load_locations

LOC = load_locations()
LOI_NGANG = [-29, 61, 151, 241, 331, 421, 511, 631, 751, 841, 931, 1021, 1111, 1201, 1291, 1381, 1471]
LOI_DOC = {'LC': 66, 'CC': 403, 'RC': 686}      # RC chỉ có ở y >= 631
DEPOT = (403, 631)

# ---------------- THAM SỐ ----------------
CAC_O = ['A-24-13', 'G-15-12', 'L-21-11', 'N-09-13']   # các ô cần ghé
SO_CA_THE = 6           # số robot mỗi thế hệ
SO_THE_HE = 5           # số thế hệ
TI_LE_LAI = 1.0         # xác suất lai ghép
TI_LE_DOT_BIEN = 0.3    # xác suất đột biến
K_TOURNAMENT = 2        # bốc k con, lấy con giỏi nhất làm bố/mẹ
SEED = 1


def diem_dung(o):
    x, y = LOC.loc[o, 'x'], LOC.loc[o, 'y']
    return (int(x), min(LOI_NGANG, key=lambda a: abs(a - y)))


DIEM = {o: diem_dung(o) for o in CAC_O}


def khoang_cach(p, q):
    (x1, y1), (x2, y2) = p, q
    if y1 == y2:
        return abs(x1 - x2)
    return min(abs(x1 - xd) + abs(y1 - y2) + abs(x2 - xd)
               for ten, xd in LOI_DOC.items() if not (ten == 'RC' and min(y1, y2) < 631))


# ===== B1. NEW GENERATION: tạo đàn đầu tiên, hoàn toàn ngẫu nhiên =====
def b1_tao_dan_ban_dau():
    return [random.sample(CAC_O, len(CAC_O)) for _ in range(SO_CA_THE)]


# ===== B2. FITNESS EVALUATION: chấm điểm = quãng đường (càng ngắn càng giỏi) =====
def b2_fitness(lo_trinh):
    tong, truoc = 0, DEPOT
    for o in lo_trinh:
        tong += khoang_cach(truoc, DIEM[o])
        truoc = DIEM[o]
    return tong + khoang_cach(truoc, DEPOT)


# ===== B3. SELECTION: tournament - bốc K con ngẫu nhiên, giữ con có fitness tốt nhất =====
def b3_chon(dan, vai):
    ung_vien = random.sample(dan, K_TOURNAMENT)
    chon = min(ung_vien, key=b2_fitness)
    if IN_CAC_BUOC:
        boc = ', '.join(f'hạng {hang(dan, r)} ({b2_fitness(r)})' for r in ung_vien)
        print(f'      B3 Selection ({vai}): bốc {boc} -> chọn hạng {hang(dan, chon)}: {ten(chon)} = {b2_fitness(chon)}')
    return chon


# ===== B4. CROSSOVER (OX): giữ 1 đoạn của bố, phần còn lại theo thứ tự của mẹ =====
def b4_lai(bo, me):
    n = len(bo)
    a, b = sorted(random.sample(range(n), 2))
    con = [None] * n
    con[a:b + 1] = bo[a:b + 1]
    con_lai = [g for g in me if g not in con]
    for i in range(n):
        if con[i] is None:
            con[i] = con_lai.pop(0)
    if IN_CAC_BUOC:
        print(f'      B4 Crossover: giữ vị trí {a + 1}..{b + 1} của bố [{ten(bo[a:b + 1])}], '
              f'các ô còn lại xếp theo thứ tự trong mẹ -> {ten(con)} = {b2_fitness(con)}')
    return con


# ===== B5. MUTATION: với xác suất TI_LE_DOT_BIEN, đổi chỗ 2 ô =====
def b5_dot_bien(con):
    if random.random() < TI_LE_DOT_BIEN:
        i, j = random.sample(range(len(con)), 2)
        con = con[:]
        con[i], con[j] = con[j], con[i]
        if IN_CAC_BUOC:
            print(f'      B5 Mutation: CÓ đột biến, đổi chỗ vị trí {i + 1} và {j + 1} -> {ten(con)} = {b2_fitness(con)}')
    elif IN_CAC_BUOC:
        print('      B5 Mutation: không đột biến lần này')
    return con


# ===== B6. OFFSPRING: gom các con thành đàn mới (giữ lại con giỏi nhất của đàn cũ) =====
def b6_dan_con(dan):
    dan_moi = [dan[0]]                       # elitism: con giỏi nhất sang thẳng đời sau
    if IN_CAC_BUOC:
        print(f'   Robot mới 1: chép nguyên robot hạng 1 (elitism): {ten(dan[0])} = {b2_fitness(dan[0])}')
    while len(dan_moi) < SO_CA_THE:
        if IN_CAC_BUOC:
            print(f'   Robot mới {len(dan_moi) + 1}:')
        bo = b3_chon(dan, 'bố')
        me = b3_chon(dan, 'mẹ')
        con = b4_lai(bo, me) if (TI_LE_LAI >= 1 or random.random() < TI_LE_LAI) else bo[:]
        con = b5_dot_bien(con)
        dan_moi.append(con)
        if IN_CAC_BUOC:
            print(f'      B6 Offspring: thêm vào đàn mới: {ten(con)} = {b2_fitness(con)}')
    return dan_moi


def ten(lo_trinh):
    return ' -> '.join(lo_trinh)


def hang(dan, r):
    return next(i for i, x in enumerate(dan) if x is r) + 1


IN_CAC_BUOC = True      # True: in chi tiết B3-B6 giữa các thế hệ; False: chỉ in danh sách từng thế hệ

if __name__ == '__main__':
    random.seed(SEED)
    dan = b1_tao_dan_ban_dau()                               # B1
    print('B1 New generation: tạo thế hệ 0 bằng cách xáo ngẫu nhiên thứ tự các ô')
    for the_he in range(SO_THE_HE + 1):
        dan.sort(key=b2_fitness)                             # B2: chấm điểm và xếp hạng
        print(f'\n===== THẾ HỆ {the_he} =====' + ('   (B2 Fitness: tính quãng đường, xếp từ ngắn đến dài)' if IN_CAC_BUOC else ''))
        for i, lo_trinh in enumerate(dan):
            print(f'    hạng {i + 1}: ' + ten(lo_trinh) + f' = {b2_fitness(lo_trinh)}')
        if the_he < SO_THE_HE:
            if IN_CAC_BUOC:
                print(f'\n--- Sinh thế hệ {the_he + 1} từ thế hệ {the_he} ---')
            dan = b6_dan_con(dan)                            # B3 -> B4 -> B5 -> B6, rồi quay lại B2
