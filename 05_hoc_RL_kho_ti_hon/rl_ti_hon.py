"""
RL TÍ HON để hiểu cách agent học cất hàng – Q-learning trên kho 4 ô.

BÀI TOÁN (thu nhỏ đúng bài của nhóm)
- Kho có 4 ô L0..L3, mỗi ô chứa tối đa 3 đơn vị. Khoảng cách từ depot: 1, 2, 3, 4.
- Có 2 mã hàng: A bán chạy (75% lượt lấy), B bán chậm (25%).
- Hàng về lần lượt: A, B, A, B, A, B (6 lô, mỗi lô 1 đơn vị).
- Mỗi lần hàng về, agent chọn 1 ô còn chỗ để cất (đó là ACTION).

HAI CHỈ SỐ
- MI  : Mixing Index của de Assis 2026. MI_l = 1 - max_i n_il / sum_i n_il, ô rỗng = 0. MI kho = trung bình 4 ô.
- D   : quãng đường lấy hàng kỳ vọng = sum_i (tỷ lệ lấy mã i) x 2 x (khoảng cách tới ô gần nhất có mã i).
        Thay cho GRO trong kho thật (ở đây kho quá nhỏ, không cần GA).
- D0  : D trung bình khi cất ngẫu nhiên (tính bằng cách liệt kê hết mọi cách cất) -> dùng để chuẩn hoá.

HÀM MỤC TIÊU VÀ REWARD
    J = lam * MI + (1 - lam) * D / D0
    reward mỗi bước  r_t = -(J_sau - J_truoc)
Cộng hết reward của 1 episode: -(J_cuoi - J_dau) = -J_cuoi (vì kho ban đầu rỗng, J_dau = 0).
=> Agent tối đa tổng reward  <=>  tối thiểu J cuối cùng. Script sẽ in ra để kiểm chứng điều này.

Q-LEARNING (thuật toán RL đơn giản nhất)
- Q(s, a) = "nếu ở trạng thái s mà chọn ô a thì tổng reward từ đây về sau kỳ vọng là bao nhiêu".
- Sau mỗi bước:  Q(s,a) <- Q(s,a) + alpha * [ r + max_a' Q(s',a') - Q(s,a) ]
- epsilon-greedy: với xác suất epsilon chọn ô ngẫu nhiên (khám phá), còn lại chọn ô có Q lớn nhất (khai thác).

CÁCH CHẠY
    python rl_ti_hon.py                      # chạy 5 mức lambda, in tóm tắt
    python rl_ti_hon.py --xem 0.5            # xem chi tiết từng bước khi học với lambda = 0.5
"""
import itertools, random, sys
from collections import defaultdict

KC = [1, 2, 3, 4]            # khoảng cách từ depot tới ô L0..L3
SUC_CHUA = 3
MA = ["A", "B"]
TY_LE_LAY = {"A": 0.75, "B": 0.25}
HANG_VE = ["A", "B", "A", "B", "A", "B"]
N_O = len(KC)

# ---------------- chỉ số ----------------
def mi_o(o):
    """o = (số A, số B) trong 1 ô"""
    tong = sum(o)
    return 0.0 if tong == 0 else 1 - max(o) / tong

def mi_kho(kho):
    return sum(mi_o(o) for o in kho) / N_O

def quang_duong(kho):
    """D: chỉ tính các mã đã có trong kho"""
    d = 0.0
    for j, m in enumerate(MA):
        o_co = [KC[l] for l in range(N_O) if kho[l][j] > 0]
        if o_co:
            d += TY_LE_LAY[m] * 2 * min(o_co)
    return d

def kho_rong():
    return tuple((0, 0) for _ in range(N_O))

def cat(kho, l, ma):
    j = MA.index(ma)
    o = list(kho[l]); o[j] += 1
    kho = list(kho); kho[l] = tuple(o)
    return tuple(kho)

def o_con_cho(kho):                       # action masking: chỉ cho chọn ô chưa đầy
    return [l for l in range(N_O) if sum(kho[l]) < SUC_CHUA]

# ---------------- liệt kê mọi cách cất (để có D0 và đáp án đúng) ----------------
def liet_ke():
    ket_qua = []
    for chon in itertools.product(range(N_O), repeat=len(HANG_VE)):
        kho, ok = kho_rong(), True
        for l, m in zip(chon, HANG_VE):
            if sum(kho[l]) >= SUC_CHUA: ok = False; break
            kho = cat(kho, l, m)
        if ok: ket_qua.append((chon, kho))
    return ket_qua

TAT_CA = liet_ke()
D0 = sum(quang_duong(k) for _, k in TAT_CA) / len(TAT_CA)

def J(kho, lam):
    return lam * mi_kho(kho) + (1 - lam) * quang_duong(kho) / D0

def dap_an_dung(lam):
    return min(TAT_CA, key=lambda x: J(x[1], lam))

# ---------------- Q-learning ----------------
def hoc(lam, so_episode=4000, alpha=0.2, eps_dau=1.0, eps_cuoi=0.02, seed=0, xem=False):
    random.seed(seed)
    Q = defaultdict(float)                # Q[(trạng thái, ô)], mặc định 0
    for ep in range(1, so_episode + 1):
        eps = max(eps_cuoi, eps_dau - (eps_dau - eps_cuoi) * ep / (0.7 * so_episode))
        in_ra = xem and ep in (1, so_episode)
        if in_ra:
            print(f"\n===== EPISODE {ep}  (epsilon = {eps:.2f}) =====")
        kho, tong_r = kho_rong(), 0.0
        for t, ma in enumerate(HANG_VE):
            s = (t, kho)                  # trạng thái = bước thứ mấy + kho đang chứa gì
            hop_le = o_con_cho(kho)
            if random.random() < eps:
                a, cach = random.choice(hop_le), "khám phá (ngẫu nhiên)"
            else:
                a, cach = max(hop_le, key=lambda l: Q[(s, l)]), "khai thác (Q lớn nhất)"
            kho_moi = cat(kho, a, ma)
            r = -(J(kho_moi, lam) - J(kho, lam))
            # giá trị tốt nhất có thể đạt từ trạng thái mới
            if t + 1 < len(HANG_VE):
                s2 = (t + 1, kho_moi)
                tuong_lai = max(Q[(s2, l)] for l in o_con_cho(kho_moi))
            else:
                tuong_lai = 0.0               # hết hàng về -> kết thúc episode
            q_str = "  ".join(f"L{l}:{Q[(s, l)]:+.3f}" if l in hop_le else f"L{l}: đầy " for l in range(N_O))
            q_cu = Q[(s, a)]
            Q[(s, a)] = q_cu + alpha * (r + tuong_lai - q_cu)
            if in_ra:
                print(f"bước {t}: hàng về {ma} | kho {fmt(kho)}")
                print(f"        Q trước khi chọn: {q_str}")
                print(f"        chọn L{a} [{cach}] | ΔMI={mi_kho(kho_moi)-mi_kho(kho):+.3f}  "
                      f"ΔD/D0={(quang_duong(kho_moi)-quang_duong(kho))/D0:+.3f}  reward={r:+.3f} | Q: {q_cu:+.3f} -> {Q[(s, a)]:+.3f}")
            tong_r += r
            kho = kho_moi
        if in_ra:
            print(f"Kết thúc: kho {fmt(kho)} | tổng reward = {tong_r:+.4f} | -J cuối = {-J(kho, lam):+.4f}  (luôn bằng nhau)")
        if xem and ep % 500 == 0:
            k = chay_tham_lam(Q)
            print(f"  ... episode {ep:5d}, epsilon {eps:.2f}: nếu dừng học ngay bây giờ, agent cho J = {J(k, lam):.4f}")
    return Q

def chay_tham_lam(Q):
    """Sau khi học: luôn chọn ô có Q lớn nhất (không khám phá nữa)."""
    kho = kho_rong()
    for t, ma in enumerate(HANG_VE):
        a = max(o_con_cho(kho), key=lambda l: Q[((t, kho), l)])
        kho = cat(kho, a, ma)
    return kho

def fmt(kho):
    return " ".join("L%d[%s]" % (l, "A" * o[0] + "B" * o[1] + "." * (SUC_CHUA - sum(o))) for l, o in enumerate(kho))

# ---------------- chạy ----------------
if __name__ == "__main__":
    print(f"D0 (cất ngẫu nhiên, trung bình trên {len(TAT_CA)} cách cất hợp lệ) = {D0:.4f}")
    if len(sys.argv) == 3 and sys.argv[1] == "--xem":
        lam = float(sys.argv[2])
        Q = hoc(lam, xem=True)
        k = chay_tham_lam(Q); _, k_dung = dap_an_dung(lam)
        print(f"\nAgent học được : {fmt(k)}  J = {J(k, lam):.4f}")
        print(f"Đáp án đúng    : {fmt(k_dung)}  J = {J(k_dung, lam):.4f}")
        sys.exit()
    print(f"\n{'lambda':>6} | {'kho agent học được':<30} | {'MI':>5} {'D/D0':>5} {'J':>6} | {'J tốt nhất (liệt kê hết)':>24} | đúng?")
    for lam in [0, 0.25, 0.5, 0.75, 1]:
        Q = hoc(lam)
        k = chay_tham_lam(Q); _, k_dung = dap_an_dung(lam)
        dung = abs(J(k, lam) - J(k_dung, lam)) < 1e-9
        print(f"{lam:>6} | {fmt(k):<30} | {mi_kho(k):.3f} {quang_duong(k)/D0:.3f} {J(k, lam):.4f} | {J(k_dung, lam):>24.4f} | {'có' if dung else 'KHÔNG'}")
