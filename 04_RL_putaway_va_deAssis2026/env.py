"""
BƯỚC 2 – Môi trường mô phỏng cất hàng – lấy hàng (putaway environment).

Một episode chạy qua một dãy wave thật theo thứ tự thời gian. Với mỗi wave:
  (1) LẤY HÀNG: mỗi dòng của wave lấy 1 đơn vị SKU từ ô gần depot nhất đang chứa SKU đó
      (quy tắc B của nhóm). Ghi lại các ô đã ghé để sau này đo bằng GRO.
  (2) NHẬP HÀNG (bổ sung lại): các đơn vị vừa lấy được sản xuất bù và về kho lần lượt.
      Mỗi đơn vị về là MỘT quyết định của agent: chọn ô nào còn chỗ để cất.
  => tồn kho từng SKU không đổi, nhưng VỊ TRÍ thay đổi theo quyết định của agent.

Hàm mục tiêu (xem mục Method):
  J(s) = lam * MI(s)/MI0 + (1-lam) * D(s)/D0
    MI(s): Mixing Index trung bình các ô (de Assis 2026, công thức 2, theo mã giày)
    D(s) : quãng đường lấy hàng kỳ vọng = sum_k p_k * 2 * d(depot, ô gần nhất chứa k)
    MI0, D0: giá trị ở trạng thái đầu (kho ngẫu nhiên) -> ở trạng thái đầu J = 1 với MỌI lam.
  reward mỗi bước = -SCALE * (J(s_{t+1}) - J(s_t))
  => tổng reward cả episode = -SCALE * (J_cuoi - J_dau)   (tổng trượt / telescoping)

  CHẾ ĐỘ reward='tour' (chế độ CHÍNH, dùng sau khi phát hiện proxy D xếp hạng sai so với GRO):
    chi phí wave w:  c_w = lam * MI_w/MI0 + (1-lam) * T_w/T0
      MI_w: MI kho lúc lấy wave w;  T_w: độ dài lộ trình THẬT của wave w (NN + 2-opt, sai khác GRO ~0,8%)
      T0  : T trung bình của chính sách Random từ S0 trên wave huấn luyện
    reward = -(c_w - 1) nhận ngay khi wave w được lấy (0 ở các bước khác)
    => tổng reward = -sum_w (c_w - 1) = -N*(J - 1),  J = trung bình c_w  => tối đa reward <=> min J
"""
import json, os
import numpy as np
from tour_fast import tour_len

SCALE = 1000.0   # chỉ đổi thang đo reward, không đổi chính sách tối ưu
T0 = json.load(open('t0.json'))['T0'] if os.path.exists('t0.json') else None
TREF = np.load('tref.npy') if os.path.exists('tref.npy') else None      # chỉ để giảm nhiễu reward khi huấn luyện


class PutawayEnv:
    N_FEAT = 12

    def __init__(self, inst, lam=0.5, s0=None, reward='tour', use_tref=False):
        self.reward_mode = reward; self.use_tref = use_tref and TREF is not None
        self.XY = inst['cell_xy']
        self.I = inst
        self.cap = int(inst['cap'])
        self.d = inst['d_cell'].astype(np.float64)
        self.dmax = self.d.max()
        self.sku_ref = inst['sku_ref']
        self.p = inst['p']
        self.L, self.K = inst['init'].shape
        self.R = self.sku_ref.max() + 1
        self.wave_ptr = inst['wave_ptr']; self.wave_sku = inst['wave_sku']
        self.lam = lam
        self.p_rank = np.argsort(np.argsort(-self.p)) / self.K   # 0 = bán chạy nhất
        self.d_rank = np.argsort(np.argsort(self.d)) / self.L     # 0 = gần depot nhất
        self.s0 = inst['init'] if s0 is None else s0      # trạng thái đầu chuẩn (kho ngẫu nhiên ổn định)
        self.set_state(self.s0)
        self.MI0 = self.mi_mean(); self.D0 = self.D
        self.J0 = 1.0

    # ---------------- trạng thái ----------------
    def set_state(self, cnt):
        self.cnt = cnt.astype(np.int16).copy()
        self.cref = np.zeros((self.L, self.R), np.int16)
        for r in range(self.R):
            self.cref[:, r] = self.cnt[:, self.sku_ref == r].sum(1)
        self.tot = self.cnt.sum(1).astype(np.int64)
        self.maxref = self.cref.max(1).astype(np.int64)
        self.heat = self.cnt @ self.p
        self.near = np.full(self.K, -1)
        self.dnear = np.full(self.K, np.inf)
        for k in range(self.K):
            self._refresh_near(k)
        self.D = float(np.sum(self.p * 2 * self.dnear))

    def _refresh_near(self, k):
        h = np.flatnonzero(self.cnt[:, k] > 0)
        if len(h):
            i = h[np.argmin(self.d[h])]; self.near[k] = i; self.dnear[k] = self.d[i]
        else:
            self.near[k] = -1; self.dnear[k] = np.inf

    def mi_cell(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(self.tot > 0, 1 - self.maxref / np.maximum(self.tot, 1), 0.0)

    def mi_mean(self):
        return float(self.mi_cell().mean())

    def J(self, lam=None):
        lam = self.lam if lam is None else lam
        return lam * self.mi_mean() / self.MI0 + (1 - lam) * self.D / self.D0

    def _add(self, l, k):
        r = self.sku_ref[k]
        self.cnt[l, k] += 1; self.cref[l, r] += 1; self.tot[l] += 1
        if self.cref[l, r] > self.maxref[l]: self.maxref[l] = self.cref[l, r]
        self.heat[l] += self.p[k]
        if self.d[l] < self.dnear[k]:
            self.D += self.p[k] * 2 * ((self.d[l]) - (self.dnear[k] if np.isfinite(self.dnear[k]) else 0))
            self.near[k] = l; self.dnear[k] = self.d[l]

    def _remove(self, l, k):
        r = self.sku_ref[k]
        self.cnt[l, k] -= 1; self.cref[l, r] -= 1; self.tot[l] -= 1
        self.maxref[l] = self.cref[l].max()
        self.heat[l] -= self.p[k]
        if self.cnt[l, k] == 0 and self.near[k] == l:
            old = self.dnear[k]; self._refresh_near(k)
            self.D += self.p[k] * 2 * (self.dnear[k] - old)

    # ---------------- vòng mô phỏng ----------------
    def reset(self, w_start, n_waves, state=None, lam=None):
        if lam is not None: self.lam = lam
        self.set_state(self.s0 if state is None else state)
        self.w = w_start; self.w_end = w_start + n_waves
        self.queue = []; self.picks = []; self.hist = []; self.pending = 0.0          # picks: danh sách ô đã ghé cho mỗi wave
        self._next_wave()
        self.pending = 0.0            # wave đầu tiên đã lấy trước mọi quyết định -> không tính cho agent
        self.Jprev = self.J(); self.Jstart = self.Jprev
        return self.obs()

    def _next_wave(self):
        while not self.queue and self.w < self.w_end:
            a, b = self.wave_ptr[self.w], self.wave_ptr[self.w + 1]
            lines = self.wave_sku[a:b]; cells = []
            for k in lines:
                l = self.near[k]; cells.append((l, k)); self._remove(l, k)
            self.picks.append(cells)
            T = tour_len(self.XY[[c for c, _ in cells]])
            mi = self.mi_mean()
            self.hist.append((mi, self.D, T))   # trạng thái kho + độ dài lộ trình lúc wave này được lấy
            if T0 is not None:
                Tc = T - TREF[self.w] + T0 if self.use_tref else T
                self.pending += -(self.lam * mi / self.MI0 + (1 - self.lam) * Tc / T0 - 1)
            self.queue = list(lines)
            self.w += 1

    def done(self):
        return not self.queue

    def mask(self):
        return self.tot < self.cap

    def obs(self):
        """Đặc trưng của TỪNG Ô cho đơn vị đang chờ cất (k = SKU, r = mã giày)."""
        k = self.queue[0]; r = self.sku_ref[k]
        tot, mx, cr = self.tot, self.maxref, self.cref[:, r].astype(np.int64)
        mi_b = self.mi_cell()
        mi_a = 1 - np.maximum(mx, cr + 1) / (tot + 1)
        dnear = self.dnear[k] if np.isfinite(self.dnear[k]) else self.dmax
        gain = np.maximum(0.0, dnear - self.d)
        g = np.ones(self.L)
        F = np.stack([
            (self.cap - tot) / self.cap,                 # 0 chỗ trống
            self.cnt[:, k] / self.cap,                   # 1 đã có cùng SKU
            cr / self.cap,                               # 2 đã có cùng mã giày
            mi_b,                                        # 3 MI ô trước khi cất
            (mi_a - mi_b) * 10,                          # 4 ΔMI ô nếu cất vào
            self.d / self.dmax,                          # 5 khoảng cách tới depot
            gain / self.dmax,                            # 6 rút ngắn khoảng cách tới SKU này bao nhiêu
            self.heat * self.L / 50,                     # 7 "độ nóng" của ô (tổng nhu cầu hàng đang chứa)
            self.d_rank,                                 # 8 thứ hạng khoảng cách
            g * (1 - self.p_rank[k]),                    # 9 SKU này bán chạy cỡ nào (1 = nhất)
            g * dnear / self.dmax,                       # 10 ô gần nhất hiện có SKU này
            g * self.lam,                                # 11 lambda (agent dùng chung cho mọi lambda)
        ], 1).astype(np.float32)
        return F

    def step(self, l):
        k = self.queue.pop(0)
        assert self.tot[l] < self.cap
        self._add(l, k)
        self._next_wave()
        Jn = self.J()
        if self.reward_mode == 'tour':
            r = self.pending; self.pending = 0.0
        else:
            r = -SCALE * (Jn - self.Jprev)
        self.Jprev = Jn
        return r, self.done()


# ---------------- chính sách cơ sở (baseline) ----------------
def pol_random(env, rng):
    return rng.choice(np.flatnonzero(env.mask()))

def pol_nearest(env, rng):
    """Closest open location: ô trống gần depot nhất."""
    m = env.mask(); d = np.where(m, env.d, np.inf); return int(np.argmin(d))

def pol_turnover(env, rng):
    """Theo vòng quay (dạng động của class-based): SKU bán chạy thứ q% -> ô trống có thứ hạng khoảng cách gần q% nhất."""
    m = env.mask(); k = env.queue[0]
    target = env.p_rank[k]
    sc = np.where(m, np.abs(env.d_rank - target), np.inf); return int(np.argmin(sc))

def pol_deassis(env, rng, eps=0.05):
    """Heuristic de Assis 2026 (công thức 5): min ΔMI_l - eps*1{đã có cùng cỡ... (ở đây: cùng SKU)}."""
    m = env.mask(); F = env.obs()
    sc = F[:, 4] / 10 - eps * (F[:, 1] > 0)
    sc = np.where(m, sc + 1e-9 * rng.random(env.L), np.inf); return int(np.argmin(sc))

def pol_greedyJ(env, rng):
    """Tham lam theo J: chọn ô làm J giảm nhiều nhất NGAY lập tức (không nhìn tương lai)."""
    m = env.mask(); F = env.obs(); k = env.queue[0]; lam = env.lam
    dMI = (F[:, 4] / 10) / env.L / env.MI0
    dD = -env.p[k] * 2 * F[:, 6] * env.dmax / env.D0
    sc = lam * dMI + (1 - lam) * dD
    sc = np.where(m, sc + 1e-12 * rng.random(env.L), np.inf); return int(np.argmin(sc))

class DeAssis2026:
    """
    Ba chế độ của de Assis et al. (2026), Algorithm 1, đặt vào môi trường của nhóm (mỗi đơn vị về = 1 lô, u = 1).
    Tham số giống deassis2026_repl.py: ω = 0,1 chọn ngẫu nhiên trong top-5; ε = 0,01; ϖ = 0,05; ϱ = 0,01;
    w0 = [1, 1, 1, 1, 1, -1]; r_t = -ΔMI của ô đã chọn.
    Khác thiết lập gốc: ở đây hàng PHẢI được cất (dòng chảy nhập – xuất liên tục), nên lô supervised chuyển
    cho người duyệt được người vận hành cất theo cách hiện tại của công ty (ô trống ngẫu nhiên).
    Thành phần "cỡ" = cỡ giày của SKU; "cỡ có ở ô" = ô đang chứa bất kỳ đôi nào cùng cỡ.
    """
    def __init__(self, mode, eps=0.01, omega=0.1, topk=5, lr=0.05, reg=0.01, w0=(1, 1, 1, 1, 1, -1),
                 theta_min=None, theta_max=None):
        # theta_min/theta_max: CÁCH ĐỌC 2 (results/deassis2026/nguyen_nhan_lech.md) – chỉ cất vào ô có
        # theta_min <= MI_l sau <= theta_max; không có ô nào hợp lệ -> người vận hành cất ngẫu nhiên.
        self.mode, self.eps, self.omega, self.topk, self.lr, self.reg = mode, eps, omega, topk, lr, reg
        self.theta_min, self.theta_max = theta_min, theta_max; self.no_cell = 0
        self.w0 = np.array(w0, float); self.w = self.w0.copy(); self.size_of = None
        self.deferred = 0

    def reset(self):
        self.w = self.w0.copy(); self.deferred = 0; self.no_cell = 0

    def __call__(self, env, rng):
        if self.size_of is None:
            sz = np.array([float(s.split('|')[1]) for s in env.I['sku_name']])
            self.size_of = np.unique(sz, return_inverse=True)[1]
            self.size_onehot = np.eye(self.size_of.max() + 1, dtype=np.int16)[self.size_of]   # K × S
        k = env.queue[0]; r = env.sku_ref[k]; m = env.mask()
        tot, mx, cr = env.tot, env.maxref, env.cref[:, r].astype(np.int64)
        mi_b = env.mi_cell(); mi_a = 1 - np.maximum(mx, cr + 1) / (tot + 1); dmi = mi_a - mi_b
        size_ok = (env.cnt[:, self.size_of == self.size_of[k]].sum(1) > 0).astype(float)
        disp = (cr > 0).sum() / env.L
        X = np.stack([-dmi, (cr > 0).astype(float), size_ok, (env.cap - tot) / env.cap, 1 - mi_b, np.full(env.L, disp)], 1)
        U = X @ self.w if self.mode == 'adaptive' else -dmi + self.eps * size_ok
        cand = np.flatnonzero(m)
        if self.theta_min is not None or self.theta_max is not None:
            ok = m & (mi_a >= (self.theta_min or -1)) & (mi_a <= (self.theta_max or 2))
            if not ok.any():
                self.no_cell += 1; return int(rng.choice(cand))           # không có ô hợp lệ -> người cất
            cand = np.flatnonzero(ok)
        order = cand[np.argsort(-U[cand])]
        l = order[0]
        if rng.random() < self.omega:
            l = rng.choice(order[:self.topk])
        if self.mode == 'supervised' and dmi[l] > 1e-12:
            self.deferred += 1
            return int(rng.choice(np.flatnonzero(m)))         # người vận hành cất ngẫu nhiên
        if self.mode == 'adaptive':
            xb = X[l] / (np.linalg.norm(X[l]) + 1e-12)
            self.w = self.w - self.lr * (2 * (self.w @ xb - (-dmi[l])) * xb + self.reg * self.w)
        return int(l)


BASELINES = {'Random': pol_random, 'Nearest': pol_nearest, 'Turnover': pol_turnover,
             'deAssis': pol_deassis, 'GreedyJ': pol_greedyJ}
DA_MODES = {'DA_heuristic': 'heuristic', 'DA_supervised': 'supervised', 'DA_adaptive': 'adaptive'}
DA2_MODES = {'DA2_heuristic': 'heuristic', 'DA2_supervised': 'supervised', 'DA2_adaptive': 'adaptive'}   # cách đọc 2
DA2_THETA = (0.82, 0.88)


def run(env, policy, w_start, n_waves, seed=0, state=None, lam=None, trace_every=0):
    rng = np.random.default_rng(seed)
    env.reset(w_start, n_waves, state, lam)
    tot_r = 0.0; tr = []
    t = 0
    while not env.done():
        r, _ = env.step(policy(env, rng)); tot_r += r; t += 1
        if trace_every and t % trace_every == 0:
            tr.append((t, env.mi_mean(), env.D / env.D0))
    h = np.array(env.hist)
    return dict(MI=env.mi_mean(), D_ratio=env.D / env.D0, J=env.J(), J_start=env.Jstart, sum_r=tot_r, trace=tr,
                picks=env.picks, MI_avg=h[:, 0].mean(), Dr_avg=h[:, 1].mean() / env.D0, T_avg=h[:, 2].mean(), hist=h)
