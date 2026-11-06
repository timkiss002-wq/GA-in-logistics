"""
Bài v7 (06/11/2026) – môi trường mở rộng để trả lời phản biện. KHÔNG đổi env.py (kết quả v6 giữ nguyên).

RobustEnv = PutawayEnv + 3 tuỳ chọn và ghi thêm số liệu cho mô hình thời gian:
  pick   : 'nearest' (giả định A4, mặc định) | 'random' (ô bất kỳ đang chứa SKU) | 'top2' (ngẫu nhiên 1 trong 2 ô gần nhất)
  inbound: 'unit'    (giả định A5, mặc định) | 'lot'   (hàng của một wave về theo lô: các đơn vị cùng SKU liền nhau,
                                                         SKU bán chậm trước) | 'delay' (hàng của wave w chỉ về SAU khi
                                                         wave w+1 đã lấy; hết hàng thì bỏ dòng và đếm)
  Ghi thêm cho mỗi wave (self.hist2): số dòng, tổng MI_l của ô được ghé (theo từng dòng, trước khi lấy),
  tổng số mã khác nhau trong ô được ghé, số ô khác nhau, số dòng hết hàng.
"""
import numpy as np
from env import PutawayEnv
from tour_fast import tour_len


class RobustEnv(PutawayEnv):
    def __init__(self, inst, lam=0.5, s0=None, reward='tour', use_tref=False, pick='nearest', inbound='unit', seed=0):
        self.pick_rule, self.inbound = pick, inbound
        self.prng = np.random.default_rng(10_000 + seed)
        super().__init__(inst, lam, s0, reward, use_tref)

    def reset(self, w_start, n_waves, state=None, lam=None):
        self.hist2 = []; self.carry = []; self.stockout = 0
        return super().reset(w_start, n_waves, state, lam)

    def _pick_cell(self, k):
        if self.pick_rule == 'nearest':
            return self.near[k]
        h = np.flatnonzero(self.cnt[:, k] > 0)
        if len(h) == 0:
            return -1
        if self.pick_rule == 'random':
            return int(self.prng.choice(h))
        if self.pick_rule == 'top2':
            o = h[np.argsort(self.d[h])[:2]]
            return int(self.prng.choice(o))
        raise ValueError(self.pick_rule)

    def _next_wave(self):
        while not self.queue and (self.w < self.w_end or self.carry):
            if self.w >= self.w_end:                     # hết đoạn: hàng còn chờ về nốt
                self.queue = list(self.carry); self.carry = []; break
            a, b = self.wave_ptr[self.w], self.wave_ptr[self.w + 1]
            lines = self.wave_sku[a:b]; cells = []; got = []
            s_mi = 0.0; s_mod = 0
            mic = self.mi_cell()
            for k in lines:
                l = self._pick_cell(k)
                if l < 0:
                    self.stockout += 1; continue
                s_mi += mic[l]; s_mod += int((self.cref[l] > 0).sum())
                cells.append((l, k)); got.append(k); self._remove(l, k)
                mic[l] = 1 - self.maxref[l] / self.tot[l] if self.tot[l] > 0 else 0.0
            self.picks.append(cells)
            T = tour_len(self.XY[[c for c, _ in cells]]) if cells else 0.0
            mi = self.mi_mean()
            self.hist.append((mi, self.D, T))
            self.hist2.append((len(cells), s_mi, s_mod, len(set(c for c, _ in cells)), len(lines) - len(cells)))
            got = list(got)
            if self.inbound == 'lot':                     # cùng SKU về liền nhau, SKU bán chậm trước
                got = sorted(got, key=lambda k: (self.p[k], k))
            if self.inbound == 'delay':
                self.queue = list(self.carry); self.carry = got
            else:
                self.queue = got
            self.w += 1

    def done(self):
        return not self.queue and not self.carry and self.w >= self.w_end
