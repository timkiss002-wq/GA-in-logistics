"""
Bài v7 – các trạng thái đầu S0 thay thế để kiểm tra độ nhạy theo giả định A3 (S0 dựng lại).
  seed1, seed2 : như S0 gốc (đổ ngẫu nhiên 80% + chạy Random 2.000 wave) nhưng hạt ngẫu nhiên khác
  fill70, fill90: như S0 gốc nhưng lấp 70% / 90% sức chứa
  class        : đổ theo lớp (SKU bán chạy vào ô gần depot, mỗi ô lấp 80%), KHÔNG chạy làm nóng
Đầu ra: s0_v7/<tên>.npz (s0)
"""
import os
import numpy as np
from env import PutawayEnv, BASELINES, run

I = dict(np.load('inst.npz', allow_pickle=True)); CAP = int(I['cap']); L, K = I['init'].shape
W0 = 2000


def stock_for(fill):
    p = I['p']; total = int(fill * CAP * L)
    ptr, sk = I['wave_ptr'], I['wave_sku']
    need = np.ones(K, int)
    for w in range(len(ptr) - 1):
        u, c = np.unique(sk[ptr[w]:ptr[w + 1]], return_counts=True); need[u] = np.maximum(need[u], c)
    need = need + 1
    stock = np.maximum(need, np.floor(p * total).astype(int))
    while stock.sum() > total:
        i = np.argmax(stock - need); stock[i] -= 1
    while stock.sum() < total:
        stock[np.argmax(p * total - stock)] += 1
    return stock


def random_fill(stock, seed):
    rng = np.random.default_rng(seed)
    slots = np.repeat(np.arange(L), CAP); rng.shuffle(slots)
    units = np.repeat(np.arange(K), stock); rng.shuffle(units)
    init = np.zeros((L, K), np.int16); np.add.at(init, (slots[:len(units)], units), 1); return init


def class_fill(stock):
    order_l = np.argsort(I['d_cell'])                       # ô gần depot trước
    per = np.full(L, len(np.repeat(np.arange(K), stock)) // L); per[:stock.sum() - per.sum()] += 1
    slots = np.repeat(order_l, per[np.arange(L)])
    units = np.repeat(np.argsort(-I['p']), stock[np.argsort(-I['p'])])   # SKU bán chạy trước, cùng SKU liền nhau
    init = np.zeros((L, K), np.int16); np.add.at(init, (slots, units), 1); return init


def warm(init, seed):
    env = PutawayEnv(I, s0=init)
    run(env, BASELINES['Random'], 0, W0, seed=seed)
    return env.cnt.copy()


if __name__ == '__main__':
    os.makedirs('s0_v7', exist_ok=True)
    st80 = I['stock']
    cases = {'seed1': lambda: warm(random_fill(st80, 1), 124), 'seed2': lambda: warm(random_fill(st80, 2), 125),
             'fill70': lambda: warm(random_fill(stock_for(0.70), 0), 123),
             'fill90': lambda: warm(random_fill(stock_for(0.90), 0), 123),
             'class': lambda: class_fill(st80)}
    for name, f in cases.items():
        fn = f's0_v7/{name}.npz'
        if os.path.exists(fn): continue
        s0 = f(); np.savez_compressed(fn, s0=s0)
        e = PutawayEnv(I, s0=s0)
        print(f'{name}: đơn vị {s0.sum()} ({s0.sum() / (CAP * L):.0%}) | MI {e.MI0:.4f} | D {e.D0:.1f}', flush=True)
