"""
Tìm nguyên nhân lệch giữa tái hiện của nhóm và de Assis 2026 bằng cách kiểm định từng giả thuyết.
Mỗi giả thuyết là một cách hiểu khác của phần bài viết không rõ. So 4 ràng buộc từ bài:
  C1 heuristic MI 0,830, critical 99,66%   C2 supervised MI 0,884 (kém nhất), ~2.000 hành động
  C3 adaptive MI 0,822 (tốt nhất)          C4 heuristic ~2.500 hành động rồi đứng yên
"""
import sys, itertools, numpy as np, pandas as pd
from deassis2026_repl import load, run, initial, DEFAULT

PAPER = dict(heuristic=(0.830, 99.66, 2500), supervised=(0.884, 99.71, 2000), adaptive=(0.822, 99.48, 2500))


def synthetic(D_real, L=847, seed=99):
    """S0 'derived from the real layout as a fully populated random configuration': 847 ô tầng chính, đổ ngẫu nhiên
    theo tần suất mã và phân phối cỡ của ảnh chụp thật."""
    rng = np.random.default_rng(seed); R, S = D_real['R'], D_real['S']
    n = np.zeros((L, R), np.int32); ns = np.zeros((L, S), np.int32); units = []
    for l in range(L):
        for _ in range(18):
            a = rng.choice(R, p=D_real['freq']); b = rng.choice(S, p=D_real['size_dist'][a])
            n[l, a] += 1; ns[l, b] += 1; units.append((l, a, b))
    D = dict(D_real); D.update(n=n, ns=ns, units=np.array(units), L=L)
    return D


def demand_freq(D):
    """Tần suất mã theo nhu cầu lấy hàng thật (Picking_Wave) thay cho tần suất tồn kho."""
    import pandas as pd
    r = pd.read_csv('data/Random_Storage.csv', dtype=str); cols = [c for c in r.columns if c.startswith('col_')]
    r = r[~r.col_1.str.contains(r'\.')]
    refs = sorted({v.split(';')[0] for v in r[cols].values.ravel()})
    w = pd.read_csv('data/Picking_Wave.csv', sep=';', encoding='utf-8-sig'); w.columns = ['wave', 'ref', 'size', 'q', 'loc', 'op']
    d = w.ref.str.strip().value_counts()
    f = np.array([d.get(x, 0) for x in refs], float) + 1; return f / f.sum()


def order_qty():
    c = pd.read_csv('data/Customer_Order.csv', sep=';', encoding='utf-8-sig'); c.columns = [x.strip() for x in c.columns]
    q = c['quantity (units)'].clip(upper=18).value_counts(normalize=True).sort_index()
    return q.index.values.astype(int), q.values


def score(res):
    """Sai lệch tổng hợp so với bài (MI tuyệt đối + critical/100 + thứ tự)."""
    e = 0
    for m, (mi, cr, ex) in PAPER.items():
        if m in res: e += abs(res[m][0] - mi) + abs(res[m][1] - cr) / 100
    order_ok = all(m in res for m in PAPER) and res['adaptive'][0] < res['heuristic'][0] < res['supervised'][0]
    return e, order_ok


if __name__ == '__main__':
    base = load('all'); syn = synthetic(base)
    qv, qp = order_qty(); dfreq = demand_freq(base)
    H = {
        'H0 chuẩn (1.093 ô thật)': (base, {}),
        'H1 847 ô đổ ngẫu nhiên (đúng quy mô bài)': (syn, {}),
        'H2 SKU = mã × cỡ': (load('all', 'sku'), {}),
        'H3 ngưỡng θ là SÀN: MI sau >= 0,8': (syn, dict(theta_min=0.8)),
        'H4 lô chờ duyệt do người cất ngẫu nhiên': (syn, dict(deferred='operator')),
        'H5 lô hàng về theo nhu cầu lấy hàng': (dict(syn, freq=dfreq), {}),
        'H6 số lượng lô theo đơn hàng (TB 1,86)': (dict(syn, qv=qv, qp=qp), {}),
        'H7 giải phóng bằng cách làm rỗng 20% số ô': (dict(syn, free_mode='cells'), {}),
        'H3+H4': (syn, dict(theta_min=0.8, deferred='operator')),
        'H3+H4+H6': (dict(syn, qv=qv, qp=qp), dict(theta_min=0.8, deferred='operator')),
    }
    rows = []
    for name, (D, ch) in H.items():
        P = dict(DEFAULT); P.update(ch); res = {}
        ini = initial(D, 0, P)
        for mode in PAPER:
            rs = [run(D, mode, s, P, trace_every=0) for s in range(3)]
            res[mode] = (np.mean([r['MI'] for r in rs]), np.mean([r['crit'] for r in rs]), np.mean([r['executed'] for r in rs]))
        e, ok = score(res)
        rows.append(dict(gia_thuyet=name, MI0=ini['MI_full'], **{f'{m}_{k}': res[m][i] for m in PAPER for i, k in enumerate(['MI', 'crit', 'exec'])},
                         sai_lech=e, dung_thu_tu=ok))
        print(f'{name:<42} MI0 {ini["MI_full"]:.3f} | H {res["heuristic"][0]:.3f}/{res["heuristic"][1]:5.1f}%/{res["heuristic"][2]:.0f} '
              f'| S {res["supervised"][0]:.3f}/{res["supervised"][1]:5.1f}%/{res["supervised"][2]:.0f} '
              f'| A {res["adaptive"][0]:.3f}/{res["adaptive"][1]:5.1f}%/{res["adaptive"][2]:.0f} | lệch {e:.3f} | thứ tự {"đúng" if ok else "sai"}', flush=True)
    pd.DataFrame(rows).to_csv('results/deassis2026/gia_thuyet.csv', index=False)
