"""
BƯỚC 1 – Chuẩn bị dữ liệu cho môi trường RL.

Đầu ra: inst.npz (dùng chung cho mọi thí nghiệm)

Quyết định thiết kế (ghi lại để đưa vào mục Method):
  D1. Kho = các ô ở tầng 1–2 (tầng lấy hàng) có toạ độ trong Storage_Location.csv: 1.692 ô × 18 ngăn.
  D2. SKU = (mã giày, cỡ) xuất hiện trong Picking_Wave.csv: 2.454 SKU.
      MI tính theo mã giày (đúng công thức 2 của de Assis 2026 và khớp 0,918 ở kiểm tra trước);
      quãng đường tính theo SKU (người lấy hàng phải lấy đúng mã và cỡ).
  D3. Không dùng nguyên Random_Storage.csv làm trạng thái đầu: vùng trộn (1.093 ô) có cỡ 4–16 phân bố đều
      và KHÔNG có dòng lấy hàng nào rơi vào nó; vùng còn lại là ô thuần 1 mã. Vì vậy trạng thái đầu được
      dựng lại theo đúng cách de Assis 2026 mô tả: kho đầy ngẫu nhiên, mỗi SKU có tồn kho tỷ lệ với nhu cầu,
      rồi giải phóng ~20% sức chứa (tức lấp 80%).
  D4. Wave được sắp theo số wave (khớp 100% thứ tự thời gian qua creationDate). Chia theo thời gian:
      70% wave đầu (05/01–17/07/2023) để huấn luyện và ước lượng nhu cầu; 30% sau để kiểm tra.
  D5. Nhu cầu p_k = tỷ lệ dòng lấy hàng của SKU k trong giai đoạn huấn luyện (không nhìn dữ liệu kiểm tra).
"""
import numpy as np, pandas as pd
from gro import pick_point, _dist, CROSS_X, CROSS_YMIN, DEPOT

DATA = 'data/'
CAP = 18
FILL = 0.80
TRAIN_FRAC = 0.70


def main(seed=0):
    loc = pd.read_csv(DATA + 'Storage_Location.csv')
    loc['originalLocation'] = loc.originalLocation.str.strip()
    loc = loc[loc.z.isin([1, 2])].reset_index(drop=True)
    pp = np.array([pick_point(x, y) for x, y in zip(loc.x, loc.y)])
    d_cell = np.array([_dist(DEPOT[0], DEPOT[1], px, pa, CROSS_X, CROSS_YMIN) for px, pa in pp])

    w = pd.read_csv(DATA + 'Picking_Wave.csv', sep=';', encoding='utf-8-sig')
    w.columns = ['wave', 'ref', 'size', 'qty', 'loc', 'op']
    w['ref'] = w.ref.str.strip()
    w['size'] = w['size'].astype(float).round(1)
    w['sku'] = w.ref + '|' + w['size'].astype(str)
    waves = np.sort(w.wave.unique())
    n_train = int(len(waves) * TRAIN_FRAC)
    train_ids = set(waves[:n_train])

    skus = sorted(w.sku.unique())
    sku_id = {s: i for i, s in enumerate(skus)}
    refs = sorted(w.ref.unique())
    ref_id = {r: i for i, r in enumerate(refs)}
    sku_ref = np.array([ref_id[s.split('|')[0]] for s in skus])

    wt = w[w.wave.isin(train_ids)]
    cnt = wt.sku.value_counts()
    p = np.array([cnt.get(s, 0) for s in skus], float)
    p = (p + 0.5) / (p + 0.5).sum()          # +0,5 để SKU chỉ có ở giai đoạn kiểm tra vẫn có tồn kho

    # tồn kho: tỷ lệ nhu cầu, ít nhất bằng số đơn vị lớn nhất SKU đó từng bị lấy trong 1 wave
    total = int(FILL * CAP * len(loc))
    per_wave_max = w.groupby(['wave', 'sku']).size().groupby('sku').max()
    need = np.array([per_wave_max.get(s, 1) for s in skus]) + 1   # +1: không bao giờ hết hàng giữa wave
    stock = np.maximum(need, np.floor(p * total).astype(int))
    while stock.sum() > total:                 # cắt bớt ở SKU tồn kho lớn nhất cho đúng tổng
        i = np.argmax(stock - need); stock[i] -= 1
    while stock.sum() < total:
        stock[np.argmax(p * total - stock)] += 1

    # đặt ngẫu nhiên từng đơn vị vào ngăn trống (chính sách random của công ty)
    rng = np.random.default_rng(seed)
    slots = np.repeat(np.arange(len(loc)), CAP)
    rng.shuffle(slots)
    units = np.repeat(np.arange(len(skus)), stock)
    rng.shuffle(units)
    init_cells = slots[:len(units)]
    init = np.zeros((len(loc), len(skus)), np.int16)
    np.add.at(init, (init_cells, units), 1)

    # wave -> danh sách SKU (mỗi dòng 1 đơn vị)
    w_sorted = w.sort_values(['wave'])
    wave_ptr = [0]; wave_sku = []
    wave_list = []
    for wid, g in w_sorted.groupby('wave', sort=True):
        wave_sku.extend(sku_id[s] for s in g.sku); wave_ptr.append(len(wave_sku)); wave_list.append(wid)

    np.savez_compressed('inst.npz', cell_name=loc.originalLocation.values, cell_xy=pp, d_cell=d_cell,
                        sku_name=np.array(skus), ref_name=np.array(refs), sku_ref=sku_ref, p=p, stock=stock,
                        init=init, wave_id=np.array(wave_list), wave_ptr=np.array(wave_ptr),
                        wave_sku=np.array(wave_sku), n_train=n_train, cap=CAP)
    print(f'ô: {len(loc)} | SKU: {len(skus)} | mã: {len(refs)} | đơn vị tồn kho: {stock.sum()} '
          f'({stock.sum() / (CAP * len(loc)):.0%} sức chứa)')
    print(f'wave: {len(wave_list)} (huấn luyện {n_train}, kiểm tra {len(wave_list) - n_train})')
    print(f'khoảng cách depot -> ô: min {d_cell.min():.0f}, TB {d_cell.mean():.0f}, max {d_cell.max():.0f}')


if __name__ == '__main__':
    main()
