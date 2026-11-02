"""
Kiểm tra lại Mixing Index (MI) của de Assis et al. (2026) trên dữ liệu thật.

Công thức (2) trong bài, cho một ô l:
    MI_l = 1 - max_i n_il / sum_i n_il      (ô rỗng thì MI_l = 0)
    n_il = số đơn vị của mã i trong ô l. Ở đây 1 ngăn (col_k) = 1 đơn vị, mỗi ô 18 ngăn.
MI toàn kho = trung bình MI_l.

Bài báo: 847 ô ở tầng chính, 209 mã, cỡ 4–16, MI = 0,918, 100% ô có MI_l > 0,8 (Bảng 2).
Chạy:  python kiem_tra_MI.py
"""
import pandas as pd

DATA = "../../02_data/order_picking/"
r = pd.read_csv(DATA + "Random_Storage.csv", dtype=str)
loc = pd.read_csv(DATA + "Storage_Location.csv")
tang = dict(zip(loc.originalLocation.str.strip(), loc.z))
COT = [c for c in r.columns if c.startswith("col_")]

def mi_o(hang, theo="ma"):
    """MI của một ô. theo='ma' : i = mã giày; theo='ma_co' : i = mã × cỡ."""
    khoa = [v.split(";")[0] if theo == "ma" else v for v in hang[COT]]
    dem = pd.Series(khoa).value_counts()
    return 1 - dem.max() / dem.sum()

r["tang"] = r.originalLocation.str.strip().map(tang)
r["co_le"] = r.col_1.str.contains(r"\.")          # cỡ ghi 7.5, 11.0 ... (khác kiểu ghi cỡ nguyên 4–16)
r["MI_ma"] = r.apply(mi_o, axis=1)
r["MI_ma_co"] = r.apply(lambda h: mi_o(h, "ma_co"), axis=1)
r["so_ma"] = r[COT].apply(lambda h: len({v.split(';')[0] for v in h}), axis=1)

def in_dong(ten, s):
    print(f"{ten:<48} {len(s):>5} ô | MI(mã) = {s.MI_ma.mean():.4f} | MI(mã×cỡ) = {s.MI_ma_co.mean():.4f}"
          f" | % ô MI>0,8 = {100*(s.MI_ma>0.8).mean():5.1f}")

print("Số mã khác nhau trong 1 ô -> số ô:")
print(r.so_ma.value_counts().sort_index().to_string(), "\n")
in_dong("Toàn bộ file Random_Storage", r)
in_dong("Tầng 1, mọi ô", r[r.tang == 1])
in_dong("Phần cỡ ghi số nguyên (vùng random thật)", r[~r.co_le])
in_dong("  - trong đó tầng 1", r[(~r.co_le) & (r.tang == 1)])
in_dong("  - trong đó tầng 1-2", r[(~r.co_le) & (r.tang <= 2)])
in_dong("Phần cỡ ghi số lẻ (7.5, 11.0...)", r[r.co_le])
print("\nBài báo de Assis 2026: 847 ô | MI = 0,918 | % ô MI>0,8 = 100,0")
