# -*- coding: utf-8 -*-
"""
Làm sạch bộ dữ liệu "Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company"
(de Assis et al., 2025, Data in Brief) và tạo các bảng tổng hợp cho phân tích SLAP.

Nguyên tắc: KHÔNG xoá dòng nào; mọi vấn đề được gắn cờ để người dùng tự lọc.
File gốc không bị sửa; kết quả ghi vào thư mục --out.

Chạy (Windows):
    python clean_data.py
    python clean_data.py --data "D:/Tài liệu NCKH =/02_data/order_picking" --out "D:/Tài liệu NCKH =/02_data/clean"
Mặc định: --data = ../02_data/order_picking, --out = ../02_data/clean (tương đối theo vị trí file .py)
Yêu cầu: pip install pandas numpy pyarrow xlsxwriter
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=HERE.parent / "02_data" / "order_picking")
ap.add_argument("--out", type=Path, default=HERE.parent / "02_data" / "clean")
ap.add_argument("--no-excel", action="store_true", help="bỏ qua bước xuất Excel (nhanh hơn)")
args = ap.parse_args()
D, OUT = args.data, args.out
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "csv").mkdir(exist_ok=True)

DEPOT_LABEL = "CC-08"                    # giả định depot (giống wave_33361_replay.py)
TRAIN_END = pd.Timestamp("2023-07-01")   # train: trước 01/07/2023; test: từ 01/07/2023
QUALITY = []                             # báo cáo chất lượng dữ liệu


def q(table, issue, rows, handling):
    QUALITY.append({"bang": table, "van_de": issue, "so_dong_anh_huong": int(rows), "cach_xu_ly": handling})


def strip_cols(df):
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].astype("string").str.strip()
    return df


# ======================================================================
# 1. Cỡ giày: chuẩn hoá 3 hệ mã hoá
# ======================================================================
def norm_size(s):
    """Trả về (size_us, size_system, size_key).
    US  : 1–16 (có thể có .5)               -> giữ nguyên
    x10 : 35–125, chia hết cho 5            -> /10 (vd 105 -> 10.5)
    mm  : >= 200 (chỉ mã PY5UPB, 215–305)   -> không quy đổi; size_key = '260mm'
    """
    if pd.isna(s):
        return np.nan, "missing", pd.NA
    s = float(s)
    if s <= 20:
        return s, "US", f"{s:g}"
    if 30 <= s <= 130 and s % 5 == 0:
        return s / 10, "x10", f"{s / 10:g}"
    if s >= 200:
        return np.nan, "mm", f"{s:g}mm"
    return np.nan, "unknown", f"?{s:g}"


def add_size(df, col):
    t = df[col].map(norm_size)
    df["size_us"] = [a for a, b, c in t]
    df["size_system"] = pd.array([b for a, b, c in t], dtype="string")
    df["size_key"] = pd.array([c for a, b, c in t], dtype="string")
    df["sku"] = (df["ref"] + "|" + df["size_key"]).astype("string")
    return df


# ======================================================================
# 2. Điểm điều hướng + đồ thị lối đi
# ======================================================================
raw_sp = (D / "Support_Points_Navigation.csv").read_text(encoding="utf-8-sig")
sp = pd.DataFrame(re.findall(r"\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\);([A-Z]{2}-\d+)", raw_sp),
                  columns=["x", "y", "z", "label"]).astype({"x": float, "y": float, "z": float})
sp["corridor"] = sp.label.str[:2].map({"LC": "Trái (LC)", "CC": "Giữa (CC)", "RC": "Phải (RC)"})
sp["cross_aisle_no"] = sp.label.str[3:].astype(int)
sp["is_depot"] = sp.label.eq(DEPOT_LABEL)
sp = sp[["label", "corridor", "cross_aisle_no", "x", "y", "z", "is_depot"]].sort_values(["label"]).reset_index(drop=True)
q("Support_Points_Navigation", "Toàn bộ dữ liệu nằm trên 1 dòng", 1, "Tách thành bảng 44 điểm: nhãn, hành lang, x, y, z")
q("Support_Points_Navigation", "Hành lang phải chỉ có RC-08 → RC-17; không có RC-01 → RC-07", 7,
  "Ghi nhận; các mã RC-01…RC-07 xuất hiện trong Picking_Wave được ước lượng toạ độ (xem coord_source)")

PTS = {r.label: (r.x, r.y) for r in sp.itertuples()}
AY = np.array(sorted(set(sp.y)))                                    # y các lối đi ngang
CORR = {x: set(sp.loc[sp.x == x, "y"]) for x in sorted(set(sp.x))}  # hành lang dọc -> các lối ngang nó nối
DEPOT = PTS[DEPOT_LABEL]


def access_y(y):
    y = np.asarray(y, dtype=float)
    return AY[np.abs(y[..., None] - AY).argmin(-1)]


def dist_matrix(x1, y1, x2, y2):
    """Khoảng cách theo lối đi (cùng mô hình với wave_33361_replay.py):
    người lấy hàng đứng trên lối ngang gần nhất; đổi lối ngang qua 1 hành lang dọc (LC/CC/RC).
    Bỏ qua khoảng cách từ lối đi vào ngăn kệ. Đơn vị: đơn vị toạ độ của bộ dữ liệu."""
    x1, x2 = np.asarray(x1, float)[:, None], np.asarray(x2, float)[None, :]
    a1, a2 = access_y(y1)[:, None], access_y(y2)[None, :]
    same = a1 == a2
    best = np.full(np.broadcast(x1, x2).shape, np.inf)
    for cx, ys in CORR.items():
        ok = np.isin(a1, list(ys)) & np.isin(a2, list(ys))
        cand = np.abs(x1 - cx) + np.abs(a1 - a2) + np.abs(x2 - cx)
        best = np.where(ok, np.minimum(best, cand), best)
    return np.where(same, np.abs(x1 - x2), best)


# ======================================================================
# 3. Vị trí lưu kho + ước lượng toạ độ cho vị trí thiếu
# ======================================================================
sl = strip_cols(pd.read_csv(D / "Storage_Location.csv", encoding="utf-8-sig"))
sl = sl.rename(columns={"originalLocation": "location"})[["location", "x", "y", "z"]]
LOC_RE = r"^([A-Z])-(\d{2})-(\d)(\d)$"
p = sl.location.str.extract(LOC_RE)
sl["rack_row"], sl["position"], sl["level"], sl["sub_slot"] = p[0], p[1].astype(int), p[2].astype(int), p[3].astype(int)
q("Storage_Location", "Toạ độ nằm trong chuỗi 'position' (vd '368, 0, 1')", len(sl), "Dùng các cột x, y, z sẵn có; bỏ cột chuỗi")
q("Storage_Location", "Tầng trong mã ô khác cột z", int((sl.level != sl.z).sum()), "Giữ z theo file gốc; cột level lấy từ mã ô")
KNOWN = {r.location: (r.x, r.y, r.z) for r in sl.itertuples()}


def impute(code):
    """Trả về (x, y, z, coord_source, coord_confidence, zone)."""
    if code in KNOWN:
        x, y, z = KNOWN[code]
        return x, y, z, "original", "high", "rack"
    m = re.match(LOC_RE, code)
    if m:
        row, pos, lv, sub = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        zz = lv if lv > 0 else 1
        same_col = sl[(sl.rack_row == row) & (sl.position == pos) & (sl.sub_slot == sub)]
        if len(same_col):            # vd K-25-20: có K-25-30, K-25-40 -> cùng x,y, khác tầng
            r = same_col.iloc[0]
            return r.x, r.y, zz, "imputed_same_column_other_level", "medium", "rack"
        # Vị trí 25/26 nằm ở đầu dãy, sát hành lang giữa (x = 403). Quy tắc kiểm chứng trên dãy K..Q:
        # y(X-25) = y(X-24); y(X-26) = y(X-24) + 32.
        if pos in (25, 26):
            p24 = sl[(sl.rack_row == row) & (sl.position == 24)]
            if len(p24):
                return 403.0, p24.y.iloc[0] + (32 if pos == 26 else 0), zz, "imputed_row_end_rule", "medium", "rack"
            rowy = sl[sl.rack_row == row]
            if len(rowy):            # dãy R không có vị trí 24 -> y của dãy (1440)
                return 403.0, rowy.y.max(), zz, "imputed_row_end_rule", "low", "rack"
        # vd R-01-xx: dãy R không có vị trí 01 -> lấy vị trí 01 của dãy liền kề (Q), cùng y = 1440
        adj = sl[(sl.rack_row == chr(ord(row) - 1)) & (sl.position == pos)]
        if len(adj):
            return adj.x.mean(), adj.y.max(), zz, "imputed_adjacent_row", "low", "rack"
    m = re.match(r"^RC-(\d{2})$", code)
    if m:                            # RC-01..RC-07 không có trong file điểm điều hướng:
        lc = PTS.get(f"LC-{int(m.group(1)):02d}")   # x = hành lang phải (686), y = y của LC-k
        if lc:
            return 686.0, lc[1], 1.0, "imputed_corridor_extrapolation", "low", "off_rack"
    return np.nan, np.nan, np.nan, "unresolved", "none", "off_rack"


# ======================================================================
# 4. Đọc các bảng gốc
# ======================================================================
co = strip_cols(pd.read_csv(D / "Customer_Order.csv", sep=";", encoding="utf-8-sig"))
pw = strip_cols(pd.read_csv(D / "Picking_Wave.csv", sep=";", encoding="utf-8-sig"))
pr = strip_cols(pd.read_csv(D / "Product.csv", sep=";", encoding="utf-8-sig"))
pol_files = {"Random": ("Random_Storage.csv", ",", None), "Class-Based": ("Class_Based_Storage.csv", ";", "ABCCOD"),
             "Dedicated": ("Dedicated_Storage.csv", ";", "XYZCOD"), "Hybrid": ("Hybrid_Storage.csv", ";", "XYZCOD")}
q("Tất cả", "Khoảng trắng thừa (mã khách đệm tới 20 ký tự, mã ô có dấu cách cuối), BOM, dấu phân cách ; và , lẫn lộn",
  len(co), "Cắt khoảng trắng; xuất thống nhất UTF-8 có BOM, dấu phẩy")

# ---------------- Customer_Order ----------------
co = co.rename(columns={"codCustomer": "customer_id", "orderNumber": "order_number", "orderToCollect": "order_line_seq",
                        "Reference": "ref", "Size (US)": "size_raw", "quantity (units)": "qty",
                        "creationDate": "created_at_raw", "waveNumber": "wave", "operator": "operator"})
co["created_at"] = pd.to_datetime(co.created_at_raw, format="%d/%m/%Y %H:%M")
co = add_size(co, "size_raw")
co["is_exact_duplicate"] = co.duplicated(keep="first")
co["is_qty_zero"] = co.qty.eq(0)
co["is_size_missing"] = co.size_raw.isna()
co["wave_in_picking"] = co.wave.isin(set(pw.waveNumber))
co["period"] = np.where(co.created_at < TRAIN_END, "train", "test")
co["year_month"] = co.created_at.dt.strftime("%Y-%m")
q("Customer_Order", "Dòng trùng hoàn toàn (mọi cột giống nhau), tính từ dòng thứ 2", co.is_exact_duplicate.sum(),
  "Giữ; cờ is_exact_duplicate = True (có thể là 2 dòng đơn thật giống hệt nhau)")
q("Customer_Order", "Số lượng = 0", co.is_qty_zero.sum(), "Giữ; cờ is_qty_zero")
q("Customer_Order", "Thiếu cỡ giày", co.is_size_missing.sum(), "Giữ; cờ is_size_missing; size_key để trống")
q("Customer_Order", "Cỡ giày mã hoá hệ ×10 (35–125 nghĩa là 3,5–12,5)", co.size_system.eq("x10").sum(), "size_us = size_raw / 10")
q("Customer_Order", "Cỡ giày hệ 215–305 (có lẽ là mm), chỉ mã PY5UPB", co.size_system.eq("mm").sum(),
  "Không quy đổi sang US (không có bảng quy đổi tin cậy); size_key dạng '260mm'")
q("Customer_Order", "Dòng thuộc wave không có trong Picking_Wave", (~co.wave_in_picking).sum(),
  f"Giữ; cờ wave_in_picking = False ({co.loc[~co.wave_in_picking, 'wave'].nunique()} wave)")
co = co[["customer_id", "order_number", "order_line_seq", "ref", "size_raw", "size_us", "size_system", "size_key", "sku",
         "qty", "created_at", "year_month", "period", "wave", "operator", "wave_in_picking",
         "is_exact_duplicate", "is_qty_zero", "is_size_missing"]]

# Ngày của wave = thời điểm tạo đơn muộn nhất trong wave (wave chỉ được lấy sau khi đơn cuối cùng được tạo)
wave_date = co.groupby("wave").created_at.max()

# ---------------- Danh mục vị trí ----------------
pol_raw = {k: strip_cols(pd.read_csv(D / f, sep=s, encoding="utf-8-sig", dtype=str)) for k, (f, s, c) in pol_files.items()}
all_codes = set(sl.location) | set(pw.locations)
for df in pol_raw.values():
    all_codes |= set(df.iloc[:, 0])
loc = pd.DataFrame([dict(zip(["location", "x", "y", "z", "coord_source", "coord_confidence", "zone"], (c,) + impute(c)))
                    for c in sorted(all_codes)])
p = loc.location.str.extract(LOC_RE)
loc["rack_row"], loc["position"], loc["level"], loc["sub_slot"] = p[0], pd.to_numeric(p[1]), pd.to_numeric(p[2]), pd.to_numeric(p[3])
loc["in_storage_location_file"] = loc.location.isin(KNOWN)
loc["used_in_picking"] = loc.location.isin(set(pw.locations))
loc["picking_units"] = loc.location.map(pw.locations.value_counts()).fillna(0).astype(int)
for k, df in pol_raw.items():
    loc[f"in_{k.replace('-', '_').lower()}_policy"] = loc.location.isin(set(df.iloc[:, 0]))
has = loc.x.notna()
loc["access_aisle_y"] = np.where(has, access_y(loc.y.fillna(0)), np.nan)
loc["dist_to_depot"] = np.nan
loc.loc[has, "dist_to_depot"] = dist_matrix([DEPOT[0]], [DEPOT[1]], loc.loc[has, "x"], loc.loc[has, "y"])[0]
SRC_VI = {"imputed_same_column_other_level": "lấy x, y của cùng cột ở tầng 3–4",
          "imputed_row_end_rule": "ô đầu dãy vị trí 25/26: x = 403, y theo vị trí 24 của dãy",
          "imputed_adjacent_row": "lấy vị trí cùng số của dãy liền kề",
          "imputed_corridor_extrapolation": "RC-k đặt tại x = 686, y = y của LC-k (ngoại suy)",
          "unresolved": "không có căn cứ ước lượng, để trống"}
for src, g in loc.groupby("coord_source"):
    if src != "original":
        q("Location", f"Vị trí thiếu toạ độ → {src} ({SRC_VI[src]})", len(g),
          ", ".join(g.location.head(10)) + (" …" if len(g) > 10 else ""))

# ---------------- Picking_Wave (mức đơn vị) ----------------
pw = pw.rename(columns={"waveNumber": "wave", "reference": "ref", "Size (US)": "size_raw",
                        "quantityToPick (units)": "qty", "locations": "location", "operator": "operator"})
pw = add_size(pw, "size_raw")
pw["unit_seq"] = pw.groupby("wave").cumcount() + 1
pw["wave_date"] = pw.wave.map(wave_date)
pw["period"] = np.where(pw.wave_date < TRAIN_END, "train", "test")
pw = pw.merge(loc[["location", "x", "y", "z", "coord_source", "coord_confidence", "zone"]], on="location", how="left")
q("Picking_Wave", "Mỗi dòng = 1 đôi (quantityToPick luôn = 1)", int(pw.qty.eq(1).sum()),
  "Giữ mức đơn vị; thêm bảng Pick_Lines gộp theo wave × sku × ô")
q("Picking_Wave", "Cỡ giày hệ 215–305 (có lẽ là mm), chỉ mã PY5UPB", pw.size_system.eq("mm").sum(), "Giữ; size_key dạng '260mm'")
q("Picking_Wave", "Dòng lấy tại vị trí đã ước lượng toạ độ", int((~pw.coord_source.isin(["original", "unresolved"])).sum()),
  "x, y theo quy tắc ước lượng; xem coord_source / coord_confidence")
q("Picking_Wave", "Dòng lấy tại vị trí KHÔNG ước lượng được (C-xxx, EX-xx, ZN-COR)", pw.coord_source.eq("unresolved").sum(),
  "Giữ; x, y để trống; phải loại khi tính quãng đường")
pw = pw[["wave", "unit_seq", "wave_date", "period", "operator", "ref", "size_raw", "size_us", "size_system", "size_key", "sku",
         "qty", "location", "zone", "x", "y", "z", "coord_source", "coord_confidence"]]

pick_lines = (pw.groupby(["wave", "wave_date", "period", "operator", "ref", "size_key", "sku", "location", "zone",
                          "coord_source"], dropna=False).size().rename("units").reset_index()
              .merge(loc[["location", "x", "y", "z", "dist_to_depot"]], on="location", how="left"))

# ---------------- Chính sách lưu kho (dạng dài) ----------------
pol_long = []
for k, df in pol_raw.items():
    cls = pol_files[k][2]
    df = df.rename(columns={df.columns[0]: "location"})
    slot_cols = [c for c in df.columns if c not in ("location", "ABCCOD", "XYZCOD")]
    m = df.melt(id_vars=["location"] + ([cls] if cls else []), value_vars=slot_cols, var_name="slot", value_name="cell")
    m["slot"] = m.slot.str.replace("col_", "", regex=False).astype(int)
    m["policy"] = k
    m["class_code"] = m[cls] if cls else pd.NA
    m["is_empty"] = m.cell.isna() | ~m.cell.fillna("").str.contains(";")
    parts = m.cell.fillna(";").str.split(";", n=1, expand=True)
    m["ref"] = parts[0].str.strip().replace("", pd.NA).astype("string")
    m["size_raw"] = pd.to_numeric(parts[1], errors="coerce")
    pol_long.append(m[["policy", "location", "class_code", "slot", "ref", "size_raw", "is_empty"]])
pol_long = pd.concat(pol_long, ignore_index=True)
pol_long = add_size(pol_long, "size_raw")
pol_long = pol_long.merge(loc[["location", "x", "y", "z", "coord_source", "dist_to_depot"]], on="location", how="left")
pol_long["ref_in_product"] = pol_long.ref.isin(set(pr.Reference))
q("Chính sách lưu kho", "README ghi col_k là 'product_code;quantity' nhưng thực tế là 'mã;cỡ giày'", len(pol_long),
  "Đặt tên cột ref + size_raw; chuẩn hoá cỡ như các bảng khác")
q("Chính sách lưu kho", "Dạng 'rộng' 18 cột ngăn, mỗi file một kiểu", len(pol_long),
  "Gộp 4 file thành 1 bảng dạng dài: policy, location, slot 1–18")
for k, g in pol_long.groupby("policy"):
    n_out = g.loc[~g.location.isin(KNOWN), "location"].nunique()
    if n_out:
        q("Chính sách lưu kho", f"{k}: {n_out}/{g.location.nunique()} ô không có trong Storage_Location", n_out * 18,
          "Gán toạ độ ước lượng (xem Location.coord_source)")
    nf = g.loc[g.ref.notna() & ~g.ref_in_product, "ref"].nunique()
    if nf:
        q("Chính sách lưu kho", f"{k}: {nf} mã không có trong Product.csv", int((g.ref.notna() & ~g.ref_in_product).sum()),
          "Giữ; cờ ref_in_product = False")
pol_long = pol_long[["policy", "location", "slot", "class_code", "ref", "size_raw", "size_us", "size_system", "size_key", "sku",
                     "is_empty", "ref_in_product", "x", "y", "z", "coord_source", "dist_to_depot"]]

# ======================================================================
# 5. Bảng tổng hợp
# ======================================================================
def pareto(series):
    c = series.sort_values(ascending=False).cumsum() / max(series.sum(), 1)
    return pd.Series(np.select([c <= 0.80, c <= 0.95], ["A", "B"], "C"), index=c.index)


# --- Nhu cầu theo SKU (mã × cỡ), từ Picking_Wave ---
g = pw.groupby(["sku", "ref", "size_key", "size_system", "period"]).size().unstack("period", fill_value=0)
g = g.reindex(columns=["train", "test"], fill_value=0)
sku = g.rename(columns={"train": "units_train", "test": "units_test"}).reset_index()
sku["units_total"] = sku.units_train + sku.units_test
sku = sku.merge(pw.groupby("sku").agg(waves_total=("wave", "nunique"), n_pick_locations=("location", "nunique")).reset_index(), on="sku")
sku = sku.sort_values(["units_total", "sku"], ascending=[False, True]).reset_index(drop=True)
sku["rank"] = np.arange(1, len(sku) + 1)
sku["cum_share"] = sku.units_total.cumsum() / sku.units_total.sum()
sku["pareto_class"] = np.select([sku.cum_share <= 0.80, sku.cum_share <= 0.95], ["A", "B"], "C")
sku["pareto_class_train"] = pareto(sku.set_index("sku").units_train).reindex(sku.sku).values
sku.loc[sku.units_train.eq(0), "pareto_class_train"] = "không có ở train"
sku["ref_abc_original"] = sku.ref.map(dict(zip(pr.Reference, pr.ABCCOD)))
sku = sku[["rank", "sku", "ref", "size_key", "size_system", "units_train", "units_test", "units_total", "cum_share",
           "pareto_class", "pareto_class_train", "ref_abc_original", "waves_total", "n_pick_locations"]]

# --- Product ---
prod = pr.rename(columns={"Reference": "ref", "ABCCOD": "abc_original", "Sector": "sector"})
pg = pw.groupby(["ref", "period"]).size().unstack("period", fill_value=0).reindex(columns=["train", "test"], fill_value=0)
prod = prod.merge(pg.rename(columns={"train": "units_train", "test": "units_test"}).reset_index(), on="ref", how="left")
prod[["units_train", "units_test"]] = prod[["units_train", "units_test"]].fillna(0).astype(int)
prod["units_total"] = prod.units_train + prod.units_test
agg = pw.groupby("ref").agg(n_sizes=("size_key", "nunique"), waves_total=("wave", "nunique"))
prod["n_sizes"] = prod.ref.map(agg.n_sizes).fillna(0).astype(int)
prod["waves_total"] = prod.ref.map(agg.waves_total).fillna(0).astype(int)
prod["order_units_total"] = prod.ref.map(co.groupby("ref").qty.sum()).fillna(0).astype(int)
prod["abc_recomputed_train"] = prod.ref.map(pareto(prod.set_index("ref").units_train))
prod.loc[prod.units_train.eq(0), "abc_recomputed_train"] = "không có ở train"
prod["abc_changed"] = prod.abc_original.ne(prod.abc_recomputed_train)
prod = prod.sort_values("units_total", ascending=False).reset_index(drop=True)
q("Product", "Chỉ có 1 sector (PF)", len(prod), "Giữ; thêm nhu cầu thực và ABC tính lại trên giai đoạn train (ngưỡng 80/95%)")
q("Product", "Mã có trong Product nhưng không xuất hiện trong Picking_Wave", int(prod.units_total.eq(0).sum()), "Giữ; units = 0")
q("Product", "ABC gốc khác ABC tính lại trên train", int(prod.abc_changed.sum()), "Giữ cả hai cột để so sánh; cờ abc_changed")

# --- Tóm tắt mỗi wave ---
pw["_orig"] = pw.coord_source.eq("original")
pw["_unres"] = pw.coord_source.eq("unresolved")
ws = pw.groupby("wave").agg(wave_date=("wave_date", "first"), period=("period", "first"), operator=("operator", "first"),
                            units=("qty", "size"), n_refs=("ref", "nunique"), n_skus=("sku", "nunique"),
                            n_locations=("location", "nunique"), units_coord_original=("_orig", "sum"),
                            units_unresolved=("_unres", "sum")).reset_index()
pw = pw.drop(columns=["_orig", "_unres"])
ws["units_coord_imputed"] = ws.units - ws.units_coord_original - ws.units_unresolved
ws["n_locations_with_coord"] = ws.wave.map(pw[pw.x.notna()].groupby("wave").location.nunique()).fillna(0).astype(int)
ws["share_units_coord_original"] = ws.units_coord_original / ws.units
ws["has_imputed_location"] = ws.units_coord_imputed > 0
ws["has_unresolved_location"] = ws.units_unresolved > 0
ws["order_units"] = ws.wave.map(co.groupby("wave").qty.sum()).fillna(0).astype(int)
ws["n_orders"] = ws.wave.map(co.groupby("wave").order_number.nunique()).fillna(0).astype(int)
ws["units_diff_order_minus_pick"] = ws.order_units - ws.units
ws["year_month"] = ws.wave_date.dt.strftime("%Y-%m")
ws["weekday"] = ws.wave_date.dt.day_name()
ws = ws[["wave", "wave_date", "year_month", "weekday", "period", "operator", "n_orders", "order_units", "units",
         "units_diff_order_minus_pick", "n_refs", "n_skus", "n_locations", "n_locations_with_coord",
         "units_coord_original", "units_coord_imputed", "units_unresolved", "share_units_coord_original",
         "has_imputed_location", "has_unresolved_location"]]
q("Picking_Wave", "Wave có tổng số đôi khác Customer_Order", int(ws.units_diff_order_minus_pick.ne(0).sum()),
  "Giữ; cột units_diff_order_minus_pick trong Wave_Summary")
q("Picking_Wave", "Wave có ít nhất 1 ô toạ độ ước lượng", int(ws.has_imputed_location.sum()), "Cờ has_imputed_location; dùng cho phân tích độ nhạy")
q("Picking_Wave", "Wave có ít nhất 1 ô không ước lượng được", int(ws.has_unresolved_location.sum()), "Cờ has_unresolved_location")

# --- Ma trận khoảng cách ---
L = loc[loc.x.notna()].reset_index(drop=True)
nodes = pd.concat([pd.DataFrame({"location": ["DEPOT"], "x": [DEPOT[0]], "y": [DEPOT[1]]}), L[["location", "x", "y"]]], ignore_index=True)
DM = dist_matrix(nodes.x, nodes.y, nodes.x, nodes.y).astype("float32")
assert np.isfinite(DM).all(), "Có cặp vị trí không có đường đi"
assert np.allclose(DM, DM.T), "Ma trận không đối xứng"
dm = pd.DataFrame(DM, index=pd.Index(nodes.location, name="from"), columns=nodes.location)

# ======================================================================
# 6. Từ điển dữ liệu
# ======================================================================
DICT = [
    ("Location", "location", "Mã ô kệ (vd M-19-12 = dãy M, vị trí 19, tầng 1, ngăn con 2) hoặc mã khu (RC-01, EX-01…)"),
    ("Location", "x, y, z", "Toạ độ theo đơn vị của bộ dữ liệu (README ghi là mét nhưng chưa được hiệu chỉnh). z = tầng"),
    ("Location", "coord_source", "original = có trong Storage_Location; imputed_same_column_other_level = lấy x, y của cùng cột ở tầng khác; "
                                 "imputed_row_end_rule = ô đầu dãy (vị trí 25/26): x = 403, y = y của vị trí 24 cùng dãy (+32 với vị trí 26), "
                                 "quy tắc khớp đúng trên dãy K→Q; imputed_adjacent_row = lấy vị trí cùng số của dãy liền kề; "
                                 "imputed_corridor_extrapolation = RC-k đặt tại x = 686, y = y của LC-k; unresolved = không có căn cứ ước lượng"),
    ("Location", "coord_confidence", "Mức tin cậy của toạ độ: high / medium / low / none"),
    ("Location", "zone", "rack = ô trên kệ; off_rack = khu ngoài kệ (RC-xx, C-xxx, EX-xx, ZN-COR)"),
    ("Location", "rack_row, position, level, sub_slot", "Các thành phần tách từ mã ô"),
    ("Location", "access_aisle_y", "y của lối đi ngang gần nhất mà người lấy hàng dùng để tiếp cận ô"),
    ("Location", "dist_to_depot", f"Khoảng cách theo lối đi từ depot ({DEPOT_LABEL}) tới ô, đơn vị toạ độ"),
    ("Location", "picking_units, used_in_picking, in_*_policy", "Số đôi từng lấy tại ô; có xuất hiện trong Picking_Wave / trong file chính sách"),
    ("Support_Points", "label, corridor, cross_aisle_no", "Điểm giao lối đi; LC/CC/RC = hành lang dọc trái/giữa/phải; số = lối ngang"),
    ("Support_Points", "is_depot", f"Depot giả định = {DEPOT_LABEL}"),
    ("Customer_Order", "customer_id, order_number, order_line_seq", "Mã khách, số đơn, thứ tự dòng trong đơn"),
    ("Customer_Order", "size_raw / size_us / size_system / size_key", "Cỡ gốc / cỡ US đã chuẩn hoá / hệ mã hoá (US, x10, mm, missing) / khoá cỡ dùng để ghép"),
    ("Customer_Order", "sku", "Khoá mã × cỡ = ref|size_key (vd 8N10W9|10.5; PY5UPB|260mm)"),
    ("Customer_Order", "created_at, year_month, period", f"Thời điểm tạo đơn; tháng; train = trước {TRAIN_END:%d/%m/%Y}, test = từ ngày đó"),
    ("Customer_Order", "wave_in_picking", "Wave của dòng đơn có tồn tại trong Picking_Wave hay không"),
    ("Customer_Order", "is_exact_duplicate / is_qty_zero / is_size_missing", "Cờ chất lượng; True = dòng có vấn đề (không bị xoá)"),
    ("Picking_Wave_units", "wave, unit_seq", "Số wave; thứ tự đơn vị trong wave (mỗi dòng = 1 đôi)"),
    ("Picking_Wave_units", "wave_date, period", "Ngày wave = thời điểm tạo đơn muộn nhất trong wave (từ Customer_Order); train/test"),
    ("Picking_Wave_units", "location, zone, x, y, z, coord_source, coord_confidence", "Ô lấy hàng thực tế và toạ độ (xem Location)"),
    ("Pick_Lines", "units", "Số đôi lấy tại cùng (wave, sku, ô)"),
    ("Wave_Summary", "order_units / units / units_diff_order_minus_pick", "Số đôi trong Customer_Order / trong Picking_Wave / chênh lệch"),
    ("Wave_Summary", "n_refs, n_skus, n_locations, n_locations_with_coord", "Số mã, số mã × cỡ, số ô khác nhau, số ô có toạ độ trong wave"),
    ("Wave_Summary", "units_coord_original / units_coord_imputed / units_unresolved", "Số đôi theo nguồn toạ độ của ô lấy"),
    ("Wave_Summary", "share_units_coord_original", "Tỷ lệ đôi lấy tại ô có toạ độ gốc"),
    ("Wave_Summary", "has_imputed_location / has_unresolved_location", "Wave có ít nhất 1 ô ước lượng / 1 ô không ước lượng được"),
    ("SKU_Demand", "units_train / units_test / units_total", "Số đôi theo giai đoạn (từ Picking_Wave)"),
    ("SKU_Demand", "rank, cum_share, pareto_class", "Hạng theo nhu cầu; tỷ lệ luỹ kế; A ≤ 80%, B ≤ 95%, C còn lại (toàn kỳ)"),
    ("SKU_Demand", "pareto_class_train", "Như trên nhưng chỉ tính trên giai đoạn train"),
    ("SKU_Demand", "waves_total, n_pick_locations", "Số wave có sku; số ô khác nhau từng lấy sku này"),
    ("Product", "abc_original / abc_recomputed_train / abc_changed", "ABC gốc của công ty / ABC tính lại theo nhu cầu train / hai cột khác nhau"),
    ("Product", "units_* / order_units_total / n_sizes / waves_total", "Số đôi lấy theo giai đoạn / số đôi trong đơn hàng / số cỡ / số wave"),
    ("Policy_Long", "policy, location, slot", "Chính sách (Random, Class-Based, Dedicated, Hybrid); ô; ngăn 1–18"),
    ("Policy_Long", "class_code", "ABCCOD (Class-Based) hoặc XYZCOD (Dedicated, Hybrid); trống với Random"),
    ("Policy_Long", "ref_in_product, is_empty", "Mã có trong Product.csv; ngăn trống"),
    ("distance_matrix.parquet", "(ma trận vuông)", "Khoảng cách theo lối đi giữa DEPOT và mọi ô có toạ độ; hàng và cột = location"),
]
dic = pd.DataFrame(DICT, columns=["bang", "cot", "mo_ta"])
qual = pd.DataFrame(QUALITY)
ORDER = ["Tất cả", "Storage_Location", "Support_Points_Navigation", "Location", "Customer_Order", "Picking_Wave", "Chính sách lưu kho", "Product"]
qual = qual.assign(_o=qual.bang.map({k: i for i, k in enumerate(ORDER)})).sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)
qual.insert(0, "stt", np.arange(1, len(qual) + 1))

# ======================================================================
# 7. Xuất CSV / Parquet
# ======================================================================
TABLES = {"Location": loc, "Support_Points": sp, "Product": prod, "SKU_Demand": sku, "Wave_Summary": ws,
          "Pick_Lines": pick_lines, "Customer_Order": co, "Picking_Wave_units": pw, "Policy_Long": pol_long,
          "Data_Quality": qual, "Data_Dictionary": dic}
for name, df in TABLES.items():
    df.to_csv(OUT / "csv" / f"{name}.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M")
dm.to_parquet(OUT / "distance_matrix.parquet")
print("CSV + parquet xong:", {k: len(v) for k, v in TABLES.items()}, "| ma trận", dm.shape)

# ======================================================================
# 8. Xuất Excel định dạng
# ======================================================================
if not args.no_excel:
    FONT, NAVY = "Arial", "#1F3A5F"
    xl_tables = [("Location", loc), ("Support_Points", sp), ("Product", prod), ("SKU_Demand", sku), ("Wave_Summary", ws),
                 ("Pick_Lines", pick_lines), ("Customer_Order", co)]
    path = OUT / "Footwear_Warehouse_Clean.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        wb = xw.book
        base = {"font_name": FONT, "font_size": 10}
        F = {k: wb.add_format({**base, **v}) for k, v in {
            "title": {"bold": True, "font_size": 16, "font_color": NAVY},
            "sub": {"italic": True, "font_color": "#555555"},
            "h": {"bold": True, "font_color": "white", "bg_color": NAVY, "text_wrap": True, "valign": "vcenter"},
            "txt": {"text_wrap": True, "valign": "top"}, "bold": {"bold": True, "valign": "top"},
            "int": {"num_format": "#,##0"}, "intl": {"num_format": "#,##0", "align": "left"},
            "dec": {"num_format": "#,##0.0"}, "pct": {"num_format": "0.0%"},
            "dt": {"num_format": "yyyy-mm-dd hh:mm"}, "def": {}}.items()}

        sh = wb.add_worksheet("00_README"); sh.hide_gridlines(2)
        sh.set_column("A:A", 3); sh.set_column("B:B", 26); sh.set_column("C:C", 100)
        sh.write("B2", "Bộ dữ liệu kho giày — bản đã làm sạch", F["title"])
        sh.write("B3", "Nguồn: de Assis et al. (2025), Data in Brief — Order Picking Dataset from a Warehouse of a Footwear "
                       "Manufacturing Company. Làm sạch bằng 03_code/clean_data.py.", F["sub"])
        info = [("Nguyên tắc", "Không xoá dòng nào. Mọi vấn đề được gắn cờ (cột is_*, coord_source…). Lọc trong Excel / Python / Power BI tuỳ mục đích."),
                ("Depot", f"{DEPOT_LABEL} tại ({DEPOT[0]:g}, {DEPOT[1]:g}) — giả định, giống wave_33361_replay.py"),
                ("Train / test", f"train = wave/đơn trước {TRAIN_END:%d/%m/%Y}; test = từ ngày đó. Ngày wave = thời điểm tạo đơn muộn nhất trong wave."),
                ("Đơn vị khoảng cách", "Đơn vị toạ độ của bộ dữ liệu. README ghi 'mét' nhưng hai ô liền kề cách 16 đơn vị — CHƯA hiệu chỉnh."),
                ("Toạ độ ước lượng", "Các ô/khu thiếu toạ độ được gán toạ độ ước lượng theo quy tắc ở sheet Data_Dictionary (cột coord_source). "
                                     "Đây là GIẢ ĐỊNH: cần nêu trong bài và chạy phân tích độ nhạy (có / không dùng các ô này)."),
                ("Mô hình khoảng cách", "Người lấy hàng đi trên lối ngang gần nhất, đổi lối qua hành lang dọc LC/CC/RC; bỏ qua đoạn từ lối đi vào ngăn kệ."),
                ("Sheet trong file này", "Data_Quality, Data_Dictionary, Location, Support_Points, Product, SKU_Demand, Wave_Summary, Pick_Lines, Customer_Order"),
                ("Chỉ có trong thư mục csv", "Picking_Wave_units.csv (mỗi dòng 1 đôi) và Policy_Long.csv (4 chính sách × ô × 18 ngăn) — quá lớn cho Excel"),
                ("Ma trận khoảng cách", "distance_matrix.parquet: DEPOT + mọi ô có toạ độ. Đọc: pd.read_parquet('distance_matrix.parquet')"),
                ("Đọc CSV bằng pandas", "pd.read_csv(..., dtype={'size_key': str, 'sku': str}) để cỡ '10' và '260mm' cùng là chuỗi"),
                ("Tái tạo", "python clean_data.py --data <thư mục CSV gốc> --out <thư mục kết quả>")]
        for i, (a, b) in enumerate(info):
            sh.write(5 + i, 1, a, F["bold"]); sh.write(5 + i, 2, b, F["txt"])
        r0 = 6 + len(info)
        sh.write(r0, 1, "Số dòng mỗi bảng", F["title"])
        for i, (n, df) in enumerate(TABLES.items()):
            sh.write(r0 + 1 + i, 1, n, F["def"]); sh.write_number(r0 + 1 + i, 2, len(df), F["intl"])

        def put(name, df, widths=None, wrap=()):
            df = df.copy()
            for c in df.columns:
                if str(df[c].dtype) in ("string", "boolean") or df[c].dtype == object:
                    df[c] = df[c].astype(object).where(df[c].notna(), None)
            df.to_excel(xw, sheet_name=name, index=False, startrow=1, header=False)
            w_ = xw.sheets[name]
            nr, nc = df.shape
            w_.add_table(0, 0, max(nr, 1), nc - 1, {"columns": [{"header": c, "header_format": F["h"]} for c in df.columns],
                                                    "style": "Table Style Light 9", "name": "t_" + name})
            w_.freeze_panes(1, 1 if nc > 6 else 0)
            w_.set_row(0, 32)
            for j, c in enumerate(df.columns):
                s = df[c]
                if widths and c in widths:
                    w = widths[c]
                else:
                    sample = s.head(1000).astype(str).str.len()
                    w = min(max(len(c) + 3, int(sample.max()) + 2, 9), 40)
                fmt = F["def"]
                if c in wrap:
                    fmt = F["txt"]
                elif pd.api.types.is_datetime64_any_dtype(s):
                    fmt, w = F["dt"], 17
                elif pd.api.types.is_bool_dtype(s):
                    fmt = F["def"]
                elif pd.api.types.is_integer_dtype(s):
                    fmt = F["int"]
                elif pd.api.types.is_float_dtype(s):
                    fmt = F["pct"] if "share" in c else (F["dec"] if c in ("size_us", "size_raw") else F["int"])
                w_.set_column(j, j, w, fmt)

        put("Data_Quality", qual, {"bang": 24, "van_de": 62, "so_dong_anh_huong": 18, "cach_xu_ly": 80}, wrap=("van_de", "cach_xu_ly"))
        put("Data_Dictionary", dic, {"bang": 22, "cot": 44, "mo_ta": 120}, wrap=("cot", "mo_ta"))
        for n, df in xl_tables:
            put(n, df)
    print("Excel xong:", path)

print(qual.to_string())
