# -*- coding: utf-8 -*-
"""
Tai tao wave picking: map wave -> o ke theo 4 chinh sach luu kho (Random, Class-Based,
Dedicated, Hybrid) + vi tri goc cua cong ty, roi tinh tuyen lay hang ngan nhat tren do thi loi di.

Gia dinh: depot = CC-08; chon o gan depot nhat, uu tien tang 1-2; bo qua khoang cach
tu loi di vao ngan ke (picker dung trong loi di).

Chay (Windows):   python wave_33361_replay.py 33361
Chay voi thu muc data khac:  python wave_33361_replay.py 33361 --data "D:/duong/dan/toi/data"
Yeu cau:  pip install pandas numpy
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------- Cau hinh ----------------
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "02_data" / "order_picking"   # tuong doi theo vi tri file .py
EXACT_MAX = 14   # <= so o nay: Held-Karp (toi uu); lon hon: Nearest Neighbor + 2-opt (heuristic)

ap = argparse.ArgumentParser()
ap.add_argument("wave", nargs="?", type=int, default=33361, help="waveNumber can tai tao")
ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="thu muc chua cac file CSV")
args = ap.parse_args()
D = args.data
if not (D / "Picking_Wave.csv").exists():
    sys.exit(f"[LOI] Khong tim thay du lieu tai: {D}\n      Dung --data de chi dung thu muc chua cac file CSV.")

# ---------------- Do thi loi di ----------------
s = (D / "Support_Points_Navigation.csv").read_text(encoding="utf-8-sig")
pts = {lab: (float(x), float(y))
       for x, y, z, lab in re.findall(r"\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\);([A-Z]{2}-\d+)", s)}
AY = sorted(set(y for x, y in pts.values()))                 # toa do y cac loi di ngang
CX = sorted(set(x for x, y in pts.values()))                 # toa do x cac hanh lang doc
corr = {x: sorted(y for (xx, y) in pts.values() if xx == x) for x in CX}
print(f"{len(pts)} diem dieu huong | {len(AY)} loi di | hanh lang:",
      {x: (min(v), max(v)) for x, v in corr.items()})

sl = pd.read_csv(D / "Storage_Location.csv", encoding="utf-8-sig")
sl["originalLocation"] = sl.originalLocation.str.strip()
XY = {r.originalLocation: (r.x, r.y, r.z) for r in sl.itertuples()}
DEPOT = pts["CC-08"]


def access(p):
    x, y = p[0], p[1]
    return x, min(AY, key=lambda a: abs(a - y))


def dist(p, q):
    (x1, a1), (x2, a2) = access(p), access(q)
    if a1 == a2:
        return abs(x1 - x2)
    best = float("inf")
    for cx, ys in corr.items():
        if a1 in ys and a2 in ys:
            best = min(best, abs(x1 - cx) + abs(a1 - a2) + abs(x2 - cx))
    return best


# ---------------- TSP ----------------
def _held_karp(d, N):
    dp = {(1 << k, k): (d[0][k + 1], None) for k in range(N)}
    for m in range(1, 1 << N):
        for k in range(N):
            if (m, k) not in dp:
                continue
            c = dp[(m, k)][0]
            for j in range(N):
                if m >> j & 1:
                    continue
                key, v = (m | 1 << j, j), c + d[k + 1][j + 1]
                if key not in dp or v < dp[key][0]:
                    dp[key] = (v, k)
    full = (1 << N) - 1
    last = min(range(N), key=lambda k: dp[(full, k)][0] + d[k + 1][0])
    tot = dp[(full, last)][0] + d[last + 1][0]
    order, m, k = [], full, last
    while k is not None:
        order.append(k)
        pk = dp[(m, k)][1]
        m ^= 1 << k
        k = pk
    return tot, order[::-1]


def _nn_2opt(d, N):
    tour, left = [0], set(range(1, N + 1))
    while left:
        nxt = min(left, key=lambda j: (d[tour[-1]][j], j))
        tour.append(nxt); left.remove(nxt)
    tour.append(0)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                if d[tour[i-1]][tour[j]] + d[tour[i]][tour[j+1]] < d[tour[i-1]][tour[i]] + d[tour[j]][tour[j+1]] - 1e-9:
                    tour[i:j + 1] = reversed(tour[i:j + 1]); improved = True
    tot = sum(d[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))
    return tot, [t - 1 for t in tour[1:-1]]


def tsp(nodes):
    """Tra ve (tong quang duong, thu tu chi so nodes). Depot o dau va cuoi."""
    if not nodes:
        return 0.0, []
    P = [DEPOT] + nodes
    d = [[dist(a, b) for b in P] for a in P]
    N = len(nodes)
    return _held_karp(d, N) if N <= EXACT_MAX else _nn_2opt(d, N)


# ---------------- Chinh sach luu kho ----------------
def load_policy(fn, sep):
    df = pd.read_csv(D / fn, sep=sep, encoding="utf-8-sig", dtype=str)
    m = {}
    for r in df.itertuples(index=False):
        r = list(r); loc = str(r[0]).strip()
        for v in r[-18:]:
            if isinstance(v, str) and ";" in v:
                ref, sz = v.split(";")
                m.setdefault((ref.strip(), float(sz)), set()).add(loc)
                m.setdefault((ref.strip(), None), set()).add(loc)
    return m, df


pols = {"Random": ("Random_Storage.csv", ","), "Class-Based": ("Class_Based_Storage.csv", ";"),
        "Dedicated": ("Dedicated_Storage.csv", ";"), "Hybrid": ("Hybrid_Storage.csv", ";")}
PM = {k: load_policy(*v) for k, v in pols.items()}
for k, (m, df) in PM.items():
    print(f"{k:12s} {df.shape}  ty le o co toa do: {df.iloc[:, 0].str.strip().isin(XY).mean():.3f}")

# ---------------- Tai tao wave ----------------
W = args.wave
pw = pd.read_csv(D / "Picking_Wave.csv", sep=";", encoding="utf-8-sig")
pw["locations"] = pw.locations.str.strip()
w = pw[pw.waveNumber == W]
if w.empty:
    sys.exit(f"[LOI] Khong co wave {W} trong Picking_Wave.csv")
items = w.groupby(["reference", "Size (US)", "locations"]).size().reset_index(name="units")
print(f"\nWave {W}: {len(items)} dong hang\n", items)

res = {}
orig = sorted(l for l in items.locations.unique() if l in XY)
miss = sorted(set(items.locations) - set(orig))
if miss:
    print("[CANH BAO] Bo qua o khong co toa do:", miss)
res["Original (company)"] = (orig, [])
for k, (m, df) in PM.items():
    chosen = []
    for r in items.itertuples():
        cands, how = m.get((r.reference, float(r[2]))), "ref+size"
        if not cands:
            cands, how = m.get((r.reference, None)), "ref only"
        if not cands:
            chosen.append((r.reference, None, "missing")); continue
        cc = sorted(c for c in cands if c in XY and XY[c][2] <= 2) or sorted(c for c in cands if c in XY)
        if not cc:
            chosen.append((r.reference, None, "no coords")); continue
        best = min(cc, key=lambda c: (dist(DEPOT, XY[c]), c))   # hoa -> chon theo ten o: ket qua on dinh
        chosen.append((r.reference, best, how))
    res[k] = (sorted(set(c[1] for c in chosen if c[1])), chosen)

print(f"\n{'Chinh sach':20s} {'Quang duong':>11s} {'So o':>5s}  Thuat toan")
for k, (locs, chosen) in res.items():
    tot, order = tsp([XY[l] for l in locs])
    algo = "Held-Karp" if len(locs) <= EXACT_MAX else "NN+2opt"
    print(f"{k:20s} {tot:11.0f} {len(locs):5d}  {algo}")
    print("   ", " -> ".join(["Depot"] + [locs[i] for i in order] + ["Depot"]))
    bad = [c for c in chosen if c[2] not in ("ref+size",)]
    if bad:
        print("    luu y:", bad)
