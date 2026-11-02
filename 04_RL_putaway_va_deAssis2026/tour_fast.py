"""
Ước lượng nhanh độ dài lộ trình 1 wave (dùng làm reward khi huấn luyện, thay cho GRO vì GRO chậm).
Nearest-neighbour + 2-opt trên đúng mô hình khoảng cách lối đi của gro.py. Kiểm định với GRO ở check_tour.py.
"""
import numpy as np
from numba import njit
from gro import dist_matrix, CROSS_X, CROSS_YMIN, DEPOT


@njit(cache=True)
def nn_2opt(D):
    n = D.shape[0]
    if n <= 1:
        return 0.0
    tour = np.empty(n, np.int64); used = np.zeros(n, np.bool_)
    tour[0] = 0; used[0] = True
    for i in range(1, n):
        prev = tour[i - 1]; best = -1; bd = 1e18
        for j in range(n):
            if not used[j] and D[prev, j] < bd:
                bd = D[prev, j]; best = j
        tour[i] = best; used[best] = True
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a = tour[i - 1]; b = tour[i]; c = tour[j]; d = tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-9:
                    lo = i; hi = j
                    while lo < hi:
                        t = tour[lo]; tour[lo] = tour[hi]; tour[hi] = t; lo += 1; hi -= 1
                    improved = True
    L = 0.0
    for i in range(n):
        L += D[tour[i], tour[(i + 1) % n]]
    return L


def tour_len(xy_points):
    pts = np.vstack([np.array(DEPOT)[None], np.unique(np.asarray(xy_points, float), axis=0)])
    D = dist_matrix(pts[:, 0].copy(), pts[:, 1].copy(), CROSS_X, CROSS_YMIN)
    return nn_2opt(D)
