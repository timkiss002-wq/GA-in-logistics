"""Warehouse distance model + Genetic Route Optimisation (GRO) re-implementation.

Layout (de Assis et al., 2025 dataset): horizontal pick aisles at the support-point y values,
three vertical cross aisles LC (x=66), CC (x=403), RC (x=686, only y>=631).
Aisles with y<=511 span x in [66, 403]; aisles with y>=631 span [66, 686].
A storage location is picked from the nearest pick aisle at its own x.
Distances are rectilinear along aisles, in dataset coordinate units.

GA follows Gnone et al. (2025) Table 2: Np=1000, G=1000, tournament T=3, Pc=0.90, Pm=0.08.
Each SKU of a wave may be stored in several locations, so a chromosome holds a visiting
order of the wave's SKUs and the location chosen for each SKU (family TSP).
"""
import numpy as np
from numba import njit

AISLES = np.array([-29., 61., 151., 241., 331., 421., 511., 631., 751., 841., 931., 1021.,
                   1111., 1201., 1291., 1381., 1471.])
CROSS_X = np.array([66., 403., 686.])
CROSS_YMIN = np.array([-29., -29., 631.])
DEPOT = (403., 631.)          # CC-08: centre of the fast-mover (X) zone in the Dedicated file


def pick_point(x, y):
    """(x, y) of a storage location -> (x, aisle_y) of its picking point."""
    a = AISLES[np.argmin(np.abs(AISLES - y))]
    return float(x), float(a)


@njit(cache=True)
def _dist(x1, a1, x2, a2, cross_x, cross_ymin):
    if a1 == a2:
        return abs(x1 - x2)
    best = 1e18
    lo = min(a1, a2)
    for c in range(cross_x.shape[0]):
        if lo < cross_ymin[c]:
            continue
        d = abs(x1 - cross_x[c]) + abs(a1 - a2) + abs(x2 - cross_x[c])
        if d < best:
            best = d
    return best


@njit(cache=True)
def dist_matrix(px, pa, cross_x, cross_ymin):
    n = px.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist(px[i], pa[i], px[j], pa[j], cross_x, cross_ymin)
            D[i, j] = d
            D[j, i] = d
    return D


def build_instance(cands):
    """cands: list (one per SKU) of lists of (x, aisle_y) candidate pick points.
    Returns D (M x M, index 0 = depot), start (offset of each SKU's candidates), k (count)."""
    pts = [DEPOT]
    start, k = [], []
    for c in cands:
        start.append(len(pts)); k.append(len(c)); pts.extend(c)
    pts = np.array(pts, dtype=np.float64)
    D = dist_matrix(pts[:, 0].copy(), pts[:, 1].copy(), CROSS_X, CROSS_YMIN)
    return D, np.array(start, np.int64), np.array(k, np.int64)


@njit(cache=True)
def _route_len(order, choice, D, start):
    n = order.shape[0]
    prev = 0
    tot = 0.0
    for t in range(n):
        s = order[t]
        node = start[s] + choice[s]
        tot += D[prev, node]
        prev = node
    tot += D[prev, 0]
    return tot


@njit(cache=True)
def _ox(p1, p2, child, a, b):
    n = p1.shape[0]
    used = np.zeros(n, np.bool_)
    for i in range(a, b + 1):
        child[i] = p1[i]
        used[p1[i]] = True
    j = (b + 1) % n
    for t in range(n):
        g = p2[(b + 1 + t) % n]
        if not used[g]:
            child[j] = g
            used[g] = True
            j = (j + 1) % n


@njit(cache=True)
def gro(D, start, k, Np=1000, G=1000, T=3, Pc=0.90, Pm=0.08, seed=0, patience=0):
    """Returns best route length and the generation it was found in.
    patience>0 stops after that many generations without improvement."""
    np.random.seed(seed)
    n = start.shape[0]
    if n == 0:
        return 0.0, 0
    if n == 1:
        best = 1e18
        for c in range(k[0]):
            v = D[0, start[0] + c] * 2
            if v < best:
                best = v
        return best, 0
    pop_o = np.empty((Np, n), np.int64)
    pop_c = np.empty((Np, n), np.int64)
    for i in range(Np):
        pop_o[i] = np.random.permutation(n)
        for s in range(n):
            pop_c[i, s] = np.random.randint(k[s])
    fit = np.empty(Np)
    for i in range(Np):
        fit[i] = _route_len(pop_o[i], pop_c[i], D, start)
    bi = np.argmin(fit)
    best = fit[bi]
    best_gen = 0
    new_o = np.empty_like(pop_o)
    new_c = np.empty_like(pop_c)
    for g in range(1, G + 1):
        # elitism: keep the best individual
        new_o[0] = pop_o[bi]
        new_c[0] = pop_c[bi]
        for i in range(1, Np):
            # tournament selection of two parents
            p = np.empty(2, np.int64)
            for q in range(2):
                w = np.random.randint(Np)
                for _ in range(T - 1):
                    r = np.random.randint(Np)
                    if fit[r] < fit[w]:
                        w = r
                p[q] = w
            if np.random.random() < Pc:
                a = np.random.randint(n)
                b = np.random.randint(n)
                if a > b:
                    a, b = b, a
                _ox(pop_o[p[0]], pop_o[p[1]], new_o[i], a, b)
                for s in range(n):
                    new_c[i, s] = pop_c[p[0], s] if np.random.random() < 0.5 else pop_c[p[1], s]
            else:
                new_o[i] = pop_o[p[0]]
                new_c[i] = pop_c[p[0]]
            if np.random.random() < Pm:
                # swap mutation on the order
                a = np.random.randint(n)
                b = np.random.randint(n)
                t = new_o[i, a]; new_o[i, a] = new_o[i, b]; new_o[i, b] = t
            if np.random.random() < Pm:
                # location-choice mutation
                s = np.random.randint(n)
                new_c[i, s] = np.random.randint(k[s])
        pop_o, new_o = new_o, pop_o
        pop_c, new_c = new_c, pop_c
        for i in range(Np):
            fit[i] = _route_len(pop_o[i], pop_c[i], D, start)
        bi = np.argmin(fit)
        if fit[bi] < best - 1e-9:
            best = fit[bi]
            best_gen = g
        elif patience > 0 and g - best_gen >= patience:
            break
    return best, best_gen


@njit(cache=True)
def exact(D, start, k):
    """Held-Karp DP with location choice (for validation, n <= ~13)."""
    n = start.shape[0]
    M = D.shape[0]
    INF = 1e18
    full = 1 << n
    # dp[mask, node] ; node indexes candidate nodes 1..M-1
    dp = np.full((full, M), INF)
    owner = np.empty(M, np.int64)
    for s in range(n):
        for c in range(k[s]):
            owner[start[s] + c] = s
            dp[1 << s, start[s] + c] = D[0, start[s] + c]
    for mask in range(1, full):
        for v in range(1, M):
            cur = dp[mask, v]
            if cur >= INF:
                continue
            for s in range(n):
                if mask & (1 << s):
                    continue
                nm = mask | (1 << s)
                for c in range(k[s]):
                    u = start[s] + c
                    val = cur + D[v, u]
                    if val < dp[nm, u]:
                        dp[nm, u] = val
    best = INF
    for v in range(1, M):
        val = dp[full - 1, v] + D[v, 0]
        if val < best:
            best = val
    return best


@njit(cache=True)
def _route_len_dp(order, D, start, k):
    """Optimal location choice for a fixed visiting order (layered shortest path)."""
    n = order.shape[0]
    kmax = 0
    for s in range(n):
        if k[s] > kmax:
            kmax = k[s]
    cur = np.empty(kmax)
    nxt = np.empty(kmax)
    s0 = order[0]
    for c in range(k[s0]):
        cur[c] = D[0, start[s0] + c]
    for t in range(1, n):
        sp = order[t - 1]; s = order[t]
        for c in range(k[s]):
            u = start[s] + c
            b = 1e18
            for cp in range(k[sp]):
                v = cur[cp] + D[start[sp] + cp, u]
                if v < b:
                    b = v
            nxt[c] = b
        for c in range(k[s]):
            cur[c] = nxt[c]
    sl = order[n - 1]
    best = 1e18
    for c in range(k[sl]):
        v = cur[c] + D[start[sl] + c, 0]
        if v < best:
            best = v
    return best


@njit(cache=True)
def two_opt(order, D, start, k, cur):
    n = order.shape[0]
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = order.copy()
                a = i; b = j
                while a < b:
                    t = cand[a]; cand[a] = cand[b]; cand[b] = t
                    a += 1; b -= 1
                v = _route_len_dp(cand, D, start, k)
                if v < cur - 1e-9:
                    cur = v; order[:] = cand; improved = True
        # or-opt: move one SKU to another position
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                cand = np.empty(n, np.int64)
                s = order[i]; p = 0
                for t in range(n):
                    if t == i:
                        continue
                    cand[p] = order[t]; p += 1
                # insert s at position j
                for t in range(n - 1, j, -1):
                    cand[t] = cand[t - 1]
                cand[j] = s
                v = _route_len_dp(cand, D, start, k)
                if v < cur - 1e-9:
                    cur = v; order[:] = cand; improved = True
    return cur


@njit(cache=True)
def gro_dp(D, start, k, Np=1000, G=1000, T=3, Pc=0.90, Pm=0.08, seed=0, patience=0, inversion=False, polish=False):
    """GRO with the order evolved by the GA and locations decoded optimally (DP) for each order."""
    np.random.seed(seed)
    n = start.shape[0]
    if n == 0:
        return 0.0, 0
    pop = np.empty((Np, n), np.int64)
    for i in range(Np):
        pop[i] = np.random.permutation(n)
    fit = np.empty(Np)
    for i in range(Np):
        fit[i] = _route_len_dp(pop[i], D, start, k)
    if n == 1:
        return fit.min(), 0
    bi = np.argmin(fit)
    best = fit[bi]
    best_gen = 0
    new = np.empty_like(pop)
    for g in range(1, G + 1):
        new[0] = pop[bi]
        for i in range(1, Np):
            p0 = np.random.randint(Np)
            for _ in range(T - 1):
                r = np.random.randint(Np)
                if fit[r] < fit[p0]:
                    p0 = r
            p1 = np.random.randint(Np)
            for _ in range(T - 1):
                r = np.random.randint(Np)
                if fit[r] < fit[p1]:
                    p1 = r
            if np.random.random() < Pc:
                a = np.random.randint(n); b = np.random.randint(n)
                if a > b:
                    a, b = b, a
                _ox(pop[p0], pop[p1], new[i], a, b)
            else:
                new[i] = pop[p0]
            if np.random.random() < Pm:
                a = np.random.randint(n); b = np.random.randint(n)
                if inversion:
                    if a > b:
                        a, b = b, a
                    while a < b:
                        t = new[i, a]; new[i, a] = new[i, b]; new[i, b] = t
                        a += 1; b -= 1
                else:
                    t = new[i, a]; new[i, a] = new[i, b]; new[i, b] = t
        pop, new = new, pop
        for i in range(Np):
            fit[i] = _route_len_dp(pop[i], D, start, k)
        bi = np.argmin(fit)
        if fit[bi] < best - 1e-9:
            best = fit[bi]; best_gen = g
        elif patience > 0 and g - best_gen >= patience:
            break
    if polish:
        order = pop[np.argmin(fit)].copy()
        best = two_opt(order, D, start, k, best)
    return best, best_gen
