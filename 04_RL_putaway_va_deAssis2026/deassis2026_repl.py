"""
TÁI HIỆN de Assis et al. (2026), Procedia CIRP 146: 643–648
"Agentic architecture for progressive autonomous storage location assignment in warehouses"

Làm đúng Algorithm 1 của bài cho cả 3 chế độ: heuristic, supervised, adaptive.
  1. Perception : MI_l trước khi cất, chỗ trống a_l = C_l - sum_i n_il
  2. Candidate  : L_t = {l : a_l > 0}
  3. Evaluation : với ô đủ chỗ cho cả lô u: MI_l sau, ΔMI_l; giữ ô có MI_l sau <= θ + ς
                  U_l = -ΔMI_l (+ ε·1{cỡ s đã có ở l}, công thức 5)      [heuristic / supervised]
                  U_l = w·x_l, x_l = [-ΔMI_l, 1{i∈l}, 1{cỡ hợp}, a_l/C_l, 1-MI_l trước, Dispersion(i)]   [adaptive]
                  Xếp hạng theo U; với xác suất ω chọn ngẫu nhiên trong top-k
  4. Look-ahead : nếu u > 1 và có >= 2 ô: thử chia lô (l1,u1),(l2,u2), chọn chia hoặc không theo tổng U
  5. Decision
  6. Supervision: chế độ supervised chỉ tự thực hiện khi mọi ô dùng có ΔMI_l <= 0, còn lại chờ người duyệt
  7. Action     : cập nhật kho
  8. Adaptive   : w <- w - ϖ(2(Û_t - r_t) x̄_t + ϱ w), x̄_t chuẩn hoá

Thiết lập như mục 3.4 của bài: kho ngẫu nhiên thực đầy, giải phóng ~20% sức chứa, 10.000 lô hàng về
(mã bốc theo tần suất tồn kho, cỡ và số lượng theo phân phối thực), mọi chế độ dùng CÙNG một chuỗi lô.

Tham số bài công bố: ϖ = 0,05; ϱ = 0,01; φ = 0,1 (nhóm hiểu là tỷ lệ khám phá ω).
Tham số bài KHÔNG công bố – nhóm tự đặt và chạy độ nhạy (xem --sens):
  θ (ngưỡng MI)  : 1,0 (không ràng buộc)      ε (thưởng cùng cỡ): 0,01      top-k: 5
  w ban đầu      : [1, 1, 1, 1, 1, -1] (ngang nhau; Dispersion mang dấu trừ như công thức 7)
  r_t (adaptive) : -tổng ΔMI_l của các ô đã dùng (đúng đại lượng bài dùng để đánh giá)
  "Thực hiện"    : lô được cất bởi agent. Lô không có ô hợp lệ, hoặc bị supervised chuyển cho người duyệt,
                   được tính là KHÔNG thực hiện và không làm thay đổi kho.
Phạm vi ô: bài dùng 847 ô "main operating level". Trong Random_Storage.csv chỉ vùng cỡ ghi số nguyên (1.093 ô,
mỗi ô 14–18 mã) cho MI ≈ 0,918 (đã kiểm tra: 0,9147); phần còn lại là ô thuần 1 mã. Nhóm chạy chính trên 1.093 ô
và kiểm tra độ nhạy trên 249 ô tầng 1 của vùng này.

Chạy:  python deassis2026_repl.py            -> 3 chế độ × 5 seed, bảng như Bảng 2 + Hình 3, Hình 4
       python deassis2026_repl.py --sens     -> độ nhạy theo các tham số không công bố
"""
import sys, os, json, itertools
import numpy as np, pandas as pd

C = 18
OMEGA_PAPER, PHI_PAPER, RHO_PAPER = 0.1, 0.05, 0.01      # φ (khám phá), ϖ (học), ϱ (điều chuẩn)
DEFAULT = dict(theta=1.0, tol=0.0, eps=0.01, topk=5, omega=OMEGA_PAPER, lr=PHI_PAPER, reg=RHO_PAPER,
               w0=[1, 1, 1, 1, 1, -1], reward='local', scope='all', n_events=10000, free=0.20, level='ref', size_hard=False, deferred='drop')


# ---------------- dữ liệu ----------------
def load(scope, level='ref'):
    r = pd.read_csv('data/Random_Storage.csv', dtype=str)
    loc = pd.read_csv('data/Storage_Location.csv'); loc['originalLocation'] = loc.originalLocation.str.strip()
    z = dict(zip(loc.originalLocation, loc.z))
    cols = [c for c in r.columns if c.startswith('col_')]
    r = r[~r.col_1.str.contains(r'\.')].copy()                     # vùng ngẫu nhiên (cỡ số nguyên)
    r['z'] = r.originalLocation.str.strip().map(z)
    if scope == 'z1': r = r[r.z == 1]
    slots = [[tuple(v.split(';')) for v in row] for row in r[cols].values]
    if level == 'sku':                                              # SKU = mã × cỡ
        slots = [[(f'{a};{b}', b) for a, b in s] for s in slots]
    refs = sorted({a for s in slots for a, _ in s}); sizes = sorted({int(b) for s in slots for _, b in s})
    ri = {x: i for i, x in enumerate(refs)}; si = {x: i for i, x in enumerate(sizes)}
    L, R, S = len(slots), len(refs), len(sizes)
    n = np.zeros((L, R), np.int32); ns = np.zeros((L, S), np.int32)
    units = []                                                      # (ô, mã, cỡ) từng đơn vị -> để giải phóng 20%
    for l, s in enumerate(slots):
        for a, b in s:
            n[l, ri[a]] += 1; ns[l, si[int(b)]] += 1; units.append((l, ri[a], si[int(b)]))
    # phân phối lô hàng về (từ ảnh chụp đầy)
    freq = n.sum(0) / n.sum()
    size_dist = np.zeros((R, S))
    for l, a, b in units: size_dist[a, b] += 1
    size_dist /= size_dist.sum(1, keepdims=True)
    q = {}
    for s in slots:
        for k, v in pd.Series([f'{a};{b}' for a, b in s]).value_counts().items(): q[v] = q.get(v, 0) + 1
    qv = np.array(sorted(q)); qp = np.array([q[k] for k in qv], float); qp /= qp.sum()
    return dict(n=n, ns=ns, units=np.array(units), freq=freq, size_dist=size_dist, qv=qv, qp=qp, L=L, R=R, S=S)


def mi_cells(n):
    tot = n.sum(1); mx = n.max(1)
    return np.where(tot > 0, 1 - mx / np.maximum(tot, 1), 0.0)


def prepare(D, seed, n_events, free):
    rng = np.random.default_rng(seed)
    n, ns = D['n'].copy(), D['ns'].copy()
    k = int(round(free * len(D['units'])))
    if D.get('free_mode', 'units') == 'cells':                       # biến thể: làm rỗng hẳn ~20% số ô
        for l in rng.choice(D['L'], int(round(free * D['L'])), replace=False):
            n[l] = 0; ns[l] = 0
    else:
        for l, a, b in D['units'][rng.choice(len(D['units']), k, replace=False)]:
            n[l, a] -= 1; ns[l, b] -= 1
    ev_ref = rng.choice(D['R'], n_events, p=D['freq'])
    ev_size = np.array([rng.choice(D['S'], p=D['size_dist'][a]) for a in ev_ref])
    ev_u = rng.choice(D['qv'], n_events, p=D['qp'])
    return n, ns, list(zip(ev_ref, ev_size, ev_u))


# ---------------- agent (Algorithm 1) ----------------
def run(D, mode, seed, P, trace_every=100):
    n, ns, events = prepare(D, seed, P['n_events'], P['free'])
    rng = np.random.default_rng(10_000 + seed)
    w = np.array(P['w0'], float); L = D['L']
    executed = deferred = rejected = 0; tr = []; W_hist = []
    for t, (i, s, u) in enumerate(events):
        tot = n.sum(1); mx = n.max(1)
        a = C - tot
        mi_b = np.where(tot > 0, 1 - mx / np.maximum(tot, 1), 0.0)

        def evaluate(uu):
            """U và ΔMI cho việc cất uu đơn vị mã i vào từng ô (np.nan nếu không hợp lệ)."""
            nmx = np.maximum(mx, n[:, i] + uu); mi_a = 1 - nmx / (tot + uu)
            dmi = mi_a - mi_b
            ok = (a >= uu) & (mi_a <= P['theta'] + P['tol'])
            if P.get('size_hard'): ok &= (ns[:, s] > 0) | (tot == 0)     # biến thể: cỡ phải tương thích
            if P.get('theta_min') is not None: ok &= mi_a >= P['theta_min']  # biến thể: ngưỡng là SÀN (MI sau >= θ)
            if P.get('dmi_max') is not None and mode != 'supervised': ok &= dmi <= P['dmi_max']   # biến thể: không cất nếu ΔMI quá lớn
            disp = (n[:, i] > 0).sum() / L
            X = np.stack([-dmi, (n[:, i] > 0).astype(float), (ns[:, s] > 0).astype(float), a / C, 1 - mi_b,
                          np.full(L, disp)], 1)
            if mode == 'adaptive':
                U = X @ w
            else:
                U = -dmi + P['eps'] * X[:, 2]
            return np.where(ok, U, np.nan), dmi, X

        U1, dmi1, X1 = evaluate(u)
        cand = np.flatnonzero(~np.isnan(U1))
        best = None                                                  # (tổng U, [(ô, số lượng)])
        if len(cand):
            order = cand[np.argsort(-U1[cand])]
            pick = order[0]
            if rng.random() < P['omega']:
                pick = rng.choice(order[:P['topk']])
            best = (U1[pick], [(pick, u)])
        # look-ahead: chia lô
        if u > 1:
            for u1 in range(1, u):
                Ua, _, _ = evaluate(u1); Ub, _, _ = evaluate(u - u1)
                ca = np.flatnonzero(~np.isnan(Ua)); cb = np.flatnonzero(~np.isnan(Ub))
                if len(ca) == 0 or len(cb) == 0: continue
                ta = ca[np.argsort(-Ua[ca])][:5]; tb = cb[np.argsort(-Ub[cb])][:5]
                for l1, l2 in itertools.product(ta, tb):
                    if l1 == l2: continue
                    val = Ua[l1] + Ub[l2]
                    if best is None or val > best[0]: best = (val, [(l1, u1), (l2, u - u1)])
        if best is None:
            rejected += 1
        else:
            alloc = best[1]
            dmis = []
            for l, uu in alloc:
                _, dm, _ = evaluate(uu); dmis.append(dm[l])
            if mode == 'supervised' and any(d > 1e-12 for d in dmis):
                deferred += 1                                        # chờ người duyệt -> agent không thực hiện
                if P['deferred'] == 'operator':                      # biến thể: người vận hành tự cất vào ô trống ngẫu nhiên
                    free_cells = np.flatnonzero(a >= u)
                    if len(free_cells):
                        l = rng.choice(free_cells); n[l, i] += u; ns[l, s] += u
            else:
                xs = []
                for (l, uu) in alloc:
                    _, _, X = evaluate(uu); xs.append(X[l])
                for l, uu in alloc:
                    n[l, i] += uu; ns[l, s] += uu
                executed += 1
                if mode == 'adaptive':
                    xbar = np.sum(xs, 0); xbar = xbar / (np.linalg.norm(xbar) + 1e-12)
                    r = -float(np.sum(dmis)) if P['reward'] == 'local' else -float(mi_cells(n).mean() - mi_b.mean()) * L
                    Uhat = float(w @ xbar)
                    w = w - P['lr'] * (2 * (Uhat - r) * xbar + P['reg'] * w)
        if trace_every and (t + 1) % trace_every == 0:
            tr.append((t + 1, float(mi_cells(n).mean()), executed))
            W_hist.append(w.copy())
    m = mi_cells(n); occ = n.sum(1) > 0
    return dict(mode=mode, seed=seed, MI=float(m.mean()), crit=float((m[occ] > 0.8).mean() * 100),
                executed=executed, deferred=deferred, rejected=rejected, trace=tr, w=w.tolist())


def initial(D, seed, P):
    n, _, _ = prepare(D, seed, P['n_events'], P['free'])
    m_full = mi_cells(D['n']); m = mi_cells(n); occ = n.sum(1) > 0
    return dict(MI_full=float(m_full.mean()), crit_full=float((m_full > 0.8).mean() * 100),
                MI_freed=float(m.mean()), crit_freed=float((m[occ] > 0.8).mean() * 100))


def main_experiment(P, seeds=range(5), tag='main'):
    D = load(P['scope'], P['level'])
    rows = []; traces = {}
    ini = [initial(D, s, P) for s in seeds]
    for mode in ['heuristic', 'supervised', 'adaptive']:
        for s in seeds:
            r = run(D, mode, s, P); rows.append(r); traces[(mode, s)] = r['trace']
            print(f'  {mode:<10} seed {s}: MI {r["MI"]:.4f} | critical {r["crit"]:.2f}% | thực hiện {r["executed"]} '
                  f'| chờ duyệt {r["deferred"]} | không có ô {r["rejected"]}', flush=True)
    return D, pd.DataFrame(rows), ini, traces


if __name__ == '__main__':
    os.makedirs('results/deassis2026', exist_ok=True)
    if '--sens' not in sys.argv:
        P = dict(DEFAULT)
        D, df, ini, traces = main_experiment(P)
        df.drop(columns=['trace']).to_csv('results/deassis2026/ket_qua_5seed.csv', index=False)
        json.dump({f'{k[0]}_{k[1]}': v for k, v in traces.items()}, open('results/deassis2026/trace.json', 'w'))
        ini = pd.DataFrame(ini)
        print(f'\nSố ô: {D["L"]} | mã: {D["R"]} | ban đầu (đầy): MI {ini.MI_full.mean():.4f}, critical {ini.crit_full.mean():.2f}% '
              f'| sau khi giải phóng 20%: MI {ini.MI_freed.mean():.4f} ± {ini.MI_freed.std():.4f}')
        g = df.groupby('mode')[['MI', 'crit', 'executed', 'deferred', 'rejected']].agg(['mean', 'std'])
        print(g.round(4).to_string())
    else:
        D_all = None; rows = []
        variants = [('chuẩn', {})] + [(f'θ = {x}', dict(theta=x)) for x in [0.95, 0.9]] + \
                   [(f'ε = {x}', dict(eps=x)) for x in [0.0, 0.05]] + [(f'top-k = {x}', dict(topk=x)) for x in [1, 10]] + \
                   [('ω = 0 (không khám phá)', dict(omega=0.0))] + \
                   [('w0 = [1,0,0,0,0,0]', dict(w0=[1, 0, 0, 0, 0, 0])), ('w0 = [1,.1,.1,.1,.1,-.1]', dict(w0=[1, .1, .1, .1, .1, -.1]))] + \
                   [('r_t = ΔMI toàn kho', dict(reward='global'))] + [('chỉ tầng 1 (249 ô)', dict(scope='z1'))] + \
                   [('SKU = mã × cỡ', dict(level='sku')), ('cỡ bắt buộc tương thích', dict(size_hard=True)),
                    ('lô chờ duyệt: người cất ngẫu nhiên', dict(deferred='operator')), ('3.000 lô thay vì 10.000', dict(n_events=3000))]
        for name, ch in variants:
            P = dict(DEFAULT); P.update(ch)
            D = load(P['scope'], P['level'])
            for mode in ['heuristic', 'supervised', 'adaptive']:
                if ('w0' in ch or 'reward' in ch) and mode != 'adaptive': continue
                if 'deferred' in ch and mode != 'supervised': continue
                rs = [run(D, mode, s, P, trace_every=0) for s in range(3)]
                rows.append(dict(bien_the=name, mode=mode, MI=np.mean([r['MI'] for r in rs]),
                                 crit=np.mean([r['crit'] for r in rs]), executed=np.mean([r['executed'] for r in rs])))
                print(f'{name:<26} {mode:<10} MI {rows[-1]["MI"]:.4f} | critical {rows[-1]["crit"]:.2f}% | thực hiện {rows[-1]["executed"]:.0f}', flush=True)
        pd.DataFrame(rows).to_csv('results/deassis2026/do_nhay.csv', index=False)
