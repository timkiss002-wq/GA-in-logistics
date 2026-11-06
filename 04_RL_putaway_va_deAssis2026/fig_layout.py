"""
Vẽ mặt bằng kho (các dãy kệ, lối đi, depot) từ toạ độ trong Storage_Location.csv / gro.py.
Hàm draw_layout(ax, values=None) dùng lại cho hình phương pháp: tô màu từng ô theo một giá trị (ví dụ điểm s_l).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gro import AISLES, CROSS_X, CROSS_YMIN, DEPOT

I = dict(np.load('inst.npz', allow_pickle=True))
XY = I['cell_xy']                       # điểm lấy hàng (x, y của lối đi)
NAMES = I['cell_name']
# Hàng kệ (toạ độ y thật của kệ trong Storage_Location.csv), mỗi lối đi phục vụ kệ hai bên
RACK_Y = [0, 32, 90, 122, 180, 212, 270, 302, 360, 392, 450, 482, 540, 572, 688, 720, 778, 810, 868, 900, 958,
          990, 1048, 1080, 1138, 1170, 1228, 1260, 1318, 1350, 1408, 1440]
DEPTH = 22


def rack_y_of_cell(i):
    """y của kệ chứa ô i: kệ gần lối đi nhất trong RACK_Y (ở phía có ô cùng dãy)."""
    x, a = XY[i]
    cands = sorted(RACK_Y, key=lambda y: abs(y - a))[:2]
    letter, bay = NAMES[i].split('-')[0], int(NAMES[i].split('-')[1])
    # quy ước tên: trong mỗi chữ cái, dãy số lẻ ở y nhỏ hơn, dãy số chẵn ở y lớn hơn (xem Storage_Location.csv)
    lo, hi = min(cands), max(cands)
    if letter == 'A': return 0
    return lo if (bay % 2 == 1) else hi


def draw_layout(ax, values=None, cmap='viridis', vmin=None, vmax=None, highlight=None, depot_label=True,
                cell_w=14):
    ys = np.array([rack_y_of_cell(i) for i in range(len(XY))])
    # kệ nền
    for y in RACK_Y:
        xs = XY[ys == y][:, 0]
        if len(xs) == 0: continue
        for blk in ([x for x in xs if x < 403], [x for x in xs if x > 403]):
            if blk:
                ax.add_patch(Rectangle((min(blk) - 8, y - DEPTH / 2), max(blk) - min(blk) + 16, DEPTH,
                                       fc='#e9e9e9', ec='#9a9a9a', lw=0.4, zorder=1))
    # ô (gộp 2 tầng: vẽ tầng 1 bên trái, tầng 2 bên phải trong cùng vị trí -> chỉ vẽ trung bình)
    if values is not None:
        import matplotlib as mpl
        norm = mpl.colors.Normalize(vmin=np.nanmin(values) if vmin is None else vmin,
                                    vmax=np.nanmax(values) if vmax is None else vmax)
        cm = plt.get_cmap(cmap)
        key = {}
        for i in range(len(XY)):
            key.setdefault((XY[i][0], ys[i]), []).append(i)
        for (x, y), idx in key.items():
            v = np.nanmax([values[j] for j in idx])
            if np.isfinite(v):
                ax.add_patch(Rectangle((x - cell_w / 2, y - DEPTH / 2), cell_w, DEPTH, fc=cm(norm(v)), ec='none', zorder=2))
    # lối đi chính và lối đi ngang
    for a in AISLES:
        xs = XY[XY[:, 1] == a][:, 0]
        if len(xs):
            ax.plot([min(xs.min(), 66), max(xs.max(), 403 if a < 631 else 686)], [a, a], color='#c9d6e8', lw=0.6, zorder=0)
    for cx, cy in zip(CROSS_X, CROSS_YMIN):
        ax.plot([cx, cx], [cy, 1471], color='#9db3d6', lw=1.2, zorder=0)
    if highlight is not None:
        x, y = XY[highlight][0], ys[highlight]
        ax.add_patch(Rectangle((x - cell_w / 2 - 6, y - DEPTH / 2 - 6), cell_w + 12, DEPTH + 12, fc='none', ec='#d62728',
                               lw=1.6, zorder=4))
    ax.plot(*DEPOT, marker='*', ms=11, color='#d62728', mec='k', mew=0.4, zorder=5)
    if depot_label:
        ax.annotate('depot', DEPOT, xytext=(DEPOT[0] + 30, DEPOT[1] - 5), fontsize=7, va='center')
    ax.set_xlim(40, 720); ax.set_ylim(-60, 1500); ax.set_aspect('equal'); ax.axis('off')
    return ys


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(3.2, 6))
    draw_layout(ax, values=I['d_cell'], cmap='Blues_r')
    fig.savefig('results/lp/layout_check.png', dpi=200, bbox_inches='tight')
