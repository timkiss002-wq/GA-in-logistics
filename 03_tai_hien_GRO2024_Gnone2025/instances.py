"""Turn waves + a storage layout into GRO instances."""
import collections
import numpy as np
from data import load_locations, load_layout, load_waves, mixing_index
from gro import pick_point, build_instance

LOC = load_locations()
PP = {l: pick_point(x, y) for l, x, y in zip(LOC.index, LOC.x, LOC.y)}


def sku_index(layout):
    """(ref, size) -> sorted unique pick points of the locations holding it."""
    idx = collections.defaultdict(set)
    for l, slots in layout.items():
        if l not in PP:
            continue
        for s in slots:
            idx[s].add(PP[l])
    return {s: sorted(v) for s, v in idx.items()}


def wave_skus(waves):
    """wave -> sorted list of distinct (ref, size) SKUs; also as-picked pick points."""
    skus, picked = {}, {}
    for w, d in waves.groupby('wave'):
        skus[w] = sorted(set(zip(d.ref, d['size'])))
        picked[w] = (d['loc'].isin(PP.keys()).all(), sorted({PP[l] for l in d['loc'] if l in PP}))
    return skus, picked


def random_layout(base, seed):
    """Random storage: shuffle every slot of `base` across all locations (same stock, same capacity)."""
    rng = np.random.default_rng(seed)
    locs = [l for l in base if l in PP]
    items = [s for l in locs for s in base[l]]
    rng.shuffle(items)
    out, i = {}, 0
    for l in locs:
        n = len(base[l]); out[l] = items[i:i + n]; i += n
    return out
