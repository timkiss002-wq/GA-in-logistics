"""Data loading for the footwear order-picking dataset (de Assis et al., 2025)."""
import os, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_data():
    """Dataset folder: env SLAP_DATA, else the first folder under ./data that holds Picking_Wave.csv."""
    if os.environ.get('SLAP_DATA'):
        return os.environ['SLAP_DATA']
    for root, _, files in os.walk(os.path.join(HERE, 'data')):
        if 'Picking_Wave.csv' in files:
            return root
    raise FileNotFoundError('Unzip the dataset into the "data" folder next to the code, or set SLAP_DATA.')


DATA = _find_data()
P = lambda f: os.path.join(DATA, f)

def size_key(s):
    return round(float(str(s).strip()), 1)

def load_locations():
    s = pd.read_csv(P('Storage_Location.csv'))
    s['originalLocation'] = s.originalLocation.str.strip()
    return s.set_index('originalLocation')[['x', 'y', 'z']]

def load_support_points():
    import re
    raw = open(P('Support_Points_Navigation.csv'), encoding='utf-8-sig').read()
    pairs = re.findall(r'\(([-\d.]+),\s*([-\d.]+),\s*[-\d.]+\);([A-Z]{2}-\d{2})', raw)
    return {lab: (float(x), float(y)) for x, y, lab in pairs}

def load_layout(name):
    """Return dict location -> list of (ref, size) per slot."""
    f = {'random': 'Random_Storage.csv', 'class': 'Class_Based_Storage.csv',
         'dedicated': 'Dedicated_Storage.csv', 'hybrid': 'Hybrid_Storage.csv'}[name]
    if name == 'random':
        d = pd.read_csv(P(f)); loccol = 'originalLocation'; cols = d.columns[1:]
    else:
        d = pd.read_csv(P(f), sep=';', encoding='utf-8-sig'); loccol = 'Location'; cols = d.columns[2:]
    out = {}
    for _, row in d.iterrows():
        slots = []
        for c in cols:
            v = row[c]
            if isinstance(v, str) and ';' in v:
                a, b = v.split(';'); slots.append((a.strip(), size_key(b)))
        out[str(row[loccol]).strip()] = slots
    return out

def load_waves():
    w = pd.read_csv(P('Picking_Wave.csv'), sep=';', encoding='utf-8-sig')
    w.columns = ['wave', 'ref', 'size', 'qty', 'loc', 'op']
    w['ref'] = w.ref.str.strip(); w['loc'] = w['loc'].str.strip()
    w['size'] = w['size'].map(size_key)
    return w

def mixing_index(layout, locs=None, level='ref'):
    """de Assis et al. (2026) Eq.(2): MI_l = 1 - max_i n_il / sum_i n_il.
    level='ref' -> SKU = reference (as in the paper); 'sku' -> reference x size."""
    vals = {}
    for l, slots in layout.items():
        if locs is not None and l not in locs: continue
        if not slots: continue
        keys = [s[0] for s in slots] if level == 'ref' else slots
        cnt = collections.Counter(keys)
        vals[l] = 1 - max(cnt.values()) / sum(cnt.values())
    return vals
