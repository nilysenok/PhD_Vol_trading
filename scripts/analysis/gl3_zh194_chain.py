# -*- coding: utf-8 -*-
"""Zh-194: the ladder of layers over H*_h5 (A-2 = calm G-3 + stress G-1 c=1) and over G-1 LOYO / A-2_22 (h = 22).
h = 5: H*, (a) A-2 with the stress expert HAR-J(b), (b) A-2 with the logistic mixture {HAR-J(b), G-1 c=1} as the stress expert, (v) EWMA points + LOYO, (g) G6, (d) seeds.
h = 22: G-1 LOYO, A-2_22 (stress = HAR-J(b)), (v), (g), (d). One call = one seed of the chain; `pick` decides (seed 42: by scores, other seeds: forced)."""
import numpy as np
import gl3_zh190_stage23_core as Q
import gl3_zh190_stage23_layers as Y3
import gl3_zh194_core as Z


class Book:
    """Series of one chain: name -> (float32 array, stage, eligible for the table of LOYO, note)."""
    def __init__(self):
        self.S = {}

    def put(self, name, p, stage, eligible=True, note=''):
        self.S[name] = (Q.f32(p), stage, eligible, note)
        return self.S[name][0]

    def __getitem__(self, name):
        return self.S[name][0]


def a2(tag, a, z, y, end, h, calm, stress, rec):
    """A-2 logistic mixture in logs; the ledger kind is renamed with the tag."""
    r = (lambda hh, k, cut, n, me, **kw: rec(hh, f'{tag}_{k}', cut, n, me, **kw)) if rec else None
    return Y3.a2_forecast(a, z, y, end, h, calm, stress, rec=r)


def chain(h, a, z, feats, y, end, comp, pick, rec=Z.record):
    """-> (Book, name of the final series before the seed average)."""
    year, bk = a.year.values, Book()
    g1 = {k: Q.f32(v) for k, v in comp['g1'].items()}
    for k, v in g1.items():
        bk.put(f'G1_{k}', v, 'stage2', False)
    if h == 5:
        bases = _bases5(a, z, y, end, comp, pick, bk, g1, rec)
    else:
        calm = bk.put('G1_LOYO', pick.loyo('G1', g1, year), 'stage2', True, 'LOYO over c')
        harj = np.where(np.isfinite(comp['harj']), comp['harj'], calm)
        bk.put('A2_22', a2('A2_22', a, z, y, end, h, calm, harj, rec), 'stage3', True, 'calm G-1 LOYO, stress HAR-J(b)')
        bases = {'G1_LOYO': bk['G1_LOYO'], 'A2_22': bk['A2_22']}
    bn = pick.best('base_v', bases)
    pts = {k: Q.f32(v) for k, v in Z.ewma_points(a, bases[bn], y, h).items()}
    for k, v in pts.items():
        bk.put(f'V_{k}', v, 'layer_v', False, f'on {bn}')
    cand = {**bases, 'V': bk.put('V', pick.loyo('v', pts, year), 'layer_v', True, f'LOYO over 6 points on {bn}')}
    gn = pick.best('base_g6', cand)
    cand['G6'] = bk.put('G6', Z.g6_forecast(a, cand[gn], y, end, h, feats, rec=rec), 'layer_g', True, f'on {gn}')
    return bk, pick.best('final', cand)


def _bases5(a, z, y, end, comp, pick, bk, g1, rec):
    year = a.year.values
    g2 = {k: Q.f32(v) for k, v in comp['g2'].items()}
    bst, har = bk.put('Bstar', comp['bst'], 'stage1'), Q.f32(comp['har'])
    cands = {'G1_LOYO': pick.loyo('G1', g1, year), 'G2_LOYO': pick.loyo('G2', g2, year),
             'G3': Q.blend_ln(bst, har, Q.g3_weight(a, bst, har, y, 5))}
    for n, v in cands.items():
        bk.put(n, v, 'stage2')
    gn = pick.best('Gstar', cands)
    calm = bk.put('Gstar', cands[gn], 'stage2', True, f'= {gn}')
    harj = np.where(np.isfinite(comp['harj']), comp['harj'], calm)   # last rows of the window have no HAR-J(b) (target not yet known): stress = calm
    st = {'A2_HARi': har, 'A2_G1c100': g1['c100']}
    for n, s in st.items():
        bk.put(n, a2(n, a, z, y, end, 5, calm, s, rec), 'stage3', False, 'stress expert ' + n[3:])
    hs = bk.put('Hstar', pick.loyo('A2', {n: bk[n] for n in st}, year), 'stage3', True, '= A2 (LOYO between the two stress experts)')
    pa = bk.put('a', a2('a', a, z, y, end, 5, calm, harj, rec), 'layer_a', True, 'A-2, stress HAR-J(b)')
    mix = bk.put('b_stress', a2('b_mix', a, z, y, end, 5, harj, g1['c100'], rec), 'layer_b', False, 'logistic mixture {HAR-J(b), G-1 c=1}')
    pb = bk.put('b', a2('b', a, z, y, end, 5, calm, mix, rec), 'layer_b', True, 'A-2, stress = logistic mixture')
    return {'Hstar': hs, 'a': pa, 'b': pb}
