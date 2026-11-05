# -*- coding: utf-8 -*-
"""Zh-193 (h = 1) one run: B1-B6, B-comb -> B**, G1-G7, summary. Usage: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh193_h1_run.py
Resumable (existing cells are kept). At most 5 processes x 1 thread (conductor limit of 05.11)."""
import json
import time
import numpy as np
import pandas as pd
import gl3_r6b_layers as L
import gl3_zh190_boost_core as K
import gl3_zh190_boost_eval as E
import gl3_zh190_stage23 as ST
import gl3_zh190_stage23_core as Q
import gl3_zh193_h1_fit as Z
import gl3_zh193_h1_layers as LY

N = Z.new_spec
BSPECS = [N('B1', x='b1'), N('B2_mono', mono=True), N('B2_lt', lt=True), N('B2_both', mono=True, lt=True), N('B3_qlike', loss='qlike'),
          N('B3_huber8', loss='huber8'), N('B4_hl1', hl=1.), N('B4_hl2', hl=2.), N('B5_d4', cat=4), N('B5_d6', cat=6),
          N('S1_cat_huber1', scheme='cat', loss='huber1')]
SEEDS = (43, 44, 45, 46)
EPS = Q.EPS


def log(msg):
    with (Z.TAB / 'run.log').open('a') as f:
        f.write(f'{pd.Timestamp.now(tz="UTC").isoformat()} {msg}\n')


class Run:
    def __init__(self):
        self.d = Z.setup()
        Z.G['b1'] = Z.b1_matrix()
        b = self.d.frames['original']
        self.b, self.m16 = b, (b.date >= '2016-01-01').values
        self.a = b[self.m16].reset_index(drop=True)
        self.key = pd.MultiIndex.from_frame(self.a[['date', 'ticker']])
        self.sc = E.Scorer(self.d, 1)
        self.ix = self.key.get_indexer(self.sc.key)
        assert (self.ix >= 0).all()
        self.etal = self.sc.etalons()
        self.y, self.end, self.year = self.a._y1.values.astype(float), self.a._end1.values, self.a.year.values
        self.rows, self.fc = [], {}
        har = Z.F.G['hari'][1][0]
        self.hari = Q.f32(ST.on_a(self.key, har, 'level'))
        self.fin = Q.f32(ST.on_a(self.key, pd.read_parquet(Z.K.OUT / 'etalony' / 'FIN_PIT_h1.parquet').rename(columns={'prediction': 'p'}), 'p'))

    def series(self, name):
        return Q.f32(ST.on_a(self.key, Z.read_cells(name), 'pred'))

    def score(self, p):
        return self.sc.score(p[self.ix])

    def row(self, name, p, stage, eligible=True, note=''):
        p = Q.f32(p)
        assert np.isfinite(p).all() and (p > 0).all(), name
        self.fc[name] = p
        r = self.sc.row(p[self.ix], self.etal)
        if not eligible:
            r['mean_LOYO'] = np.nan
        h, f = self.etal['HARJb'], self.etal['FIN_PIT']
        eh, ef = self.sc.score(h), self.sc.score(f)
        r['cel'] = bool(r['mean_QLIKE'] < ef['mean_QLIKE'] - EPS and r['stress2020'] <= eh['stress2020'] + EPS and r['stressRVI'] <= eh['stressRVI'] + EPS)
        self.rows.append(dict(h=1, stage=stage, config=name, eligible=eligible, note=note, **r))
        return r

    def mean(self, name):
        return self.score(self.fc[name])['mean_QLIKE']


def b_stage(R):
    t = time.time()
    log(f'B fits {len(BSPECS)} specs x 6 cutoffs {Z.run_pool(BSPECS, Z.TAB / "run.log"):.0f}s')
    for n, p in R.etal.items():
        R.rows.append(dict(h=1, stage='etalon', config={'HARJb': 'HAR-J(b)', 'HARi_LOYO': 'HAR_i'}.get(n, n), eligible=True, note='', **R.sc.row(p, R.etal)))
    bst = Q.f32(ST.on_a(R.key, pd.read_parquet(Q.ROOT / 'data/regen/gl3_zh190/bstar_h1.parquet'), 'pred'))
    R.row('Bstar_h1', bst, 'B', True, 'ticker-huber4 (stage 1)')
    for s in BSPECS:
        R.row(s['name'], R.series(s['name']), 'B', True, s['scheme'] if s['scheme'] != 'ticker' else '')
    allfc = pd.read_parquet(Q.ROOT / 'data/regen/gl3_zh190/forecasts_h1.parquet')
    cont = Q.f32(ST.on_a(R.key, allfc[(allfc.stage == 'stage1') & (allfc.config == 'cont-huber2')], 'pred'))
    R.row('B6', (bst + R.fc['S1_cat_huber1'] + cont) / 3, 'B', True, 'mean of levels: Bstar, cat-huber1 (recomputed on 6 cutoffs), cont-huber2')
    return t


def best_of(R, names, base):
    ok = [n for n in names if R.mean(n) < R.mean(base) - EPS]
    return min(ok, key=R.mean) if ok else None


def comb_spec(R):
    """B-comb: B* + every B1-B4 item that alone gave a lower mean QLIKE than B* (the best variant inside an item)."""
    pick = dict(b1=best_of(R, ['B1'], 'Bstar_h1'), b2=best_of(R, ['B2_mono', 'B2_lt', 'B2_both'], 'Bstar_h1'),
                b3=best_of(R, ['B3_qlike', 'B3_huber8'], 'Bstar_h1'), b4=best_of(R, ['B4_hl1', 'B4_hl2'], 'Bstar_h1'))
    if not any(pick.values()):
        return None, pick
    kw = {}
    if pick['b1']:
        kw['x'] = 'b1'
    if pick['b2']:
        kw.update(mono=pick['b2'] in ('B2_mono', 'B2_both'), lt=pick['b2'] in ('B2_lt', 'B2_both'))
    if pick['b3']:
        kw['loss'] = {'B3_qlike': 'qlike', 'B3_huber8': 'huber8'}[pick['b3']]
    if pick['b4']:
        kw['hl'] = {'B4_hl1': 1., 'B4_hl2': 2.}[pick['b4']]
    return N('Bcomb', **kw), pick


def select_bss(R, cand):
    base = R.score(R.fc['Bstar_h1'])
    ok = [n for n in cand if R.score(R.fc[n])['stress2020'] <= base['stress2020'] + EPS]
    return min(ok or cand, key=R.mean), bool(ok)


B6_MEMBERS = [N('m_tk'), N('m_cat', scheme='cat', loss='huber1'), N('m_cont', scheme='cont', loss='huber2')]


def member_specs(mem, prefix, seed, **kw):
    """Fit specs of a B** (one model, or the three members of B6 whose levels are averaged) with the G modification `kw`."""
    return [{**m, 'name': f'{prefix}_m{k}', 'seed': seed, **kw} for k, m in enumerate(mem)]


def mean_level(series):
    return Q.f32(np.mean(series, axis=0))


def seed_inputs(R, bss, nm, g1p, g2p):
    """Inputs of the G chain: B** level (given), G1 points and G2 as the mean of levels of the `nm` members."""
    pts = []
    for k in range(nm):
        n = f'{g1p}_m{k}'
        val = pd.concat([pd.read_parquet(Z.cell_path(n + '_val', Y)).assign(year=Y) for Y in Q.ALL_YEARS], ignore_index=True)
        pts.append(ST.g1_points(Z.read_cells(n), val, R.key))
    g1 = {c: mean_level([p[c] for p in pts]) for c in pts[0]}
    return dict(fin=R.fin, hari=R.hari, bss=bss, g1=g1, g2=mean_level([R.series(f'{g2p}_m{k}') for k in range(nm)]))


def aux_for(R):
    b = R.b
    z2 = LY.Y3.z_features(b)[R.m16]
    return dict(z2=z2, stress=L.stress_hard(b)[R.m16], z3=b[['_ln_close_RVI', '_ln_rq', '_ln_rv_w']].values[R.m16], rec=Z.ledger)


def main():
    R = Run()
    t0 = b_stage(R)
    sel = dict(cat_deviation='stage 1 passed no cat row at h=1 (screen_h1.csv: cat-huber1 best of cat, rank 9, only 2019-2020 cells); cat-huber1 recomputed on all 6 cutoffs by the stage-1 module')
    spec, pick = comb_spec(R)
    sel['Bcomb_items'] = pick
    if spec:
        log(f'Bcomb fit {Z.run_pool([spec], Z.TAB / "run.log"):.0f}s')
        R.row('Bcomb', R.series('Bcomb'), 'B', True, json.dumps({k: v for k, v in spec.items() if k in ('x', 'loss', 'mono', 'lt', 'hl')}))
    cand = [s['name'] for s in BSPECS if s['name'] != 'S1_cat_huber1'] + ['B6'] + (['Bcomb'] if spec else [])
    bss, passed = select_bss(R, cand)
    sel.update(Bss=bss, Bss_stress_filter_nonempty=passed)
    specs = {s['name']: s for s in BSPECS + ([spec] if spec else [])}
    R.row('Bss', R.fc[bss], 'B', True, f'= {bss}')
    mem = B6_MEMBERS if bss == 'B6' else [specs[bss]]
    sel['Bss_members'] = [m['name'] for m in mem]
    g42 = member_specs(mem, 'G1', 42, off='hari') + member_specs(mem, 'G2', 42, g2=True)
    gs = {s: member_specs(mem, f'G1_s{s}', s, off='hari') + member_specs(mem, f'G2_s{s}', s, g2=True) + member_specs(mem, f'Bss_s{s}', s) for s in SEEDS}
    log(f'G fits {Z.run_pool(g42, Z.TAB / "run.log"):.0f}s')
    inp = seed_inputs(R, R.fc[bss], len(mem), 'G1', 'G2')
    aux = aux_for(R)
    ev = LY.Ev(R.year, R.score, adder=lambda n, p, el, note: R.row(n, p, 'G', el, note))
    LY.chain(R.a, inp, R.y, R.end, aux, ev)
    sel.update(ev.sel)
    # G7: 5 seeds of the best of G1-G6, same selection replayed
    log(f'G7 fits {Z.run_pool([s for q in SEEDS for s in gs[q]], Z.TAB / "run.log"):.0f}s')
    finals = [ev.fc[ev.sel['G7_of']]]
    for s in SEEDS:
        e2 = LY.Ev(R.year, sel=ev.sel)
        bs = mean_level([R.series(f'Bss_s{s}_m{k}') for k in range(len(mem))])
        LY.chain(R.a, seed_inputs(R, bs, len(mem), f'G1_s{s}', f'G2_s{s}'), R.y, R.end, aux, e2)
        finals.append(e2.fc[ev.sel['G7_of']])
    R.row('G7', np.mean(finals, axis=0), 'G', True, f'5 seeds 42-46 of {ev.sel["G7_of"]}, mean of levels')
    sel['seeds'] = [42, *SEEDS]
    finish(R, sel, time.time() - t0)


def finish(R, sel, sec):
    T, tab = Z.TAB, pd.DataFrame(R.rows)
    tab.to_csv(T / 'summary.csv', index=False)
    tab[tab.stage.isin(['etalon', 'B'])].to_csv(T / 'b_rows.csv', index=False)
    tab[tab.stage.isin(['etalon', 'G'])].to_csv(T / 'g_rows.csv', index=False)
    f = pd.concat([pd.DataFrame(dict(config=n, date=R.a.date.values, ticker=R.a.ticker.values, pred=p.astype('float32'))) for n, p in R.fc.items()], ignore_index=True)
    f.to_parquet(Z.OUT / 'forecasts_h1.parquet', index=False)
    (T / 'selection.json').write_text(json.dumps(sel, indent=1, default=str))
    log(f'finished rows={len(tab)} series={len(R.fc)} seconds={sec:.0f}')


if __name__ == '__main__':
    main()
