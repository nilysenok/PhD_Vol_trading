# -*- coding: utf-8 -*-
"""Round 6b, item 2b (Zh-188): PIT rebuild of the non-PIT archive inputs of the stack -- linear HAR-X+ and FIN as an input
(D12_h5, frozen warmup-2016 parameters, early stopping on purged windows) -- inside our expanding window, and recomputation of
G3/G5/G6/G7a on them. HAR-J(b), HAR-CJ _emb_d0 and K9 are PIT by the audit of 2a and are used as they are; ITER is excluded. DEV only (date <= 2021-12-31), no control reader, no model is saved.

Pure part (no heavy imports): train_rows, hxp_*, cj_*, pc_scores, calibrate_scalar -- every fit uses only rows whose
target END lies before the cutoff (U8/U16). Heavy part (Study of round 6 reused by inheritance, outputs redirected
to docs/tables/gl3_r6b_h5/pit and data/regen/gl3_r6b_h5/pit) starts in main().
"""
import argparse
import contextlib
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gl3_r6b_common as C

ROOT = HERE.parents[1]
OUT = ROOT / 'data/regen/gl3_r6b_h5/pit'
TAB = ROOT / 'docs/tables/gl3_r6b_h5/pit'
ALPHAS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, .1, .3, 1.)
L1S = (.1, .5, .9, 1.)
GRID = [(l1, al) for l1 in L1S for al in ALPHAS] + [(1., 0.)]
DEFAULT_EN = (.1, 1.)  # (alpha, l1) of round 6, used only when the inner window is too short
PC_FIELDS = ['yahoo_sp500_vol22', 'yahoo_eem_vol22', 'yahoo_gold_vol22', 'yahoo_wti_vol22', 'yahoo_brent_vol22', 'yahoo_dxy_vol22', 'vix']
ARCHIVE_KEYS = ['FIN', 'HAR', 'CJ', 'HXP', 'K9', 'ITER']
PIT_KEYS = ['FIN', 'HAR', 'CJ', 'HXP', 'K9']  # FIN=FIN_PIT, HXP=HXP_PIT; ITER (D26 x6) excluded (needs the whole h=1 chain), see GATES.md


# ------------------------------------------------------------------ pure PIT core
def _ns(x):
    return np.asarray(x, dtype='datetime64[ns]')


def train_rows(date, end, y, cut):
    """Training rows at cutoff: row date and target END both before cut, positive finite target (U8)."""
    c = pd.Timestamp(cut).to_datetime64()
    return (_ns(date) < c) & (_ns(end) < c) & np.isfinite(np.asarray(y, float)) & (np.asarray(y, float) > 1e-12)


def validation_rows(date, end, y, inner, cut):
    """Inner-selection validation rows: the year [inner, cut) with target END before cut."""
    c = pd.Timestamp(cut).to_datetime64(); i = pd.Timestamp(inner).to_datetime64()
    return (_ns(date) >= i) & (_ns(date) < c) & (_ns(end) < c) & np.isfinite(np.asarray(y, float)) & (np.asarray(y, float) > 1e-12)


def max_end(end, mask, cut):
    m = _ns(end)[mask].max()
    assert m < pd.Timestamp(cut).to_datetime64(), 'target end >= cutoff in a fit'
    return pd.Timestamp(m)


def hxp_fit(H, Z, y, tr, alpha, l1):
    """HAR-J (OLS in logs) + elastic net on the HAR-orthogonalised standardised extras; log forecast for all rows."""
    logy = np.log(y[tr]); coef = np.linalg.lstsq(H[tr], logy, rcond=None)[0]; base = H @ coef
    mean = Z[tr].mean(0); sd = Z[tr].std(0); sd[sd < 1e-8] = 1.; zz = (Z - mean) / sd
    proj = np.linalg.lstsq(H[tr], zz[tr], rcond=None)[0]; res = zz - H @ proj
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        en = ElasticNet(alpha=max(alpha, 1e-8), l1_ratio=l1, max_iter=3000, tol=1e-5, fit_intercept=False).fit(res[tr], logy - base[tr])
    return base + en.predict(res)


def hxp_select(H, build_z, y, date, end, cut, record):
    """Inner choice of (alpha, l1): fit on END < inner cut = cut - 1 year, MSE of ln y on the year before cut (END < cut)."""
    inner = pd.Timestamp(cut) - pd.DateOffset(years=1)
    tr = train_rows(date, end, y, inner); va = validation_rows(date, end, y, inner, cut)
    if tr.sum() < 100 or va.sum() < 20:
        return DEFAULT_EN[0], DEFAULT_EN[1], float('nan')
    Z = build_z(tr); mx = max_end(end, tr, inner); best = None; ly = np.log(y[va])
    for l1, al in GRID:
        mse = float(np.mean((ly - hxp_fit(H, Z, y, tr, al, l1)[va]) ** 2))
        record('PIT_HAR-X+', inner, int(tr.sum()), mx, 'inner_grid', dict(alpha=al, l1=l1, mse=mse))
        if best is None or mse < best[2]:
            best = (al, l1, mse)
    return best


def hxp_predict(H, build_z, y, date, end, cut, record):
    al, l1, mse = hxp_select(H, build_z, y, date, end, cut, record)
    tr = train_rows(date, end, y, cut); mx = max_end(end, tr, cut)
    pred = hxp_fit(H, build_z(tr), y, tr, al, l1)
    record('PIT_HAR-X+', cut, int(tr.sum()), mx, 'refit', dict(alpha=al, l1=l1, inner_mse=mse))
    return pred, dict(alpha=al, l1=l1, inner_mse=mse, n=int(tr.sum()))


def pc_scores(values, train_mask):
    """First principal component of the daily market block; mean, scale and loading from training dates only."""
    past = values[train_mask]; pm = np.nan_to_num(np.nanmean(past, axis=0)); ps = np.nan_to_num(np.nanstd(past, axis=0)); ps[ps < 1e-8] = 1.
    load = np.linalg.svd(np.nan_to_num((past - pm) / ps), full_matrices=False)[2][0]
    if load[np.argmax(np.abs(load))] < 0:
        load = -load
    return np.nan_to_num((values - pm) / ps) @ load


def annual_periods():
    d = pd.date_range('2016-01-01', '2022-01-01', freq='YS')
    return list(zip(d[:-1], d[1:]))


def calib_rows(date, end, y, raw, cut):
    """Rows for the s2 correction: three years back, target END before cut (closed windows)."""
    c = pd.Timestamp(cut); lo = (c - pd.DateOffset(years=3)).to_datetime64(); y = np.asarray(y, float)
    return (_ns(date) < c.to_datetime64()) & (_ns(end) < c.to_datetime64()) & (_ns(date) >= lo) & np.isfinite(raw) & np.isfinite(y) & (y > 1e-12)


def calibrate_scalar(date, end, y, raw, periods=None):
    """Level forecast exp(raw + s2/2), s2 = ddof0 variance of the ln error on closed windows (same as Engine.calibrate 'scalar')."""
    out = np.full(len(raw), np.nan); date = _ns(date); y = np.asarray(y, float)
    for cut, e in periods or annual_periods():
        tr = calib_rows(date, end, y, raw, cut); va = (date >= cut.to_datetime64()) & (date < e.to_datetime64())
        err = np.log(y[tr]) - raw[tr]
        v = float(np.mean((err - err.mean()) ** 2)) if len(err) >= 50 else 0.
        out[va] = np.exp(np.clip(raw[va] + v / 2, -60, 60))
    return out


# ------------------------------------------------------------------ builders on real dev data
class Extras:
    """HAR-X+ extras (as Engine.hxp_current); only the PC column depends on the training mask."""
    def __init__(self, b):
        self.b = b; imoex = b.groupby('date').rv_IMOEX.first().sort_index()
        im = np.log((imoex ** 2).rolling(22, min_periods=13).mean().clip(lower=1e-12)); market = b.groupby('date').rv_m.mean()
        fields = [c for c in PC_FIELDS if c in b]; assert fields, 'no market block for PC1'
        self.daily = b.groupby('date')[fields].first().sort_index()
        rrv = [b[c].values for c in b if c.startswith('rrv_') and 'target' not in c]
        self.static = np.column_stack([2 * b._ln_close_RVI.values, b.date.map(im).values, np.log(b.date.map(market).clip(lower=1e-12)),
                                       b._LHAR1, b._LHAR5, b._LHAR22, np.zeros(len(b)), b._HARQ, np.log(b.rsv_pos.clip(lower=1e-12)),
                                       np.log(b.rsv_neg.clip(lower=1e-12)), b.r2_on] + rrv)

    def __call__(self, tr):
        use = np.isin(self.daily.index.values, self.b.date.values[tr])
        pc = pd.Series(pc_scores(self.daily.values, use), index=self.daily.index)
        Z = self.static.copy(); Z[:, 6] = pc.reindex(self.b.date.values).values
        return np.nan_to_num(Z, nan=0., posinf=0., neginf=0.)


def build_hxp(d, record, log=print):
    """PIT linear HAR-X+ on the annual schedule 2016..2021 -> raw ln forecast, s2-corrected level, chosen (alpha, l1) per year."""
    a = d.a; y = d.y(5, 'original'); date = a.date.values; end = a._end5.values
    H = np.nan_to_num(d.harx(5, 'original'), nan=0., posinf=0., neginf=0.); ex = Extras(a)
    raw = np.full(len(a), np.nan); chosen = []
    for cut, e in annual_periods():
        va = ((a.date >= cut) & (a.date < e)).values
        p, meta = hxp_predict(H, ex, y, date, end, cut, record); raw[va] = p[va]; chosen.append(dict(year=cut.year, **meta))
        log(f'PIT HAR-X+ {cut.year}: {meta}')
    return raw, calibrate_scalar(date, end, y, raw), chosen


# ------------------------------------------------------------------ redirect of the round-6 modules
@contextlib.contextmanager
def redirect(state):
    """Point round-6 modules at our own folders for the duration of the run, tag every ledger row with state['version']; restore on exit."""
    import gl3_r6h5_data as D, gl3_r6h5_engine as E, gl3_r6h5_run as R, gl3_r56_engine as B
    for p in (TAB, OUT / 'cache', OUT / 'engine'):
        p.mkdir(parents=True, exist_ok=True)
    manifest = TAB / 'diagnostics_manifest.json'
    if not manifest.exists():
        manifest.write_text('{"control_read":false}')
    saved = [(m, k, getattr(m, k)) for m, ks in ((D, ('TAB', 'OUT')), (E, ('TAB', 'OUT')), (R, ('TAB', 'OUT', 'NUMERICAL_CONTRACT')),
                                                  (B, ('TAB', 'OUT', 'EVENTS', 'record'))) for k in ks]
    original = B.record

    def tagged(*args, **kwargs):
        kwargs['detail'] = {**(kwargs.get('detail') or {}), 'version': state['version']}
        return original(*args, **kwargs)
    try:
        for m in (D, E, R):
            m.TAB = TAB; m.OUT = OUT / 'cache'
        B.TAB = TAB; B.OUT = OUT / 'engine'; B.EVENTS = TAB / 'fit_events.jsonl'; B.record = tagged
        yield D, E, R, B
    finally:
        for m, k, v in saved:
            setattr(m, k, v)


def study_class(R, D, B, state):
    from sklearn.linear_model import Ridge
    from gl3_r56_data import periods, mask_train
    from gl3_r56_diagnostics import read, REG

    def root(spec):
        while 'parent' in spec:
            spec = spec['parent']
        return spec['core']

    class PitStudy(R.Study):
        """Round-6 Study with: G1/G4 read from the saved round-6 dev forecasts (they contain no archive input),
        G3 with a configurable expert list, outputs in our own folders."""
        tag = 'x'; g3_keys = ARCHIVE_KEYS

        def save(self):
            pd.DataFrame(self.rows).to_csv(TAB / f'dev_candidates_{self.tag}.csv', index=False)
            (TAB / f'selection_{self.tag}.json').write_text(json.dumps(dict(status=self.status, recipes=self.rec, control_read=False), ensure_ascii=False, indent=2))

        def compute(self, s):
            core = s['core']
            if core in ('G1', 'G4'):
                name = 'G4' if core == 'G4' else 'G1_' + s['variant']
                z = read(REG / 'gl3_r6_h5' / f'{name}_dev.parquet', ['date', 'ticker', 'prediction'])
                return self.d.series(z, 'prediction').astype('float32'), np.full(len(self.d.a), np.nan, 'float32')
            if core in ('G5', 'G6') and root(s['parent']) in ('G1', 'G4'):
                raise NotImplementedError('parent G1/G4 selected: needs the 10-seed bank rebuild, not done in this sheet')
            if core == 'G3':
                return self.g3(s)
            return super().compute(s)

        def g3(self, s):
            a = self.d.a; x = np.column_stack([np.log(self.inputs[k]) for k in self.g3_keys]); raw = np.full(len(a), np.nan); y = self.d.y(5, 'original')
            for cut, end in periods(s['mode']):
                tr = mask_train(a, 5, cut, s['mode'], y) & np.isfinite(x).all(axis=1); va = ((a.date >= cut) & (a.date < end)).values
                if tr.sum() < 100:
                    raw[va] = np.log(self.inputs['FIN'][va]); continue
                m = Ridge(alpha=1., positive=True, max_iter=3000, tol=1e-8).fit(x[tr], np.log(y[tr])); ok = va & np.isfinite(x).all(axis=1); raw[ok] = m.predict(x[ok])
                B.record('G3_positive_ridge', 5, cut, s['mode'], 'original', 42, int(tr.sum()), a.loc[tr, '_end5'].max(),
                         detail={'weights': m.coef_.tolist(), 'intercept': float(m.intercept_), 'keys': list(self.g3_keys)})
            return self.e.scalar(raw, mode=s['mode']), raw

        def run_pass(self, version, prefix, keys):
            state['version'] = version; self.tag = version; self.g3_keys = keys; R.NUMERICAL_CONTRACT = 'r6b-' + version
            self.rows = []; self.rec = {}; self.pcache = {}; self.rawcache = {}; self.scorecache = {}
            D.disk('pass ' + version)
            YEARS = list(D.YEARS) + ['all']; ks = [self.add(prefix + 'G1_' + v, dict(core='G1', variant=v)) for v in ('standard', 'regularized')]
            for gate in ('hard', 'logistic'):
                rec = {}
                for year in YEARS:
                    cols = [str(y) for y in D.YEARS if y != year]
                    expert = min(('K9', 'K10', 'HAR', 'HXP'), key=lambda k: np.mean([self.d.score5(self.inputs[k])[c] for c in cols]))
                    rec[str(year)] = dict(core='G2', gate=gate, expert=expert)
                ks.append(self.add(prefix + 'G2_' + gate, rec))
            for mode in ('annual', 'quarterly'):
                ks.append(self.add(prefix + 'G3_' + mode, dict(core='G3', mode=mode)))
            ks.append(self.add(prefix + 'G4', dict(core='G4')))
            ks.append(self.add(prefix + 'G5', self.choose(ks, lambda p: dict(core='G5', parent=p))))
            ks.append(self.add(prefix + 'G6', self.choose(ks, lambda p: dict(core='G6', parent=p))))
            parents = self.choose(ks)
            self.add(prefix + 'G7a_BEST', {y: {'core': 'G7', 'kind': 's2', 'parent': r} for y, r in parents.items()})
            return pd.DataFrame(self.rows)
    return PitStudy


# ------------------------------------------------------------------ run: archive control, PIT build, PIT pass, tables
CANDS = ['G3_annual', 'G3_quarterly', 'G5', 'G6', 'G7a_BEST']
NUM = ['2017', '2018', '2019', '2020', '2021', 'mean_QLIKE', 'stress2020', 'stressRVI', 'ratio_stress2020', 'ratio_stressRVI']
VERSIONS = {'archive': ('archive_', 'FIN_arch,HAR,CJ_arch,HXP_arch,K9,ITER_arch'), 'pit': ('pit_', 'FIN_PIT,HAR,CJ_arch,HXP_PIT,K9'),
            'pit_iter': ('pititer_', 'FIN_PIT,HAR,CJ_arch,HXP_PIT,K9,ITER_arch')}


def forecast_file(a, level, path):
    """date, ticker, prediction (float32), 2016-2021, finite and positive."""
    z = a[['date', 'ticker']].copy(); z['prediction'] = np.asarray(level, 'float32'); z = z[(z.date >= '2016-01-01') & (z.date < '2022-01-01')]
    assert (z.date < '2022-01-01').all(); z = z[np.isfinite(z.prediction) & (z.prediction > 0)]
    path.parent.mkdir(parents=True, exist_ok=True); z.to_parquet(path, index=False)


def table(d, passes):
    ref = d.ref; idx = d.idx; base = d.base; final = C.ticker_means(ref.rv_actual, ref.rv_pred, ref.year, ref.ticker)
    m20 = ref.date.between('2020-03-01', '2020-06-30').values; mrvi = d.a.loc[idx, '_stress'].values.astype(bool); out = []
    for version, df in passes.items():
        prefix, inputs = VERSIONS[version]
        for cand in CANDS:
            r = df[df.candidate == prefix + cand].iloc[0]; p = pd.read_parquet(OUT / 'cache' / f'{prefix + cand}_dev.parquet').prediction.values
            sc = C.score(ref.rv_actual, p, ref.year, m20, mrvi)
            assert all(abs(sc[k] - r[k]) < 1e-9 for k in NUM), (version, cand)
            k, n = C.tickers_better(C.ticker_means(ref.rv_actual, p, ref.year, ref.ticker), final)
            out.append(dict(candidate=cand, version=version, inputs=inputs, eligible=True, **{c: sc[c] for c in NUM}, **C.admission(sc, base, k),
                            tickers_better=k, tickers_n=n, binom_p=C.binom_sf(k, n), gain_vs_final=base['mean_QLIKE'] - sc['mean_QLIKE']))
    t = pd.DataFrame(out)
    for c in ('leak_vs_pit', 'leak_reason_pit', 'leak_vs_pit_iter', 'leak_reason_pit_iter'):
        t[c] = None
    for cand in CANDS:
        m = {v: float(t[(t.candidate == cand) & (t.version == v)].mean_QLIKE.iloc[0]) for v in passes}
        for v in ('pit', 'pit_iter'):
            leak, why = C.leak_verdict(m['archive'], m[v], base['mean_QLIKE'])
            t.loc[(t.candidate == cand) & (t.version.isin(['archive', v])), ['leak_vs_' + v, 'leak_reason_' + v]] = [leak, why]
    return t


def main():
    state = {'version': 'setup'}; log = lambda m: print(m, flush=True)
    with redirect(state) as (D, E, R, B):
        s = study_class(R, D, B, state)(); d = s.d; a = d.a; D.disk('start')
        rec = lambda kind, cut, n, mx, phase, detail: B.record(kind, 5, cut, 'annual', 'original', 42, n, mx, phase=phase, detail=detail)
        old = {k: v.copy() for k, v in s.inputs.items()}; passes = {'archive': s.run_pass('archive', 'archive_', ARCHIVE_KEYS)}
        r6 = pd.read_csv(ROOT / 'docs/tables/gl3_r6_h5/dev_candidates.csv').set_index('candidate')
        ctrl = [dict(candidate=c, **{k: abs(float(passes['archive'].set_index('candidate').loc['archive_' + c, k]) - float(r6.loc[c, k])) for k in NUM}) for c in CANDS]
        ctrl = pd.DataFrame(ctrl); ctrl['max_abs_diff'] = ctrl[NUM].max(axis=1); ctrl['ok'] = ctrl.max_abs_diff <= 1e-9
        ctrl.to_csv(TAB / 'control_reproduction.csv', index=False)
        state['version'] = 'pit_build'; D.disk('build HAR-X+')
        hraw, hlev, chosen = build_hxp(d, rec, log)
        assert np.allclose(s.e.scalar(hraw), hlev, rtol=1e-5, equal_nan=True), 'own s2 correction differs from Engine.calibrate'
        D.disk('build FIN_PIT'); fraw = s.e.bank('final_fixed', 42); flev = s.e.scalar(fraw)
        for k, lev in (('HXP', hlev), ('FIN', flev)):
            forecast_file(a, lev, OUT / 'experts' / f'{k}_PIT.parquet')
        ex = [dict(expert=f'{k}_{v}', **d.score5(lev)) for k, v, lev in (('HXP', 'archive', old['HXP']), ('HXP', 'PIT', hlev), ('FIN', 'archive_input', old['FIN']), ('FIN', 'PIT', flev))]
        pd.DataFrame(ex).to_csv(TAB / 'experts_summary.csv', index=False)
        s.inputs['HXP'] = hlev.astype(float); s.rawinputs['HXP'] = hraw.astype(float); s.inputs['FIN'] = flev.astype(float); s.rawinputs['FIN'] = fraw.astype(float)
        passes['pit'] = s.run_pass('pit', 'pit_', PIT_KEYS)
        passes['pit_iter'] = s.run_pass('pit_iter', 'pititer_', ARCHIVE_KEYS)
        for c in passes['pit'].candidate:
            z = pd.read_parquet(OUT / 'cache' / f'{c}_dev.parquet'); (OUT / 'candidates').mkdir(exist_ok=True)
            z[['date', 'ticker', 'prediction']].astype({'prediction': 'float32'}).to_parquet(OUT / 'candidates' / f'{c}.parquet', index=False)
        t = table(d, passes)
        t[t.version != 'pit_iter'].drop(columns=['leak_vs_pit_iter', 'leak_reason_pit_iter']).rename(columns={'leak_vs_pit': 'leak', 'leak_reason_pit': 'leak_reason'}).to_csv(TAB / 'archive_vs_pit.csv', index=False)
        t[t.version != 'pit'].to_csv(TAB / 'archive_vs_pit_iter_variant.csv', index=False)
        ev = [json.loads(x) for x in (TAB / 'fit_events.jsonl').read_text().splitlines()]
        counts = pd.Series([f"{e['component']}|{e['phase']}" for e in ev]).value_counts().to_dict()
        (TAB / 'provenance.json').write_text(json.dumps(dict(hxp_chosen=chosen, n_events=len(ev), events_by_component=counts, control_read=False, control_ok=bool(ctrl.ok.all())), indent=2, default=str))
        log(t.to_string())


def check(gate):
    if gate == 'g2':
        c = pd.read_csv(TAB / 'control_reproduction.csv'); print(c[['candidate', 'max_abs_diff', 'ok']].to_string()); assert c.ok.all() and len(c) == 5
    elif gate == 'g3':
        ev = [json.loads(x) for x in (TAB / 'fit_events.jsonl').read_text().splitlines()]
        bad = [e for e in ev if not (e['h'] == 5 and pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) < pd.Timestamp('2022-01-01'))]
        print('events', len(ev), 'bad', len(bad)); assert ev and not bad
    elif gate == 'g4':
        t = pd.read_csv(TAB / 'archive_vs_pit.csv'); print(len(t), t.leak.isna().sum()); assert len(t) == 10 and t.leak.notna().all()
    elif gate == 'g5':
        from gl3_r56_diagnostics import read
        ref = read(ROOT / 'data/regen/gl3_r3/dev/predictions_h5.parquet', ['date', 'ticker'], [('date', '>=', pd.Timestamp('2017-01-01')), ('date', '<=', pd.Timestamp('2021-12-31'))])
        files = sorted((OUT / 'experts').glob('*.parquet')) + sorted((OUT / 'candidates').glob('*.parquet'))
        for f in files:
            z = pd.read_parquet(f); assert (z.date < '2022-01-01').all() and np.isfinite(z.prediction).all() and (z.prediction > 0).all(), f
            m = ref.merge(z[['date', 'ticker']], how='left', indicator=True); assert (m._merge == 'both').all(), f
        print('files', len(files), 'ok')
    else:
        raise SystemExit('gate?')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', action='store_true'); ap.add_argument('--check')
    args = ap.parse_args()
    main() if args.run else check(args.check)
