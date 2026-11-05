# -*- coding: utf-8 -*-
"""Ж-190, агент 2: потикерный HAR-J(б) со сжатием (HAR_i, λ) и эталоны FIN_PIT / HAR-J(б) для h = 1, 5, 22. DEV 2016–2021, без тестового ридера.
Запуск (из корня): PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_zh190_har.py {har|fin1|assemble|audit} <h|gate>. Модели не сохраняются."""
import os
import sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[k] = '1'
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gl3_r6b_common as C

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/regen/gl3_zh190/etalony'; TAB = ROOT / 'docs/tables/gl3_zh190/etalony'; EVENTS = TAB / 'fit_events.jsonl'
REG = ROOT / 'data/regen'; FIN5_SHA = '474e92e5fcf9f0c9'
HARJ = ['rv_d', 'rv_w', 'rv_m', 'jump', 'jump_annualized', 'jump_lag1', 'jump_lag5', 'jump_lag22']
LAMS = (0., .25, .5, .75, 1.); STEPS = range(2016, 2022); MIN_OBS = 30; YEARS = C.YEARS
NUM = [*map(str, YEARS), 'mean_QLIKE', 'stress2020', 'stressRVI', 'ratio_stress2020', 'ratio_stressRVI']
LN = lambda l: f'lam{int(round(l * 100))}'


# ------------------------------------------------------------------ ledger, диск
def record(h, kind, cut, n, maxend, phase='fit', detail=None):
    assert pd.Timestamp(maxend) < pd.Timestamp(cut) < pd.Timestamp('2022-01-01'), ('конец цели >= cut', kind, cut, maxend)
    r = dict(component=kind, h=h, cutoff=str(cut), n=int(n), max_target_end=str(maxend), phase=phase, detail=detail or {}, time_utc=pd.Timestamp.now(tz='UTC').isoformat())
    with EVENTS.open('a') as f:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')


def disk(stage):
    TAB.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, check=True).stdout
    with (TAB / 'disk.log').open('a') as f:
        f.write(str(pd.Timestamp.now(tz='UTC')) + ' ' + stage + '\n' + out)
    if shutil.disk_usage('/').free < 20 * 1024 ** 3:
        raise RuntimeError('DISK_STOP: less than 20GiB free')


# ------------------------------------------------------------------ чистое ядро HAR_i (без данных)
def design(a):
    return np.nan_to_num(np.log(a[HARJ].values.astype(float).clip(1e-10)), nan=0., posinf=0., neginf=0.)


def train_mask(year, end_y, y, vy, emb=True):
    """Обучение на отсечке 1 января vy: год строки < vy, цель есть; при emb конец цели тоже < vy (r2_nogi.trmask)."""
    m = (np.asarray(year) < vy) & np.isfinite(y)
    return m & (np.asarray(end_y) < vy) if emb else m


def ols(X, ly, m):
    lr = LinearRegression().fit(X[m], ly[m]); return np.r_[lr.intercept_, lr.coef_]


def lin(X, b):
    return b[0] + X @ b[1:]


def fit_cutoff(tick, X, ly, tr):
    """-> (β_pool, {тикер: β_i или None}); None, если строк тикера в окне < MIN_OBS (берётся β_pool)."""
    bp = ols(X, ly, tr)
    return bp, {tk: (ols(X, ly, tr & (tick == tk)) if (tr & (tick == tk)).sum() >= MIN_OBS else None) for tk in np.unique(tick)}


def ln_models(tick, X, bp, bi):
    lp = lin(X, bp); lt = lp.copy()
    for tk, b in bi.items():
        if b is not None:
            m = tick == tk; lt[m] = lin(X[m], b)
    return lp, lt


def past_s2(errs, vy, first, emb=True):
    """s²_pool и s²_i по ошибкам ≤ 3 прошлых val-лет; эмбарго: конец цели < 1 января vy (шаг 1 — без поправки)."""
    pool = [errs[k] for k in range(max(first, vy - 3), vy)]
    if not pool:
        return 0., {}
    P = pd.concat(pool)
    if emb:
        P = P[P.te < np.datetime64(f'{vy}-01-01')]
    s2p = float(np.var(P.ep.values)) if len(P) >= MIN_OBS else 0.
    return s2p, {tk: (float(np.var(g.et.values)) if len(g) >= MIN_OBS else s2p) for tk, g in P.groupby('ticker')}


def forecast(tick, lp, lt, s2p, si, lam):
    """β* = λβ_i + (1−λ)β_pool (линейно по ln), s²* = λs²_i + (1−λ)s²_pool; уровень exp(ln + s²*/2)."""
    ln = lam * lt + (1 - lam) * lp; s2 = lam * pd.Series(tick).map(si).fillna(s2p).values + (1 - lam) * s2p
    return ln, np.exp(ln + .5 * s2)


def run_h(a, X, y, end, emb=True, rec=None, steps=STEPS):
    """a: date, ticker, year; X: регрессоры; y: цель (NaN — нет); end: конец окна цели. -> pred[λ], inwin (без s²), s2, beta."""
    assert a.date.max() < pd.Timestamp('2022-01-01'), 'данные >= 2022'
    tick = a.ticker.values; year = a.year.values; te = np.asarray(end, 'datetime64[ns]'); end_y = np.asarray(pd.DatetimeIndex(te).year, float)
    ly = np.log(y + 1e-10); errs = {}; pred = {l: np.full(len(a), np.nan) for l in LAMS}; inw = []; s2r = []; br = []
    for vy in steps:
        tr = train_mask(year, end_y, y, vy, emb); bp, bi = fit_cutoff(tick, X, ly, tr); lp, lt = ln_models(tick, X, bp, bi)
        sel = (year == vy) & np.isfinite(y) & (y > 1e-12); v = year == vy
        errs[vy] = pd.DataFrame(dict(ticker=tick[sel], te=te[sel], ep=(ly - lp)[sel], et=(ly - lt)[sel]))
        s2p, si = past_s2(errs, vy, min(steps), emb)
        if rec:
            rec('HARi_pool', vy, int(tr.sum()), te[tr].max())
        for tk, b in bi.items():
            n = int((tr & (tick == tk)).sum()); br.append(dict(cutoff=vy, ticker=tk, n=n, fallback_pool=b is None, **{f'b{j}': float((bp if b is None else b)[j]) for j in range(len(bp))}))
            s2r.append(dict(cutoff=vy, ticker=tk, s2_i=si.get(tk, s2p), s2_pool=s2p))
            if rec and b is not None:
                rec('HARi_ticker', vy, n, te[tr & (tick == tk)].max(), dict(ticker=tk))
        cols = {}
        for l in LAMS:
            ln, p = forecast(tick, lp, lt, s2p, si, l); pred[l][v] = p[v]; cols[f'ln_{LN(l)}'] = ln[tr].astype('float32')
        inw.append(pd.DataFrame(dict(cutoff=vy, date=a.date.values[tr], ticker=tick[tr], **cols)))
    return dict(pred=pred, inwin=pd.concat(inw, ignore_index=True), s2=pd.DataFrame(s2r), beta=pd.DataFrame(br))


def loyo(scores, preds, year):
    """О5: прогноз года Y — точка с наименьшим средним QLIKE по остальным четырём годам."""
    out = np.full(len(year), np.nan); pick = {}
    for Y in YEARS:
        pick[Y] = min(preds, key=lambda k: np.mean([scores[k][str(y)] for y in YEARS if y != Y])); m = np.asarray(year) == Y; out[m] = preds[pick[Y]][m]
    return out, pick


# ------------------------------------------------------------------ данные и файлы
def load():
    """Data раундов 5–6 (кадр 'original'). Файл сдвинутого измерения 'alt' в репозитории отсутствует и здесь не нужен: подставляется пустой, кадр 'alt' не используется."""
    import gl3_r56_data as D
    D.disk_guard = disk; real = D.read

    def read_(path, cols, filters=None):
        if Path(path).name == 'grid10_shift_mean.parquet' and not Path(path).exists():
            return pd.DataFrame({c: pd.Series(dtype='datetime64[ns]' if c == 'date' else 'object' if c == 'ticker' else 'float64') for c in cols})
        return real(path, cols, filters)
    D.read = read_
    try:
        return D.Data()
    finally:
        D.read = real


def pfile(name, h):
    return OUT / f'{name}_h{h}.parquet'


def forecast_file(a, level, path):
    """date, ticker, prediction (float32), 2016–2021, конечный и положительный."""
    z = a[['date', 'ticker']].copy(); z['prediction'] = np.asarray(level, 'float32'); z = z[(z.date >= '2016-01-01') & (z.date < '2022-01-01')]
    z = z[np.isfinite(z.prediction) & (z.prediction > 0)]; path.parent.mkdir(parents=True, exist_ok=True); z.to_parquet(path, index=False)


def on_ref(path, ref):
    m = ref[['date', 'ticker']].merge(pd.read_parquet(path), on=['date', 'ticker'], how='left', validate='1:1')
    assert m.prediction.notna().all() and len(m) == len(ref), path; return m.prediction.values.astype(float)


def ref_masks(d, h):
    ref, idx = d.refs[h]
    return ref, idx, ref.date.between('2020-03-01', '2020-06-30').values, d.frames['original'].loc[idx, '_stress'].values.astype(bool)


def har(h):
    """HAR_i по h: гейт λ=0 = har_corr, прогнозы λ и LOYO, in-window, s²_i, β по отсечкам."""
    disk(f'har h{h}'); d = load(); b = d.frames['original']; ref, idx, m20, mrvi = ref_masks(d, h)
    ev = pd.read_parquet(REG / f'gl3_r3/dev/predictions_h{h}.parquet', columns=['date', 'ticker', 'har_corr']); ev = ev[ev.date <= '2021-12-31']
    arch = np.full(len(b), np.nan); arch[pd.MultiIndex.from_frame(b[['date', 'ticker']]).get_indexer(pd.MultiIndex.from_frame(ev[['date', 'ticker']]))] = ev.har_corr.values
    TAB.mkdir(parents=True, exist_ok=True); res = run_h(b, design(b), b[f'_y{h}'].values, b[f'_end{h}'].values, rec=lambda k, vy, n, mx, dt=None: record(h, k, f'{vy}-01-01', n, mx, 'fit', dt))
    m = np.isfinite(arch); rel = float(np.max(np.abs(res['pred'][0.][m] / arch[m] - 1))); assert rel <= 1e-8, ('λ=0 ≠ har_corr', h, rel)
    (TAB / f'g2_lam0_h{h}.json').write_text(json.dumps(dict(h=h, max_rel=rel, n=int(m.sum()), tol=1e-8)))
    f32 = lambda p: p.astype('float32').astype('float64'); P = {l: f32(res['pred'][l]) for l in LAMS}
    sc = {LN(l): C.score(ref.rv_actual, P[l][idx], ref.year, m20, mrvi) for l in LAMS}; lo, pick = loyo(sc, {LN(l): P[l] for l in LAMS}, b.year.values)
    for l in LAMS:
        forecast_file(b, P[l], pfile(f'HARi_{LN(l)}', h))
    lo[b.year.values < 2017] = np.nan; forecast_file(b, lo, pfile('HARi_LOYO', h)); res['inwin'].to_parquet(pfile('HARi_inwin', h), index=False)
    res['s2'].to_csv(TAB / f'har_i_s2_h{h}.csv', index=False); res['beta'].to_csv(TAB / f'har_i_beta_h{h}.csv', index=False)
    pd.DataFrame([dict(h=h, year=Y, lam=float(pick[Y][3:]) / 100) for Y in YEARS]).to_csv(TAB / f'loyo_pick_h{h}.csv', index=False)
    print('HAR_i', h, 'rel', rel, 'LOYO', {Y: pick[Y] for Y in YEARS})


# ------------------------------------------------------------------ FIN_PIT h=1: честное переобучение финала (путь FIN_fresh раундов 5–6, ES по О2 с embargo 1)
def gap_boundary(dates_all, cut, gap):
    dates = np.sort(np.unique(dates_all)); pos = np.searchsorted(dates, np.datetime64(cut), 'left'); assert pos > gap; return pd.Timestamp(dates[pos - gap])


def fin1():
    import xgboost as xgb
    import gl3_r56_engine as B
    import gl3_r56_data as D56
    disk('fin1'); (TAB / 'diagnostics_manifest.json').write_text('{"control_read":false}')
    B.EVENTS = EVENTS; B.TAB = TAB; B.OUT = OUT; B.disk_guard = disk; d = load()
    fz = [{k: (np.nan if v is None else v) for k, v in p.items()} for p in json.loads((ROOT / 'docs/tables/gl3_r5_r6/frozen_warmup_configuration.json').read_text())['fin1params']]
    assert all(np.allclose([p[k] for k in ('learning_rate', 'reg_lambda', 'max_depth')], [q[k] for k in ('learning_rate', 'reg_lambda', 'max_depth')]) for p, q in zip(fz, d.fin1params))

    class Fin1(B.Engine):
        def boost(self, h, mode, measure, cut, seed, lib, x, y, tr, w, style='a', objective='mse', base=None, c=1., warm=None, huber=False, tag=''):
            assert (h, lib, style, objective, huber) == (1, 'xgb', 'a', 'mse', False)
            b = self.data.frames[measure]; vs = pd.Timestamp(np.sort(b.loc[tr, 'date'].unique())[-126]); edge = gap_boundary(b.date.values, vs, h)
            ti = tr & (b.date < vs).values & (b[f'_end{h}'] < edge).values; vi = tr & (b.date >= vs).values; assert b.loc[ti, f'_end{h}'].max() < edge <= vs
            p = dict(tree_method='hist', max_depth=5, eta=.05, min_child_weight=20, reg_lambda=10., subsample=.8, colsample_bytree=.8, nthread=1, seed=seed, verbosity=0,
                     objective='reg:squarederror', eval_metric='rmse')
            for dest, src in [('max_depth', 'max_depth'), ('eta', 'learning_rate'), ('min_child_weight', 'min_child_weight'), ('reg_lambda', 'reg_lambda'), ('alpha', 'reg_alpha'), ('subsample', 'subsample'), ('colsample_bytree', 'colsample_bytree')]:
                if src in warm and np.isfinite(warm[src]):
                    p[dest] = int(warm[src]) if dest == 'max_depth' else float(warm[src])
            first = xgb.train(p, xgb.DMatrix(x[ti], label=y[ti], weight=w[ti]), 300, evals=[(xgb.DMatrix(x[vi], label=y[vi]), 'past')], early_stopping_rounds=20, verbose_eval=False)
            trees = int(first.best_iteration) + 1; B.record(tag + lib, h, cut, mode, measure, seed, int(ti.sum()), b.loc[ti, f'_end{h}'].max(), phase='inner_ES', detail=dict(trees=trees, embargo=h, edge=str(edge)))
            model = xgb.train(p, xgb.DMatrix(x[tr], label=y[tr], weight=w[tr]), trees); B.record(tag + lib, h, cut, mode, measure, seed, int(tr.sum()), b.loc[tr, f'_end{h}'].max(), phase='refit', detail=dict(trees=trees))
            return model.predict(xgb.DMatrix(x), output_margin=True).astype(float)

    e = Fin1(d); b = d.frames['original']; r = np.full(len(b), np.nan)
    for cut, end in D56.periods('annual'):
        va = ((b.date >= cut) & (b.date < end)).values; _, y, tr, w = e.context(1, 'annual', 'original', cut); x, names, _ = e.features(1, 'annual', 'original', cut)
        x = x[:, [i for i, n in enumerate(names) if n != 'ln_HAR_b']]; yy = np.log(np.clip(y, 1e-12, None)); q = pd.Series(yy[tr]).groupby(b.loc[tr, 'ticker'].values).quantile([.01, .99]).unstack()
        yy[tr] = np.clip(yy[tr], b.loc[tr, 'ticker'].map(q[.01]), b.loc[tr, 'ticker'].map(q[.99]))
        pp = [e.boost(1, 'annual', 'original', cut, 42, 'xgb', x, yy, tr, w, warm=p, tag='FIN_') for p in fz]; r[va] = np.log(np.mean([B.level(z) for z in pp], axis=0))[va]
    forecast_file(b, e.calibrate(1, 'annual', 'original', r, 'scalar')[0], pfile('FIN_PIT', 1)); shutil.rmtree(OUT / 'cache', ignore_errors=True); print('FIN_PIT h1 written')


# ------------------------------------------------------------------ сборка эталонов и таблицы
def assemble(h):
    disk(f'assemble h{h}'); d = load(); b = d.frames['original']; ref, idx, m20, mrvi = ref_masks(d, h)
    z = pd.read_parquet(REG / f'gl3_r3/dev/predictions_h{h}.parquet', columns=['date', 'ticker', 'har_corr']); z = z[z.date <= '2021-12-31']
    forecast_file(z, z.har_corr.values, pfile('HARJb', h))
    if h == 5:
        shutil.copyfile(REG / 'gl3_r6b_h5/pit/experts/FIN_PIT.parquet', pfile('FIN_PIT', 5))
    if h == 22:
        shutil.copyfile(pfile('HARJb', 22), pfile('FIN_PIT', 22))
    P = {'FIN_PIT': on_ref(pfile('FIN_PIT', h), ref), 'HARJb': on_ref(pfile('HARJb', h), ref), **{f'HARi_{LN(l)}': on_ref(pfile(f'HARi_{LN(l)}', h), ref) for l in LAMS},
         'HARi_LOYO': on_ref(pfile('HARi_LOYO', h), ref), 'final_saved': ref.rv_pred.values.astype(float)}
    tm = {k: C.ticker_means(ref.rv_actual, P[k], ref.year, ref.ticker) for k in P}; rows = []
    for k, p in P.items():
        sc = C.score(ref.rv_actual, p, ref.year, m20, mrvi); r = dict(h=h, candidate=k, eligible=k in ('FIN_PIT', 'HARJb', 'HARi_LOYO'), note='справочно' if k == 'final_saved' else '', **{c: sc[c] for c in NUM})
        for e in ('FIN_PIT', 'HARJb', 'HARi_LOYO'):
            n, m = C.tickers_better(tm[k], tm[e]); r[f'n_better_vs_{e}'] = n; r[f'binom_p_vs_{e}'] = C.binom_sf(n, m)
        rows.append(r)
    pd.DataFrame(rows).to_csv(TAB / f'etalony_h{h}.csv', index=False); print(pd.DataFrame(rows)[['candidate', 'mean_QLIKE', 'stress2020', 'stressRVI']].to_string())


# ------------------------------------------------------------------ аудит G2–G5
def audit(gate):
    HS = (1, 5, 22); ev = [json.loads(s) for s in EVENTS.read_text().splitlines()]
    if gate == 'g2':
        for h in HS:
            g = json.loads((TAB / f'g2_lam0_h{h}.json').read_text()); print(g); assert g['max_rel'] <= 1e-8
    elif gate == 'g3':
        bad = [e for e in ev if not pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) < pd.Timestamp('2022-01-01')]
        f1 = [e for e in ev if e['component'] == 'FIN_xgb' and e['h'] == 1]; print('events', len(ev), 'bad', len(bad), 'FIN_h1 refit', sum(e['phase'] == 'refit' for e in f1), 'ES', sum(e['phase'] == 'inner_ES' for e in f1))
        assert ev and not bad and sum(e['phase'] == 'refit' for e in f1) == 30 and sum(e['phase'] == 'inner_ES' for e in f1) == 30
        import hashlib
        sha = hashlib.sha256(pfile('FIN_PIT', 5).read_bytes()).hexdigest()[:16]; print('FIN_PIT_h5 sha', sha); assert sha == FIN5_SHA
        assert pfile('FIN_PIT', 5).read_bytes() == (REG / 'gl3_r6b_h5/pit/experts/FIN_PIT.parquet').read_bytes() and pfile('FIN_PIT', 22).read_bytes() == pfile('HARJb', 22).read_bytes(); print('FIN_PIT_h22 == HARJb_h22 побайтно')
    elif gate == 'g4':
        from gl3_r56_diagnostics import read
        f = read(ROOT / 'data/regen/features_pit_v3.parquet', ['date', 'close_RVI']).groupby('date').close_RVI.first().sort_index()
        stress = f.rolling(253, min_periods=253).apply(lambda z: np.mean(z[:-1] <= z[-1]), raw=True) > 2 / 3
        for h in HS:
            t = pd.read_csv(TAB / f'etalony_h{h}.csv'); ref = read(REG / f'gl3_r3/dev/predictions_h{h}.parquet', ['date', 'ticker', 'year', 'rv_actual', 'rv_pred'], [('date', '>=', pd.Timestamp('2017-01-01')), ('date', '<=', pd.Timestamp('2021-12-31'))])
            ms = {}
            for r in t.to_dict('records'):
                p = ref.rv_pred.values.astype(float) if r['candidate'] == 'final_saved' else None
                if p is None:
                    z = pd.read_parquet(pfile(r['candidate'], h)); assert z.prediction.dtype == np.dtype('float32') and z.date.max() < pd.Timestamp('2022-01-01') and np.isfinite(z.prediction).all() and (z.prediction > 0).all()
                    p = ref[['date', 'ticker']].merge(z, on=['date', 'ticker'], how='left', validate='1:1').prediction.values.astype(float); assert np.isfinite(p).all(), (h, r['candidate'])
                q = ref.rv_actual.values / p; l = pd.Series(q - np.log(q) - 1); yq = l.groupby(ref.year.values).mean(); st = ref.date.map(stress).fillna(False).values.astype(bool); m20 = ref.date.between('2020-03-01', '2020-06-30').values
                for y, v in yq.items():
                    assert abs(v - r[str(y)]) <= 1e-11, (h, r['candidate'], y)
                for c, v in (('mean_QLIKE', yq.mean()), ('stress2020', l[m20].mean()), ('stressRVI', l[st].mean()), ('ratio_stress2020', q[m20].mean()), ('ratio_stressRVI', q[st].mean())):
                    assert abs(v - r[c]) <= 1e-11, (h, r['candidate'], c, v, r[c])
                ms[r['candidate']] = l.groupby([ref.ticker.values, ref.year.values]).mean().groupby(level=0).mean()
            for r in t.to_dict('records'):
                for e in ('FIN_PIT', 'HARJb', 'HARi_LOYO'):
                    k = int((ms[r['candidate']] < ms[e] - 1e-12).sum()); assert k == r[f'n_better_vs_{e}'], (h, r['candidate'], e)
        print('AUDIT G4 OK')
    elif gate == 'g5':
        for h in HS:
            t = pd.read_csv(TAB / f'etalony_h{h}.csv'); assert len(t) == 9 and set(t.candidate) >= {'FIN_PIT', 'HARJb', 'HARi_LOYO', 'final_saved', 'HARi_lam0', 'HARi_lam100'}, h
            w = pd.read_parquet(pfile('HARi_inwin', h)); assert sorted(w.cutoff.unique()) == list(STEPS) and w.filter(like='ln_').notna().all().all(), h
            assert all(pd.Timestamp(f'{c}-01-01') > g.date.max() for c, g in w.groupby('cutoff')) and len(w.filter(like='ln_').columns) == 5
        print('AUDIT G5 OK')


if __name__ == '__main__':
    {'har': har, 'fin1': fin1, 'assemble': assemble, 'audit': audit}[sys.argv[1]](*(int(x) if x.isdigit() else x for x in sys.argv[2:]))
