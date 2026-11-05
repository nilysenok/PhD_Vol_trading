# -*- coding: utf-8 -*-
"""Раунд 6б, h=5, волна 2: слои поверх сохранённых прогнозов — Г9, Г10, Г11, Г12, Г14_FIN, Г15 (У8, У21–У25, У27, У28). DEV 2017–2021.
Все слои работают только по закрытым окнам: на строке i (день d) используются строки j <= i-LAG внутри тикера (конец цели <= d).
Запуск: PYTHONPATH=scripts/analysis python3 scripts/analysis/gl3_r6b_layers.py run | audit
"""
# xtask: файл длиннее 200 строк — один модуль во владении листа (слои, прогон, аудит)
import os
import sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[k] = '3'
import json, hashlib, shutil, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gl3_r6b_common as C

ROOT = Path(__file__).resolve().parents[2]; REG = ROOT / 'data/regen'
OUT = REG / 'gl3_r6b_h5/layers'; TAB = ROOT / 'docs/tables/gl3_r6b_h5/layers'; EVENTS = TAB / 'fit_events.jsonl'
YEARS = tuple(range(2017, 2022))
LAG = 5                      # У8: прогноз из s разрешается в s+5; на d — только s <= d-5
RVI_LEAD = 0                 # мутация У13: 1 = режим по RVI дня d+1
K11_CLIP = (.8, 1.25); K14_CLIP = (.7, 1.5)
W10 = (.25, .5, .75); HL14 = (20, 60, 120); W14 = (.5, .75)
MIN_OBS = 30; WIN_CALM, WIN_STRESS, WIN_15 = 252, 63, 252
PIT_KEYS = ['G1_standard', 'G1_regularized', 'G2_hard', 'G2_logistic', 'G3_annual', 'G3_quarterly', 'G4', 'G5', 'G6']
CALM_KEYS = ['G3_annual', 'G3_quarterly', 'G7a_BEST']


# ------------------------------------------------------------------ закрытые окна по тикеру
def _grp(a):
    """Ключ группировки; мутация У13 подменяет его (перемешанные тикеры)."""
    return a.ticker.values


def by_ticker(a, x, fn):
    return pd.Series(np.asarray(x, float)).groupby(_grp(a)).transform(fn).values


def closed(a, x, lag=None):
    """Значение x у строки s становится известным на строке s+lag (внутри тикера)."""
    lag = LAG if lag is None else lag
    return by_ticker(a, x, lambda s: s.shift(lag))


def roll_mean(a, x, n):
    return by_ticker(a, x, lambda s: s.rolling(n, min_periods=MIN_OBS).mean())


def date_pool(a, x):
    return pd.Series(np.asarray(x, float)).groupby(a.date.values).transform('mean').values


def ratio(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float); ok = np.isfinite(y) & np.isfinite(p) & (p > 0) & (y > 0)
    r = np.full(len(y), np.nan); r[ok] = y[ok] / p[ok]; return r


def qlike(r):
    return r - np.log(r) - 1


# ------------------------------------------------------------------ режим стресса (У21)
def rvi_pair(a, lead=None):
    """RVI дня d (при lead=1 — дня d+1, мутация) и медиана RVI по датам строго до d (>= 22 дат)."""
    lead = RVI_LEAD if lead is None else lead
    rvi = a.groupby('date').close_RVI.first().sort_index(); med = rvi.expanding(min_periods=22).median().shift(1)
    x = rvi.shift(-lead) if lead else rvi
    return a.date.map(x).values.astype(float), a.date.map(med).values.astype(float)


def rvw_pair(a):
    """RV_w тикера на d и 2/3-квантиль его RV_w за предыдущие 252 даты (d не входит)."""
    q = by_ticker(a, a.rv_w.values, lambda s: s.shift(1).rolling(252, min_periods=252).quantile(2 / 3))
    return a.rv_w.values.astype(float), q


def stress_rvi(a):
    x, m = rvi_pair(a); return x > m


def stress_hard(a):
    w, q = rvw_pair(a); return stress_rvi(a) | (w > q)


def z_feats(a):
    """z_RVI = ln RVI_d - ln медианы; z_RVw = ln RV_w - ln 2/3-квантиля (У22)."""
    x, m = rvi_pair(a); w, q = rvw_pair(a); lg = lambda v: np.log(np.clip(v, 1e-12, None))
    return np.column_stack([lg(x) - lg(m), lg(w) - lg(q)])


# ------------------------------------------------------------------ слои
def layer10(a, level, y, state, w, lag=None):
    """У23: k_i = среднее RV/σ̂² по 252 закрытым окнам (спокойно) или 63 (стресс по RVI); k = w k_i + (1-w) k_pool."""
    rl = closed(a, ratio(y, level), lag); ki = np.where(state, roll_mean(a, rl, WIN_STRESS), roll_mean(a, rl, WIN_CALM))
    pool = date_pool(a, ki); k = w * np.where(np.isnan(ki), pool, ki) + (1 - w) * pool
    return np.asarray(level, float) * np.where(np.isfinite(k), k, 1.)


def layer11(a, level, y, raw=None):
    """У24: k_i = clip(exp(s²_i/2)), s²_i — дисперсия (ddof 0) лог-ошибки тикера за три прошлых года, окно на годовой отсечке (конец цели < cut).
    raw=None (PIT-кандидаты без raw): лог-ошибка по уровню, а raw = уровень / exp(s²_пул/2). -> (прогноз, события)"""
    out = np.full(len(a), np.nan); ev = []; end = a._end5.values; date = a.date.values; level = np.asarray(level, float)
    e = np.log(np.clip(y, 1e-12, None)) - (np.log(level) if raw is None else raw)
    for Y in YEARS:
        cut = pd.Timestamp(f'{Y}-01-01'); va = a.year.values == Y
        tr = (end < np.datetime64(cut)) & (date >= np.datetime64(cut - pd.DateOffset(years=3))) & np.isfinite(e); n = int(tr.sum())
        if n < MIN_OBS:
            out[va] = level[va] if raw is None else np.exp(raw[va]); ev.append(_event('G11', cut, n, None, 'fallback_no_history')); continue
        pooled = float(np.var(e[tr])); g = pd.DataFrame({'t': _grp(a)[tr], 'e': e[tr]}).groupby('t').e
        local = g.var(ddof=0).where(g.count() >= MIN_OBS, pooled)  # ключ тикера для обучения и для применения берётся отдельно (мутация У13)
        s2 = pd.Series(_grp(a)[va]).map(local).fillna(pooled).values; k = np.clip(np.exp(s2 / 2), *K11_CLIP)
        base = level[va] * np.exp(-pooled / 2) if raw is None else np.exp(raw[va]); out[va] = base * k
        ev.append(_event('G11', cut, n, a._end5[tr].max(), 'calibration', dict(pooled_s2=pooled, raw=raw is not None)))
    return out, ev


def layer14(a, level, y, hl, w, lag=None):
    """У27: k_i,t = EWMA(RV/σ̂²) по закрытым наблюдениям тикера (полураспад hl), k_t — та же EWMA по пулу; k = clip(w k_i + (1-w) k_t)."""
    rl = closed(a, ratio(y, level), lag); ew = lambda s: s.ewm(halflife=hl, ignore_na=True, min_periods=MIN_OBS).mean()
    ki = by_ticker(a, rl, ew); pl = pd.Series(rl).groupby(a.date.values).mean().sort_index()
    kt = a.date.map(ew(pl)).values.astype(float)
    k = w * np.where(np.isnan(ki), kt, ki) + (1 - w) * kt; k = np.where(np.isfinite(k), np.clip(k, *K14_CLIP), 1.)
    return np.asarray(level, float) * k


def layer15(a, fin, harj, y, lag=None):
    """У28: w_i = Q^HAR/(Q^HAR+Q^hyb) по 252 закрытым окнам, w = 0,5 w_i + 0,5 w_пул, [0;1]; σ̂² = w·финал + (1-w)·HAR-J(б)."""
    ok = np.isfinite(fin) & np.isfinite(harj); fin = np.where(ok, fin, np.nan); harj = np.where(ok, harj, np.nan)
    qh = roll_mean(a, closed(a, qlike(ratio(y, fin)), lag), WIN_15); qr = roll_mean(a, closed(a, qlike(ratio(y, harj)), lag), WIN_15)
    wi = qr / (qr + qh); wt = date_pool(a, qr) / (date_pool(a, qr) + date_pool(a, qh))
    w = np.clip(.5 * np.where(np.isnan(wi), wt, wi) + .5 * wt, 0, 1); w = np.where(np.isfinite(w), w, .5)
    return w * fin + (1 - w) * harj


def fit_logistic(zz, y, f, e):
    """Как G2_logistic раунда 6: вес финала g = σ(zz·β), p = e + g (f - e) (f — финал, e — спокойная модель); QLIKE + 0,5e-4 Σβ₁², BFGS."""
    d = f - e
    def fg(b):
        g = expit(zz @ b); p = e + g * d; r = y / p
        return np.mean(r - np.log(r) - 1) + .5e-4 * np.sum(b[1:] ** 2), zz.T @ ((1 - r) / p * d * g * (1 - g)) / len(y) + np.r_[0., 1e-4 * b[1:]]
    with np.errstate(all='ignore'):  # пробные шаги линейного поиска BFGS могут переполнять σ — на результат не влияет
        return minimize(fg, np.zeros(zz.shape[1]), method='BFGS', jac=True, options={'maxiter': 200, 'gtol': 1e-8})


def gate_logistic(a, fin, calm, y, hard, z, comp):
    """У22: вес финала σ(a + b1 z_RVI + b2 z_RVw), подбор на отсечке 1 января по закрытым окнам трёх прошлых лет. -> (прогноз, события).
    Мало данных (< 100 строк) — жёсткий вариант, как в раунде 6 (событие fallback)."""
    out = np.where(hard, fin, calm).astype(float); ev = []; end = a._end5.values; date = a.date.values
    for Y in YEARS:
        cut = pd.Timestamp(f'{Y}-01-01'); va = a.year.values == Y
        tr = ((end < np.datetime64(cut)) & (date >= np.datetime64(cut - pd.DateOffset(years=3))) & np.isfinite(fin) & np.isfinite(calm)
              & np.isfinite(y) & (y > 0) & np.isfinite(z).all(axis=1)); n = int(tr.sum())
        if n < 100:
            ev.append(_event(comp, cut, n, None, 'fallback_hard', dict(reason='n<100'))); continue
        mu = z[tr].mean(0); sd = z[tr].std(0); sd[sd < 1e-8] = 1.
        zz = np.column_stack([np.ones(len(a)), np.nan_to_num((z - mu) / sd)])
        r = fit_logistic(zz[tr], y[tr], fin[tr], calm[tr])
        with np.errstate(all='ignore'): g = expit(zz[va] @ r.x)  # предупреждения BLAS (Accelerate) на конечных данных
        ok = np.isfinite(fin[va]) & np.isfinite(calm[va])
        o = np.full(va.sum(), np.nan); o[ok] = (calm[va] + g * (fin[va] - calm[va]))[ok]; out[va] = o
        ev.append(_event(comp, cut, n, a._end5[tr].max(), 'fit', dict(coef=r.x.tolist(), success=bool(r.success))))
    return out, ev


def _event(kind, cut, n, maxend, phase, detail=None):
    return dict(component=kind, h=5, cutoff=str(cut), n=int(n), max_target_end=None if maxend is None else str(maxend), phase=phase, detail=detail or {})


# ------------------------------------------------------------------ оценка и LOYO
f32 = lambda p: np.asarray(p, float).astype('float32').astype('float64')


class Scorer:
    """Метрики строк эталона dev (idx — позиции строк эталона в массивах длины a)."""
    def __init__(s, y, year, ticker, idx, m20, mrvi, final):
        s.y, s.year, s.ticker, s.idx, s.m20, s.mrvi = y, year, ticker, idx, m20, mrvi
        s.base = s.score(final); s.tb = C.ticker_means(y, final[idx], year, ticker)

    def score(s, p):
        return C.score(s.y, p[s.idx], s.year, s.m20, s.mrvi)

    def row(s, name, eligible, p):
        sc = s.score(p); k, n = C.tickers_better(C.ticker_means(s.y, p[s.idx], s.year, s.ticker), s.tb); ad = C.admission(sc, s.base, k, eligible)
        keys = [*map(str, YEARS), 'mean_QLIKE', 'stress2020', 'stressRVI', 'ratio_stress2020', 'ratio_stressRVI']
        return dict(candidate=name, eligible=eligible, **{c: sc[c] for c in keys}, better_years=ad['better_years'], stress_pass=ad['stress_pass'],
                    better_mean=ad['better_mean'], n_better_tickers=k, binom_p=C.binom_sf(k, n), admitted=ad['admitted'])


def loyo(names, scores, preds, a):
    """У4: прогноз года Y — вариант с наименьшим средним QLIKE по остальным четырём годам."""
    out = np.full(len(a), np.nan); pick = {}
    for Y in YEARS:
        best = min(names, key=lambda k: np.mean([scores[k][str(y)] for y in YEARS if y != Y])); pick[Y] = best
        m = a.year.values == Y; out[m] = preds[best][m]
    return out, pick


class Book:
    """Реестр прогнозов, оценок и строк таблицы."""
    def __init__(s, sc, a): s.sc, s.a, s.preds, s.scores, s.rows, s.picks = sc, a, {}, {}, [], {}

    def keep(s, name, p):
        s.preds[name] = f32(p); s.scores[name] = s.sc.score(s.preds[name]); return s.preds[name]

    def add(s, name, eligible, p):
        p = s.keep(name, p); s.rows.append(s.sc.row(name, eligible, p)); return p

    def pick(s, name, names, eligible=True):
        p, s.picks[name] = loyo(names, s.scores, s.preds, s.a); return s.add(name, eligible, p)


# ------------------------------------------------------------------ данные
def disk(stage):
    TAB.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    out = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, check=True).stdout
    with (TAB / 'disk.log').open('a') as f: f.write(str(pd.Timestamp.now(tz='UTC')) + ' ' + stage + '\n' + out)
    if shutil.disk_usage('/').free < 20 * 1024 ** 3: raise RuntimeError('DISK_STOP: less than 20GiB free')


def load():
    import gl3_r6h5_data as D
    from gl3_r56_diagnostics import read
    D.disk = disk; d = D.Data(); a = d.a; src = REG / 'gl3_r3/dev/predictions_h5.parquet'
    p = read(src, ['date', 'ticker', 'raw', 'rv_pred', 'har_corr'])
    inp = dict(fin=d.series(p, 'rv_pred'), raw=np.log(d.series(p, 'raw')), harj=d.series(p, 'har_corr'), y=a._y5.values.astype(float))
    pit = REG / 'gl3_r6b_h5/pit/candidates'
    inp['pit'] = {k: d.series(read(pit / f'pit_{k}.parquet', ['date', 'ticker', 'prediction']), 'prediction') for k in PIT_KEYS + ['G7a_BEST']}
    return d, inp


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    want = {'data/regen/gl3_r3/dev/predictions_h5.parquet': '3eb3822fc5c36bce', 'data/regen/gl3_r6b_h5/pit/candidates/pit_G3_annual.parquet': '5e23b717adc3b85d',
            'data/regen/gl3_r6b_h5/pit/candidates/pit_G3_quarterly.parquet': '2e933487509d2aa3', 'data/regen/gl3_r6b_h5/pit/candidates/pit_G7a_BEST.parquet': '5117e27f05c8962f',
            'scripts/analysis/gl3_r6h5_data.py': 'b4ff178a4fe12ddd', 'scripts/analysis/gl3_r6b_common.py': 'e17e408bff8c1376'}
    got = {k: sha(ROOT / k) for k in want}
    for k, v in want.items(): assert got[k].startswith(v), (k, got[k])
    return got


def best_pit(book, a, pit):
    """BEST_PIT: LOYO-выбор из pit_G1…pit_G6 (как Study.choose раунда 6) + сверка с выбором, записанным в pit_G7a_BEST."""
    for k in PIT_KEYS: book.keep(k, pit[k])
    p, pick = loyo(PIT_KEYS, book.scores, book.preds, a)
    rec = json.loads((ROOT / 'docs/tables/gl3_r6b_h5/pit/selection_pit.json').read_text())['recipes']
    parent = {Y: rec['pit_G7a_BEST'][str(Y)]['parent'] for Y in YEARS}
    recorded = {Y: [k for k in PIT_KEYS if rec['pit_' + k][str(Y)] == parent[Y]] for Y in YEARS}
    return f32(p), pick, recorded


def record(ev):
    TAB.mkdir(parents=True, exist_ok=True)
    with EVENTS.open('a') as f:
        for e in ev:
            if e['max_target_end'] is not None: assert pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) < pd.Timestamp('2022-01-01')
            f.write(json.dumps({**e, 'time_utc': pd.Timestamp.now(tz='UTC').isoformat()}, ensure_ascii=False) + '\n')


def build_all(a, inp, book):
    """Все строки таблицы. -> (picks, events, best_pit_info)."""
    fin, raw, harj, y, pit = inp['fin'], inp['raw'], inp['harj'], inp['y'], inp['pit']; ev = []
    best, bp_pick, bp_rec = best_pit(Book(book.sc, a), a, pit); stress = stress_hard(a); st_rvi = stress_rvi(a); z = z_feats(a)
    for k in CALM_KEYS: book.keep('calm_' + k, pit[k])
    calm9, book.picks['calm_G9'] = loyo(['calm_' + k for k in CALM_KEYS], book.scores, book.preds, a)
    calm9 = f32(calm9)
    book.add('Г9_hard', False, hard_gate(fin, calm9, stress))
    pl, e2 = gate_logistic(a, fin, calm9, y, stress, z, 'G9'); book.add('Г9_logistic', False, pl); ev += e2
    book.pick('Г9', ['Г9_hard', 'Г9_logistic'])
    for base, lev in (('FIN', fin), ('BEST', best)):
        for w in W10: book.add(f'Г10_{base}_w{w}', False, layer10(a, lev, y, st_rvi, w))
        book.pick(f'Г10_{base}_LOYO', [f'Г10_{base}_w{w}' for w in W10])
    for base, lev, rw in (('FIN', fin, raw), ('BEST', best, None)):
        p, e = layer11(a, lev, y, rw); book.add(f'Г11_{base}', True, p); ev += e
    c12 = book.preds['Г10_BEST_LOYO']; book.add('Г12_hard', False, hard_gate(fin, c12, stress))
    pl, e3 = gate_logistic(a, fin, c12, y, stress, z, 'G12'); book.add('Г12_logistic', False, pl); ev += e3
    book.pick('Г12', ['Г12_hard', 'Г12_logistic'])
    grid = [f'Г14_FIN_hl{h}_w{w}' for h in HL14 for w in W14]
    for h in HL14:
        for w in W14: book.add(f'Г14_FIN_hl{h}_w{w}', False, layer14(a, fin, y, h, w))
    book.pick('Г14_FIN_LOYO', grid); book.add('Г15', True, layer15(a, fin, harj, y))
    return ev, dict(best_pit_pick=bp_pick, best_pit_recorded_in_pit_G7a_BEST=bp_rec)


def hard_gate(fin, calm, stress):
    """У22, жёсткий: в стрессе — финал, иначе — спокойная модель."""
    return np.where(stress, fin, calm)


def run():
    assert not (TAB / 'candidates.csv').exists(), 'уже посчитано; повторный запуск запрещён (У12)'
    disk('load'); prov = provenance(); d, inp = load(); a = d.a; ref = d.ref; t0 = pd.Timestamp.now()
    assert np.array_equal(stress_rvi(a), a._state.values), 'режим RVI не совпал с Data._state'
    sc = Scorer(ref.rv_actual.values, ref.year.values, ref.ticker.values, d.idx, ref.date.between('2020-03-01', '2020-06-30').values,
                a.loc[d.idx, '_stress'].values.astype(bool), inp['fin'])
    assert abs(sc.base['mean_QLIKE'] - .209965392) < 1e-8 and abs(sc.base['stress2020'] - .342295788) < 1e-8, sc.base
    book = Book(sc, a); disk('layers'); ev, info = build_all(a, inp, book); record(ev)
    t = pd.DataFrame(book.rows); OUT.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True); t.to_csv(TAB / 'candidates.csv', index=False)
    for r in book.rows:
        z = ref[['date', 'ticker']].copy(); z['prediction'] = book.preds[r['candidate']][d.idx].astype('float32'); z.to_parquet(OUT / (fname(r['candidate']) + '_dev.parquet'), index=False)
    fits = [e for e in ev if e['phase'] == 'fit']; fb = [e for e in ev if e['phase'].startswith('fallback')]
    (TAB / 'selection.json').write_text(json.dumps(dict(picks={k: {str(y): v for y, v in p.items()} for k, p in book.picks.items()},
        **{k: {str(y): v for y, v in p.items()} for k, p in info.items()}, fallbacks=fb, control_read=False), ensure_ascii=False, indent=1))
    (TAB / 'attempt_counts.json').write_text(json.dumps(dict(candidate_rows=len(t), grid_points=int(t.candidate.str.contains('_w|_hl').sum()), logistic_fits=len(fits),
        fallbacks=len(fb), calibrations_G11=sum(e['phase'] == 'calibration' for e in ev), events=len(ev), seconds=(pd.Timestamp.now() - t0).total_seconds())))
    (TAB / 'provenance.json').write_text(json.dumps(dict(sha256=prov, control_read=False), indent=1)); print(t[['candidate', 'mean_QLIKE', 'better_years', 'n_better_tickers', 'admitted']].to_string())


def fname(name): return name.replace('Г', 'G').replace('λ', 'lam')


def audit():
    """G4/G5: независимый пересчёт метрик из parquet (≤ 1e-11), покрытие, конечность, ledger."""
    from gl3_r56_diagnostics import read
    t = pd.read_csv(TAB / 'candidates.csv'); ref = read(REG / 'gl3_r3/dev/predictions_h5.parquet', ['date', 'ticker', 'year', 'rv_actual'],
                                                      [('date', '>=', pd.Timestamp('2017-01-01')), ('date', '<=', pd.Timestamp('2021-12-31'))])
    for r in t.to_dict('records'):
        p = pd.read_parquet(OUT / (fname(r['candidate']) + '_dev.parquet')); assert p.prediction.dtype == np.dtype('float32') and p.date.max() < pd.Timestamp('2022-01-01')
        z = ref.merge(p, on=['date', 'ticker'], validate='1:1'); assert len(z) == len(ref) == len(p) and np.isfinite(z.prediction).all() and (z.prediction > 0).all()
        l = qlike(z.rv_actual / z.prediction.astype(float)); ys = l.groupby(z.year).mean()
        for y, q in ys.items(): np.testing.assert_allclose(q, r[str(y)], rtol=0, atol=1e-11)
        np.testing.assert_allclose(ys.mean(), r['mean_QLIKE'], rtol=0, atol=1e-11)
        m = z.date.between('2020-03-01', '2020-06-30').values; np.testing.assert_allclose(l[m].mean(), r['stress2020'], rtol=0, atol=1e-11)
    ev = [json.loads(s) for s in EVENTS.read_text().splitlines()]
    for e in ev:
        assert e['h'] == 5 and (e['max_target_end'] is None or pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) < pd.Timestamp('2022-01-01'))
    assert (TAB / 'selection.json').exists() and len(t) == 24, len(t)
    print('AUDIT OK', len(t), 'rows', len(ev), 'events')


if __name__ == '__main__':
    {'run': run, 'audit': audit}[sys.argv[1]]()
