#!/usr/bin/env python3
"""Д22, слой 1: HAR-X⁺ = HAR-J + расширенный реестр слагаемых (эластичная сеть по времени, как harx.py), цели h ∈ {1, 5, 22} и подгоризонты
2–5, 6–22 (ln среднего RV за дни t+2..t+5 и t+6..t+22), эмбарго на конце обучающего окна (строки, чей горизонт заходит в год оценки, не обучают).
Реестр TERMS расширяется здесь (harx.py не меняется):
  harx.TERMS (RVI, RV_m IMOEX, RV рынка, LHAR 1/5/22, PC1, HARQ, RSV±, RV66/132)
  + RRV Parkinson / Garman–Klass d/w/m (features_pit_v3_rrv; окна кончаются днём t, как rv дня t в матрице)
  + календарь окна горизонта [t+a, t+b] (a, b — границы цели в торговых днях, по будним дням): календарных дней в окне (выходные), число заседаний ЦБ / экспираций / FOMC в окне, дней до ближайшего — только события с known_from ≤ t (дивидендов нет: файлы пусты)
  + ночной гэп ln(r²_on + 1e-6), r_on = ln(open_t / close_{t−1}) (D13 intraday1850; |r| ≥ 0,4 — пропуск).
Поправка s² (Б0.1): rv_pred_{ord,rob,sm,smrob} (val шага k — оценка шага k−1; тест — шага k), rv_pred — без поправки.
Выход: data/regen/D22/HXP_{tgt}_val.parquet, test_locked/D22_HXP/HXP_{tgt}_test.parquet, HXP_lnall.parquet (ln-прогноз модели шага k на всех
строках: hxp_{tgt}_{k} — признак Д8.12 и база классификатора), HXP_coefs.csv.
python3 scripts/analysis/d22_harxp.py [--tgts 1 5 22 2_5 6_22] [--leak-test]"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harx as HX
import gibrid_har as GH

DATA = HX.DATA
REGEN = HX.REGEN
FEAT = REGEN / 'features_pit_v3_rrv.parquet'
OUT = REGEN / 'D22'
LOCK = REGEN / 'test_locked' / 'D22_HXP'
INTRA = REGEN / 'D13' / 'intraday1850.parquet'
TGTS = ('1', '5', '22', '2_5', '6_22')
BOUNDS = {'1': (1, 1), '5': (1, 5), '22': (1, 22), '2_5': (2, 5), '6_22': (6, 22)}  # дни горизонта [a, b] в торговых днях
CAL_TYPES = ('cbr', 'expirations', 'fomc')


# ---------------------------------------------------------------- цели подгоризонтов
def add_targets(A):
    """rv_target_h{1,5,22} (load_pit) + rv_target_h2_5 = (5·m5 − RV(t+1))/4, rv_target_h6_22 = (22·m22 − 5·m5)/17 (уровни; ≤ 0 -> NaN)."""
    m1, m5, m22 = A['rv_target_h1'], A['rv_target_mean_h5'], A['rv_target_mean_h22']
    A['rv_target_h2_5'] = ((5 * m5 - m1) / 4).where(lambda s: s > 1e-12)
    A['rv_target_h6_22'] = ((22 * m22 - 5 * m5) / 17).where(lambda s: s > 1e-12)
    # год последнего дня горизонта (эмбарго): дата строки t+b тикера
    for tg, (a, b) in BOUNDS.items():
        e = A.groupby('ticker')['date'].shift(-b)
        A[f'_tend_{tg}'] = pd.to_datetime(e).dt.year
    return A


# ---------------------------------------------------------------- новые слагаемые
def _col(c):
    return lambda A: A[c].astype('float64')


def t_r2on(A):
    I = pd.read_parquet(INTRA, columns=['date', 'ticker', 'gap_on'])
    I['date'] = pd.to_datetime(I['date'])
    k = A[['date', 'ticker']].assign(date=pd.to_datetime(A['date'])).merge(I, on=['date', 'ticker'], how='left')
    return pd.Series(np.log(k['gap_on'].values ** 2 + 1e-6), index=A.index)


_EV = {}


def _events(cal_dir=DATA / 'calendar'):
    if 'ev' not in _EV:
        ev = {}
        for ty in CAL_TYPES:
            d = pd.read_csv(cal_dir / f'{ty}.csv')
            ev[ty] = (pd.to_datetime(d['date']).values.astype('datetime64[D]'), pd.to_datetime(d['known_from']).values.astype('datetime64[D]'))
        _EV['ev'] = ev
    return _EV['ev']


def cal_window(dates, a, b, ev=None):
    """По уникальным датам t: окно горизонта по будним дням — с (t + a−1 будних дней) + 1 день по t + b будних дней (np.busday_offset;
    будни известны в t, праздники биржи не учитываются). cal_bd — календарных дней в окне (выходные внутри горизонта);
    по типу события: число событий (known_from ≤ t) в окне, дней до ближайшего будущего (known_from ≤ t; нет -> 90, cap 90)."""
    ev = _events() if ev is None else ev
    ud = np.array(sorted(pd.unique(pd.to_datetime(dates))), dtype='datetime64[D]')
    st = np.busday_offset(ud, 0, roll='backward')
    lo = (np.busday_offset(st, a - 1) if a > 1 else ud) + np.timedelta64(1, 'D')
    hi = np.busday_offset(st, b)
    res = {'cal_bd': ((hi - lo) / np.timedelta64(1, 'D') + 1).astype(float)}
    for ty, (d, kf) in ev.items():
        n, dte = np.zeros(len(ud)), np.full(len(ud), 90.0)
        for i, t in enumerate(ud):
            k = d[(kf <= t) & (d > t)]
            n[i] = ((k >= lo[i]) & (k <= hi[i])).sum()
            if len(k):
                dte[i] = min(90.0, float((k.min() - t) / np.timedelta64(1, 'D')))
        res[f'cal_n_{ty}'], res[f'cal_dte_{ty}'] = n, dte
    return pd.DataFrame(res, index=pd.DatetimeIndex(ud))


def _cal(tg, field):
    def f(A):
        key = ('cal', tg, len(A), str(A['date'].iloc[-1]))
        if key not in _EV:
            _EV[key] = cal_window(A['date'], *BOUNDS[tg])
        G = _EV[key]
        return pd.Series(G[field].reindex(pd.to_datetime(A['date'])).values, index=A.index)
    return f


TERMS = dict(HX.TERMS)
TERMS.pop('ln_sig1', None)  # σ̂₁ финала второго круга в Д22 не входит (сигнал h=1 — слой 3, r̂₁ из слоя 2)
for m in ('pk', 'gk'):
    for w in ('d', 'w', 'm'):
        TERMS[f'rrv_{m}_{w}'] = _col(f'rrv_{m}_{w}')
TERMS['r2_on'] = t_r2on
CAL_FIELDS = ['cal_bd'] + [f'cal_{k}_{ty}' for ty in CAL_TYPES for k in ('n', 'dte')]
for tg in TGTS:
    for fld in CAL_FIELDS:
        TERMS[f'{fld}@{tg}'] = _cal(tg, fld)

BASE = ['ln_rvi2', 'ln_rvm_imoex', 'ln_rvm_mkt', 'lhar_1', 'lhar_5', 'lhar_22', 'pc1', 'harq', 'ln_rsv_pos', 'ln_rsv_neg',
        'rrv_pk_d', 'rrv_pk_w', 'rrv_pk_m', 'rrv_gk_d', 'rrv_gk_w', 'rrv_gk_m', 'r2_on']


def terms_for(tg, drop=()):
    """Слагаемые цели tg: BASE + календарь окна этой цели + RV66/132 (h=22 и подгоризонт 6–22)."""
    t = BASE + [f'{f}@{tg}' for f in CAL_FIELDS] + (['ln_rv66', 'ln_rv132'] if tg in ('22', '6_22') else [])
    return [x for x in t if x not in set(drop) and x.split('@')[0] not in set(drop)]


def build_terms(A, names):
    out = {n: (TERMS[n](A) if n in TERMS else A[n].astype('float64')) for n in names}
    return pd.DataFrame(out, index=A.index).replace([np.inf, -np.inf], np.nan)


def load():
    A, harj = HX.load(FEAT)
    A = add_targets(A)
    return A, harj


# ---------------------------------------------------------------- шаг (harx.forecast_step + эмбарго)
def forecast_step(A, tg, step, harj, terms, Xt, log=None, grid=True):
    """-> (val_df, test_df, ln_all Series). Val-год vy = 2015+k, тест ty = 2016+k; обучение — строки года < vy, у которых последний день
    горизонта тоже в году < vy (эмбарго); α, l1 — внутренняя валидация (обучение < vy−1 с тем же эмбарго, MSE ln на годе vy−1)."""
    target = f'rv_target_h{tg}'
    Xt = Xt[terms]
    Hf = lambda idx: GH.clean(np.log(A.loc[idx, harj].values.clip(1e-10)))
    Zf = lambda idx, ref: Xt.loc[idx].fillna(Xt.loc[ref].mean()).fillna(0.0).values.astype('float64')
    ok = A[target].notna()
    tend = A[f'_tend_{tg}']
    vy, ty = 2015 + step, 2016 + step
    tr_mask = lambda y: (A.year < y) & ok & (tend < y)
    tp = A.index[tr_mask(vy)]; vi = A.index[(A.year == vy) & ok]; td = A.index[A.year == ty]
    best = (0.0, 1.0, np.nan)
    if grid and terms:
        i_tr = A.index[tr_mask(vy - 1)]; i_va = A.index[(A.year == vy - 1) & ok]
        if len(i_tr) and len(i_va):
            yt, yv = np.log(A.loc[i_tr, target].values + 1e-10), np.log(A.loc[i_va, target].values + 1e-10)
            Htr, Ztr, Hva, Zva = Hf(i_tr), Zf(i_tr, i_tr), Hf(i_va), Zf(i_va, i_tr)
            res = []
            for l1, al in [(l1, al) for l1 in HX.L1S for al in HX.ALPHAS] + [(1.0, 0.0)]:
                p, _ = HX.fit_en(Htr, Ztr, yt, al, l1)
                res.append((al, l1, float(np.mean((yv - p(Hva, Zva)) ** 2))))
            best = min(res, key=lambda r: r[2])
    y = np.log(A.loc[tp, target].values + 1e-10)
    pred, coef = HX.fit_en(Hf(tp), Zf(tp, tp), y, best[0], best[1])
    if log is not None:
        log.append(dict(tgt=tg, step=step, val_year=vy, alpha=best[0], l1_ratio=best[1], inner_mse=best[2], n_train=len(tp),
                        n_nonzero=int((np.abs(coef['g_std']) > 1e-10).sum()), **{f'g_{t}': v for t, v in zip(terms, coef['g_raw'])},
                        **{f'b_{c}': v for c, v in zip(['const'] + list(harj), coef['b'])}))
    out = []
    for idx, yr in ((vi, vy), (td, ty)):
        out.append(pd.DataFrame({'date': A.loc[idx, 'date'].values, 'ticker': A.loc[idx, 'ticker'].values, 'year': yr, 'step': step,
                                 'rv_actual': A.loc[idx, target].values, 'rv_pred': np.clip(np.exp(pred(Hf(idx), Zf(idx, tp))), 1e-10, None)}))
    lnall = pd.Series(pred(Hf(A.index), Zf(A.index, tp)), index=A.index)
    return out[0], out[1], lnall, coef


def run(tgts, tag='HXP', drop=(), steps=range(1, 11), A=None, harj=None, Xall=None, write=True):
    if A is None:
        A, harj = load()
    lock = REGEN / 'test_locked' / f'D22_{tag}'
    OUT.mkdir(parents=True, exist_ok=True); lock.mkdir(parents=True, exist_ok=True)
    names = sorted(set(sum([terms_for(tg, drop) for tg in tgts], [])))
    Xall = build_terms(A, names) if Xall is None else Xall
    lnall = A[['date', 'ticker']].copy()
    res, logs = {}, []
    for tg in tgts:
        terms = terms_for(tg, drop)
        vals, tests = [], []
        for k in steps:
            v, t, la, _ = forecast_step(A, tg, k, harj, terms, Xall, logs)
            vals.append(v); tests.append(t); lnall[f'{tag.lower()}_{tg}_{k}'] = la.values
        val, test = HX.add_bias(pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True))
        res[tg] = (val, test)
        if write:
            val.to_parquet(OUT / f'{tag}_{tg}_val.parquet', index=False)
            test.to_parquet(lock / f'{tag}_{tg}_test.parquet', index=False)
        print(f'{tag} {tg}: готово', flush=True)
    if write:
        lnall.to_parquet(OUT / f'{tag}_lnall.parquet', index=False)
        pd.DataFrame(logs).to_csv(OUT / f'{tag}_coefs.csv', index=False)
    return res, lnall, pd.DataFrame(logs)


# ---------------------------------------------------------------- тест «будущее не влияет» (слагаемые)
def leak_test(A, names, cuts=('2019-06-30', '2021-12-31', '2023-06-30'), seed=20261026):
    """Как harx.leak_test: числовые входы строк date > t -> бутстрап; календарь — события с date > t и known_from > t сдвигаются шумом;
    гэп — файл intraday1850 подменяется (строки > t -> шум). Слагаемые на строках ≤ t должны совпасть побайтно."""
    global INTRA
    rng = np.random.default_rng(seed)
    base = build_terms(A, names)
    rows = []
    saved_intra, saved_ev = INTRA, _EV.pop('ev', None)
    for t in cuts:
        t = pd.Timestamp(t)
        B = A.copy(); fut = B.date > t
        for c in B.columns:
            if c in ('date', 'ticker', 'year') or c.startswith('_tend') or not np.issubdtype(B[c].dtype, np.number):
                continue
            pool = B.loc[~fut, c].dropna().values
            if len(pool):
                B.loc[fut, c] = rng.choice(pool, fut.sum())
        I = pd.read_parquet(saved_intra); f = pd.to_datetime(I.date) > t
        I.loc[f, 'gap_on'] = rng.normal(0, 0.01, f.sum())
        tmp = OUT / '_leak_intra.parquet'; I.to_parquet(tmp, index=False); INTRA = tmp
        ev = _events(); ev2 = {}
        for ty, (d, kf) in ev.items():  # будущие события, о которых в t ещё не знали, — случайный сдвиг дат
            d2 = d.copy(); m = kf > np.datetime64(t.date())
            d2[m] = d2[m] + rng.integers(-20, 20, m.sum()).astype('timedelta64[D]')
            ev2[ty] = (d2, kf)
        for k in [k for k in _EV if k != 'ev']:
            _EV.pop(k)
        _EV['ev'] = ev2
        X2 = build_terms(B, names)
        _EV.clear(); _EV['ev'] = ev; INTRA = saved_intra
        past = ~fut
        for n in names:
            a1, a2 = base.loc[past, n].values, X2.loc[past, n].values
            rows.append(dict(t=str(t.date()), term=n, n=int(past.sum()), mismatch=int((~((a1 == a2) | (np.isnan(a1) & np.isnan(a2)))).sum())))
    (OUT / '_leak_intra.parquet').unlink(missing_ok=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tgts', nargs='*', default=list(TGTS))
    ap.add_argument('--leak-test', action='store_true')
    ap.add_argument('--hj', action='store_true', help='абляция «минус слой 1»: HAR-J с тем же эмбарго (без слагаемых), tag HJ')
    a = ap.parse_args()
    t0 = time.time()
    A, harj = load()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.leak_test:
        names = sorted(set(sum([terms_for(tg) for tg in TGTS], [])))
        r = leak_test(A, names)
        r.to_csv(OUT / 'leak_test_hxp.csv', index=False)
        print(r.groupby('t').mismatch.sum().to_string()); print('всего расхождений:', int(r.mismatch.sum())); return
    if a.hj:
        run(a.tgts, tag='HJ', drop=BASE + CAL_FIELDS + ['ln_rv66', 'ln_rv132'], A=A, harj=harj)
    else:
        run(a.tgts, A=A, harj=harj)
    print(f'слой 1 готов, {time.time() - t0:.0f} c', flush=True)


if __name__ == '__main__':
    main()
