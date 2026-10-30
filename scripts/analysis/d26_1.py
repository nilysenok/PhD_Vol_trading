#!/usr/bin/env python3
"""Д26.1 и Д26.2 (круг h=5 на разработке, агент Б; поручение «Ъ Д26», владелец 26.10, 09:35/09:45).

Д26.1 — D22-слой-2 на цель h=5 (ln среднего RV за [t+1, t+5]): рецепт D22_h1 (Д8.12 на HAR-X⁺), но основа — HAR-X⁺₅ слоя 1 Д22
  (HAR-J + реестр Д22, календарь окна [t+1, t+5], ночной гэп; эмбарго; data/regen/D22/HXP_lnall.parquet, столбцы hxp_5_{k}); ln HAR-X⁺₅
  модели шага — признак бустинга (на обучении — внутри выборки, как в Д8.12); признаки как у D22_h1: v3 + RRV PK/GK d/w/m + гэп + календарь
  окна горизонта (@5); винзоризация цели 1/99 по тикеру; подбор — протокол B v2 (tune_L2.py, подмены l2_lib только в своём процессе);
  итог — среднее LGBM + XGB.
Д26.2 — итерированный D22_h1 на 5 шагов (точки D22_h1, модели шагов переобучаются ровно как в D22_h1 и сверяются с его val-файлами):
  внешние признаки заморожены на t, лаги RV обновляются собственным прогнозом; прогноз h=5 — ln среднего exp(прогнозов) с поправкой по лиду.

Жёсткие запреты круга: метрики — только val шагов 1–6 (2016–2021), строки шагов ≥ 7 в оценку не попадают (assert в dev_only);
test_locked/ не читается — только запись тестовых прогнозов своих дизайнов, без метрик.

Режимы:
  --tune [--var v]   Д26.1: протокол B v2 (tag D26_1{var}) -> data/regen/D26_1{var}_{xgb,lgbm}_h5{,_top5,_seeds4344}_val.parquet
  ...                (остальные режимы — ниже, в main)
python3 -W ignore scripts/analysis/d26_1.py --tune --workers 4"""
import os
os.environ['L2_FEAT'] = '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3_rrv.parquet'
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, hashlib, json, subprocess, sys, time, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l2_lib as L
import d22_harxp as P
import d73_features as D73

R = L.REGEN
D22 = R / 'D22'
LOCK = R / 'test_locked'          # только запись
SEEDS = (42, 43, 44)
SEEDS10 = tuple(range(42, 52))
KEY = ['date', 'ticker', 'step']
DEV_MAX = 6

# ---------------------------------------------------------------- варианты Д26.1
# основа (столбец ln HAR-X⁺ шага), календарь окна, cfg l2_lib
VARS = {
    '': dict(lncol='hxp_5_{k}', cal='5', cfg={'harf': True, 'wins': True}),   # рецепт D22_h1 на h=5
    'nw': dict(lncol='hxp_5_{k}', cal='5', cfg={'harf': True}),                # без винзоризации цели (2020: обрезка 99% у h=5 срезает шок)
    'v9': dict(lncol='hxp_5_{k}', cal='5', cfg={'harf': True, 'v9': True}),    # структура D12_h5 (V9 + Д8.12) на основе HAR-X⁺₅, без винзоризации
    'H1': dict(lncol='hxp_1_{k}', cal='1', cfg={'harf': True, 'wins': True}),  # D22_h1 как есть (основа Д26.2)
}
_V = {'v': VARS['']}
_har_ln0, _split0, _bf0, _lb0 = L.har_ln, L.split, L.build_features, L.load_base


def extra_cols(var=None):
    v = _V['v'] if var is None else VARS[var]
    return ['rrv_pk_d', 'rrv_pk_w', 'rrv_pk_m', 'rrv_gk_d', 'rrv_gk_w', 'rrv_gk_m', 'r2_on'] + [f'{f}@{v["cal"]}' for f in P.CAL_FIELDS]


def har_ln_x(tp, parts, h, hc, w=None, huber=False, y=None):
    """ln HAR-X⁺ модели шага k = (последний год обучения + 1) − 2015 на строках каждой части (обучение — внутри выборки)."""
    k = int(tp['year'].max()) + 1 - 2015
    col = _V['v']['lncol'].format(k=k)
    return [d[col].values for d in parts]


def _split10(A, h, step):
    if step != 10:
        return _split0(A, h, step)
    t = L.tcol(h)
    tr = A[A.year < 2026]
    return tr[tr.year != 2025].dropna(subset=[t]), tr[tr.year == 2025].dropna(subset=[t]), A[A.year == 2026], 2025, 2026


def load_base_x(matrix=None, blocks=(), h=None):
    A, fc = _lb0(matrix, blocks, h)
    A = A.merge(pd.read_parquet(D22 / 'HXP_lnall.parquet'), on=['date', 'ticker'], how='left', validate='one_to_one')
    assert A['hxp_5_1'].notna().all() and A['hxp_1_1'].notna().all()
    return A, fc


def build_features_x(A, FC, h, cfg):
    A2, cols, cats, mono = _bf0(A, FC, h, cfg)
    ex = extra_cols()
    X = P.build_terms(A2, ex)
    for c in ex:
        A2[c] = X[c].values
    return A2, cols + [c for c in ex if c not in cols], cats, mono


def patch(var=''):
    _V['v'] = VARS[var]
    L.har_ln, L.split, L.build_features, L.load_base = har_ln_x, _split10, build_features_x, load_base_x


def tag1(var=''):
    return f'D26_1{var}'


def tune(workers, var=''):
    patch(var)
    import tune_L2 as T
    sys.argv = ['tune_L2.py', '--h', '5', '--cfg', json.dumps(VARS[var]['cfg']), '--space', 'B', '--workers', str(workers), '--tag', tag1(var)]
    T.main()


# ---------------------------------------------------------------- оценка OOS (шаги 2–6, val 2017–2021); как d26_lib агента А
CODES = ('ord', 'rob', 'sm', 'smrob')
D12_REF = dict(design='wrap2|V0_XGB|B-sm|L0', m1720=0.215906, q2021=0.156640)  # final3/ranking_h5.csv
ETALON_H5 = dict(code='sm', m1720=0.2410, q2021=0.1727)                      # HAR-J с поправкой s² sm


def dev_only(df):
    """Строки шагов 1–6 (val 2016–2021). Всё, что идёт в метрики, проходит через эту функцию."""
    d = df[df.step <= DEV_MAX].copy()
    assert d['step'].max() <= DEV_MAX and (pd.to_datetime(d['date']).dt.year <= 2021).all()
    return d


def qlike(y, f):
    r = y / np.clip(f, 1e-12, None)
    return float(np.mean(r - np.log(r) - 1))


def s2_est(code, E, st):
    """ord/rob — остатки val шага st; sm/smrob — объединённые остатки шагов st−2..st (как l2_lib.s2_est, gibrid_bias)."""
    e = np.concatenate([E[j] for j in range(max(1, st - 2) if code in ('sm', 'smrob') else st, st + 1) if j in E])
    if code in ('rob', 'smrob'):
        return float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2)
    return float(np.var(e))


def years(val, code='sm'):
    """OOS QLIKE по годам 2017–2021 (шаги 2–6), Б0.1 кода code с s² шага t−1 (raw — без поправки)."""
    v = dev_only(val)
    v = v[v.rv_actual > 1e-12]
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
    out = {}
    for st in range(2, DEV_MAX + 1):
        g = v[v.step == st]
        F = 1.0 if code == 'raw' else np.exp(s2_est(code, E, st - 1) / 2)
        out[2015 + st] = qlike(g.rv_actual.values, g.rv_pred.values * F)
    return out


def m1720(q):
    return float(np.mean([q[y] for y in range(2017, 2021)]))


def avg_lvl(a, b):
    """Среднее двух прогнозов по уровню RV (как нога wrap2 '2' у D12_h5)."""
    m = a[KEY + ['year', 'rv_actual', 'rv_pred']].merge(b[KEY + ['rv_pred']], on=KEY, suffixes=('', '_b'), validate='one_to_one')
    return m.assign(rv_pred=(m.rv_pred + m.rv_pred_b) / 2).drop(columns='rv_pred_b')


def avg_ln(a, b):
    m = a[KEY + ['year', 'rv_actual', 'rv_pred']].merge(b[KEY + ['rv_pred']], on=KEY, suffixes=('', '_b'), validate='one_to_one')
    return m.assign(rv_pred=np.sqrt(m.rv_pred * m.rv_pred_b)).drop(columns='rv_pred_b')


def read_seeds(prefix, m, h, regen=R):
    """{seed: val DataFrame шагов 1–6} из {prefix}_{m}_h{h}_val (seed 42) и _seeds4344_val."""
    a = pd.read_parquet(regen / f'{prefix}_{m}_h{h}_val.parquet')
    a = a.assign(seed=42) if 'seed' not in a else a
    fr = [a]
    p = regen / f'{prefix}_{m}_h{h}_seeds4344_val.parquet'
    if p.exists():
        fr.append(pd.read_parquet(p))
    d = dev_only(pd.concat(fr, ignore_index=True))
    return {int(s): g.drop(columns=[c for c in ('trial', 'rank', 'point') if c in g]).reset_index(drop=True) for s, g in d.groupby('seed')}


def d12_ref():
    """D12_h5 по seed: среднее XGB и LGBM по уровню (wrap2 '2'), Б0.1 sm. seed 42 обязан дать 0,215906 / 0,156640."""
    X, Lg = read_seeds('D12_h5', 'xgb', 5), read_seeds('D12_h5', 'lgbm', 5)
    out = {s: years(avg_lvl(X[s], Lg[s]), 'sm') for s in SEEDS}
    assert abs(m1720(out[42]) - D12_REF['m1720']) < 5e-6 and abs(out[42][2021] - D12_REF['q2021']) < 5e-6, out[42]
    return out


def entry(q, ref):
    """Условие входа по 2017–2020: Δ̄ = среднее Q(s)/среднее ref(s) − 1 < −3 %; d_s = Q(s) − ref(s), σ = СКО d_s (ddof = 1), |d̄| > 2σ."""
    q, rr = np.asarray(q, float), np.asarray(ref, float)
    d = q - rr
    db, sd = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else float('nan')
    rel = float(q.mean() / rr.mean() - 1)
    return dict(n=len(q), mean_q=float(q.mean()), ref=float(rr.mean()), rel=rel, d_mean=db, sigma=sd, two_sigma=2 * sd,
                rel_s=[float(x) for x in q / rr - 1], pass3=rel < -0.03, pass2s=(db < 0) and (abs(db) > 2 * sd))


# ================================================================ Д26.2: итерированный D22_h1 на 5 шагов
OUT2 = R / 'D26_2'
LOCK2 = LOCK / 'D26_2'
TAG_H1 = 'D22_h1'
J = 5
RVF = ('rv', 'rv_d', 'rv_w', 'rv_2w', 'rv_m', 'rv_q', 'rv_annualized')
COMP = ('bv', 'rsv_pos', 'rsv_neg', 'rv_morning', 'rv_midday', 'rv_evening')   # компоненты дня: R̂ × доля (среднее за 22 строки ≤ t)
DIML = ('rskew', 'rkurt')                                                     # безразмерные: среднее за 22 строки ≤ t
RRVD = ('rrv_pk', 'rrv_gk')
HARJ = ('rv_d', 'rv_w', 'rv_m', 'jump', 'jump_annualized', 'jump_lag1', 'jump_lag5', 'jump_lag22')
ITV = ('lvl', 'full')   # lvl — обновляются только уровни RV (rv, rv_d..rv_q, √252rv; HAR-часть и HARQ в HAR-X⁺₁); full — ещё компоненты дня
PLUG = ('med', 'mean')  # подстановка R̂ в лаги: med = exp(ℓ) (медиана); mean = exp(ℓ + s₁²/2) (условное среднее уровня, s₁² — Б0.1 sm hyb₁)
G2 = {}


def cal_lead(dates, j, ev=None):
    """Календарь окна h=1 для дня состояния t' = t + (j−1) будних дней: окно (t', t'+1 будний], события только с known_from ≤ t
    (что известно в день прогноза t), дней до ближайшего — от t'. При j = 1 совпадает с d22_harxp.cal_window(dates, 1, 1)."""
    ev = P._events() if ev is None else ev
    ud = np.array(sorted(pd.unique(pd.to_datetime(dates))), dtype='datetime64[D]')
    st = np.busday_offset(ud, 0, roll='backward')
    tp = ud if j == 1 else np.busday_offset(st, j - 1)
    lo, hi = tp + np.timedelta64(1, 'D'), np.busday_offset(st, j)
    res = {'cal_bd': ((hi - lo) / np.timedelta64(1, 'D') + 1).astype(float)}
    for ty, (d, kf, kt) in ev.items():
        n, dte = np.zeros(len(ud)), np.full(len(ud), 90.0)
        for i, t in enumerate(ud):
            k = d[D73.izvestna(kf, kt, t) & (d > tp[i])]
            n[i] = ((k >= lo[i]) & (k <= hi[i])).sum()
            if len(k):
                dte[i] = min(90.0, float((k.min() - tp[i]) / np.timedelta64(1, 'D')))
        res[f'cal_n_{ty}'], res[f'cal_dte_{ty}'] = n, dte
    return pd.DataFrame(res, index=pd.DatetimeIndex(ud))


def history(A):
    """Прошлое тикера (строки ≤ t, сдвиги только назад): лаги RV 0..66, скачка 0..22, RRV дня 0..21, доли компонент и средние за 22 строки."""
    o = A.sort_values(['ticker', 'date']).index
    g = A.loc[o].groupby('ticker')
    H = {'rv': np.column_stack([g['rv'].shift(m).reindex(A.index).values for m in range(67)]),
         'jump': np.column_stack([g['jump'].shift(m).reindex(A.index).values for m in range(23)])}
    for c in RRVD:
        H[c] = np.column_stack([g[f'{c}_d'].shift(m).reindex(A.index).values for m in range(22)])
    rv = A['rv'].where(A['rv'] > 1e-12)
    roll = lambda s: s.loc[o].groupby(A.loc[o, 'ticker']).transform(lambda x: x.rolling(22, min_periods=1).mean()).reindex(A.index).values
    H['share'] = {c: roll(A[c] / rv) for c in COMP + tuple(f'{c}_d' for c in RRVD)}
    H['mean'] = {c: roll(A[c]) for c in DIML}
    return H


def synth(idx, j, Rhat, H, var):
    """Признаки дня состояния t+j−1 для строк idx (позиции в A): ряд s = (RV_{t−66..t}, R̂_1..R̂_{j−1}).
    -> dict {столбец: значения}; для j = 1 — пустой (признаки строки как есть)."""
    if j == 1:
        return {}
    rv = H['rv'][idx]                                   # [:, m] = RV_{t−m}
    s = np.column_stack([rv[:, ::-1], Rhat[:, :j - 1]])  # s[:, 66] = RV_t, s[:, 66+i] = R̂_i
    cur = s.shape[1] - 1                                # индекс дня состояния t+j−1
    win = lambda n: np.nanmean(s[:, cur - n:cur], axis=1)  # лаговые окна (без дня состояния), как rv_d/rv_w/... в матрице
    out = {'rv': s[:, cur], 'rv_d': s[:, cur - 1], 'rv_w': win(5), 'rv_2w': win(10), 'rv_m': win(22), 'rv_q': win(66)}
    out['rv_annualized'] = np.sqrt(252 * np.clip(out['rv'], 0, None))
    if var == 'full':
        sh = {c: H['share'][c][idx] for c in H['share']}
        for c in COMP:
            out[c] = out['rv'] * sh[c]
        for c in DIML:
            out[c] = H['mean'][c][idx]
        # скачки дня: реальные для дней ≤ t, для дней прогноза — max(R̂ − BV̂, 0), BV̂ = R̂ × доля BV
        jr = H['jump'][idx][:, ::-1]                                    # [:, 22] = J_t
        jf = np.clip(Rhat[:, :j - 1] * (1 - sh['bv'][:, None]), 0, None)
        js = np.column_stack([jr, jf]); jc = js.shape[1] - 1
        out['jump'] = js[:, jc]
        out['jump_annualized'] = np.sqrt(252 * out['jump'])
        out['bv_annualized'] = np.sqrt(252 * np.clip(out['bv'], 0, None))
        for lg in (1, 5, 22):
            out[f'jump_lag{lg}'] = js[:, jc - lg]
        for c in RRVD:
            rr = H[c][idx][:, ::-1]                                     # [:, 21] = RRV_d дня t
            rs = np.column_stack([rr, Rhat[:, :j - 1] * sh[f'{c}_d'][:, None]]); rc = rs.shape[1] - 1
            out[f'{c}_d'] = rs[:, rc]
            out[f'{c}_w'] = np.nanmean(rs[:, rc - 4:rc + 1], axis=1)
            out[f'{c}_m'] = np.nanmean(rs[:, rc - 21:rc + 1], axis=1)
    return out


def hxp_delta(rows, k, syn, cal, var, q, mkt):
    """Сдвиг ln HAR-X⁺₁ модели шага k при переходе к состоянию t+j−1: Σ b_c Δ ln HAR-J_c + Σ g_τ Δ слагаемого (обновляемые слагаемые:
    HARQ, календарь окна; full — ещё RSV±, RRV, ln RV рынка). Остальные слагаемые X заморожены на t (смещение δ_t, как x6 Д22)."""
    cf = G2['coef'][k]
    d = np.zeros(len(rows))
    lnc = lambda x: np.log(np.clip(np.asarray(x, float), 1e-10, None))
    for c in HARJ:
        if c in syn:
            d += cf[f'b_{c}'] * np.nan_to_num(lnc(syn[c]) - lnc(rows[c].values))
    upd = {}
    if 'rv_d' in syn:
        upd['harq'] = (lnc(syn['rv_d']) * q, np.log(rows['rv_d'].astype(float).clip(lower=1e-12).values) * q)
    for f in P.CAL_FIELDS:
        upd[f'{f}@1'] = (cal[f'{f}@1'], rows[f'{f}@1'].values)
    if var == 'full':
        for c in ('rsv_pos', 'rsv_neg'):
            upd[f'ln_{c}'] = (np.log(np.clip(syn[c], 1e-12, None)), np.log(rows[c].astype(float).clip(lower=1e-12).values))
        for c in RRVD:
            for w in ('d', 'w', 'm'):
                upd[f'{c}_{w}'] = (syn[f'{c}_{w}'], rows[f'{c}_{w}'].values)
        upd['ln_rvm_mkt'] = mkt
    for tname, (new, old) in upd.items():
        g = cf.get(f'g_{tname}', 0.0)
        if g and np.isfinite(g):
            d += g * np.nan_to_num(np.asarray(new, float) - np.asarray(old, float))
    return d


_CAP = {}


def _capture():
    import xgboost, lightgbm
    if 'x0' not in _CAP:
        _CAP['x0'], _CAP['l0'] = xgboost.train, lightgbm.train

        def xt(*a, **k):
            b = _CAP['x0'](*a, **k); _CAP['b'] = b; return b

        def lt(*a, **k):
            b = _CAP['l0'](*a, **k); _CAP['b'] = b; return b
        xgboost.train, lightgbm.train = xt, lt


def fit_h1(m, step, seed):
    """Модель D22_h1 шага (точка D22_h1; шаг 10 — точка шага 6), обучение как в d22_h1.py. -> (predict(X) -> ln, names)."""
    import xgboost as xgb
    from tune_L2 import SPACEB
    _capture()
    A, cols = G2['A'], G2['cols']
    p = {k: G2['sel'][(m, min(step, 9) if step <= 9 else 6)][k] for k, *_ in SPACEB[m]}
    D = L.step_data(A, cols, 1, step, m, VARS['H1']['cfg'], TAG_H1)
    ref = L.make_ref(m, A, cols, 1, VARS['H1']['cfg'], TAG_H1) if step > 1 else None
    pv, pt, bi, _, _ = L.fit(m, p, D, VARS['H1']['cfg'], seed=seed, ref=ref)
    b = _CAP.pop('b')
    if m == 'xgb':
        ft = ['q'] * len(D['names'])
        pr = lambda X: b.predict(xgb.DMatrix(X, feature_types=ft), iteration_range=(0, bi + 1), output_margin=True)
    else:
        pr = lambda X: b.predict(X, raw_score=True)
    chk = float(np.max(np.abs(pr(D['X']['va']) - pv)))
    assert chk < 1e-5, ('захват модели', m, step, seed, chk)
    return pr, D['names']


def iterate_rows(idx, step, models, var, plant=False, pf=1.0):
    """idx — позиции строк в G2['A'] (один год). -> массив ℓ [n, J] (hyb₁ = среднее ln XGB и ln LGBM) и по моделям."""
    A, H = G2['A'], G2['H']
    rows = A.iloc[idx]
    k = step
    q = G2['q'][idx]
    Rhat = np.zeros((len(idx), J))
    ell = np.zeros((len(idx), J)); per = {m: np.zeros((len(idx), J)) for m in models}
    dates = pd.to_datetime(rows['date']).values
    for j in range(1, J + 1):
        syn = synth(idx, j, Rhat, H, var)
        if j == 1:
            cal = {f'{f}@1': rows[f'{f}@1'].values for f in P.CAL_FIELDS}
            dl = np.zeros(len(idx))
        else:
            C = G2['cal'][j]
            cal = {f'{f}@1': C[f].reindex(pd.DatetimeIndex(dates)).values for f in P.CAL_FIELDS}
            mkt = None
            if var == 'full':
                rm = pd.Series(syn['rv_m']).groupby(dates).transform('mean').values
                mkt = (np.log(np.clip(rm, 1e-12, None)), G2['lnmkt'][idx])
            dl = hxp_delta(rows, k, syn, cal, var, q, mkt)
        for m, (pr, names) in models.items():
            X = np.empty((len(idx), len(names)), dtype=np.float64)
            for i, c in enumerate(names):
                if c == 'har_ln':
                    X[:, i] = rows[f'hxp_1_{k}'].values + dl
                elif c in syn:
                    X[:, i] = syn[c]
                elif c in cal:
                    X[:, i] = cal[c]
                else:
                    X[:, i] = rows[c].values
            per[m][:, j - 1] = pr(L.clean(X).astype(np.float32))
        ell[:, j - 1] = np.mean([np.log(np.clip(np.exp(per[m][:, j - 1]), 1e-10, None)) for m in models], axis=0)
        Rhat[:, j - 1] = np.exp(ell[:, j - 1]) * pf
        if plant and j == 1:  # подложенная утечка: вместо R̂₁ — фактический RV(t+1)
            Rhat[:, 0] = np.nan_to_num(G2['fut1'][idx])
    return ell, per


def prep_h1(A=None):
    """Матрица D22_h1 (v3 + RRV + гэп + календарь @1 + ln HAR-X⁺) + история, календарь лидов, коэффициенты HAR-X⁺₁."""
    patch('H1')
    if A is None:
        A, FC = L.load_base()
    else:
        FC = G2['FC']
    A2, cols, _, _ = L.build_features(A, FC, 1, VARS['H1']['cfg'])
    A2 = A2.reset_index(drop=True)
    from tune_L2 import SPACEB  # noqa: F401
    cf = pd.read_csv(D22 / 'HXP_coefs.csv'); cf = cf[cf.tgt.astype(str) == '1']
    T = P.build_terms(A2, ['harq', 'ln_rvm_mkt'])
    lrd = np.log(A2['rv_d'].astype(float).clip(lower=1e-12))
    q = np.where(np.abs(lrd) > 0, T['harq'].values / lrd.values, 0.0)
    fut1 = A2.sort_values(['ticker', 'date']).groupby('ticker')['rv'].shift(-1).reindex(A2.index).values  # только для подложенной утечки
    G2.update(A=A2, cols=cols, FC=FC, H=history(A2), fut1=fut1, q=np.nan_to_num(q), lnmkt=T['ln_rvm_mkt'].values,
              cal={j: cal_lead(A2['date'], j) for j in range(2, J + 1)},
              coef={int(r.step): r._asdict() for r in cf.itertuples(index=False)}, sel=L.selected_params(TAG_H1, 1))
    c1 = cal_lead(A2['date'], 1)
    for f in P.CAL_FIELDS:
        assert np.allclose(c1[f].reindex(pd.to_datetime(A2['date'])).values, A2[f'{f}@1'].values, equal_nan=True), f
    return A2


def s2_plug(seed):
    """s₁² hyb₁ D22_h1 (Б0.1 sm) для подстановки 'mean': val шага k — остатки val шагов k−3..k−1 (шаг 1 — свои), тест шага k — k−2..k.
    Процедура (как s² тестовых прогнозов в final2): остатки val шагов 7–9 входят только в s² шагов 8–9 и теста, не печатаются.
    Для добавочных seed (45–51) сохранённого D22_h1 нет — берётся среднее s² seed 42/43/44 того же шага (↺ в отчёте)."""
    if seed not in SEEDS:
        ss = [s2_plug(s) for s in SEEDS]
        return {k: float(np.mean([s[k] for s in ss])) for k in ss[0]}
    fr = {}
    for m in ('xgb', 'lgbm'):
        a = pd.read_parquet(R / f'{TAG_H1}_{m}_h1_val.parquet').assign(seed=42)
        b = pd.read_parquet(R / f'{TAG_H1}_{m}_h1_seeds4344_val.parquet')
        d = pd.concat([a, b], ignore_index=True)
        fr[m] = d[d.seed == seed][KEY + ['rv_actual', 'rv_pred']]
    w = fr['xgb'].merge(fr['lgbm'], on=KEY + ['rv_actual'], suffixes=('_x', '_l'), validate='one_to_one')
    w = w[w.rv_actual > 1e-12]
    w['e'] = np.log(w.rv_actual + 1e-10) - (np.log(w.rv_pred_x) + np.log(w.rv_pred_l)) / 2
    E = {int(st): g.e.values for st, g in w.groupby('step')}
    out = {}
    for k in range(1, 10):
        out[('val', k)] = s2_est('sm', E, k - 1 if k > 1 else 1)
        out[('test', k)] = s2_est('sm', E, k)
    return out


def _job_iter(a):
    step, seed, parts, plant = a
    models = {m: fit_h1(m, step, seed) for m in ('xgb', 'lgbm')}
    A = G2['A']
    s2p = s2_plug(seed)
    vy, ty = (2015 + step, 2016 + step) if step <= 9 else (2025, 2026)
    out = []
    for part, yr in (('val', vy), ('test', ty)):
        if part not in parts:
            continue
        idx = np.where(A.year.values == yr)[0]
        for var, plug in [(v, p) for v in ITV for p in PLUG]:
            pf = np.exp(s2p[(part, step)] / 2) if plug == 'mean' else 1.0
            ell, per = iterate_rows(idx, step, models, var, plant, pf)
            d = pd.DataFrame({'date': A['date'].values[idx], 'ticker': A['ticker'].values[idx], 'year': yr, 'step': step, 'seed': seed,
                              'var': f'{var}_{plug}', 'part': part})
            for jj in range(J):
                d[f'l{jj + 1}'] = ell[:, jj]
            d['lx1'], d['ll1'] = per['xgb'][:, 0], per['lgbm'][:, 0]
            out.append(d)
    return out


def run_iter(workers, steps=range(1, 10), seeds=SEEDS, parts=('val', 'test'), sfx=''):
    """sfx='_s10' — добавочные seed 45–51 (только val, только вариант lvl): iter_val_s10.parquet, файлы кандидатов не пишутся."""
    global ITV
    prep_h1()
    if sfx:
        ITV = ('lvl',)
    jobs = [(s, sd, parts, False) for s in steps for sd in seeds]
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_job_iter, jobs, chunksize=1)
    D = pd.concat([d for r in res for d in r], ignore_index=True)
    OUT2.mkdir(parents=True, exist_ok=True); LOCK2.mkdir(parents=True, exist_ok=True)
    if sfx:
        D[D.part == 'val'].drop(columns='part').to_parquet(OUT2 / f'iter_val{sfx}.parquet', index=False)
        print(f'итерация {sfx}: val {int((D.part == "val").sum())} строк', flush=True)
        return
    V, T = D[D.part == 'val'].drop(columns='part'), D[D.part == 'test'].drop(columns='part')
    V.to_parquet(OUT2 / 'iter_val.parquet', index=False)
    T.to_parquet(LOCK2 / 'iter_test.parquet', index=False)
    write_h5(V, T)
    print(f'итерация готова: val {len(V)} строк, тест {len(T)} строк (метрики теста не считались)', flush=True)


def boost2(var, corr):
    return f'D26_2_{var}_{corr}'


def write_h5(V, T):
    """Файлы кандидатов Д26.2 в формате final3 (одна модель в обоих слотах; rv_pred до внешней Б0.1): val шагов 1–9 —
    data/regen/D26_2/{boost}_{xgb|lgbm}_h5{,_seeds4344}_val.parquet; тест шагов 1–9 — test_locked/{boost}/ (собирается в памяти, тест не читается).
    σ_j² лидов шагов ≤ 6 — в D26_2/s2lead_dev.csv (шаги 7–9 — только процедура, не пишутся)."""
    F = future_rv()
    cols = ['date', 'ticker', 'year', 'step', 'seed', 'rank', 'rv_actual', 'rv_pred']
    s2rows = []
    for var in sorted(V['var'].unique()):
        out = {'raw': ([], []), 'cj': ([], [])}
        for sd in sorted(V.seed.unique()):
            rv_, cv_, s2l = build_h5(V[(V['var'] == var) & (V.seed == sd)], F, 'val')
            if len(T):
                rt_, ct_, _ = build_h5(T[(T['var'] == var) & (T.seed == sd)], F, 'test', s2l)
            else:
                rt_ = ct_ = None
            for corr, a, b in (('raw', rv_, rt_), ('cj', cv_, ct_)):
                out[corr][0].append(a.assign(seed=sd, rank=1))
                if b is not None:
                    out[corr][1].append(b.assign(seed=sd, rank=1))
            s2rows += [dict(var=var, seed=sd, step=k, lead=j, s2=v) for (kind, k, j), v in s2l.items() if kind == 'val' and k <= DEV_MAX]
        for corr, (vv, tt) in out.items():
            b = boost2(var, corr)
            vv = pd.concat(vv, ignore_index=True)[cols]
            lk = LOCK / b; lk.mkdir(parents=True, exist_ok=True)
            tt = pd.concat(tt, ignore_index=True)[cols] if tt else None
            for m in ('xgb', 'lgbm'):
                for sfx, sel in (('', [42]), ('_seeds4344', [43, 44])):
                    vv[vv.seed.isin(sel)].to_parquet(OUT2 / f'{b}_{m}_h5{sfx}_val.parquet', index=False)
                    if tt is not None:
                        tt[tt.seed.isin(sel)].to_parquet(lk / f'{b}_{m}_h5{sfx}_test.parquet', index=False)
    pd.DataFrame(s2rows).to_csv(OUT2 / 's2lead_dev.csv', index=False)


def check_lead1():
    """ℓ₁ итерации (val шагов 1–6) обязан совпасть с сохранённым D22_h1 (ln XGB, ln LGBM по seed)."""
    V = dev_only(pd.read_parquet(OUT2 / 'iter_val.parquet'))
    rows = []
    for m, col in (('xgb', 'lx1'), ('lgbm', 'll1')):
        S = read_seeds(TAG_H1, m, 1)
        for sd, g in S.items():
            x = V[(V.seed == sd) & (V['var'] == 'lvl_med')].merge(g[KEY + ['rv_pred']], on=KEY)
            rows.append(dict(model=m, seed=sd, n=len(x), n_saved=len(g), max_abs=float(np.max(np.abs(np.log(np.clip(np.exp(x[col]), 1e-10, None)) - np.log(x.rv_pred))))))
    T = pd.DataFrame(rows)
    print(T.to_string())
    return T


# ---------------------------------------------------------------- гибрид gibrid_v17 (c1–c4; OOS, эталон с той же s²), как d26_lib агента А
CR = ('c1', 'c2', 'c3', 'c4')
FEAT_V3 = R / 'features_pit_v3.parquet'


def gib_run(boost, xgb_df, lgb_df, gdir, legs=('single', 'wrap'), same_slot=False):
    """Входы гибрида — только val шагов 1–6 ({boost}_{xgb|lgbm}_h5_val.parquet в gdir/gib_in); -> сводка по дизайнам (как final3.summarize)."""
    gin, gout = gdir / 'gib_in', gdir / 'gibrid' / boost
    gin.mkdir(parents=True, exist_ok=True); gout.mkdir(parents=True, exist_ok=True)
    cols = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
    for m, d in (('xgb', xgb_df), ('lgbm', lgb_df)):
        dev_only(d)[cols].to_parquet(gin / f'{boost}_{m}_h5_val.parquet', index=False)
    env = dict(os.environ, GIBRID_SAME_SLOT='1') if same_slot else dict(os.environ)
    cmd = [sys.executable, '-W', 'ignore', str(HERE / 'gibrid_v17.py'), '--boost', boost, '--h', '5', '--features', str(FEAT_V3),
           '--regen', str(gin), '--legs', *legs, '--out', str(gout), '--devlog', str(gout / 'devlog.md')]
    with open(gout / 'run.log', 'w') as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE.parents[1])).returncode
    if rc:
        raise RuntimeError(f'gibrid {boost}: код {rc}, см. {gout / "run.log"}')
    return gib_summary(gout / f'designs_val_{boost}_h5.csv')


def gib_summary(path):
    cols = ['design', 'step', 'val_year', 'q_hyb'] + [k + '_h' for k in CR]
    D = pd.read_csv(path, usecols=cols)
    assert D.loc[D.step > DEV_MAX, 'q_hyb'].isna().all()
    D = D[~D.design.str.contains(r'\|V0_(?:HAR|HCJ)\|')]
    D = D[D.step.between(2, DEV_MAX)]
    q = D.pivot_table(index='design', columns='step', values='q_hyb', sort=False)
    fl = D.groupby('design', sort=False)[[k + '_h' for k in CR]].first()
    S = pd.DataFrame({'design': q.index, 'm1720': q[[2, 3, 4, 5]].mean(axis=1).values, 'q2021': q[6].values})
    for s in range(2, DEV_MAX + 1):
        S[str(2015 + s)] = q[s].values
    for k in CR:
        S[k] = fl.loc[S.design, k + '_h'].astype(bool).values
    S['n_failed'] = (~S[list(CR)]).sum(axis=1)
    S['failed'] = S.apply(lambda r: ','.join(k for k in CR if not r[k]), axis=1)
    return S.sort_values(['n_failed', 'm1720'], kind='stable').reset_index(drop=True)


def seeds_table(S_by_seed):
    """{seed: сводка} -> таблица по дизайнам: число нарушений (максимум по seed), средние по seed; правило: n_failed, затем 2017–2020."""
    rows = []
    for d in set.intersection(*[set(S.design) for S in S_by_seed.values()]):
        r = {s: S.set_index('design').loc[d] for s, S in S_by_seed.items()}
        rows.append(dict(design=d, n_failed=max(int(x.n_failed) for x in r.values()),
                         failed=','.join(sorted(set(','.join(x.failed for x in r.values() if x.failed).split(',')) - {''})),
                         m1720=float(np.mean([x.m1720 for x in r.values()])), q2021=float(np.mean([x.q2021 for x in r.values()])),
                         **{f'm1720_s{s}': float(x.m1720) for s, x in r.items()}, **{f'q2021_s{s}': float(x.q2021) for s, x in r.items()},
                         **{f'{y}': float(np.mean([x[str(y)] for x in r.values()])) for y in range(2017, 2022)}))
    return pd.DataFrame(rows).sort_values(['n_failed', 'm1720'], kind='stable').reset_index(drop=True)


def entry_both(qs, ref_s):
    """Правило дирижёра: парное (D12_h5 тех же seed) и против фиксированного 0,2159; оба — 3% и 2σ."""
    p = entry(qs, ref_s)
    f = entry(qs, [D12_REF['m1720']] * len(qs))
    return dict(rel_pair=p['rel'], d_pair=p['d_mean'], sig2_pair=p['two_sigma'], rel_s=p['rel_s'], pass_pair=p['pass3'] and p['pass2s'],
                rel_fix=f['rel'], d_fix=f['d_mean'], sig2_fix=f['two_sigma'], pass_fix=f['pass3'] and f['pass2s'],
                passed=p['pass3'] and p['pass2s'] and f['pass3'] and f['pass2s'])


# ---------------------------------------------------------------- детектор «будущее не влияет» (новые контуры Д26.1, Д26.2)
def _perturb(A, cut, rng):
    """Строки date > cut: все числовые столбцы (включая цели, RV, ln HAR-X⁺) -> бутстрап значений столбца до cut."""
    B = A.copy()
    fut = (pd.to_datetime(B.date) > cut).values
    for c in B.columns:
        if c in ('date', 'ticker', 'year') or not np.issubdtype(B[c].dtype, np.number):
            continue
        pool = B.loc[~fut, c].dropna().values
        if len(pool):
            B.loc[fut, c] = rng.choice(pool, fut.sum())
    return B, fut


def _perturb_events(cut, rng):
    ev = P._events(); ev2 = {}
    for ty, (d, kf, kt) in ev.items():  # будущие события, о которых в cut ещё не знали, — случайный сдвиг дат
        d2 = d.copy(); m = kf > np.datetime64(cut.date())
        d2[m] = d2[m] + rng.integers(-20, 20, m.sum()).astype('timedelta64[D]')
        ev2[ty] = (d2, kf, kt)
    return ev, ev2


def leak1(var='', cut='2019-06-30', step=3, plant=False, seed=42):
    """Д26.1: модели шага 3 (обучение < 2018, ранняя остановка на 2018, тест 2019) переобучаются на матрице, где строки > cut заменены
    шумом; тестовые прогнозы строк ≤ cut обязаны совпасть. plant — в признаки подложена цель rv_target_mean_h5 (детектор обязан поймать).
    Год теста шага 3 (2019) — разработочный; сравниваются прогнозы между собой, метрики не считаются."""
    from tune_L2 import SPACEB
    patch(var)
    cut = pd.Timestamp(cut); rng = np.random.default_rng(20261026)
    A0, FC = L.load_base()
    sel = L.selected_params(tag1(var), 5)
    cfg = VARS[var]['cfg']
    res = []
    for pert in (False, True):
        A = _perturb(A0, cut, rng)[0] if pert else A0
        A2, cols, _, _ = L.build_features(A, FC, 5, cfg)
        if plant:
            nxt = A2.sort_values(['ticker', 'date']).groupby('ticker')['rv'].shift(-1).reindex(A2.index)  # из (возмущённого) ряда RV
            A2['_leak'] = np.log(nxt.clip(lower=1e-10).fillna(1e-4)); cols = cols + ['_leak']
        out = {}
        for m in ('xgb', 'lgbm'):
            D = L.step_data(A2, cols, 5, step, m, cfg, tag1(var))
            ref = L.make_ref(m, A2, cols, 5, cfg, tag1(var))
            p = {k: sel[(m, step)][k] for k, *_ in SPACEB[m]}
            _, pt, _, _, _ = L.fit(m, p, D, cfg, seed=seed, ref=ref)
            out[m] = pd.DataFrame({'date': D['parts']['te']['date'].values, 'ticker': D['parts']['te']['ticker'].values, 'p': pt})
        res.append(out)
    rows = []
    for m in ('xgb', 'lgbm'):
        a, b = res[0][m], res[1][m]
        x = a.merge(b, on=['date', 'ticker'], suffixes=('', '_2'))
        past = pd.to_datetime(x.date) <= cut
        df = np.abs(x.p - x.p_2)
        rows.append(dict(contour=f'D26_1{var}', model=m, cut=str(cut.date()), plant=plant, n_past=int(past.sum()),
                         mismatch=int((df[past] > 1e-9).sum()), changed_after=int((df[~past] > 1e-9).sum())))
    return pd.DataFrame(rows)


def leak2(cut='2019-06-30', plant=False, seed=42):
    """Д26.2: модели D22_h1 шагов 3 и 4 (seed 42) фиксированы; матрица (RV, компоненты, ln HAR-X⁺, цели) на строках > cut и будущие
    события календаря (known_from > cut) заменены шумом; итерация (ℓ₁..ℓ₅) на строках 2019 с датой ≤ cut обязана совпасть.
    plant — вместо R̂₁ подставляется фактический RV(t+1) (детектор обязан поймать)."""
    cut = pd.Timestamp(cut); rng = np.random.default_rng(7)
    prep_h1()
    A0, FC = L.load_base()
    models = {st: {m: fit_h1(m, st, seed) for m in ('xgb', 'lgbm')} for st in (3, 4)}
    def run():
        A = G2['A']; idx = np.where(A.year.values == 2019)[0]; o = []
        for st, part in ((3, 'test'), (4, 'val')):
            for var in ITV:
                ell, _ = iterate_rows(idx, st, models[st], var, plant)
                o.append(pd.DataFrame(ell, columns=[f'l{j}' for j in range(1, J + 1)]).assign(
                    date=A['date'].values[idx], ticker=A['ticker'].values[idx], step=st, var=var))
        return pd.concat(o, ignore_index=True)
    base = run()
    ev, ev2 = _perturb_events(cut, rng)
    B, _ = _perturb(A0, cut, rng)
    P._EV.clear(); P._EV['ev'] = ev2
    try:
        prep_h1(B)
        pert = run()
    finally:
        P._EV.clear(); P._EV['ev'] = ev
        prep_h1()
    x = base.merge(pert, on=['date', 'ticker', 'step', 'var'], suffixes=('', '_2'))
    past = (pd.to_datetime(x.date) <= cut).values
    rows = []
    for (st, var), g in x.groupby(['step', 'var']):
        pm = (pd.to_datetime(g.date) <= cut).values
        df = np.max(np.abs(g[[f'l{j}' for j in range(1, J + 1)]].values - g[[f'l{j}_2' for j in range(1, J + 1)]].values), axis=1)
        rows.append(dict(contour='D26_2', step=st, var=var, cut=str(cut.date()), plant=plant, n_past=int(pm.sum()),
                         mismatch=int((df[pm] > 1e-9).sum()), changed_after=int((df[~pm] > 1e-9).sum())))
    return pd.DataFrame(rows)


def future_rv():
    """RV_{t+j} (j = 1..5, сдвиг строк тикера вперёд) и цель h=5 — только для остатков лидов и оценки."""
    A = pd.read_parquet(L.FEAT, columns=['date', 'ticker', 'rv', 'rv_target_mean_h5']).sort_values(['ticker', 'date'])
    g = A.groupby('ticker')['rv']
    for j in range(1, J + 1):
        A[f'f{j}'] = g.shift(-j)
    chk = A[[f'f{j}' for j in range(1, 6)]].mean(axis=1, skipna=False)
    ok = chk.notna() & A.rv_target_mean_h5.notna()
    assert np.allclose(chk[ok], A.rv_target_mean_h5[ok], rtol=1e-9), 'цель h=5 ≠ среднее RV следующих 5 строк'
    return A.drop(columns='rv')


def build_h5(I, F, kind='val', s2lead=None):
    """I — итерация (один var, один seed, шаги 1..9), F — future_rv. -> (raw, cj, s2lead):
    raw: rv_pred = (1/5)·Σ exp(ℓ_j) — поправку делает Б0.1 оценки (s² шага t−1, как у всех);
    cj:  rv_pred = (1/5)·Σ exp(ℓ_j + σ_j²/2) — поправка по лиду (условное среднее уровня при ln RV_{t+j} ~ N(ℓ_j, σ_j²));
         σ_j² — sm по остаткам лида j на val шагов k−3..k−1 (шаг 1 — свои); тест шага k — шагов k−2..k; внешняя Б0.1 не применяется.
    s2lead — словарь σ_j² из val (для теста)."""
    d = I.merge(F, on=['date', 'ticker'], how='left', validate='many_to_one')
    lj = d[[f'l{j}' for j in range(1, J + 1)]].values
    raw = d[KEY + ['year']].assign(rv_actual=d.rv_target_mean_h5.values, rv_pred=np.exp(lj).mean(axis=1))
    if s2lead is None:
        s2lead = {}
        for j in range(1, J + 1):
            ok = (d[f'f{j}'] > 1e-12).values
            E = {int(st): (np.log(d.loc[ok & (d.step == st).values, f'f{j}'].values) - lj[ok & (d.step == st).values, j - 1]) for st in sorted(d.step.unique())}
            for k in sorted(d.step.unique()):
                s2lead[('val', k, j)] = s2_est('sm', E, k - 1 if k > 1 else 1)
                s2lead[('test', k, j)] = s2_est('sm', E, k)
    sj = np.column_stack([d.step.map(lambda k: s2lead[(kind, int(k), j)]).values for j in range(1, J + 1)])
    cj = raw.assign(rv_pred=np.exp(lj + sj / 2).mean(axis=1))
    raw = raw[raw.rv_actual.notna()].reset_index(drop=True); cj = cj[cj.rv_actual.notna()].reset_index(drop=True)
    return raw, cj, s2lead


def eval2():
    """Д26.2: QLIKE 2017–2021 по seed для вариантов итерации (lvl/full × med/mean) и поправок (raw + Б0.1 sm; cj без внешней поправки)."""
    V = pd.read_parquet(OUT2 / 'iter_val.parquet')
    F = future_rv()
    ref = d12_ref()
    rows, frames, S2 = [], {}, {}
    for var in sorted(V['var'].unique()):
        for sd in SEEDS:
            I = V[(V['var'] == var) & (V.seed == sd)]
            raw, cj, s2l = build_h5(I, F, 'val')
            frames[(var, 'raw', sd)], frames[(var, 'cj', sd)], S2[(var, sd)] = raw, cj, s2l
        for corr, code in (('raw', 'sm'), ('raw', 'smrob'), ('cj', 'raw')):
            q = {sd: years(frames[(var, corr, sd)], code) for sd in SEEDS}
            e = entry([m1720(q[s]) for s in SEEDS], [m1720(ref[s]) for s in SEEDS])
            rows.append(dict(design=f'D26_2/{var}/{corr}/{code}', **{f'm1720_s{s}': m1720(q[s]) for s in SEEDS},
                             **{f'q2021_s{s}': q[s][2021] for s in SEEDS}, m1720=e['mean_q'], q2021=float(np.mean([q[s][2021] for s in SEEDS])),
                             rel=e['rel'], **{str(y): float(np.mean([q[s][y] for s in SEEDS])) for y in range(2017, 2022)}))
    T = pd.DataFrame(rows)
    T.to_csv(OUT2 / 'd26_2_eval.csv', index=False)
    print(T.round(4).to_string())
    return T, frames, S2


# ---------------------------------------------------------------- Д26.1: 10 seed для лучшего варианта (точки подбора, без нового перебора)
G1 = {}


def _job_seed(a):
    m, step, seed = a
    from tune_L2 import SPACEB
    var, cfg = G1['var'], VARS[G1['var']]['cfg']
    p = {k: G1['sel'][(m, step)][k] for k, *_ in SPACEB[m]}
    D = L.step_data(G1['A'], G1['cols'], 5, step, m, cfg, tag1(var))
    ref = L.make_ref(m, G1['A'], G1['cols'], 5, cfg, tag1(var)) if step > 1 else None
    pv, pt, bi, _, _ = L.fit(m, p, D, cfg, seed=seed, ref=ref)
    fr = lambda part, pred, yr: pd.DataFrame({'date': D['parts'][part]['date'].values, 'ticker': D['parts'][part]['ticker'].values, 'year': yr,
                                              'step': step, 'seed': seed, 'rank': 1, 'rv_actual': D['parts'][part][L.tcol(5)].values,
                                              'rv_pred': np.clip(np.exp(pred), 1e-10, None)})
    return m, fr('va', pv, D['vy']), fr('te', pt, D['ty'])


def seeds10(var, workers, steps=range(1, 10), seeds=SEEDS10[3:]):
    patch(var)
    A, FC = L.load_base()
    A2, cols, _, _ = L.build_features(A, FC, 5, VARS[var]['cfg'])
    G1.update(var=var, A=A2, cols=cols, sel=L.selected_params(tag1(var), 5))
    jobs = [(m, s, sd) for m in ('lgbm', 'xgb') for s in steps for sd in seeds]
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_job_seed, jobs, chunksize=1)
    lk = LOCK / tag1(var); lk.mkdir(parents=True, exist_ok=True)
    for m in ('xgb', 'lgbm'):
        pd.concat([r[1] for r in res if r[0] == m]).to_parquet(R / f'{tag1(var)}_{m}_h5_seeds10_val.parquet', index=False)
        pd.concat([r[2] for r in res if r[0] == m]).to_parquet(lk / f'{tag1(var)}_{m}_h5_seeds10_test.parquet', index=False)
    print(f'{tag1(var)}: seed {seeds[0]}–{seeds[-1]} готовы (тест записан, метрики не считались)', flush=True)


def eval1(variants=('', 'nw', 'v9'), extra10=False):
    """Д26.1: по варианту и seed — XGB, LGBM, среднее по уровню (основное) и в логах; Б0.1 sm/smrob; сравнение с D12_h5 (оба правила)."""
    ref = d12_ref()
    rows = []
    for var in variants:
        if not (R / f'{tag1(var)}_xgb_h5_val.parquet').exists():
            continue
        X, Lg = read_seeds(tag1(var), 'xgb', 5), read_seeds(tag1(var), 'lgbm', 5)
        seeds = list(SEEDS)
        if extra10 and (R / f'{tag1(var)}_xgb_h5_seeds10_val.parquet').exists():
            for m, S in (('xgb', X), ('lgbm', Lg)):
                d = dev_only(pd.read_parquet(R / f'{tag1(var)}_{m}_h5_seeds10_val.parquet'))
                S.update({int(s): g.drop(columns=['rank']).reset_index(drop=True) for s, g in d.groupby('seed')})
            seeds = sorted(X)
        for nm, f in (('avg', avg_lvl), ('avgln', avg_ln), ('xgb', lambda x, l: x), ('lgbm', lambda x, l: l)):
            for code in ('sm', 'smrob'):
                q = {s: years(f(X[s], Lg[s]), code) for s in seeds}
                qs = [m1720(q[s]) for s in SEEDS]
                e = entry_both(qs, [m1720(ref[s]) for s in SEEDS])
                r = dict(design=f'{tag1(var)}/{nm}/{code}', n_seed=len(seeds), m1720=float(np.mean(qs)),
                         q2021=float(np.mean([q[s][2021] for s in SEEDS])), **{f'm1720_s{s}': m1720(q[s]) for s in SEEDS},
                         **{f'q2021_s{s}': q[s][2021] for s in SEEDS}, **{str(y): float(np.mean([q[s][y] for s in SEEDS])) for y in range(2017, 2022)},
                         **{k: v for k, v in e.items() if k != 'rel_s'}, rel_s=' / '.join(f'{100 * x:+.2f}%' for x in e['rel_s']))
                if len(seeds) > 3:
                    q10 = [m1720(q[s]) for s in seeds]
                    r.update(m1720_10=float(np.mean(q10)), sd_10=float(np.std(q10, ddof=1)), q2021_10=float(np.mean([q[s][2021] for s in seeds])),
                             rel_10_fix=float(np.mean(q10) / D12_REF['m1720'] - 1))
                rows.append(r)
    T = pd.DataFrame(rows)
    (R / 'D26_1').mkdir(exist_ok=True)
    T.to_csv(R / 'D26_1' / ('d26_1_eval10.csv' if extra10 else 'd26_1_eval.csv'), index=False)
    print(T[['design', 'm1720', 'q2021', 'rel_pair', 'rel_s', 'sig2_pair', 'rel_fix', 'passed'] + (['m1720_10', 'sd_10', 'q2021_10'] if extra10 and 'm1720_10' in T else [])].round(4).to_string())
    return T


def gib_all(specs, workers=3):
    """specs: [(имя, {seed: (xgb_df, lgb_df)}, gdir, legs, same_slot)] -> {имя: таблица по дизайнам (seeds_table)} + лучший по правилу."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(nm, sd, xl, gdir, legs, ss) for nm, by, gdir, legs, ss in specs for sd, xl in by.items()]
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda j: (j[0], j[1], gib_run(f'{j[0]}s{j[1]}', j[2][0], j[2][1], j[3], j[4], j[5])), jobs))
    out = {}
    ref = d12_ref()
    for nm, *_ in specs:
        T = seeds_table({sd: S for n, sd, S in res if n == nm})
        e = [entry_both([T.loc[i, f'm1720_s{s}'] for s in SEEDS], [m1720(ref[s]) for s in SEEDS]) for i in T.index]
        T['rel_pair'] = [x['rel_pair'] for x in e]; T['sig2_pair'] = [x['sig2_pair'] for x in e]; T['rel_fix'] = [x['rel_fix'] for x in e]
        T['rel_s'] = [' / '.join(f'{100 * v:+.2f}%' for v in x['rel_s']) for x in e]; T['passed_entry'] = [x['passed'] for x in e]
        T.insert(0, 'kandidat', nm)
        out[nm] = T
    return out


def gib_main(v1=('', 'nw', 'v9'), v2=None, workers=3):
    specs = []
    for var in v1:
        if (R / f'{tag1(var)}_xgb_h5_val.parquet').exists():
            X, Lg = read_seeds(tag1(var), 'xgb', 5), read_seeds(tag1(var), 'lgbm', 5)
            specs.append((tag1(var), {s: (X[s], Lg[s]) for s in SEEDS}, R / 'D26_1', ('single', 'wrap'), False))
    for b in (v2 or []):
        S = read_seeds(b, 'xgb', 5, OUT2)
        specs.append((b, {s: (S[s], S[s]) for s in SEEDS}, OUT2, ('single',), True))
    out = gib_all(specs, workers)
    T = pd.concat(out.values(), ignore_index=True)
    T.to_csv(R / 'D26_1' / 'd26_b_gibrid.csv', index=False)
    best = T.groupby('kandidat', sort=False).head(1)
    print(best[['kandidat', 'design', 'n_failed', 'failed', 'm1720', 'q2021', 'rel_pair', 'rel_s', 'sig2_pair', 'rel_fix', 'passed_entry']].round(4).to_string())
    return T


# ---------------------------------------------------------------- Д26.2 + Д26.1: среднее
def _read_all(prefix, m, h, regen=R):
    """Все val-строки шагов 1–9 по seed (для экспорта и s² процедуры; метрики — только через years()/dev_only)."""
    a = pd.read_parquet(regen / f'{prefix}_{m}_h{h}_val.parquet')
    a = a.assign(seed=42) if 'seed' not in a else a
    d = pd.concat([a, pd.read_parquet(regen / f'{prefix}_{m}_h{h}_seeds4344_val.parquet')], ignore_index=True)
    return {int(s): g[KEY + ['year', 'rv_actual', 'rv_pred']].reset_index(drop=True) for s, g in d.groupby('seed')}


def b01(d, code='sm'):
    """Б0.1 вне выборки: val шага k × exp(s²/2), s² — остатки val шагов до k (sm: k−3..k−1; шаг 1 — свои, 2016 не оценивается)."""
    v = d[d.rv_actual > 1e-12]
    E = {int(st): np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
    F = {k: np.exp(s2_est(code, E, k - 1 if k > 1 else 1) / 2) for k in E}
    return d.assign(rv_pred=d.rv_pred * d.step.map(F).values)


def combo(var1='v9', b2='D26_2_lvl_mean_cj'):
    """Среднее Д26.1 (вариант var1, среднее XGB и LGBM по уровню, с Б0.1 sm) и Д26.2 (b2, с поправкой по лиду): mix_lvl — по уровню,
    mix_ln — геометрическое. Оба уже с поправкой, внешняя Б0.1 не применяется. Val шагов 1–9 -> D26_2/{boost}_*; тест не строится."""
    X, Lg = _read_all(tag1(var1), 'xgb', 5), _read_all(tag1(var1), 'lgbm', 5)
    S2 = _read_all(b2, 'xgb', 5, OUT2)
    ref = d12_ref()
    cols = ['date', 'ticker', 'year', 'step', 'seed', 'rank', 'rv_actual', 'rv_pred']
    rows = []
    for nm, f in (('mix_lvl', avg_lvl), ('mix_ln', avg_ln)):
        b = f'D26_2_{nm}'
        fr = {sd: f(b01(avg_lvl(X[sd], Lg[sd])), S2[sd]) for sd in SEEDS}
        V = pd.concat([d.assign(seed=sd, rank=1) for sd, d in fr.items()], ignore_index=True)[cols]
        for m in ('xgb', 'lgbm'):
            for sfx, sel in (('', [42]), ('_seeds4344', [43, 44])):
                V[V.seed.isin(sel)].to_parquet(OUT2 / f'{b}_{m}_h5{sfx}_val.parquet', index=False)
        q = {sd: years(fr[sd], 'raw') for sd in SEEDS}
        e = entry_both([m1720(q[s]) for s in SEEDS], [m1720(ref[s]) for s in SEEDS])
        rows.append(dict(design=f'{b}/raw', **{f'm1720_s{s}': m1720(q[s]) for s in SEEDS}, m1720=float(np.mean([m1720(q[s]) for s in SEEDS])),
                         q2021=float(np.mean([q[s][2021] for s in SEEDS])), **{str(y): float(np.mean([q[s][y] for s in SEEDS])) for y in range(2017, 2022)},
                         **{k: v for k, v in e.items() if k != 'rel_s'}, rel_s=' / '.join(f'{100 * x:+.2f}%' for x in e['rel_s'])))
    T = pd.DataFrame(rows)
    T.to_csv(OUT2 / 'd26_2_combo.csv', index=False)
    print(T.round(4).to_string())
    return T


def eval10_2(var1='v9'):
    """10 seed (42–51): Д26.2 lvl_med/cj, lvl_mean/cj и mix_lvl (с Д26.1 var1 тех же seed). Только val шагов 1–6."""
    F = future_rv()
    V = pd.concat([pd.read_parquet(OUT2 / 'iter_val.parquet'), pd.read_parquet(OUT2 / 'iter_val_s10.parquet')], ignore_index=True)
    V = V[V.step <= DEV_MAX]
    d1 = {}
    for m in ('xgb', 'lgbm'):
        a = _read_all(tag1(var1), m, 5)
        b = pd.read_parquet(R / f'{tag1(var1)}_{m}_h5_seeds10_val.parquet')
        a.update({int(s): g[KEY + ['year', 'rv_actual', 'rv_pred']].reset_index(drop=True) for s, g in b.groupby('seed')})
        d1[m] = {s: dev_only(x) for s, x in a.items()}
    rows = []
    for var in ('lvl_med', 'lvl_mean'):
        qs, qm = {}, {}
        for sd in SEEDS10:
            _, cj, _ = build_h5(V[(V['var'] == var) & (V.seed == sd)], F, 'val')
            qs[sd] = years(cj, 'raw')
            if var == 'lvl_mean':
                qm[sd] = years(avg_lvl(b01(avg_lvl(d1['xgb'][sd], d1['lgbm'][sd])), cj), 'raw')
        for nm, q in ((f'D26_2_{var}_cj', qs), ('D26_2_mix_lvl', qm)):
            if not q:
                continue
            x = [m1720(q[s]) for s in SEEDS10]
            rows.append(dict(design=nm, n_seed=len(x), m1720_10=float(np.mean(x)), sd_10=float(np.std(x, ddof=1)),
                             q2021_10=float(np.mean([q[s][2021] for s in SEEDS10])), rel_10_fix=float(np.mean(x) / D12_REF['m1720'] - 1),
                             **{str(y): float(np.mean([q[s][y] for s in SEEDS10])) for y in range(2017, 2022)}))
    T = pd.DataFrame(rows)
    T.to_csv(OUT2 / 'd26_2_eval10.csv', index=False)
    print(T.round(4).to_string())
    return T


# ---------------------------------------------------------------- yaml кандидатов (формат final3.py)
KAND = R / 'final3' / 'kandidaty'
YAML1 = """boost: {boost}
h: 5
semeystvo: "Д8.12 на HAR-X⁺ (D22-слой-2 на h=5)"
rychag: Д26.1
opisanie: "{opis}"
regen: data/regen
legs: [single, top5, seeds3]
odinochnaya: false
chistyy: false
test_dir: data/regen/test_locked/{boost}
osnova: D12_h5
n_variantov: {nv}
god2021: "только отложенный, одна проверка на вариант"
kod: scripts/analysis/d26_1.py --tune --var {var}
uslovie_vhoda: "{uv}"
"""
YAML2 = """boost: {boost}
h: 5
semeystvo: "итерированный D22_h1 (h=1 -> 5 шагов)"
rychag: Д26.2
opisanie: "{opis}"
regen: data/regen/D26_2
legs: [single, seeds3]
odinochnaya: false
chistyy: false
test_dir: data/regen/test_locked/{boost}
osnova: D22_h1
n_variantov: {nv}
god2021: "только отложенный, одна проверка на вариант"
kod: scripts/analysis/d26_1.py --iter
uslovie_vhoda: "{uv}"
"""


def write_yaml(kind, boost, fname, opis, nv, uv, var=''):
    """fname — имя файла (D26_1_*.yaml / D26_2_*.yaml по поручению); boost — префикс файлов прогнозов."""
    txt = (YAML1 if kind == 1 else YAML2).format(boost=boost, opis=opis.replace('"', "'"), nv=nv, uv=uv.replace('"', "'"), var=var)
    KAND.mkdir(parents=True, exist_ok=True)
    (KAND / f'{fname}.yaml').write_text(txt)
    print('yaml ->', KAND / f'{fname}.yaml')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tune', action='store_true'); ap.add_argument('--var', default='')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--iter', action='store_true'); ap.add_argument('--check1', action='store_true')
    ap.add_argument('--eval2', action='store_true'); ap.add_argument('--eval1', action='store_true')
    ap.add_argument('--seeds10', action='store_true'); ap.add_argument('--leak', action='store_true')
    ap.add_argument('--iter10', action='store_true')
    ap.add_argument('--steps', type=int, nargs='*', default=list(range(1, 10)))
    a = ap.parse_args()
    if a.tune:
        tune(a.workers, a.var)
    if a.iter:
        run_iter(a.workers, a.steps)
    if a.iter10:
        run_iter(a.workers, a.steps, seeds=SEEDS10[3:], parts=('val',), sfx='_s10')
    if a.check1:
        check_lead1()
    if a.eval2:
        eval2()
    if a.eval1:
        eval1()
    if a.seeds10:
        seeds10(a.var, a.workers)
        eval1((a.var,), extra10=True)
    if a.leak:
        T = pd.concat([leak1(a.var), leak1(a.var, plant=True), leak2(), leak2(plant=True)], ignore_index=True)
        T.to_csv(R / 'D26_1' / 'leak_d26_b.csv', index=False)
        print(T.to_string())


if __name__ == '__main__':
    main()
