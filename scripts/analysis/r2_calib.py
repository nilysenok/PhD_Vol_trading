#!/usr/bin/env python3
"""R2, агент L: слой калибровки (кластер 3) и calib.update (кластер 15) по схеме models/designs/r2/schema_v1.yaml, протокол v3.
Вход — сырой OOS-ряд ноги или итога (val-строки date, ticker, year, step, rv_actual, rv_pred; шаги 1..t). Всё, что оценивается по
ошибкам, берётся только по прошлым OOS-строкам:
  update = step  — параметры шага t — по строкам шагов < t (как Б0.1 первого раунда: шаг 1 — без поправки);
  update = daily — параметры дня d — по строкам, чьё окно цели кончилось до d (дата строки t+h < d), включая уже разрешённые строки
                   текущего года (эмбарго h).
Порядок слоёв (calib.order, fixed): s2_state_kappa > c_qlike > mz.
  s2_est (оценка s² по остаткам e = ln y − ln ŷ окна):
    sm — дисперсия, окно шагов t−3..t−1 (Б0.1 R1); smrob — (1,4826·MAD)², то же окно; sm_pooled (❓) — дисперсия по всем прошлым шагам;
    sm_by_ticker — дисперсия по тикеру, окно t−3..t−1 (< 50 строк — общая); sm_roll1y / sm_roll2y — окно 1 / 2 года;
    sm_ewma097 — EWMA (λ = 0,97 на торговый день) средней по сечению (e − ē)² по датам окна «все прошлые».
    В режиме daily окно в годах считается от дня d (3, 1, 2 года; «все»).
  s2_state (форма s² по состоянию; уровень — от s2_est): const; RVI — (e − ē)² = a + b·ln RVI²; RVI_volvol — + вол-оф-вол (СКО ln RV
    тикера за 22 дня ≤ d, как Д26 b); regime2 — дисперсия по режиму RVI > медианы за 252 прошлых дня. s²(d) = s̄²·clip(f(z_d)/f̄_окна, 0,25; 4).
  kappa: 0.5 — F = exp(s²/2); fit — κ ∈ [0; 1,5] по минимуму QLIKE ŷ·exp(κ·s²) на окне (сами строки окна — со своей s² по состоянию).
  c_qlike: off / roll_all / roll_2y — аддитивная калибровка уровня ŷ + c, c = argmin QLIKE на окне (все прошлые / 2 года), как Д26 c.
  mz: off / on — Минцер–Зарновиц в логах: ln y = a + b·ln ŷ на всех прошлых строках (Д24.11).
Таблица стабильности: параметры по шагам (в daily — медиана по дням шага), относительный скачок к шагу t−1; скачок > 50% — флаг
(аналог c4, протокол п. 2.2 (в)).
Интерфейс для r2_core: calibrate(design, V, S) -> (Vc, stab); S = state_frame(M) из r2_nogi.mat_L.
python3 scripts/analysis/r2_calib.py dym [--hs 1 5 22] | kontrol"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import sys, argparse, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r2_nogi as N

COLS = N.COLS
WIN_Y = {'sm': 3, 'smrob': 3, 'sm_by_ticker': 3, 'sm_roll1y': 1, 'sm_roll2y': 2, 'sm_pooled': None, 'sm_ewma097': None}
PLANT = {'on': False}


def state_frame(M, h):
    A = M['A']
    lr = lambda c: np.log(A[c].values.astype(float).clip(1e-10))
    return pd.DataFrame({'date': A['date'].values, 'ticker': A['ticker'].astype(str).values, 'lnrvi2': A['lnrvi2'].values,
                         'vov': A['vov'].values, 'I2': A['I2'].values, 'tend': A[f'_tend{h}'].values,
                         'lrv_d': lr('rv_d'), 'lrv_w': lr('rv_w'), 'lrv_m': lr('rv_m')})


# ---------------------------------------------------------------- кластер 12: вероятностный слой (scale_har)
QZ = (np.arange(200) + 0.5) / 200
PROB_Q = (0.05, 0.25, 0.5, 0.75, 0.95)


def prob_layer(design, d, ok, lny, p0, step):
    """boost.prob = scale_har: масштаб остатка — HAR по ln|e − ē| на [1, ln RV_d, ln RV_w, ln RV_m] (d0) по прошлым OOS-строкам окна
    (boost.prob_window: 2y — шаги t−2..t−1, all — все прошлые), нормировка — E[(e − ē)/σ]² = 1 на окне; распределение z = (e − ē)/σ —
    boost.prob_dist: empirical (квантили z окна) или student (t с df по окну). Точечный прогноз под QLIKE — E[RV] = ŷ·e^{ē}·E[e^{σ_d z}]
    (по 200 квантилям z); квантили q05–q95 ŷ·e^{ē + σ_d·Q_z(α)}. Шаг 1 (нет истории) — ŷ без изменений.
    scale_boost и quantile5 — бустинг (агент B): NotImplementedError. -> (точка, DataFrame квантилей, строки параметров)."""
    b = N.complete(design)['boost']
    if b['prob'] in ('scale_boost', 'quantile5'):
        raise NotImplementedError(f"boost.prob = {b['prob']} — бустинг, агент B")
    from scipy import stats
    X = np.column_stack([np.ones(len(d)), d[['lrv_d', 'lrv_w', 'lrv_m']].fillna(0).values])
    pt = p0.copy(); Q = np.full((len(d), len(PROB_Q)), np.nan); par = []
    e_all = lny - np.log(p0)
    for t in sorted(np.unique(step)):
        ms = step == t
        hi = t if PLANT['on'] else t - 1
        win = ok & (step <= hi) & (step >= (1 if b['prob_window'] == 'all' else t - 2))
        if win.sum() < 100:
            continue
        e = e_all[win]; eb = float(e.mean())
        beta = np.linalg.lstsq(X[win], np.log(np.abs(e - eb) + 1e-6), rcond=None)[0]
        sg = np.exp(X[win] @ beta)
        cadj = float(np.sqrt(np.mean(((e - eb) / sg) ** 2)))
        z = (e - eb) / (sg * cadj)
        if b['prob_dist'] == 'student':
            df_, _, sc = stats.t.fit(z, floc=0)
            zq = stats.t.ppf(QZ, df_) * sc; qf = lambda a: stats.t.ppf(a, df_) * sc
            par.append((t, 'prob_df', df_)); par.append((t, 'prob_t_scale', sc))
        else:
            zq = np.quantile(z, QZ); qf = lambda a: np.quantile(z, a)
        sd = np.exp(X[ms] @ beta) * cadj
        pt[ms] = p0[ms] * np.exp(eb) * np.exp(sd[:, None] * zq[None, :]).mean(1)
        for j, a in enumerate(PROB_Q):
            Q[ms, j] = p0[ms] * np.exp(eb + sd * qf(a))
        par += [(t, 'prob_ebar', eb), (t, 'prob_cadj', cadj)] + [(t, f'prob_b{i}', float(v)) for i, v in enumerate(beta)]
    return pt, pd.DataFrame(Q, columns=[f'q{int(100 * a):02d}' for a in PROB_Q]), par


def _qlike(y, p):
    r = y / np.clip(p, 1e-12, None)
    return float(np.mean(r - np.log(r) - 1))


def _s2(code, e, tk, dates):
    """s² по окну: скаляр или массив по тикерам (dict)."""
    if len(e) < 30:
        return None
    if code == 'smrob':
        return float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2)
    if code == 'sm_by_ticker':
        base = float(np.var(e)); s = pd.Series(e).groupby(tk)
        return {k: (float(np.var(v.values)) if len(v) >= 50 else base) for k, v in s}
    if code == 'sm_ewma097':
        d = pd.DataFrame({'d': dates, 'e2': (e - e.mean()) ** 2}).groupby('d').e2.mean().sort_index()
        return float(d.ewm(alpha=0.03, adjust=True).mean().iloc[-1])
    return float(np.var(e))


class Params:
    def __init__(self, s2=None, kappa=0.5, beta=None, fbar=1.0, reg=None, c=0.0, mz=None):
        self.s2, self.kappa, self.beta, self.fbar, self.reg, self.c, self.mz = s2, kappa, beta, fbar, reg, c, mz


def _state_z(cal, S):
    if cal['s2_state'] == 'RVI':
        return np.column_stack([np.ones(len(S)), np.nan_to_num(S['lnrvi2'].values, nan=np.nanmean(S['lnrvi2'].values))])
    if cal['s2_state'] == 'RVI_volvol':
        return np.column_stack([np.ones(len(S)), np.nan_to_num(S['lnrvi2'].values, nan=np.nanmean(S['lnrvi2'].values)),
                                np.nan_to_num(S['vov'].values, nan=np.nanmean(S['vov'].values))])
    return None


def _s2_rows(P, cal, tk, Z, I2):
    """s² по строкам при параметрах P."""
    n = len(tk)
    base = np.array([P.s2.get(t, np.mean(list(P.s2.values()))) for t in tk]) if isinstance(P.s2, dict) else np.full(n, P.s2 or 0.0)
    st = cal['s2_state']
    if st in ('RVI', 'RVI_volvol') and P.beta is not None:
        return base * np.clip((Z @ P.beta) / P.fbar, 0.25, 4)
    if st == 'regime2' and P.reg is not None:
        return base * np.where(I2 > 0, P.reg[1], P.reg[0])
    return base


def _fit(cal, pool, Z, I2, tk, dates, y, p0, p1prev, p2prev, lny):
    """Параметры по окну pool (маски для каждого слоя)."""
    if pool['s2'].sum() < 30:
        return Params(s2=None)
    e = lny[pool['s2']] - np.log(p0[pool['s2']])
    s2 = _s2(cal['s2_est'], e, tk[pool['s2']], dates[pool['s2']])
    P = Params(s2=s2)
    st = cal['s2_state']
    if st in ('RVI', 'RVI_volvol'):
        em = (e - e.mean()) ** 2
        P.beta = np.linalg.lstsq(Z[pool['s2']], em, rcond=None)[0]
        P.fbar = max(float(np.mean(Z[pool['s2']] @ P.beta)), 1e-8)
    elif st == 'regime2':
        ii = I2[pool['s2']] > 0; vt = float(np.var(e))
        P.reg = (float(np.var(e[~ii])) / vt if (~ii).sum() > 30 else 1.0, float(np.var(e[ii])) / vt if ii.sum() > 30 else 1.0)
    if cal['kappa'] == 'fit':
        m = pool['k']
        s2r = _s2_rows(P, cal, tk[m], None if Z is None else Z[m], I2[m])
        r = minimize_scalar(lambda k: _qlike(y[m], p0[m] * np.exp(k * s2r)), bounds=(0.0, 1.5), method='bounded')
        P.kappa = float(r.x)
    else:
        P.kappa = 0.5
    if cal['c_qlike'] != 'off':
        m = pool['c']
        if m.sum() >= 30:
            pp = p1prev[m]; lo = -0.9 * float(np.min(pp))
            P.c = float(minimize_scalar(lambda c: _qlike(y[m], pp + c), bounds=(lo, float(np.median(pp))), method='bounded').x)
    if cal['mz'] == 'on':
        m = pool['mz']
        if m.sum() >= 30:
            P.mz = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), np.log(p2prev[m])]), lny[m], rcond=None)[0]
    return P


def _apply(P, cal, p0, tk, Z, I2):
    if P.s2 is None:
        return p0, p0, p0
    p1 = p0 if cal.get('_no_s2') else p0 * np.exp(P.kappa * _s2_rows(P, cal, tk, Z, I2))
    p2 = np.clip(p1 + P.c, 1e-9, None) if cal['c_qlike'] != 'off' else p1
    p3 = np.exp(P.mz[0] + P.mz[1] * np.log(p2)) if (cal['mz'] == 'on' and P.mz is not None) else p2
    return p1, p2, p3


def _pools_step(cal, step, t, d_years=None):
    """Маски окон в режиме step для шага t (по шагам < t; подлог — включая сам шаг t)."""
    hi = t if PLANT['on'] else t - 1
    w = WIN_Y[cal['s2_est']]
    s2m = (step <= hi) & (step >= (1 if w is None else t - w))
    cm = (step <= hi) & (step >= (1 if cal['c_qlike'] == 'roll_all' else t - 2))
    return {'s2': s2m, 'k': s2m, 'c': cm, 'mz': step <= hi}


def _pools_day(cal, dates, tend, d):
    """Маски окон в режиме daily для дня d: строки, чьё окно цели кончилось до d (подлог — все строки с датой < d)."""
    res = (dates <= d) if PLANT['on'] else (tend < d)  # подлог: и строки самого дня d (их цель ещё в будущем)
    w = WIN_Y[cal['s2_est']]
    yrs = lambda n: dates >= (d - np.timedelta64(int(365.25 * n), 'D'))
    s2m = res & (True if w is None else yrs(w))
    cm = res & (True if cal['c_qlike'] == 'roll_all' else yrs(2))
    return {'s2': s2m, 'k': s2m, 'c': cm, 'mz': res}


def calibrate(design, V, S):
    """design — дизайн по схеме v1 (используются calib.*); V — сырой ряд; S — state_frame. -> (Vc, таблица стабильности)."""
    full = N.complete(design)
    cal = dict(full['calib'])
    d = V[COLS].merge(S, on=['date', 'ticker'], how='left').sort_values(['date', 'ticker']).reset_index(drop=True)
    ok = (d.rv_actual > 1e-12).values
    y = np.where(ok, d.rv_actual.values, np.nan); lny = np.log(np.where(ok, y, 1.0) + 1e-10)  # +1e-10 — как harx.add_bias (Б0.1 R1)
    p0 = d.rv_pred.values.astype(float)
    step, tk = d.step.values, d.ticker.values
    Qp, ppar = None, []
    if full['boost']['prob'] != 'off':  # кластер 12: вероятностный слой заменяет s²/κ (E[e^{σz}] — та же поправка уровня), c и mz — дальше
        p0, Qp, ppar = prob_layer(design, d, ok, lny, p0, step)
        cal['_no_s2'] = True
    dates, tend = d.date.values, pd.to_datetime(d.tend).values
    tend = np.where(pd.isna(tend), np.datetime64('2100-01-01'), tend)
    Z, I2 = _state_z(cal, d), d.I2.fillna(0).values
    p1, p2, p3 = p0.copy(), p0.copy(), p0.copy()
    stab = []
    for t in sorted(np.unique(step)):
        ms = step == t
        if cal['update'] == 'step':
            pool = {k: v & ok for k, v in _pools_step(cal, step, t).items()}
            P = _fit(cal, pool, Z, I2, tk, dates, y, p0, p1, p2, lny)
            a, b, c = _apply(P, cal, p0[ms], tk[ms], None if Z is None else Z[ms], I2[ms])
            p1[ms], p2[ms], p3[ms] = a, b, c
            stab.append(_prow(t, P))
        else:
            rows = []
            for dd in np.unique(dates[ms]):
                md = ms & (dates == dd)
                pool = {k: v & ok for k, v in _pools_day(cal, dates, tend, dd).items()}
                P = _fit(cal, pool, Z, I2, tk, dates, y, p0, p1, p2, lny)
                a, b, c = _apply(P, cal, p0[md], tk[md], None if Z is None else Z[md], I2[md])
                p1[md], p2[md], p3[md] = a, b, c
                rows.append(_prow(t, P))
            R = pd.DataFrame(rows)
            stab.append(R.median(numeric_only=True).to_dict() | {'step': t})
    out = d[COLS].copy(); out['rv_pred'] = p3
    if Qp is not None:
        for c in Qp.columns:
            out[c] = Qp[c].values
    T = pd.DataFrame(stab).sort_values('step').reset_index(drop=True)
    if ppar:
        Pp = pd.DataFrame(ppar, columns=['step', 'param', 'value']).pivot_table(index='step', columns='param', values='value').reset_index()
        T = T.merge(Pp, on='step', how='left')
    for c in [c for c in T.columns if c != 'step']:
        prev = T[c].shift(1)
        T[f'skachok_{c}'] = (T[c] - prev).abs() / prev.abs().replace(0, np.nan)
    sk = [c for c in T.columns if c.startswith('skachok_')]
    T['flag_50'] = (T[sk].fillna(0) > 0.5).any(axis=1) if sk else False
    return out.sort_values(['step', 'date', 'ticker']).reset_index(drop=True), T


def _prow(t, P):
    s2 = (float(np.mean(list(P.s2.values()))) if isinstance(P.s2, dict) else P.s2)
    r = dict(step=t, s2=s2, kappa=P.kappa if P.s2 is not None else np.nan, c=P.c)
    if P.beta is not None:
        r.update({f'beta{i}': float(b) for i, b in enumerate(P.beta)})
    if P.reg is not None:
        r.update(reg_lo=P.reg[0], reg_hi=P.reg[1])
    if P.mz is not None:
        r.update(mz_a=float(P.mz[0]), mz_b=float(P.mz[1]))
    return r


def fn_v0(design, seed, steps):
    """Модуль для r2_core.run_phase: V0-ячейка (boost.role = none) — нога L × калибровка дизайна; ряд v0 / v0__<тег варианта ноги>.
    Данные — r2_core.SHARED['M'] (r2_nogi.mat_from). Вариант калибровки выбирается ядром/O по дизайну (ключи calib.*)."""
    import r2_core as RC
    M = RC.SHARED['M']
    h = int(design['h'])
    S = state_frame(M, h)
    out, par = {}, []
    for tag, (V, _) in N.run_leg(design, M=M, steps=steps, params=par).items():
        Vc, T = calibrate(design, V, S)
        out[N.series_name(tag, 'v0')] = Vc
        par += calib_params(tag, T)
    N.write_params(design, h, par, kind='v0')  # params.csv: <ряд>:b_*, … и <ряд>:calib_s2, calib_kappa, calib_c, calib_mz_a/b, …
    return out


def _state_M():
    """Матрица L для state_frame (RVI, вол-оф-вол, режим, ln RV_d/w/m): r2_core.SHARED['M'] или строится один раз (17 тикеров, ≤ 2021)."""
    import r2_core as RC
    if RC.SHARED.get('M') is None:
        A, harj = RC.load_d0()
        RC.share(M=N.mat_from(A, harj))
    return RC.SHARED['M']


def fn_v0_panel(design, seed, steps):
    """Модуль для r2_core.run_phase: сырой ряд агента P (r2_panel.phase_fn, leg.panel ≠ off) × общий слой калибровки L (calib.*, boost.prob).
    Ряды — <ряд P>__cal; параметры калибровки — params.csv (param = <ряд P>__cal:calib_*). Данные P — SHARED['A'] из r2_panel.load_A()."""
    import r2_panel as RP
    h = int(design['h'])
    S = state_frame(_state_M(), h)
    out, par = {}, []
    for name, V in RP.phase_fn(design, seed, steps).items():
        Vc, T = calibrate(design, V, S)
        out[f'{name}__cal'] = Vc
        par += [(f'{name}__cal', s, n, v) for _, s, n, v in calib_params('base', T)]
    _write_cal_params(design, h, par)
    return out


def _write_cal_params(design, h, rows):
    """rows: (ряд, шаг, параметр, значение) -> params.csv (формат D) через r2_nogi.write_params (ряд — как есть)."""
    by = {}
    for r, s, n, v in rows:
        by.setdefault(r, []).append(('base', s, n, v))
    for r, rr in by.items():
        N.write_params(design, h, rr, kind=r)


def calib_params(tag, T):
    """Таблица стабильности -> строки (тег, шаг, calib_<параметр>, значение) для write_params (формат D)."""
    cols = [c for c in T.columns if c not in ('step', 'flag_50') and not c.startswith('skachok_')]
    return [(tag, int(r['step']), f'calib_{c}', r[c]) for _, r in T.iterrows() for c in cols]


# ---------------------------------------------------------------- контроль, детектор, дым
def kontrol():
    """calib по умолчанию (sm, κ = 0,5, const, step) = Б0.1 sm первого раунда: ряд Э HARJ_emb_d0 × V0_F (sm) на 17 тикерах, шаги 1–6."""
    rows = []
    M = N.mat_L()
    for h in N.HS:
        V = pd.read_parquet(N.CACHE / f'HARJ_emb_d0_h{h}_sdet_val.parquet')
        F = pd.read_csv(N.CACHE / f'V0_F_h{h}.csv'); F = F[(F.ryad == 'HARJ_emb_d0') & (F.s2 == 'sm')].set_index('step').F
        Vc, _ = calibrate({'h': h}, V, state_frame(M, h))
        ref = V.assign(rv_pred=V.rv_pred * V.step.map(F))
        m = Vc.merge(ref, on=N.KEY, suffixes=('', '_r'))
        d = float(np.abs(m.rv_pred / m.rv_pred_r - 1).max())
        rows.append(dict(h=h, n=len(m), max_rel=d, ok=bool(len(m) == len(V) and d <= 1e-10)))
        print(rows[-1], flush=True)
    return rows


def _perturb_V(V, S, cut, rng):
    """Будущее к дате cut: rv_actual и rv_pred строк date > cut и rv_actual строк, чьё окно цели заходит за cut, × U(0,5; 2)."""
    d = V.merge(S[['date', 'ticker', 'tend']], on=['date', 'ticker'], how='left').copy()
    fut = (d.date > cut).values; crs = (pd.to_datetime(d.tend) > cut).fillna(True).values
    d.loc[fut, 'rv_pred'] = d.loc[fut, 'rv_pred'].values * rng.uniform(0.5, 2, fut.sum())
    d.loc[fut | crs, 'rv_actual'] = d.loc[fut | crs, 'rv_actual'].values * rng.uniform(0.5, 2, (fut | crs).sum())
    return d[COLS]


DYM = {
    'default_sm': {}, 'smrob': {'calib': {'s2_est': 'smrob'}}, 'sm_pooled': {'calib': {'s2_est': 'sm_pooled'}},
    'by_ticker': {'calib': {'s2_est': 'sm_by_ticker'}}, 'roll1y': {'calib': {'s2_est': 'sm_roll1y'}}, 'roll2y': {'calib': {'s2_est': 'sm_roll2y'}},
    'ewma': {'calib': {'s2_est': 'sm_ewma097'}}, 'kappa_fit': {'calib': {'kappa': 'fit'}},
    'state_RVI': {'calib': {'s2_state': 'RVI'}}, 'state_volvol': {'calib': {'s2_state': 'RVI_volvol', 'kappa': 'fit'}},
    'state_regime2': {'calib': {'s2_state': 'regime2'}}, 'c_all': {'calib': {'c_qlike': 'roll_all'}}, 'c_2y': {'calib': {'c_qlike': 'roll_2y'}},
    'mz': {'calib': {'mz': 'on'}}, 'all_step': {'calib': {'s2_state': 'RVI_volvol', 'kappa': 'fit', 'c_qlike': 'roll_2y', 'mz': 'on'}},
    'daily_sm': {'calib': {'update': 'daily'}},
    'all_daily': {'calib': {'update': 'daily', 's2_est': 'sm_ewma097', 's2_state': 'RVI', 'kappa': 'fit', 'c_qlike': 'roll_all', 'mz': 'on'}},
    'prob_emp_all': {'boost': {'prob': 'scale_har'}}, 'prob_t_2y': {'boost': {'prob': 'scale_har', 'prob_dist': 'student', 'prob_window': '2y'}},
    'prob_emp_c_mz': {'boost': {'prob': 'scale_har'}, 'calib': {'c_qlike': 'roll_2y', 'mz': 'on'}},
}


def dym(hs=N.HS, names=None, designs=None, tag='dym'):
    """Дым: нога HARJ (d0, эмбарго) на 3 тикерах, шаги 1–3, h; каждая калибровка: детектор (срезы 2017-06-30 и 2017-12-31, подлог —
    окно с самим шагом t / неразрешёнными строками), проверка сдвига, таблица стабильности. designs — {имя: дизайн} вместо DYM (детектор
    драйвера: калибровки из файла дизайнов)."""
    import r2_core as RC
    out = N.LDIR / tag; out.mkdir(parents=True, exist_ok=True)
    if (out / 'blok_sdvig_dym_calib.csv').exists():
        (out / 'blok_sdvig_dym_calib.csv').unlink()
    rng = np.random.default_rng(20261026)
    M = N.mat_L(tickers=N.DYM_TK)
    rows, stabs = [], []
    for h in hs:
        S = state_frame(M, h)
        V = N.run_leg({'h': h}, M=M, steps=N.DYM_STEPS)['base'][0]
        pert = {c: _perturb_V(V, S, pd.Timestamp(c), rng) for c in ('2017-06-30', '2017-12-31')}
        for nm, des in (designs or DYM).items():
            if names and nm not in names:
                continue
            t0 = time.time()
            des = dict(des, h=h)
            Vc, T = calibrate(des, V, S)
            dt = time.time() - t0
            stabs.append(T.assign(h=h, calib=nm))
            leak = {}
            for c, Vp in pert.items():
                cut = pd.Timestamp(c)
                a = calibrate(des, Vp, S)[0]
                m = Vc.merge(a, on=N.KEY, suffixes=('', '_p')); p = (m.date <= cut).values
                leak[c] = float(np.abs(np.log(m.rv_pred[p]) - np.log(m.rv_pred_p[p])).max())
                if c == '2017-06-30':
                    PLANT['on'] = True
                    try:
                        b0, b1 = calibrate(des, V, S)[0], calibrate(des, Vp, S)[0]
                    finally:
                        PLANT['on'] = False
                    mm = b0.merge(b1, on=N.KEY, suffixes=('', '_p')); pp = (mm.date <= cut).values
                    leak['podlog'] = float(np.abs(np.log(mm.rv_pred[pp]) - np.log(mm.rv_pred_p[pp])).max())
            v = Vc[(Vc.step >= 2) & (Vc.rv_actual > 1e-12)]
            q = {int(y): _qlike(g.rv_actual.values, g.rv_pred.values) for y, g in v.groupby('year')}
            sd = RC.shift_check(Vc, M['A'], h, f'calib:{nm}', RC.validate(des), block_file=out / 'blok_sdvig_dym_calib.csv')
            sd = {f'sdvig_{k}': v for k, v in sd.items() if k in ('argmax', 'ok')}
            ok_leak = leak['2017-06-30'] == 0.0 and leak['2017-12-31'] == 0.0 and leak['podlog'] > 0
            rows.append(dict(calib=nm, h=h, sek=round(dt, 1), q2017=q.get(2017), q2018=q.get(2018), nan=int((~np.isfinite(Vc.rv_pred)).sum()),
                             leak_mid=leak['2017-06-30'], leak_ye=leak['2017-12-31'], podlog=leak['podlog'], leak_ok=ok_leak,
                             flag_50=bool(T.flag_50.any()), **sd))
            rows[-1]['ok'] = bool(ok_leak and rows[-1]['nan'] == 0 and rows[-1]['sdvig_ok'])
            print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in rows[-1].items()}, flush=True)
    R = pd.DataFrame(rows); R.to_csv(out / 'dym_L_calib.csv', index=False)
    pd.concat(stabs).to_csv(out / 'dym_L_calib_stabilnost.csv', index=False)
    print(f'дым калибровки: строк {len(R)}, не OK {int((~R.ok).sum())}'); return R


# ---------------------------------------------------------------- контракт драйвера S (r2_driver.py, модуль L3)
def _prev_module(design):
    import r2_driver as DR
    ch = DR.route(design)
    i = ch.index('L3') if 'L3' in ch else len(ch)
    return ch[i - 1] if i > 0 else 'L'


def fn_cal(design, seed, steps):
    """L3 в цепочке драйвера: ряды предыдущего модуля цепочки (L, P, R, B, BG, M — по r2_driver.route) этого дизайна и seed из кэша
    (r2_core.load_series) × калибровка дизайна (calib.*, boost.prob = scale_har). Ряды — <ряд>__cal; параметры — params.csv."""
    import r2_core as RC
    h = int(design['h'])
    prev = _prev_module(design)
    L = RC.load_series(design, modul=prev, seed=seed, h=h, steps=list(steps))
    if L.empty:
        L = RC.load_series(design, modul=prev, seed='det', h=h, steps=list(steps))
    if L.empty:
        raise FileNotFoundError(f'L3: нет рядов модуля {prev} для {design.get("design")} h={h} seed={seed}')
    S = state_frame(_state_M(), h)
    out, par = {}, []
    for ryad, g in L.groupby('ryad'):
        Vc, T = calibrate(design, g[COLS].copy(), S)
        out[f'{ryad}__cal'] = Vc
        par += [(f'{ryad}__cal', s, n, v) for _, s, n, v in calib_params('base', T)]
    _write_cal_params(design, h, par)
    return out


def cmd_run(designs, h, seeds, steps, workers):
    """run --designs <yaml> --h --seeds --steps --workers: калибровка (модуль 'L3') рядов предыдущего модуля цепочки в кэш по хэшу;
    шаги 1–6 (режим R2). -> 0 / 1."""
    import r2_core as RC
    D = N.read_designs_file(designs, h)
    RC.share(M=None)
    _state_M()
    T = RC.run_phase('L3', h, D, fn_cal, seeds=tuple(seeds), steps=tuple(steps), workers=workers, module='L3', check_gate=False,
                     block_file=os.environ.get('R2_BLOCK_FILE'))
    bad = int((T.get('ryad') == 'ОШИБКА').sum()) if len(T) else 0
    RC.log(f'L3 run h={h}: дизайнов {len(D)}, рядов {len(T) - bad}, ошибок {bad}')
    return 1 if bad else 0


def cmd_leak(designs, h):
    """leak --designs <yaml> --h: детектор калибровки для всех различных настроек calib.* / boost.prob дизайнов файла
    (нога HARJ, 3 тикера, шаги 1–3; порча ряда после среза, подлог — окно с самим шагом). -> 0 / 1."""
    import r2_core as RC
    D = N.read_designs_file(designs, h)
    cfg = {}
    for d in D:
        f = N._onoff(d)
        key = repr(sorted(f['calib'].items())) + repr((f['boost']['prob'], f['boost']['prob_dist'], f['boost']['prob_window']))
        cfg.setdefault(key, {'calib': {k: v for k, v in f['calib'].items() if k != 'order'},
                             'boost': {k: f['boost'][k] for k in ('prob', 'prob_dist', 'prob_window')}})
    designs_ = {f'cfg{i:02d}': c for i, c in enumerate(cfg.values())}
    R = dym((h,), designs=designs_, tag='leak')
    return 0 if bool(R.leak_ok.all()) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'leak', 'dym', 'kontrol'])
    ap.add_argument('--hs', type=int, nargs='*', default=list(N.HS))
    ap.add_argument('--names', nargs='*', default=None)
    ap.add_argument('--designs'); ap.add_argument('--h', type=int)
    ap.add_argument('--seeds', type=int, nargs='*', default=[42]); ap.add_argument('--steps', type=int, nargs='*', default=list(N.STEPS))
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    if a.mode == 'run':
        sys.exit(cmd_run(a.designs, a.h, a.seeds, a.steps, a.workers))
    if a.mode == 'leak':
        sys.exit(cmd_leak(a.designs, a.h))
    if a.mode == 'dym':
        dym(tuple(a.hs), a.names)
    else:
        r = kontrol(); sys.exit(0 if all(x['ok'] for x in r) else 1)


if __name__ == '__main__':
    main()
