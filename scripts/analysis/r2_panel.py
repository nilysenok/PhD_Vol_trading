#!/usr/bin/env python3
"""R2, агент P: кластер 14 — панельно-факторная архитектура (протокол v3 docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md;
схема models/designs/r2/schema_v1.yaml: leg.panel, leg.panel_beta_window, leg.panel_market; определения — models/designs/r2/r2_P_panel.yaml,
записаны до счёта). Ядро — scripts/analysis/r2_core.py (агент S); код первого раунда и модули соседей только импортируются.

Архитектура (всё по день d включительно, версия d0, эмбарго h):
  1. Разложение: ln RV_i,t = α_i,d + β_i,d·ln m_t + u_i,t на окне W = panel_beta_window строк тикера по день d; β̂ усаживается к 1 по
     Васичеку (c = se²/(se² + τ²_d), τ²_d — дисперсия β̂ по тикерам дня d); m — RV_IMOEX или равновзвешенная RV 17 тикеров (panel_market).
  2. Рыночная компонента F_mkt (прогноз ln m̄ на d+1..d+h): HAR / HAR-X рынка (harx.fit_en, как ноги L) | RVI-якорь (d25_1.rvi_leg, как L)
     с калибровкой сдвигом по обучению | factor = их среднее в логах. Ряд по датам, обучение — даты, чьё окно цели кончилось до 1 янв. vy.
  3. Идиосинкразия F_u: HAR на ũ_d, ũ_w, ũ_m с частичным объединением κ = leg.pooled_kappa (пул ↔ тикер) или + пулированный LightGBM
     на остатке (boost.role = residual, num_threads = 1).
  4. Прогноз ln ŷ_i = α̃ + β̃·F_mkt + F_u; кросс-секционная усадка ln ŷ ← (1 − w)·ln ŷ + w·(α̃^(h) + β̃·F_mkt), α̃^(h) = α̃ + ū^h
     (α тикера в масштабе цели h), w = 0,25 | 0,5 | w(ρ̄_d),
     ρ̄ — g5_corr_m (средняя корреляция 22 дня, features_r2_v2). panel = off — та же HAR без разложения (абляция).
Ряд P сырой (без s²): калибровка кластера 3 — общий слой L. Параметры по шагам — R2/cache/<hash>/params.csv (design_sha, h, step, param, value).
Ничего за годы ≥ 2022 не читается: матрица — r2_core.load_d0 (строки ≤ 2021), G5 — фильтр date ≤ 2021-12-31; пишется только val шагов 1–6.
test_locked/ не читается. Метрик модуль не считает (выбор — агент O).

python3 scripts/analysis/r2_panel.py selftest
python3 scripts/analysis/r2_panel.py dym                       # 3 тикера × шаги 1–3 × seed 42 × h ∈ {1, 5, 22}: ряды, сдвиг, детектор, params
python3 scripts/analysis/r2_panel.py run --designs <f.yaml> --h 5 [--seeds 42 43 44] [--steps 1 .. 6] [--phase C] [--workers N]
python3 scripts/analysis/r2_panel.py leak --designs <f.yaml> [--tickers ...] [--steps ...]
Для r2_driver (агент S): applicable(design) -> (ok, причина); phase_fn(design, seed, steps) -> {ряд: V} (данные — r2_core.SHARED['A'] из load_A)."""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import yaml

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import r2_core as C  # noqa: E402

YAML = REPO / 'models' / 'designs' / 'r2' / 'r2_P_panel.yaml'
CFG = yaml.safe_load(YAML.read_text())
MODULE = 'P'
MAXSTEP = int(CFG['max_step'])
STEPS = tuple(range(1, MAXSTEP + 1))
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
KEY = ['date', 'ticker', 'step']
EPS = 1e-10
G5_COLS = list(CFG['data']['g5_cols'])
TAU2_MIN = float(CFG['decomp']['tau2_min'])
MIN_TK = int(CFG['u_leg']['min_ticker_rows'])
W_LO, W_HI = float(CFG['panel']['w_lo']), float(CFG['panel']['w_hi'])
Z_MKT = {k: list(v) for k, v in CFG['market_leg']['regressors_Z'].items()}
LGBM = dict(CFG['u_leg']['lgbm'])
H_MKT = ['x_d', 'x_w', 'x_m']
SHRINK_W = {'factor': 0.0, 'factor_shrink_025': 0.25, 'factor_shrink_05': 0.5}


class NotApplicable(Exception):
    """Сочетание ключей не относится к модулю P (дизайн отклоняется до счёта)."""


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def is_off(v):
    return v in (False, None, 'off')


def panel_of(d):
    p = d['leg']['panel']
    return 'off' if is_off(p) else p


# ================================================================ применимость
def check(d):
    """d — полный дизайн (C.validate). NotApplicable — ключи, которые модуль P не реализует (у них свои модули)."""
    d = C.validate(d)
    leg, boost, data = d['leg'], d['boost'], d['data']
    dflt = C.validate({'h': int(d['h'])})
    bad = []
    if leg['type'] not in CFG['applicable']['leg.type']:
        bad.append(f'leg.type={leg["type"]}')
    for k in CFG['applicable']['leg_default_only']:
        if leg[k] != dflt['leg'][k]:
            bad.append(f'leg.{k}={leg[k]}')
    for k, v in data.items():
        if k not in CFG['applicable']['data_supported'] and v != dflt['data'][k]:
            bad.append(f'data.{k}={v}')
    if boost['role'] not in CFG['applicable']['boost.role']:
        bad.append(f'boost.role={boost["role"]}')
    if boost['role'] == 'residual':
        if boost['lib'] not in CFG['applicable']['boost.lib']:
            bad.append(f'boost.lib={boost["lib"]}')
        for k in CFG['applicable']['boost_residual_only_default']:
            if boost[k] != dflt['boost'][k]:
                bad.append(f'boost.{k}={boost[k]}')
    for k, v in d['mh'].items():
        if v != dflt['mh'][k]:
            bad.append(f'mh.{k}={v}')
    extra = [g for g in d['features'] if g not in CFG['applicable']['features']]
    if extra:
        bad.append(f'features {extra}')
    gk = set(C.grid_keys(d)) - {'leg.enet.lambda', 'leg.enet.alpha', 'leg.pooled_kappa'}
    if gk:
        bad.append(f'сетки {sorted(gk)} не относятся к P')
    if bad:
        raise NotApplicable('; '.join(bad))
    return d


def applicable(design):
    try:
        check(design)
        return True, ''
    except NotApplicable as e:
        return False, str(e)


def grid_keys_P(d):
    leg = d['leg']
    k = []
    if panel_of(d) != 'off' and leg['type'] in ('HARX', 'factor') and leg['regressors'] != 'none':
        k += ['leg.enet.lambda', 'leg.enet.alpha']
    k.append('leg.pooled_kappa')
    return k


def _scal(x):
    return x[0] if isinstance(x, list) else x


def series_name(tag):
    return MODULE if tag == 'base' else f'{MODULE}__{tag}'


# ================================================================ данные
_G5 = {}
_D0 = {}


def load_g5():
    """date, ticker, g5_corr_m, g5_corr_w из features_r2_v2 (агент F), только date ≤ 2021-12-31; SHA сверяется с yaml."""
    if 'v' not in _G5:
        import pyarrow.parquet as pq
        p = C.MAC_ROOT / CFG['data']['g5']
        got = C.sha_file(p)
        if got != CFG['data']['g5_sha256']:
            raise AssertionError(f'{p.name}: SHA {got[:16]}… ≠ yaml {CFG["data"]["g5_sha256"][:16]}…')
        t = pq.read_table(p, columns=['date', 'ticker'] + G5_COLS,
                          filters=[('date', '<=', pd.Timestamp(f'{C.MAX_YEAR}-12-31'))]).to_pandas()
        t['date'] = pd.to_datetime(t['date'])
        assert t['date'].dt.year.max() <= C.MAX_YEAR
        _G5['v'] = t
    return _G5['v']


def load_A():
    """Полная матрица d0 (17 тикеров, ≤ 2021) + G5. Срез тикеров — только на выходе (рынок eq17 и τ² β — по всем 17)."""
    if 'A' not in _D0:
        A, harj = C.load_d0()
        C.check_d0(A)
        A = A.merge(load_g5(), on=['date', 'ticker'], how='left')
        _D0['A'], _D0['harj'] = A, harj
    return _D0['A'].copy(), _D0['harj']


def _roll(s, by, n, minp, fn='mean'):
    return getattr(s.groupby(by).rolling(n, min_periods=minp), fn)().reset_index(level=0, drop=True).sort_index()


class Data:
    """Всё, что строится из матрицы A (чистая функция A: детектор пересобирает Data на испорченной A).
    D — строки (ticker, date); mk — даты: обе меры рынка, ln RVI², ρ̄; кэш разложения по (рынок, W)."""

    def __init__(self, A, tickers=None):
        A = A.copy()
        A['date'] = pd.to_datetime(A['date'])
        A['year'] = A['date'].dt.year
        A = A.sort_values(['ticker', 'date']).reset_index(drop=True)
        tk = A['ticker']
        rvp = A['rv'].where(A['rv'] > 0).groupby(tk).ffill().fillna(A['rv']).clip(lower=EPS)
        A['_rvp'] = rvp
        A['l_d'] = np.log(rvp)
        A['l_w'] = np.log(_roll(rvp, tk, 5, 1))
        A['l_m'] = np.log(_roll(rvp, tk, 22, 1))
        dates = np.sort(A['date'].unique())
        g = A.groupby('date')
        imo = g['rv_IMOEX'].first().astype(float)
        if 'imoex_low_bars' in A:
            imo = imo.where(g['imoex_low_bars'].first() != 1)
        m_imo = (imo.where(imo > 0) ** 2 / 252).ffill()
        m_eq = np.exp(g['l_d'].mean())  # равные веса в логах (геометрическое среднее RV тикеров дня) — см. yaml decomp.market.RV_eq17
        rvi = g['close_RVI'].first().astype(float).ffill()
        mk = pd.DataFrame(index=pd.DatetimeIndex(dates))
        mk['m_RV_IMOEX'], mk['m_RV_eq17'] = m_imo.reindex(dates).values, m_eq.reindex(dates).values
        mk['ln_rvi2'] = np.log(rvi.reindex(dates).clip(lower=1e-6).values ** 2)
        mk['rvi'] = rvi.reindex(dates).values
        for c in G5_COLS:
            mk[c] = g[c].mean().reindex(dates).values if c in A else np.nan
        self.A, self.mk, self.dates = A, mk, mk.index
        A['_di'] = np.searchsorted(self.dates.values, A['date'].values)
        self.tickers = tuple(sorted(tickers)) if tickers else tuple(sorted(A['ticker'].unique()))
        self.out = A['ticker'].isin(self.tickers).values
        self._mkt, self._ab = {}, {}

    def at(self, s):
        """Ряд по датам -> значения по строкам D."""
        return np.asarray(s, dtype='float64')[self.A['_di'].values]

    def market(self, which):
        """Даты: x_d, x_w, x_m (ln m, ln среднего 5/22), x_q/x_h (66/132), x_o (ln 22-дн. второй меры), ym{h} (ln m̄_{d+1..d+h}), tend{h}."""
        if which not in self._mkt:
            mk = self.mk
            m = mk[f'm_{which}'].astype(float)
            o = mk['m_RV_eq17' if which == 'RV_IMOEX' else 'm_RV_IMOEX'].astype(float)
            F = pd.DataFrame(index=mk.index)
            F['x_d'] = np.log(m.clip(lower=EPS))
            F['x_w'] = np.log(m.rolling(5, min_periods=1).mean().clip(lower=EPS))
            F['x_m'] = np.log(m.rolling(22, min_periods=1).mean().clip(lower=EPS))
            F['ln_m_q'] = np.log(m.rolling(66, min_periods=40).mean().clip(lower=EPS))
            F['ln_m_h'] = np.log(m.rolling(132, min_periods=80).mean().clip(lower=EPS))
            F['ln_m_other_m'] = np.log(o.rolling(22, min_periods=1).mean().clip(lower=EPS))
            F['ln_rvi2'] = mk['ln_rvi2'].values
            F['rvi'] = mk['rvi'].values
            dts = pd.Series(mk.index, index=mk.index)
            for h in C.HS:
                fut = sum(m.shift(-j) for j in range(1, h + 1)) / h
                F[f'ym{h}'] = np.log(fut.clip(lower=EPS))
                F[f'tend{h}'] = dts.shift(-h).values
            self._mkt[which] = F
        return self._mkt[which]

    def ab(self, which, W):
        """α̃, β̃ (усадка к 1 по Васичеку), β̂, c по строкам D; окно — W строк тикера по день d."""
        key = (which, int(W))
        if key not in self._ab:
            A = self.A
            x = self.at(self.market(which)['x_d'].values)
            y = A['l_d'].values.copy()
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = np.where(ok, x, np.nan), np.where(ok, y, np.nan)
            tk = A['ticker']
            fr = pd.DataFrame({'x': x, 'y': y, 'xy': x * y, 'xx': x * x, 'yy': y * y})
            r = fr.groupby(tk).rolling(int(W), min_periods=int(W) // 2).mean().reset_index(level=0, drop=True).sort_index()
            n = pd.Series(ok.astype(float)).groupby(tk).rolling(int(W), min_periods=1).sum().reset_index(level=0, drop=True).sort_index()
            n = n.values
            sxx = (r.xx - r.x ** 2).values
            sxy = (r.xy - r.x * r.y).values
            syy = (r.yy - r.y ** 2).values
            with np.errstate(all='ignore'):
                bh = sxy / sxx
                s2 = np.clip(syy - bh * sxy, 0, None) * n / np.maximum(n - 2, 1)
                se2 = s2 / (n * sxx)
            bad = ~(np.isfinite(bh) & np.isfinite(se2) & (sxx > 1e-12))
            bh[bad], se2[bad] = np.nan, np.nan
            tau2 = pd.Series(bh).groupby(A['date'].values).transform('var').values  # ddof = 1
            tau2 = np.where(np.isfinite(tau2), np.maximum(tau2, TAU2_MIN), TAU2_MIN)
            c = se2 / (se2 + tau2)
            bt = c * 1.0 + (1 - c) * bh
            at = r.y.values - bt * r.x.values
            self._ab[key] = dict(alpha=at, beta=bt, beta_raw=bh, c=c)
        return self._ab[key]


# ================================================================ линейные подгонки
def _w_recency(dates, cutoff, data):
    hl = data.get('recency_halflife', 'inf')
    if hl in ('inf', None) or (isinstance(hl, float) and not np.isfinite(hl)):
        return None
    yrs = float(str(hl).rstrip('y'))
    age = (np.datetime64(cutoff) - np.asarray(dates, dtype='datetime64[ns]')) / np.timedelta64(1, 'D') / 365.25
    return 0.5 ** (age / yrs)


def _ols(M, y, w=None):
    Mc = np.column_stack([np.ones(len(M)), M])
    sw = np.ones(len(y)) if w is None else np.sqrt(w)
    return np.linalg.lstsq(Mc * sw[:, None], y * sw, rcond=None)[0]


def _pred(b, M):
    return np.column_stack([np.ones(len(M)), M]) @ b


def _fit_en(H, Z, y, lam, l1, w=None):
    """HAR-члены H без штрафа + Z со штрафом эластичной сети (Z стандартизуется по обучению). w=None — сам harx.fit_en (как ноги L);
    с весами давности — тот же алгоритм (Фриш–Во) с весами наблюдений. -> (predict(H, Z), коэффициенты [1, H, Z в исходном масштабе])."""
    import harx as HX
    if w is None:
        p, co = HX.fit_en(H, Z, y, lam, l1)
        return p, np.concatenate([co['b'], co['g_raw']])
    from sklearn.linear_model import ElasticNet
    ws = w / w.sum()
    mu = ws @ Z
    sd = np.sqrt(ws @ (Z - mu) ** 2)
    keep = sd > 1e-12
    Zs = np.zeros_like(Z)
    Zs[:, keep] = (Z[:, keep] - mu[keep]) / sd[keep]
    Hc = np.column_stack([np.ones(len(H)), H])
    sw = np.sqrt(w)[:, None]
    B = np.linalg.lstsq(Hc * sw, np.column_stack([y, Zs]) * sw, rcond=None)[0]
    R = np.column_stack([y, Zs]) - Hc @ B
    g = np.zeros(Z.shape[1])
    if keep.any():
        en = ElasticNet(alpha=lam, l1_ratio=l1, fit_intercept=False, max_iter=20000, tol=1e-7, selection='cyclic', precompute=True)
        en.fit(R[:, 1:][:, keep], R[:, 0], sample_weight=w * len(w) / w.sum())
        g[keep] = en.coef_
    b = B[:, 0] - B[:, 1:] @ g

    def pred(H2, Z2):
        Z2s = np.zeros_like(Z2)
        Z2s[:, keep] = (Z2[:, keep] - mu[keep]) / sd[keep]
        return np.column_stack([np.ones(len(H2)), H2]) @ b + Z2s @ g
    return pred, np.concatenate([b, np.where(keep, g / np.where(keep, sd, 1), 0.0)])


# ================================================================ рыночная компонента
def mkt_forecast(Dt, which, h, leg, lam, l1, cutoff, data, par, pre):
    """F_mkt по датам (ln m̄_{d+1..d+h}) для модели шага с точкой переобучения cutoff. par — список (имя, значение) параметров."""
    F = Dt.market(which)
    c = np.datetime64(cutoff)
    y = F[f'ym{h}'].values
    Hm = F[H_MKT].values
    tr = np.isfinite(y) & (F[f'tend{h}'].values < c) & np.isfinite(Hm).all(1)
    w = _w_recency(F.index.values[tr], cutoff, data)
    par.append((f'{pre}mkt_n_train', float(tr.sum())))
    t = leg['type']
    zc = Z_MKT[leg['regressors']] if t in ('HARX', 'factor') else []

    def har():
        b = _ols(Hm[tr], y[tr], w)
        return _pred(b, np.nan_to_num(Hm)), b

    out = {}
    if t in ('HARJ', 'HARX', 'factor'):
        if zc:
            Zdf = F[zc]
            Z = Zdf.fillna(Zdf[tr].mean()).fillna(0.0).values.astype('float64')
            p, b = _fit_en(Hm[tr], Z[tr], y[tr], lam, l1, w)
            fx = p(np.nan_to_num(Hm), Z)
            names = ['const'] + H_MKT + zc
            par += [(f'{pre}mkt_lambda', lam), (f'{pre}mkt_alpha', l1)]
        else:
            fx, b = har()
            names = ['const'] + H_MKT
        par += [(f'{pre}mkt_b_{n}', float(v)) for n, v in zip(names, b)]
        out['harx'] = fx
    if t in ('RVI_anchor', 'factor'):
        import d25_1 as D25
        B = pd.DataFrame({'date': F.index, 'ticker': 'MKT', 'close_RVI': F['rvi'].values, f'rv_target_mean_h{h}': np.exp(y)})
        L = D25.rvi_leg(B, h).set_index('date').reindex(F.index)
        la = np.log(np.clip(L['leg'].values, EPS, None))
        la[~np.isfinite(L['leg'].values)] = np.nan
        ok = tr & np.isfinite(la)
        a = float(np.average(y[ok] - la[ok], weights=None if w is None else w[np.isfinite(la[tr])])) if ok.any() else 0.0
        fa = la + a
        fh = har()[0]
        miss = ~np.isfinite(fa)
        fa = np.where(miss, fh, fa)
        par += [(f'{pre}anchor_a', a), (f'{pre}anchor_n_train', float(ok.sum()))]
        out['anchor'], out['anchor_miss'], out['anchor_k'] = fa, miss, L['k'].values
    if t == 'factor':
        f = 0.5 * out['harx'] + 0.5 * out['anchor']
    elif t == 'RVI_anchor':
        f = out['anchor']
    else:
        f = out['harx']
    return f, out


# ================================================================ идиосинкразия
def _har_kappa(X, y, tk, tr, pr, kappa, w):
    """F = (1 − κ)·пул-МНК + κ·МНК тикера (у тикера < MIN_TK строк — пул). -> (прогноз на строках pr, коэффициенты пула, число тикеров с МНК)."""
    bp = _ols(X[tr], y[tr], w)
    out = _pred(bp, X[pr])
    nt = 0
    if kappa > 0:
        tk_tr, tk_pr = tk[tr], tk[pr]
        wt = None if w is None else w
        for t in np.unique(tk_pr):
            mi = tk_tr == t
            if mi.sum() >= MIN_TK:
                bi = _ols(X[tr][mi], y[tr][mi], None if wt is None else wt[mi])
                pj = tk_pr == t
                out[pj] = (1 - kappa) * out[pj] + kappa * _pred(bi, X[pr][pj])
                nt += 1
    return out, bp, nt


def _boost(Xb, r, tr, pr, seed, w):
    import lightgbm as lgb
    p = {k: v for k, v in LGBM.items() if k != 'n_estimators'}
    p['seed'] = int(seed)
    ds = lgb.Dataset(Xb[tr], r, weight=w, free_raw_data=False)
    bst = lgb.train(p, ds, num_boost_round=int(LGBM['n_estimators']))
    return bst.predict(Xb[pr], num_threads=1), bst.num_trees()


# ================================================================ дизайн -> ряды
def run_design(design, Dt, seed=42, steps=STEPS, want_all=False):
    """-> ({ряд: V}, {ряд: W (date, ticker, step, rv_pred на строках year ≤ vy) | None}, [(ряд, шаг, параметр, значение)]).
    Строки — тикеры Dt.tickers; разложение и рынок — по всей матрице Dt."""
    d = check(design)
    h, leg, data, boost = int(d['h']), d['leg'], d['data'], d['boost']
    pan = panel_of(d)
    which, W = leg['panel_market'], int(leg['panel_beta_window'])
    A = Dt.A
    tk = A['ticker'].values
    tcol = f'rv_target_h{h}'
    lnY = np.log(A[tcol].values + EPS)
    F = Dt.market(which)
    ab = Dt.ab(which, W)
    al, be = ab['alpha'], ab['beta']
    xr = {c: Dt.at(F[c].values) for c in ('x_d', 'x_w', 'x_m', 'ln_rvi2', f'ym{h}')}
    tend_m = pd.to_datetime(F[f'tend{h}'])  # конец окна рыночной цели (по датам); NaT — цели нет
    U = np.column_stack([A['l_d'].values - al - be * xr['x_d'], A['l_w'].values - al - be * xr['x_w'],
                         A['l_m'].values - al - be * xr['x_m']])
    tu = lnY - al - be * xr[f'ym{h}']
    Xoff = A[['l_d', 'l_w', 'l_m']].values
    g5 = [c for c in G5_COLS if 'G5' in d['features']]
    base_b = [xr['x_d'], xr['x_w'], xr['x_m'], xr['ln_rvi2']] + [A[c].values for c in g5]
    Xb_fac = np.column_stack([U, xr['x_d'], xr['x_w'], xr['x_m'], be, xr['ln_rvi2']] + [A[c].values for c in g5])
    Xb_off = np.column_stack([Xoff] + base_b)
    rho = Dt.at(Dt.mk[G5_COLS[0]].values)
    vrs = C.variants(d, keys=grid_keys_P(d))
    outV, outW, par = {}, {}, []
    accV = {series_name(tg): [] for tg, _, _ in vrs}
    accW = {series_name(tg): [] for tg, _, _ in vrs}
    for k in steps:
        C.assert_dev_step(k)
        vy = C.step_years(k)[0]
        cut = pd.Timestamp(f'{vy}-01-01')
        base_tr = C.train_mask(A, h, vy) & Dt.out & np.isfinite(lnY)
        tr_off = base_tr & np.isfinite(Xoff).all(1)
        mkt_ok = Dt.at((tend_m < cut).fillna(False).values.astype(float)) > 0.5
        tr_fac = base_tr & mkt_ok & np.isfinite(U).all(1) & np.isfinite(tu)
        vm = C.val_mask(A, h, vy) & Dt.out
        pr = ((A['year'].values <= vy) & Dt.out) if want_all else vm
        w_off = _w_recency(A['date'].values[tr_off], cut, data)
        w_fac = _w_recency(A['date'].values[tr_fac], cut, data)
        # усадка: w по строкам
        if pan == 'factor_shrink_state':
            dtr = Dt.dates[Dt.dates < cut]
            hist = Dt.mk.loc[dtr, G5_COLS[0]].dropna().values
            ecdf = np.searchsorted(np.sort(hist), rho, side='right') / max(len(hist), 1)
            wsh = np.where(np.isfinite(rho) & (len(hist) > 0), W_LO + (W_HI - W_LO) * ecdf, 0.5 * (W_LO + W_HI))
        else:
            wsh = np.full(len(A), SHRINK_W.get(pan, 0.0))
        mcache, ucache = {}, {}
        for tg, _, dv in vrs:
            nm = series_name(tg)
            lg = dv['leg']
            kap = float(_scal(lg['pooled_kappa']))
            e = lg['enet']
            lam, l1 = float(_scal(e['lambda'])), float(_scal(e['alpha']))
            p = []
            # F_u (или HAR без разложения) — по κ; запасной прогноз строк без α/β — HAR без разложения
            if kap not in ucache:
                q = []
                off_pr, boff, ntf = _har_kappa(Xoff, lnY, tk, tr_off, pr, kap, w_off)
                q += [(f'u_off_b_{n}', float(v)) for n, v in zip(['const', 'l_d', 'l_w', 'l_m'], boff)]
                if pan == 'off':
                    fu, trx, Xb, yy, ww = off_pr, tr_off, Xb_off, lnY, w_off
                    q += [('u_n_train', float(tr_off.sum())), ('u_n_ticker_fit', float(ntf))]
                else:
                    fu, bu, ntu = _har_kappa(U, tu, tk, tr_fac, pr, kap, w_fac)
                    trx, Xb, yy, ww = tr_fac, Xb_fac, tu, w_fac
                    q += [(f'u_b_{n}', float(v)) for n, v in zip(['const', 'u_d', 'u_w', 'u_m'], bu)]
                    q += [('u_n_train', float(tr_fac.sum())), ('u_n_ticker_fit', float(ntu))]
                if boost['role'] == 'residual':
                    ins, _, _ = _har_kappa(Xoff if pan == 'off' else U, yy, tk, trx, trx, kap, ww)
                    bp, nt = _boost(Xb, yy[trx] - ins, trx, pr, seed, ww)
                    fu = fu + float(boost['shrink']) * bp
                    q += [(f's{seed}:boost_n_trees', float(nt)), (f's{seed}:boost_resid_sd_train', float(np.std(yy[trx] - ins))),
                          (f's{seed}:boost_pred_sd', float(np.std(bp)))]
                # ū^h тикера (κ-смесь пула и тикера) — уровень u в масштабе цели h: α̃^(h) = α̃ + ū^h (цель усадки)
                cu = np.full(int(pr.sum()), np.nan)
                if pan != 'off':
                    m0 = float(np.average(yy[trx], weights=ww))
                    cu[:] = m0
                    tk_tr, tk_pr = tk[trx], tk[pr]
                    for t in np.unique(tk_pr):
                        mi = tk_tr == t
                        if kap > 0 and mi.sum() >= MIN_TK:
                            cu[tk_pr == t] = (1 - kap) * m0 + kap * float(np.average(yy[trx][mi], weights=None if ww is None else ww[mi]))
                    q.append(('u_mean_pool', m0))
                q.append(('kappa', kap))
                ucache[kap] = (fu, off_pr, cu, q)
            fu, off_pr, cu, q = ucache[kap]
            p += q
            if pan == 'off':
                ln = fu
                nfb = 0
            else:
                mkey = (lam, l1)
                if mkey not in mcache:
                    mp = []
                    fm, parts = mkt_forecast(Dt, which, h, leg, lam, l1, cut, data, mp, '')
                    mcache[mkey] = (Dt.at(fm), parts, mp)
                fmr, parts, mp = mcache[mkey]
                p += mp
                sysc = al[pr] + cu + be[pr] * fmr[pr]  # α̃^(h) + β̃·F_mkt
                ln = sysc + (1 - wsh[pr]) * (fu - cu)  # = (1 − w)·(α̃ + β̃F_mkt + F_u) + w·(α̃^(h) + β̃F_mkt)
                fb = ~np.isfinite(ln)
                ln = np.where(fb, off_pr, ln)
                nfb = int((fb & vm[pr]).sum())
                vv = vm[pr]
                if 'anchor' in parts:
                    di = np.unique(A['_di'].values[pr][vv])
                    kk = parts['anchor_k'][di]
                    p += [('anchor_k_med_val', float(np.nanmedian(kk)) if np.isfinite(kk).any() else np.nan),
                          ('anchor_miss_share_val', float(np.mean(parts['anchor_miss'][di])))]
                bv, cv = be[pr][vv], ab['c'][pr][vv]
                p += [('beta_med_val', float(np.nanmedian(bv))), ('beta_p10_val', float(np.nanpercentile(bv, 10))),
                      ('beta_p90_val', float(np.nanpercentile(bv, 90))), ('beta_raw_med_val', float(np.nanmedian(ab['beta_raw'][pr][vv]))),
                      ('shrink_c_mean_val', float(np.nanmean(cv))), ('alpha_med_val', float(np.nanmedian(al[pr][vv]))),
                      ('w_mean_val', float(np.mean(wsh[pr][vv]))), ('w_sd_val', float(np.std(wsh[pr][vv])))]
            p.append(('n_fallback_val', float(nfb)))
            par += [(nm, k, a, v) for a, v in p]  # имя параметра в params.csv — <ряд>:<имя> (write_params)
            rows = pr
            fr = pd.DataFrame({'date': A['date'].values[rows], 'ticker': A['ticker'].values[rows].astype(str), 'year': A['year'].values[rows],
                               'step': k, 'rv_actual': A[tcol].values[rows], 'rv_pred': np.clip(np.exp(ln), EPS, None)})
            accV[nm].append(fr[vm[pr]].assign(year=vy))
            if want_all:
                accW[nm].append(fr[['date', 'ticker', 'step', 'rv_pred']])
    for nm in accV:
        V = pd.concat(accV[nm], ignore_index=True).sort_values(KEY).reset_index(drop=True)[COLS]
        assert V['year'].max() <= C.MAX_YEAR and np.isfinite(V['rv_pred']).all(), f'{nm}: ≥ 2022 или нечисловой прогноз'
        outV[nm] = V
        outW[nm] = pd.concat(accW[nm], ignore_index=True) if want_all else None
    return outV, outW, par


# ================================================================ параметры по шагам (R2/cache/<hash>/params.csv)
def write_params(design, h, rows, root=None):
    """design_sha, h, step, param, value; param = <ряд>:<имя>. Строки с теми же (h, step, param) заменяются (flock + os.replace)."""
    import fcntl
    p = C.cache_dir(design, root=root) / 'params.csv'
    sha = C.design_hash(design)
    new = pd.DataFrame([dict(design_sha=sha, h=int(h), step=int(s), param=f'{nm}:{a}', value=float(v)) for nm, s, a, v in rows
                        if v is not None and np.isfinite(float(v))], columns=['design_sha', 'h', 'step', 'param', 'value'])
    with open(str(p) + '.lock', 'a') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        old = pd.read_csv(p) if p.exists() else new.iloc[:0]
        key = lambda x: x.h.astype(str) + '|' + x.step.astype(str) + '|' + x.param
        out = pd.concat([old[~key(old).isin(set(key(new)))], new], ignore_index=True).sort_values(['h', 'param', 'step'])
        tmp = p.with_name(f'.params.{os.getpid()}.tmp')
        out.to_csv(tmp, index=False)
        os.replace(tmp, p)
        fcntl.flock(lk, fcntl.LOCK_UN)
    return p


# ================================================================ run_phase ядра
def phase_fn(design, seed, steps):
    """fn для r2_core.run_phase: SHARED['A'] — load_A() (17 тикеров + G5), SHARED['P_data'] — Data (строится в родителе до пула),
    SHARED['P_root'] — корень кэша (необязательно)."""
    S = C.SHARED
    Dt = S.get('P_data') or Data(S['A'])
    V, _, par = run_design(design, Dt, seed, tuple(steps))
    write_params(design, int(design['h']), par, root=S.get('P_root'))
    return V


def load_designs(p):
    Y = yaml.safe_load(Path(p).read_text())
    return Y['designs'] if isinstance(Y, dict) and 'designs' in Y else (Y if isinstance(Y, list) else [Y])


def run(designs, h, seeds, steps, workers=None, phase='C', root=None, check_gate=True, block_file=None):
    """Боевой счёт (сервер) через r2_core.run_phase: ворота, кэш, запись, сдвиг, DONE. Неприменимые дизайны — в лог и мимо."""
    A, harj = load_A()
    Dt = Data(A)
    for w in sorted({int(_scal(C.validate({**d, 'h': h})['leg']['panel_beta_window'])) for d in designs}):
        for m in ('RV_IMOEX', 'RV_eq17'):
            Dt.ab(m, w)
    C.share(A=A, harj=harj, P_data=Dt, P_root=root)
    ok = []
    for d in designs:
        full = C.validate({**d, 'h': h})
        a, why = applicable(full)
        if a:
            ok.append(full)
        else:
            log(f'не применим: {full.get("design")} — {why}')
    return C.run_phase(phase, h, ok, phase_fn, seeds=tuple(seeds), steps=tuple(steps), workers=workers, module=MODULE, root=root,
                       block_file=block_file, check_gate=check_gate)


# ================================================================ детектор и сдвиг
def leak_build(design, seed, steps, tickers):
    """build(A) для r2_core.leak_check: val (part = val) и прогнозы моделей шагов на всех строках year ≤ vy (part = all)."""
    def build(A):
        V, Wd, _ = run_design(design, Data(A, tickers), seed, steps, want_all=True)
        return {nm: pd.concat([V[nm][['date', 'ticker', 'step', 'rv_pred']].assign(part='val'), Wd[nm].assign(part='all')], ignore_index=True)
                for nm in V}
    return build


def leak_design(design, A, seed, steps, tickers, cuts, name):
    return C.leak_check(leak_build(design, seed, steps, tickers), A, f'P:{name}', cuts=cuts)


# ================================================================ самопроверки
def selftest():
    rng = np.random.default_rng(7)
    # 1) β: точная линейная связь -> β̂ = b, se² = 0, c = 0, α̃ = a
    n, dates = 400, pd.bdate_range('2015-01-01', periods=400)
    x = rng.normal(-9, 0.4, n)
    rows = []
    for i, (a, b) in enumerate([(1.0, 0.8), (-0.5, 1.3), (0.2, 1.0)]):
        rows.append(pd.DataFrame({'date': dates, 'ticker': f'T{i}', 'rv': np.exp(a + b * x), 'rv_IMOEX': np.sqrt(np.exp(x) * 252),
                                  'close_RVI': 30.0, **{c: 0.2 for c in G5_COLS}}))
    A = pd.concat(rows, ignore_index=True)
    Dt = Data(A)
    ab = Dt.ab('RV_IMOEX', 126)
    last = Dt.A.groupby('ticker').tail(1).index
    for (a, b), i in zip([(1.0, 0.8), (-0.5, 1.3), (0.2, 1.0)], last):
        assert abs(ab['beta'][i] - b) < 1e-6 and abs(ab['alpha'][i] - a) < 1e-5 and ab['c'][i] < 1e-6, (ab['beta'][i], ab['alpha'][i], ab['c'][i])
    # 2) усадка к 1: шумный β̂ с большой se² тянется к 1
    y = 0.3 + 1.6 * x + rng.normal(0, 2.0, n)
    A2 = A.copy()
    A2.loc[A2.ticker == 'T0', 'rv'] = np.exp(y)
    ab2 = Data(A2).ab('RV_IMOEX', 126)
    i0 = Data(A2).A.query('ticker == "T0"').index[-1]
    assert abs(ab2['beta'][i0] - 1) < abs(ab2['beta_raw'][i0] - 1) and 0 < ab2['c'][i0] < 1
    # 3) рынок eq17 = среднее RV тикеров дня; ym{h} — среднее следующих h дней
    F = Dt.market('RV_eq17')
    m = Dt.mk['m_RV_eq17']
    assert abs(np.exp(F['ym5'].iloc[10]) - m.iloc[11:16].mean()) < 1e-15
    # 4) применимость
    for na in ({'h': 5, 'leg': {'type': 'HEAVY'}}, {'h': 5, 'leg': {'refit': 'monthly'}}, {'h': 5, 'boost': {'role': 'feature'}},
               {'h': 5, 'boost': {'role': 'residual', 'lib': 'both_avg'}}, {'h': 5, 'mh': {'persist': 'OLS'}},
               {'h': 5, 'data': {'history_start': 'ISS_start'}}, {'h': 5, 'leg': {'regime_count': [2, 3]}}):
        assert not applicable(na)[0], f'не поймано: {na}'
    assert applicable({'h': 5, 'leg': {'panel': 'factor_shrink_state', 'pooled_kappa': [0, 0.5, 1]}, 'data': {'recency_halflife': '4y'}})[0]
    d = C.validate({'h': 5, 'leg': {'panel': 'factor', 'type': 'HARX', 'regressors': 'S1'}})
    assert len(C.variants(d, keys=grid_keys_P(d))) == 9
    d = C.validate({'h': 5, 'leg': {'panel': 'off', 'type': 'HARX', 'regressors': 'S1'}})
    assert len(C.variants(d, keys=grid_keys_P(d))) == 1
    # 5) взвешенная эластичная сеть с равными весами = harx.fit_en
    H, Z = rng.normal(size=(300, 3)), rng.normal(size=(300, 2))
    yy = H @ [0.3, 0.2, 0.1] + Z @ [0.5, 0.0] + rng.normal(0, 0.1, 300)
    p0, b0 = _fit_en(H, Z, yy, 0.01, 0.5)
    p1, b1 = _fit_en(H, Z, yy, 0.01, 0.5, np.ones(300))
    assert np.abs(b0 - b1).max() < 1e-6, np.abs(b0 - b1).max()
    log('selftest OK: β и α разложения, усадка к 1, рынок eq17 и цель m̄_h, применимость, сетки, взвешенная EN = harx.fit_en')


# ================================================================ дым
def dym():
    D = CFG['dym']
    tick, steps, seed = list(D['tickers']), tuple(D['steps']), int(D['seed'])
    cuts = tuple(CFG['guards']['leak']['cuts_dym'])
    out = C.MAC_ROOT / CFG['out']['dym']
    out.mkdir(parents=True, exist_ok=True)
    blk = out / 'blok_sdvig_dym.csv'
    selftest()
    A, harj = load_A()
    Dt = Data(A, tick)
    summ, leak_rows, sdvig, par_rows = [], [], [], []
    for h in D['h']:
        ref = C._smoke_series(Dt.A, harj, h, steps)
        ref = ref[ref.ticker.isin(tick)][KEY + ['rv_pred']]
        for spec in D['designs']:
            if spec.get('only_h') and h not in spec['only_h']:
                continue
            nm = spec['name']
            des = C.validate({'design': nm, 'h': h, 'seed': seed, **{k: v for k, v in spec.items() if k not in ('name', 'only_h')}})
            ok, why = applicable(des)
            if not ok:
                summ.append(dict(h=h, design=nm, status='ne_primenim', note=why))
                continue
            t0 = time.time()
            V, _, par = run_design(des, Dt, seed, steps)
            t1 = time.time() - t0
            R = leak_design(des, A, seed, steps, tick, cuts, nm)
            for r in R.to_dict('records'):
                leak_rows.append(dict(h=h, design=nm, **r))
            sha = C.design_hash(des)
            par_rows += [dict(design=nm, design_sha=sha, h=h, step=s, param=f'{sn}:{a}', value=v) for sn, s, a, v in par]
            for sn, v in V.items():
                v.to_parquet(out / f'{nm}_{sn}_h{h}_s{seed}_val.parquet', index=False)
                sc = C.shift_check(v, A, h, f'P:{nm}:{sn}', des, block_file=blk)
                sdvig.append(dict(h=h, design=nm, ryad=sn, **{k: x for k, x in sc.items() if k.startswith('corr') or k in ('argmax', 'ok')}))
                rr = v.merge(ref, on=KEY, suffixes=('', '_harj'))
                dl = np.log(rr.rv_pred / rr.rv_pred_harj)
                summ.append(dict(h=h, design=nm, ryad=sn, status='ok', rows=len(v), years=f'{v.year.min()}-{v.year.max()}',
                                 dln_vs_harj_mean=float(dl.mean()), dln_vs_harj_sd=float(dl.std()),
                                 leak_ok=bool(R[R.ryad == sn].ok.all()), shift_ok=sc['ok'], shift_argmax=sc['argmax'], sec=round(t1, 1),
                                 sha_design=sha[:16], slozhnost=C.complexity(des)))
                log(f'h={h} {nm} {sn}: строк {len(v)}, детектор {"OK" if summ[-1]["leak_ok"] else "ОШИБКА"}, '
                    f'сдвиг argmax={sc["argmax"]} {"OK" if sc["ok"] else "БЛОК"}, {t1:.1f} с')
        # положительный контроль сдвига: тот же дизайн на матрице с подлогом «вход d := d+1» должен блокироваться
        des = C.validate({'design': D['designs'][0]['name'], 'h': h, 'seed': seed, 'leg': D['designs'][0]['leg']})
        Vp, _, _ = run_design(des, Data(C.plant_frame(A), tick), seed, steps)
        sc = C.shift_check(Vp['P'], A, h, 'P:kontrol_podlog', des, block_file=blk)
        summ.append(dict(h=h, design='kontrol_podlog_d+1', ryad='P', status='ok', rows=len(Vp['P']), leak_ok=None,
                         shift_ok=sc['ok'], shift_argmax=sc['argmax'], note='ожидается БЛОК (+1)'))
        sdvig.append(dict(h=h, design='kontrol_podlog_d+1', ryad='P', **{k: x for k, x in sc.items() if k.startswith('corr') or k in ('argmax', 'ok')}))
        log(f'h={h} контроль подлога: argmax={sc["argmax"]} {"БЛОК — ожидаемо" if not sc["ok"] else "НЕ ПОЙМАН — ОШИБКА"}')
    S = pd.DataFrame(summ)
    S.to_csv(out / 'dym_summary.csv', index=False)
    pd.DataFrame(leak_rows).to_csv(out / 'leak.csv', index=False)
    pd.DataFrame(par_rows).to_csv(out / 'params.csv', index=False)
    pd.DataFrame(sdvig).to_csv(out / 'sdvig.csv', index=False)
    ok = S[(S.status == 'ok') & (S.design != 'kontrol_podlog_d+1')]
    kp = S[S.design == 'kontrol_podlog_d+1']
    good = bool(ok.leak_ok.all() and (ok.shift_ok == True).all() and (kp.shift_ok == False).all())  # noqa: E712
    log(f'дым: рядов {len(ok)}, детектор {int(ok.leak_ok.sum())}/{len(ok)}, сдвиг {int((ok.shift_ok == True).sum())}/{len(ok)}, '  # noqa: E712
        f'контроль подлога пойман {int((kp.shift_ok == False).sum())}/{len(kp)}, не применимо {int((S.status == "ne_primenim").sum())} — '  # noqa: E712
        f'{"ВСЕ OK" if good else "ЕСТЬ ОШИБКИ"}')
    return S, good


def leak_cmd(designs, steps, tickers, seed):
    A, _ = load_A()
    rows = []
    cuts = tuple(CFG['guards']['leak']['cuts_full'])
    for d in load_designs(designs):
        des = C.validate({**d, 'seed': seed})
        R = leak_design(des, A, seed, tuple(steps), tickers, cuts, des.get('design') or 'P')
        rows.append(R.assign(design=des.get('design'), h=int(des['h'])))
    R = pd.concat(rows, ignore_index=True)
    print(R.to_string(index=False))
    assert R.ok.all(), 'детектор: ОШИБКА'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['selftest', 'dym', 'run', 'leak'])
    ap.add_argument('--designs')
    ap.add_argument('--h', type=int)
    ap.add_argument('--phase', default='C')
    ap.add_argument('--steps', type=int, nargs='+', default=list(STEPS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tickers', nargs='+')
    ap.add_argument('--workers', type=int)
    a = ap.parse_args()
    if a.cmd == 'selftest':
        selftest()
    elif a.cmd == 'dym':
        sys.exit(0 if dym()[1] else 1)
    elif a.cmd == 'run':
        print(run(load_designs(a.designs), a.h, a.seeds, a.steps, a.workers, a.phase).to_string(index=False))
    elif a.cmd == 'leak':
        leak_cmd(a.designs, a.steps, a.tickers, a.seed)


if __name__ == '__main__':
    main()
