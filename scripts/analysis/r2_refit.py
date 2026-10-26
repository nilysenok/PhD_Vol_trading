#!/usr/bin/env python3
"""R2, агент R: кластер 15 — адаптивное обновление и оценивание под критерий (протокол v3 docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md,
разделы 1, 2.6, 4, 6; схема models/designs/r2/schema_v1.yaml; определения — models/designs/r2/r2_R_refit.yaml, записаны до счёта).
Ядро — scripts/analysis/r2_core.py (агент S): validate / design_hash / cache_dir / write_series, load_d0, perturb_frame / plant_frame,
shift_check, run_phase. Код первого раунда не меняется (harx, gibrid_har только импортируются).

Ключи:
  leg.refit        yearly | quarterly | monthly — переоценка ноги внутри val-года на точках c (1 янв., начала кварталов, месяцев):
                   строки обучения — дата < c, цель есть, конец окна цели < c (эмбарго h по концу цели); прогноз строк [c_i, c_{i+1}).
  leg.rls_lambda   1 | 0,995 | 0,99 — RLS с забыванием в точной форме: коэффициенты на КАЖДЫЙ торговый день t — взвешенный МНК (эл. сеть)
                   по строкам с концом цели < t, вес λ^(возраст в торговых днях). Для HARJ и HARX (S1–S5).
  leg.estimator    log_ols_calib — МНК в логах + слой калибровки L (calib.*); qmle_level — QMLE в уровнях E[RV̄_h] = exp(x'β), гамма-потеря.
  boost.boost_refit  yearly | quarterly_warm | monthly — остаточный ученик LightGBM (boost.role = residual) на ноге дизайна:
                   quarterly_warm — тёплый старт init_model, +warm_add деревьев, lr тот же; num_threads = 1, без ранней остановки.
  calib.update     step | daily — слой калибровки L (r2_calib.calibrate); пока его нет в базе — свой слой для calib.* = default.
Ряды: R[__<вариант>] — итог (для O s2 = bez), R_raw[__<вариант>] — до калибровки. Параметры по шагам — R2/cache/<hash>/params.csv.
Год валидации — только для оценки: деревьев фиксированное число, гиперпараметров по нему нет; ничего ≥ 2022 не читается и не пишется.

python3 scripts/analysis/r2_refit.py selftest
python3 scripts/analysis/r2_refit.py control                    # дизайн по умолчанию = нога L HARJ_emb_d0 (17 тикеров, шаги 1–6) и × V0_F sm
python3 scripts/analysis/r2_refit.py dym                        # 3 тикера × шаги 1–3 × seed 42 × h ∈ {1, 5, 22}: детектор, сдвиг, params
python3 scripts/analysis/r2_refit.py run --designs <f.yaml> --h 5 [--seeds 42 43 44] [--steps 1 .. 6] [--workers N] [--phase C]
python3 scripts/analysis/r2_refit.py leak --designs <f.yaml> [--tickers ...] [--steps ...]"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import yaml

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import r2_core as C  # noqa: E402
import harx as HX  # noqa: E402
import gibrid_har as GH  # noqa: E402

YAML = REPO / 'models' / 'designs' / 'r2' / 'r2_R_refit.yaml'
CFG = yaml.safe_load(YAML.read_text())
F_YAML = REPO / 'models' / 'designs' / 'r2' / 'r2_F_features.yaml'
MODULE = 'R'
MAXSTEP = int(CFG['max_step'])
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
KEY = ['date', 'ticker', 'step']
EPS = 1e-10
S_TERMS = {k: list(v) for k, v in CFG['leg']['s_terms'].items()}
FAR = np.datetime64('2100-01-01', 'ns')
PLANT = {'layer': None, 'mode': None}   # подлог детектора: слой leg | boost | calib, режим lookahead | noemb


class NotApplicable(Exception):
    """Сочетание ключей вне модуля R (дизайн отклоняется до счёта)."""


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def _plant(layer):
    return PLANT['mode'] if PLANT['layer'] == layer else None


# ================================================================ применимость
def _is_default(design, dotted):
    L = C.load_schema().leaves[tuple(dotted.split('.'))]
    return C._norm(C.get(design, dotted)) == L.ndefault


def check(design):
    """-> spec модуля для полного дизайна; NotApplicable, если сочетание не для R."""
    d = C.validate(design)
    leg, b, cal, data = d['leg'], d['boost'], d['calib'], d['data']
    bad = []
    if leg['type'] not in ('HARJ', 'HARX'):
        bad.append(f'leg.type = {leg["type"]} (R: HARJ, HARX)')
    if leg['type'] == 'HARJ' and leg['regressors'] != 'none':
        bad.append('HARJ с leg.regressors ≠ none')
    if leg['type'] == 'HARX' and leg['regressors'] == 'none':
        bad.append('HARX без leg.regressors (S1–S5)')
    for k in ('leg.robust', 'leg.struct', 'leg.struct_split', 'leg.tau_window', 'leg.tau_extend', 'leg.moe', 'leg.moe_gate',
              'leg.moe_calib', 'leg.moe_hard', 'leg.panel', 'leg.panel_beta_window', 'leg.panel_market', 'leg.regime_count',
              'leg.pooled_kappa', 'data.target', 'data.train_target_estimator', 'data.history_start'):
        if isinstance(C.get(d, k), list) or not _is_default(d, k):
            bad.append(f'{k} ≠ default')
    for k in ('mh.scheme', 'mh.decomp_k', 'mh.iterated_k', 'mh.persist', 'mh.slope', 'mh.reconcile'):
        if not _is_default(d, k):
            bad.append(f'{k} ≠ default (модуль M)')
    for k in ('leg.refit', 'leg.rls_lambda', 'leg.estimator', 'boost.boost_refit', 'calib.update', 'data.recency_halflife',
              'data.window', 'data.winsor', 'boost.role'):
        if isinstance(C.get(d, k), list):
            bad.append(f'{k}: сетка внутри складки в R не поддерживается — отдельные дизайны')
    if bad:
        raise NotApplicable('; '.join(bad))
    lam = float(leg['rls_lambda'])
    if lam != 1 and leg['refit'] != 'yearly':
        bad.append('rls_lambda < 1 только с leg.refit = yearly (обновление и так ежедневное)')
    role = b['role']
    if role not in ('none', 'residual'):
        bad.append(f'boost.role = {role} (R: none, residual; остальное — модуль B)')
    if role == 'none' and b['boost_refit'] != 'yearly':
        bad.append('boost.boost_refit без бустинга (boost.role = none)')
    if role == 'residual':
        if b['lib'] != 'lightgbm':
            bad.append('boost.lib ≠ lightgbm')
        for k in ('boost.resid_source', 'boost.grid', 'boost.early_stop', 'boost.monotone', 'boost.objective', 'boost.clip_sigma',
                  'boost.shrink', 'boost.prob'):
            if not _is_default(d, k):
                bad.append(f'{k} ≠ default')
        for g in d['features']:
            if g != 'G0' and not _group_file(g)[0].exists():
                bad.append(f'нет файла группы {g}')
    cal_default = all(_is_default(d, f'calib.{k}') for k in ('s2_est', 'kappa', 's2_state', 'c_qlike', 'mz'))
    if leg['estimator'] == 'qmle_level' and not (cal_default and cal['update'] == 'step'):
        bad.append('qmle_level: калибровки нет — calib.* должны быть default')
    if leg['estimator'] == 'log_ols_calib' and not cal_default and calib_layer() != 'L':
        bad.append('calib.* ≠ default: нужен слой r2_calib агента L (в базе его нет)')
    if bad:
        raise NotApplicable('; '.join(bad))
    if leg['type'] == 'HARX' and leg['estimator'] == 'log_ols_calib':
        lams, als = leg['enet']['lambda'], leg['enet']['alpha']
        lams, als = (lams if isinstance(lams, list) else [lams]), (als if isinstance(als, list) else [als])
        many = len(lams) * len(als) > 1
        variants = [((f'l{float(l):g}_a{float(a):g}' if many else 'base'), float(l), float(a)) for l in lams for a in als]
    else:
        variants = [('base', None, None)]
    return dict(h=int(d['h']), type=leg['type'], regs=leg['regressors'], refit=leg['refit'], lam=lam, est=leg['estimator'],
                role=role, boost_refit=b['boost_refit'], features=list(d['features']), update=cal['update'], cal_default=cal_default,
                variants=variants, data=dict(recency=str(data['recency_halflife']), window=data['window'], winsor=data['winsor']))


# ================================================================ данные
def _group_file(g):
    F = yaml.safe_load(F_YAML.read_text())['groups'][g]
    return C.MAC_ROOT / F['file'], F.get('columns', [])


class Ctx:
    """Матрица d0 одного h (возможно, с шумом/подлогом детектора), упорядоченная по (date, ticker)."""

    def __init__(self, A, harj, h):
        A = A.sort_values(['date', 'ticker'], kind='mergesort').reset_index(drop=True)
        self.A, self.h, self.harj = A, int(h), list(harj)
        self.date = pd.to_datetime(A['date']).values.astype('datetime64[ns]')
        self.ticker = A['ticker'].astype(str).values
        self.year = A['year'].values.astype(int)
        te = pd.to_datetime(A[f'_tend{h}']).values.astype('datetime64[ns]')
        self.tend = np.where(np.isnat(te), FAR, te)
        self.ylvl = A[f'rv_target_h{h}'].values.astype('float64')
        self.has_y = np.isfinite(self.ylvl)
        self.yln = np.log(np.where(self.has_y, self.ylvl, 0.0) + EPS)
        self.udates = np.unique(self.date)
        self.di = np.searchsorted(self.udates, self.date)
        self.H = GH.clean(np.log(A[self.harj].values.astype('float64').clip(1e-10)))
        self._Z, self._F, self._S = {}, {}, None

    def dnum(self, c):
        """Номер первого торгового дня ≥ c."""
        return int(np.searchsorted(self.udates, np.datetime64(pd.Timestamp(c), 'ns')))

    def Z(self, regs):
        if regs not in self._Z:
            self._Z[regs] = HX.build_terms(self.A, S_TERMS[regs]).values.astype('float64')
        return self._Z[regs]

    def F(self, groups):
        key = tuple(groups)
        if key not in self._F:
            cfg = json.loads(C.FEAT.with_suffix('.config.json').read_text())
            cols = [c for c in cfg['feature_cols'] if c in self.A.columns and not c.startswith(('rv_target', '_'))
                    and pd.api.types.is_numeric_dtype(self.A[c])]
            X = self.A[cols].astype('float64')
            for g in groups:
                if g == 'G0':
                    continue
                p, gc = _group_file(g)
                G = pd.read_parquet(p)
                G['date'] = pd.to_datetime(G['date'])
                gc = gc or [c for c in G.columns if c not in ('date', 'ticker')]
                m = self.A[['date', 'ticker']].merge(G[['date', 'ticker'] + gc], on=['date', 'ticker'], how='left')
                X = pd.concat([X, m[gc].astype('float64').set_index(X.index)], axis=1)
            self._F[key] = (np.ascontiguousarray(X.values), list(X.columns))
        return self._F[key]

    def S(self):
        """Состояние для слоя калибровки L (как r2_nogi.mat_L черновика L + state_frame)."""
        if self._S is None:
            A = self.A
            rvp = A['rv'].where(A['rv'] > 0).groupby(A['ticker']).ffill().fillna(A['rv']).clip(lower=1e-10)
            vov = np.log(rvp).groupby(A['ticker']).transform(lambda s: s.rolling(22, min_periods=11).std())
            dd = A.groupby('date')['close_RVI'].first().sort_index().astype(float).ffill()
            past = dd.shift(1).rolling(252, min_periods=126)
            I2 = (dd > past.median()).astype(float).where(past.count().ge(126), 0.0)
            self._S = pd.DataFrame({'date': pd.to_datetime(A['date']).values, 'ticker': self.ticker,
                                    'lnrvi2': HX.t_rvi(A).values, 'vov': vov.values,
                                    'I2': I2.reindex(pd.to_datetime(A['date'])).values, 'tend': self.tend})
        return self._S


# ================================================================ линейное ядро: взвешенный МНК / эл. сеть (Фриш–Во) / QMLE
def _prep_z(Z, W):
    """Пропуски — взвешенным средним обучения (как harx: fillna(mean обучения)), затем взвешенные μ, σ заполненного (ddof 0)."""
    nn = ~np.isnan(Z)
    ws = (W[:, None] * nn).sum(0)
    fill = np.where(ws > 0, (W[:, None] * np.where(nn, Z, 0.0)).sum(0) / np.where(ws > 0, ws, 1.0), 0.0)
    Zf = np.where(nn, Z, fill)
    mu = W @ Zf
    sd = np.sqrt(W @ (Zf - mu) ** 2)
    keep = sd > 1e-12
    return fill, mu, np.where(keep, sd, 1.0), keep


def _zs(Z, zp):
    fill, mu, sd, keep = zp
    return ((np.where(np.isnan(Z), fill, Z) - mu) / sd)[:, keep]


def enet_cd(G, b, lam, l1, tol=1e-12, maxit=50000):
    """min ½g'Gg − b'g + lam·l1·|g|₁ + ½·lam·(1−l1)·|g|² (= ElasticNet sklearn на остатках Фриша–Во, G = R'R/n)."""
    p = len(b)
    if p == 0:
        return np.zeros(0)
    if lam * l1 == 0.0:
        return np.linalg.lstsq(G + lam * (1 - l1) * np.eye(p), b, rcond=None)[0]
    g = np.zeros(p)
    den = np.diag(G) + lam * (1 - l1)
    thr = lam * l1
    for _ in range(maxit):
        mx = 0.0
        for j in range(p):
            r = b[j] - G[j] @ g + G[j, j] * g[j]
            nj = np.sign(r) * max(abs(r) - thr, 0.0) / den[j] if den[j] > 0 else 0.0
            mx = max(mx, abs(nj - g[j]))
            g[j] = nj
        if mx < tol:
            break
    return g


def _newton(D1, yl, W, beta, loss):
    """QMLE в уровнях, Ньютон с дроблением шага. gamma: Σ W(y e^{−η} + η); poisson: Σ W(e^η − y η)."""
    q = CFG['estimator']['qmle_level']['newton']

    def f(b):
        eta = D1 @ b
        return float(W @ (yl * np.exp(-eta) + eta)) if loss == 'gamma' else float(W @ (np.exp(eta) - yl * eta))
    f0, it = f(beta), 0
    for it in range(1, int(q['max_iter']) + 1):
        eta = D1 @ beta
        if loss == 'gamma':
            u = yl * np.exp(-eta)
            grad = D1.T @ (W * (1.0 - u))
        else:
            u = np.exp(eta)
            grad = D1.T @ (W * (u - yl))
        Hs = D1.T @ (D1 * (W * u)[:, None])
        step = np.linalg.lstsq(Hs, grad, rcond=None)[0]
        t = 1.0
        for _ in range(int(q['halving'])):
            nb = beta - t * step
            f1 = f(nb)
            if np.isfinite(f1) and f1 <= f0 + 1e-15 * abs(f0):
                break
            t /= 2
        else:
            break
        beta, f0 = nb, f1
        if np.max(np.abs(t * step)) < float(q['tol_step']):
            break
    return beta, it


class Lin:
    """Модель ноги: ln ŷ = a + [H, Zs]·beta; Zs — стандартизация слагаемых по обучению этой точки."""

    def __init__(self, a, beta, zp, ph, it=0, n=0):
        self.a, self.beta, self.zp, self.ph, self.it, self.n = float(a), beta, zp, ph, it, n

    def predict(self, H, Z):
        D = H if self.zp is None else np.hstack([H, _zs(Z, self.zp)])
        return self.a + D @ self.beta

    def coef(self, harj, terms):
        """Коэффициенты в исходных единицах: const, harj, слагаемые (g/σ; выкинутые — 0)."""
        bH = self.beta[:self.ph]
        out = {}
        const = self.a
        if self.zp is not None:
            fill, mu, sd, keep = self.zp
            g = np.zeros(len(keep))
            g[keep] = self.beta[self.ph:] / sd[keep]
            const -= float(g @ mu)
            out.update({f'g_{t}': float(v) for t, v in zip(terms, g)})
        out = {'const': const, **{f'b_{c}': float(v) for c, v in zip(harj, bH)}, **out}
        return out


def fit_lin(H, Z, y, w, variants, est, ylvl=None, prev=None, loss=None):
    """-> {вариант: Lin}. log_ols_calib: МНК (HARJ) / эл. сеть со штрафом только на слагаемые; qmle_level: гамма-QMLE без штрафа."""
    W = w / w.sum()
    zp = _prep_z(Z, W) if Z is not None else None
    D = H if zp is None else np.hstack([H, _zs(Z, zp)])
    ph = H.shape[1]
    mD, my = W @ D, float(W @ y)
    Dc, yc = D - mD, y - my
    G = Dc.T @ (Dc * W[:, None])
    b = Dc.T @ (W * yc)
    if est == 'qmle_level':
        sdD = np.sqrt(np.diag(G))
        kp = sdD > 1e-12
        Ds = np.column_stack([np.ones(len(D)), (Dc[:, kp] / sdD[kp])])
        if prev is not None and prev.zp is not None and zp is not None and not np.array_equal(prev.zp[3], zp[3]):
            prev = None
        if prev is not None:   # тёплый старт: прошлые коэффициенты в координатах этой точки
            bp = prev.beta if prev.zp is None else np.concatenate([prev.beta[:ph], prev.beta[ph:] * zp[2][zp[3]] / prev.zp[2][prev.zp[3]]])
            ap = prev.predict(H[:1], None if Z is None else Z[:1])[0] - (D[:1] @ bp)[0]
            beta0 = np.concatenate([[ap + mD @ bp], (bp * sdD)[kp]])
        else:
            bl = np.zeros(D.shape[1])
            bl[kp] = np.linalg.lstsq(G[np.ix_(kp, kp)], b[kp], rcond=None)[0]
            beta0 = np.concatenate([[my], (bl * sdD)[kp]])
            eta = Ds @ beta0
            beta0[0] += np.log(W @ (ylvl * np.exp(-eta))) if (loss or 'gamma') == 'gamma' else np.log((W @ ylvl) / (W @ np.exp(eta)))
        bs, it = _newton(Ds, ylvl, W, beta0, loss or CFG['estimator']['qmle_level']['loss'])
        beta = np.zeros(D.shape[1])
        beta[kp] = bs[1:] / sdD[kp]
        return {'base': Lin(bs[0] - mD @ beta, beta, zp, ph, it=it, n=len(y))}
    out = {}
    Ghh, bh = G[:ph, :ph], b[:ph]
    if zp is None or D.shape[1] == ph:
        bH = np.linalg.lstsq(Ghh, bh, rcond=None)[0]
        for tag, _, _ in variants:
            beta = np.concatenate([bH, np.zeros(D.shape[1] - ph)])
            out[tag] = Lin(my - mD @ beta, beta, zp, ph, n=len(y))
        return out
    Ghz, Gzz, bz = G[:ph, ph:], G[ph:, ph:], b[ph:]
    Pz = np.linalg.lstsq(Ghh, np.column_stack([Ghz, bh]), rcond=None)[0]
    Gr = Gzz - Ghz.T @ Pz[:, :-1]
    br = bz - Ghz.T @ Pz[:, -1]
    for tag, lam, l1 in variants:
        g = enet_cd(Gr, br, lam, l1) if lam is not None else np.linalg.lstsq(Gr, br, rcond=None)[0]
        bH = Pz[:, -1] - Pz[:, :-1] @ g
        beta = np.concatenate([bH, g])
        out[tag] = Lin(my - mD @ beta, beta, zp, ph, n=len(y))
    return out


# ================================================================ точки переоценки и строки обучения
def _pts(vy, names):
    return [pd.Timestamp(f'{vy}-{md}') for md in names]


def leg_cuts(spec, vy, vdates):
    if spec['lam'] != 1:
        return [pd.Timestamp(x) for x in np.unique(vdates)]
    return _pts(vy, CFG['refit'][spec['refit']])


def boost_cuts(spec, vy):
    m = spec['boost_refit']
    return _pts(vy, CFG['refit']['quarterly' if m == 'quarterly_warm' else m])


def _assign(dates, cuts):
    return np.searchsorted(np.array([np.datetime64(c, 'ns') for c in cuts]), dates, side='right') - 1


def train_rows(ctx, c, c_next, data, layer):
    """Строки обучения в точке c: дата < c, цель есть, конец окна цели < c. Подлог детектора: lookahead — дата < следующей точки без
    эмбарго; noemb — дата < c без условия на конец цели."""
    c64 = np.datetime64(pd.Timestamp(c), 'ns')
    mode = _plant(layer)
    if mode == 'lookahead':
        m = ctx.has_y & (ctx.date < np.datetime64(pd.Timestamp(c_next), 'ns'))
    elif mode == 'noemb':
        m = ctx.has_y & (ctx.date < c64)
    else:
        m = ctx.has_y & (ctx.date < c64) & (ctx.tend < c64)
    if data['window'] == 'rolling3y':
        m &= ctx.date >= np.datetime64(pd.Timestamp(c) - pd.DateOffset(years=3), 'ns')
    return np.where(m)[0]


def weights(ctx, rows, c, lam, data):
    w = np.ones(len(rows))
    if lam != 1:
        w *= lam ** (ctx.dnum(c) - ctx.di[rows]).astype('float64')
    if data['recency'] not in ('inf', 'None'):
        hl = float(data['recency'].rstrip('y'))
        age = (np.datetime64(pd.Timestamp(c), 'ns') - ctx.date[rows]) / np.timedelta64(1, 'D') / 365.25
        w *= 0.5 ** (age / hl)
    return w


def _y_train(ctx, rows, data):
    y = ctx.yln[rows]
    if data['winsor'] == 'p1_99_by_ticker':
        s = pd.Series(y)
        tk = ctx.ticker[rows]
        y = s.clip(s.groupby(tk).transform(lambda v: v.quantile(0.01)), s.groupby(tk).transform(lambda v: v.quantile(0.99))).values
    return y


# ================================================================ нога одного шага
class LegStep:
    def __init__(self, ctx, spec, k):
        self.ctx, self.spec, self.k = ctx, spec, k
        self.vy = 2015 + k
        self.vidx = np.where(C.val_mask(ctx.A, ctx.h, self.vy))[0]
        self.Z = ctx.Z(spec['regs']) if spec['type'] == 'HARX' else None
        self.cuts = leg_cuts(spec, self.vy, ctx.date[self.vidx])
        self.models = {}
        self.gaps, self.ntrain = [], {}
        self.pred = {t: np.full(len(self.vidx), np.nan) for t, _, _ in spec['variants']}
        asg = _assign(ctx.date[self.vidx], self.cuts)
        prev = None
        for i, c in enumerate(self.cuts):
            sel = np.where(asg == i)[0]
            if not len(sel):
                continue
            nxt = self.cuts[i + 1] if i + 1 < len(self.cuts) else pd.Timestamp(f'{self.vy + 1}-01-01')
            ms = self.fit_at(c, nxt, prev)
            prev = ms.get('base')
            rows_v = self.vidx[sel]
            for t, m in ms.items():
                self.pred[t][sel] = m.predict(ctx.H[rows_v], None if self.Z is None else self.Z[rows_v])
            self.gaps.append((ctx.date[rows_v].min() - self._last_tend) / np.timedelta64(1, 'D'))

    def fit_at(self, c, nxt, prev=None):
        ctx, spec = self.ctx, self.spec
        rows = train_rows(ctx, c, nxt, spec['data'], 'leg')
        if _plant('leg') is None:   # охрана: цели обучения известны строго до c
            assert ctx.tend[rows].max() < np.datetime64(pd.Timestamp(c), 'ns') and ctx.date[rows].max() < np.datetime64(pd.Timestamp(c), 'ns'), c
        self._last_tend = ctx.tend[rows].max()
        w = weights(ctx, rows, c, spec['lam'], spec['data'])
        ms = fit_lin(ctx.H[rows], None if self.Z is None else self.Z[rows], _y_train(ctx, rows, spec['data']), w, spec['variants'],
                     spec['est'], ylvl=np.maximum(ctx.ylvl[rows], 0.0), prev=prev)
        self.models[pd.Timestamp(c)] = ms
        self.ntrain[pd.Timestamp(c)] = (len(rows), float(w.sum() ** 2 / (w ** 2).sum()))
        return ms

    def in_effect(self, c):
        """Модель ноги, действующая в точке c (последняя точка переоценки ≤ c); нет — подгонка в c."""
        ks = [x for x in self.models if x <= pd.Timestamp(c)]
        if ks:
            return self.models[max(ks)]
        return self.fit_at(c, pd.Timestamp(c) + pd.Timedelta(days=1))

    def params(self, tag_names):
        ctx, spec = self.ctx, self.spec
        terms = S_TERMS[spec['regs']] if self.Z is not None else []
        keys = sorted(self.models)
        snap = keys if len(keys) <= 12 else [x for i, x in enumerate(keys) if i == 0 or x.month != keys[i - 1].month]
        rows = []
        add = lambda p, v: rows.append((self.k, f'R.{p}', float(v)))
        add('leg.n_refit', len(keys))
        add('guard.min_gap_days', min(self.gaps) if self.gaps else np.nan)
        for t in tag_names:
            ser = {x: self.models[x][t].coef(ctx.harj, terms) for x in keys if t in self.models[x]}
            for x in snap:
                if x in ser:
                    for nm, v in ser[x].items():
                        add(f'leg.{t}.{nm}@{x:%Y-%m-%d}', v)
            if len(keys) > 12 and ser:
                T = pd.DataFrame([ser[x] for x in keys if x in ser])
                for nm in T.columns:
                    v = T[nm].values
                    rel = np.abs(np.diff(v)) / np.maximum(np.abs(v[:-1]), 1e-12) if len(v) > 1 else np.array([0.0])
                    add(f'leg.{t}.{nm}.mean', v.mean())
                    add(f'leg.{t}.{nm}.sd', v.std())
                    add(f'leg.{t}.{nm}.max_rel_jump', rel.max())
            if spec['est'] == 'qmle_level':
                add(f'leg.{t}.qmle_iter_max', max(self.models[x][t].it for x in keys))
        for x in snap:
            add(f'leg.n_train@{x:%Y-%m-%d}', self.ntrain[x][0])
            if spec['lam'] != 1:
                add(f'leg.n_eff@{x:%Y-%m-%d}', self.ntrain[x][1])
        return rows


# ================================================================ остаточный ученик LightGBM с переоценкой
def boost_step(ls, seed):
    """-> ({вариант: f на строках val}, params). Цель — ln y − ln ноги, действующей в точке c; без ранней остановки."""
    import lightgbm as lgb
    ctx, spec = ls.ctx, ls.spec
    B = CFG['boost']
    mode = spec['boost_refit']
    cuts = boost_cuts(spec, ls.vy)
    asg = _assign(ctx.date[ls.vidx], cuts)
    X, _ = ctx.F(spec['features'])
    prm = dict(B['params'], seed=int(seed), bagging_seed=int(seed), feature_fraction_seed=int(seed), data_random_seed=int(seed))
    dprm = {'max_bin': int(prm['max_bin']), 'verbosity': -1, 'feature_pre_filter': False}
    out = {t: np.zeros(len(ls.vidx)) for t, _, _ in spec['variants']}
    rows_p = []
    for t, _, _ in spec['variants']:
        bst = None
        for i, c in enumerate(cuts):
            nxt = cuts[i + 1] if i + 1 < len(cuts) else pd.Timestamp(f'{ls.vy + 1}-01-01')
            rows = train_rows(ctx, c, nxt, spec['data'], 'boost')
            if _plant('boost') is None:
                assert ctx.tend[rows].max() < np.datetime64(c, 'ns'), c
            legm = ls.in_effect(c)[t]
            r = ctx.yln[rows] - legm.predict(ctx.H[rows], None if ls.Z is None else ls.Z[rows])
            w = weights(ctx, rows, c, 1.0, spec['data'])
            ds = lgb.Dataset(X[rows], r, weight=None if np.all(w == 1) else w, params=dprm, free_raw_data=False)
            if mode == 'quarterly_warm' and bst is not None:
                bst = lgb.train(prm, ds, num_boost_round=int(B['warm_add']), init_model=bst, keep_training_booster=True)
            else:
                bst = lgb.train(prm, ds, num_boost_round=int(B['n_trees']), keep_training_booster=True)
            sel = np.where(asg == i)[0]
            if len(sel):
                out[t][sel] = bst.predict(X[ls.vidx[sel]])
            rows_p += [(ls.k, f'R.boost.{t}.n_trees@{c:%Y-%m-%d}', float(bst.current_iteration())),
                       (ls.k, f'R.boost.{t}.n_train@{c:%Y-%m-%d}', float(len(rows))),
                       (ls.k, f'R.boost.{t}.resid_mean@{c:%Y-%m-%d}', float(r.mean()))]
    return out, rows_p


# ================================================================ калибровка (слой L; свой — только для calib.* = default)
_LAYER = {}


def calib_layer():
    """'L' — r2_calib.calibrate агента L работает; 'R' — свой слой для calib.* = default."""
    if 'v' not in _LAYER:
        try:
            import r2_calib as LC
            d = pd.DataFrame({'date': pd.to_datetime(['2017-01-03'] * 40), 'ticker': [f't{i}' for i in range(40)], 'year': 2017, 'step': 2,
                              'rv_actual': 1e-4, 'rv_pred': 1e-4})
            s = d[['date', 'ticker']].assign(lnrvi2=0.0, vov=0.0, I2=0.0, tend=pd.Timestamp('2017-01-04'))
            LC.calibrate({'h': 1}, d, s)
            _LAYER['v'], _LAYER['mod'] = 'L', LC
        except Exception as e:  # noqa: BLE001 — слоя L нет в базе или API другой: свой слой только для default
            _LAYER['v'], _LAYER['err'] = 'R', f'{type(e).__name__}: {e}'
    return _LAYER['v']


def calib_fallback(V, ctx, update):
    """calib.* = default (sm, κ = 0,5, const, off, off) по определению L: step — окно шагов t−3..t−1 (шаг 1 — без поправки);
    daily — строки с концом цели < d за 3 года до d; ≥ 30 строк; s² = Var(ln(y + 1e-10) − ln ŷ); ŷ·exp(κ·s²)."""
    d = V[COLS].merge(pd.DataFrame({'date': ctx.date, 'ticker': ctx.ticker, 'tend': ctx.tend}), on=['date', 'ticker'], how='left')
    d = d.sort_values(['date', 'ticker']).reset_index(drop=True)
    ok = (d.rv_actual > 1e-12).values
    e = np.log(np.where(ok, d.rv_actual.values, 1.0) + EPS) - np.log(d.rv_pred.values)
    p0 = d.rv_pred.values.astype('float64')
    p = p0.copy()
    step, dates = d.step.values, d.date.values.astype('datetime64[ns]')
    tend = np.where(pd.isna(d.tend), FAR, pd.to_datetime(d.tend).values.astype('datetime64[ns]'))
    mode = _plant('calib')
    stab = []
    if update == 'step':
        for t in sorted(np.unique(step)):
            hi = t if mode == 'lookahead' else t - 1
            pool = ok & (step <= hi) & (step >= t - 3)
            s2 = float(np.var(e[pool])) if pool.sum() >= 30 else None
            if s2 is not None:
                p[step == t] = p0[step == t] * np.exp(0.5 * s2)
            stab.append(dict(step=int(t), s2=np.nan if s2 is None else s2, kappa=0.5 if s2 is not None else np.nan))
    else:
        win = np.timedelta64(int(365.25 * 3), 'D')
        for t in sorted(np.unique(step)):
            s2s = []
            for dd in np.unique(dates[step == t]):
                md = dates == dd
                if mode == 'lookahead':
                    res = dates <= dd
                elif mode == 'noemb':
                    res = dates < dd
                else:
                    res = tend < dd
                pool = ok & res & (dates >= dd - win)
                if pool.sum() >= 30:
                    s2 = float(np.var(e[pool]))
                    p[md] = p0[md] * np.exp(0.5 * s2)
                    s2s.append(s2)
            stab.append(dict(step=int(t), s2=float(np.median(s2s)) if s2s else np.nan, kappa=0.5 if s2s else np.nan))
    out = d[COLS].copy()
    out['rv_pred'] = p
    return out, pd.DataFrame(stab)


def calibrate(design, V, ctx, spec):
    if spec['est'] == 'qmle_level':
        return V, None, 'none'
    if calib_layer() == 'L':
        LC = _LAYER['mod']
        old = dict(LC.PLANT) if hasattr(LC, 'PLANT') else None
        try:
            if old is not None and _plant('calib') is not None:
                LC.PLANT['on'] = True
            Vc, stab = LC.calibrate(design, V, ctx.S())
        finally:
            if old is not None:
                LC.PLANT.clear()
                LC.PLANT.update(old)
        return Vc[COLS], stab, 'L'
    if not spec['cal_default']:
        raise NotApplicable('calib.* ≠ default без слоя L')
    Vc, stab = calib_fallback(V, ctx, spec['update'])
    return Vc, stab, 'R'


# ================================================================ один дизайн
def _name(base, tag):
    return base if tag == 'base' else f'{base}__{tag}'


def run_one(design, A, harj, seed, steps, ctx=None):
    """-> ({ряд: DataFrame COLS}, [(step, param, value)]). Шаги 1..max(steps) считаются все (калибровка — по прошлым шагам)."""
    spec = check(design)
    d = C.validate(design)
    ctx = ctx or Ctx(A, harj, spec['h'])
    tags = [t for t, _, _ in spec['variants']]
    raw = {t: [] for t in tags}
    prows = []
    for k in range(1, max(steps) + 1):
        C.assert_dev_step(k)
        if k > MAXSTEP:
            raise AssertionError(f'шаг {k} > {MAXSTEP}')
        ls = LegStep(ctx, spec, k)
        lnp = {t: ls.pred[t].copy() for t in tags}
        prows += ls.params(tags)
        if spec['role'] == 'residual':
            f, pb = boost_step(ls, seed)
            prows += pb
            for t in tags:
                lnp[t] += f[t]
        for t in tags:
            raw[t].append(pd.DataFrame({'date': ctx.date[ls.vidx], 'ticker': ctx.ticker[ls.vidx], 'year': ls.vy, 'step': k,
                                        'rv_actual': ctx.ylvl[ls.vidx], 'rv_pred': np.clip(np.exp(lnp[t]), 1e-10, None)}))
    out = {}
    keep = set(steps)
    for t in tags:
        Vr = pd.concat(raw[t], ignore_index=True)
        assert int(pd.to_datetime(Vr.date).dt.year.max()) <= C.MAX_YEAR
        Vc, stab, layer = calibrate(d, Vr, ctx, spec)
        prows.append((min(steps), f'R.calib.{t}.layer_L', float(layer == 'L')))
        if stab is not None:
            for r in stab.to_dict('records'):
                for nm, v in r.items():
                    if nm != 'step' and isinstance(v, (int, float, np.floating, np.integer, bool, np.bool_)) and not isinstance(v, str):
                        prows.append((int(r['step']), f'R.calib.{t}.{nm}', float(v)))
        srt = lambda V: V[V.step.isin(keep)].sort_values(KEY).reset_index(drop=True)
        out[_name('R', t)] = srt(Vc)
        out[_name('R_raw', t)] = srt(Vr)
    prows = [r for r in prows if r[0] in keep]
    return out, prows


# ================================================================ params.csv (формат D: design_sha, h, step, param, value + seed)
def params_frame(design, h, seed, prows):
    return pd.DataFrame([dict(design_sha=C.design_hash(design)[:16], h=int(h), step=int(s), param=p, value=v, seed=seed) for s, p, v in prows],
                        columns=['design_sha', 'h', 'step', 'param', 'value', 'seed'])


def write_params(design, h, seed, prows, path):
    """Свои строки (h, seed, param R.*) заменяются; запись под flock + os.replace (несколько seed / модулей одного дизайна)."""
    import fcntl
    new = params_frame(design, h, seed, prows)
    path = Path(path)
    with open(str(path) + '.lock', 'a') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        old = pd.read_csv(path) if path.exists() else new.iloc[:0]
        mine = (old.h == int(h)) & (old.seed.astype(str) == str(seed)) & old.param.astype(str).str.startswith('R.')
        tmp = path.with_name(f'.params.{os.getpid()}.tmp')
        pd.concat([old[~mine], new], ignore_index=True).sort_values(['h', 'seed', 'step', 'param']).to_csv(tmp, index=False)
        os.replace(tmp, path)
        fcntl.flock(lk, fcntl.LOCK_UN)


# ================================================================ run_phase ядра
def phase_fn(design, seed, steps):
    """fn для r2_core.run_phase: SHARED['A'], SHARED['harj'] (load_d0 до пула)."""
    S = C.SHARED
    h = int(design['h'])
    out, prows = run_one(design, S['A'], S['harj'], seed, steps)
    write_params(design, h, seed, prows, C.cache_dir(design, root=S.get('R_root')) / 'params.csv')
    return out


def load_designs(p):
    Y = yaml.safe_load(Path(p).read_text())
    return Y['designs'] if isinstance(Y, dict) and 'designs' in Y else (Y if isinstance(Y, list) else [Y])


def run(designs, h, seeds, steps, workers=None, phase='C', root=None, check_gate=True, block_file=None, tickers=None, A=None, harj=None):
    """Боевой счёт (сервер) через r2_core.run_phase: ворота, кэш, запись, сдвиг, DONE. Только применимые дизайны."""
    if A is None:
        A, harj = C.load_d0(tickers)
    C.share(A=A, harj=harj, R_root=root)
    ok = []
    for d in designs:
        full = C.validate({**d, 'h': h})
        try:
            check(full)
            ok.append(full)
        except NotApplicable as e:
            log(f'не применим: {full.get("design")} — {e}')
    return C.run_phase(phase, h, ok, phase_fn, seeds=tuple(seeds), steps=tuple(steps), workers=workers, module=MODULE, root=root,
                       block_file=block_file, check_gate=check_gate)


# ================================================================ детектор «будущее не влияет»
def _dl(a, b, cut, kmax):
    m = a[KEY + ['rv_pred']].merge(b[KEY + ['rv_pred']], on=KEY, suffixes=('', '_p'))
    d = np.abs(np.log(m.rv_pred.values) - np.log(m.rv_pred_p.values))
    p = (m.date <= pd.Timestamp(cut)).values & (m.step.values <= kmax)
    return (float(d[p].max()) if p.any() else 0.0), (float(d[~p].max()) if (~p).any() else float('nan')), int(p.sum())


def _first_td(A, p):
    u = np.sort(pd.to_datetime(A['date']).unique())
    return pd.Timestamp(u[np.searchsorted(u, np.datetime64(pd.Timestamp(p), 'ns'))])


def _layer_points(spec, layer, vy):
    if layer == 'leg':
        return None if spec['lam'] != 1 else _pts(vy, CFG['refit'][spec['refit']])
    if layer == 'boost':
        return boost_cuts(spec, vy)
    return None if spec['update'] == 'daily' else 'step'


def leak_bench(design, A, harj, seed, steps, cuts_mid, refit_points, plant_point):
    """-> (базовые ряды, параметры, строки детектора). Срез: Δ ln на строках ≤ среза шагов с vy ≤ год среза = 0 точно; подлоги ловятся."""
    spec = check(design)
    h = spec['h']
    base, prows = run_one(design, A, harj, seed, steps)
    tol = float(CFG['guards']['leak']['tol'])
    rows = []
    cuts = [(pd.Timestamp(c), 'seredina') for c in cuts_mid] + [(_first_td(A, p), f'pereocenka {p}') for p in refit_points]
    for i, (cut, kind) in enumerate(cuts):
        kmax = cut.year - 2015
        st = [s for s in steps if s <= kmax]
        if not st:
            continue
        o, _ = run_one(design, C.perturb_frame(A, cut, 20261026 + i), harj, seed, st)
        raw_ok = {}
        for nm in sorted(base, key=lambda x: (not x.startswith('R_raw'), x)):
            dl, da, n = _dl(base[nm], o[nm], cut, kmax)
            ok, why = bool(dl <= tol and n > 0), ''
            if nm.startswith('R_raw'):
                raw_ok[nm.replace('R_raw', 'R', 1)] = ok
            elif not ok and cut.month == 1 and spec['update'] == 'step' and raw_ok.get(nm):
                ok, why = True, 'calib_step_yanvar: калибровка step (Б0.1) берёт строки шага t−1 с концом цели в январе vy (нога и бустинг: Δ = 0)'
            rows.append(dict(test=f'srez {cut.date()} ({kind})', ryad=nm, kmax=kmax, strok_do=n, max_dln_do=dl, max_dln_posle=da, ok=ok,
                             ogovorka=why))
    # подлог look1: входы дня d := d+1 (после шума) — детектор обязан видеть
    c0 = pd.Timestamp(cuts_mid[0])
    st0 = [s for s in steps if s <= c0.year - 2015]
    o0, _ = run_one(design, C.plant_frame(A), harj, seed, st0)
    o1, _ = run_one(design, C.plant_frame(C.perturb_frame(A, c0, 20261026)), harj, seed, st0)
    for nm in ('R', 'R_raw'):
        nm = nm if nm in o0 else sorted(x for x in o0 if x.startswith(nm + '__'))[0]
        dl, _, n = _dl(o0[nm], o1[nm], c0, c0.year - 2015)
        rows.append(dict(test=f'podlog look1 {c0.date()}', ryad=nm, strok_do=n, max_dln_do=dl, ok=bool(dl > 0)))
    # подлоги слоёв: lookahead (ловится всегда), noemb (ловится при h ≥ 5)
    pp = pd.Timestamp(plant_point)
    layers = ['leg'] + (['boost'] if spec['role'] == 'residual' else []) + (['calib'] if spec['est'] == 'log_ols_calib' else [])
    for layer in layers:
        pts = _layer_points(spec, layer, pp.year)
        for mode in ('lookahead', 'noemb'):
            if mode == 'noemb' and pts == 'step':
                rows.append(dict(test=f'podlog {layer}/{mode}', ryad='-', ok=True, ogovorka='calib step: условия на конец цели нет (Б0.1)'))
                continue
            if mode == 'noemb' and pts is not None:
                cut = _first_td(A, max(x for x in pts if x <= pp))
            else:
                cut = _first_td(A, pp)
            stp = [s for s in steps if s <= cut.year - 2015]
            if not stp:
                continue
            PLANT.update(layer=layer, mode=mode)
            try:
                a0, _ = run_one(design, A, harj, seed, stp)
                a1, _ = run_one(design, C.perturb_frame(A, cut, 20261099), harj, seed, stp)
            finally:
                PLANT.update(layer=None, mode=None)
            nm = 'R' if layer == 'calib' else 'R_raw'
            nm = nm if nm in a0 else sorted(x for x in a0 if x.startswith(nm + '__'))[0]
            dl, _, n = _dl(a0[nm], a1[nm], cut, cut.year - 2015)
            need = not (mode == 'noemb' and h == 1)
            rows.append(dict(test=f'podlog {layer}/{mode} {cut.date()}', ryad=nm, strok_do=n, max_dln_do=dl,
                             ok=bool(dl > 0) if need else True,
                             ogovorka='' if need else f'h=1: цель строки d−1 известна в день d — не утечка (Δ = {dl:.2e})'))
    return base, prows, rows


# ================================================================ selftest
def selftest():
    rng = np.random.default_rng(20261026)
    n, ph, pz = 3000, 4, 5
    H = rng.normal(size=(n, ph))
    Z = rng.normal(size=(n, pz)) + 0.5 * H[:, :1]
    Z[rng.random((n, pz)) < 0.03] = np.nan
    y = 0.3 + H @ rng.normal(size=ph) + np.nan_to_num(Z) @ np.array([0.4, 0, -0.2, 0.1, 0]) + rng.normal(0, 0.5, n)
    # 1) эл. сеть = harx.fit_en (без весов), все точки схемы
    tr = np.arange(n) < 2500
    Ztr = pd.DataFrame(Z[tr])
    Zf_tr = Ztr.fillna(Ztr.mean()).values
    Zf_all = pd.DataFrame(Z).fillna(Ztr.mean()).values
    for lam in (0.01, 0.1, 1.0):
        for al in (0.0, 0.5, 1.0):
            ref = HX.fit_en(H[tr], Zf_tr, y[tr], lam, al)[0](H, Zf_all)
            mine = fit_lin(H[tr], Z[tr], y[tr], np.ones(tr.sum()), [('v', lam, al)], 'log_ols_calib')['v'].predict(H, Z)
            assert np.abs(ref - mine).max() < 1e-5, (lam, al, np.abs(ref - mine).max())
    # 2) МНК с весами = lstsq на √w
    w = rng.uniform(0.1, 2, n)
    m = fit_lin(H, None, y, w, [('base', None, None)], 'log_ols_calib')['base']
    Hc = np.column_stack([np.ones(n), H])
    b = np.linalg.lstsq(Hc * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    assert np.abs(m.predict(H, None) - Hc @ b).max() < 1e-9
    # 3) RLS с забыванием: рекурсия по дням (диффузный старт) = взвешенный МНК λ^возраст
    lam, days = 0.99, 300
    dd = np.repeat(np.arange(days), 10)
    X = np.column_stack([np.ones(len(dd)), rng.normal(size=(len(dd), 3))])
    beta_t = np.cumsum(rng.normal(0, 0.02, (days, 4)), 0)
    yy = np.einsum('ij,ij->i', X, beta_t[dd]) + rng.normal(0, 0.3, len(dd))
    th, P = np.zeros(4), np.eye(4) * 1e4   # диффузный старт: априорная точность 1e-4·λ^300 ≪ данных
    for t in range(days):
        P = P / lam
        Xi, yi = X[dd == t], yy[dd == t]
        K = P @ Xi.T @ np.linalg.inv(np.eye(len(yi)) + Xi @ P @ Xi.T)
        th = th + K @ (yi - Xi @ th)
        P = P - K @ Xi @ P
        P = (P + P.T) / 2
    wls = fit_lin(X[:, 1:], None, yy, lam ** (days - dd).astype(float), [('base', None, None)], 'log_ols_calib')['base']
    assert abs(wls.a - th[0]) < 1e-6 and np.abs(wls.beta - th[1:]).max() < 1e-6, (wls.a, th)
    # 4) QMLE в уровнях: гамма и пуассон состоятельны для E[y|x] = exp(x'β); ln-МНК смещает константу на E ln ε
    Xq = rng.normal(size=(20000, 3))
    bq = np.array([-8.0, 0.5, -0.3, 0.2])
    mu = np.exp(bq[0] + Xq @ bq[1:])
    yq = mu * rng.gamma(2.0, 0.5, len(mu))
    for loss in ('gamma', 'poisson'):
        mq = fit_lin(Xq, None, np.log(yq), np.ones(len(yq)), [('base', None, None)], 'qmle_level', ylvl=yq, loss=loss)['base']
        assert abs(mq.a - bq[0]) < 0.03 and np.abs(mq.beta - bq[1:]).max() < 0.03, (loss, mq.a, mq.beta)
    ml = fit_lin(Xq, None, np.log(yq), np.ones(len(yq)), [('base', None, None)], 'log_ols_calib')['base']
    from scipy.special import digamma
    assert abs((ml.a - bq[0]) - (digamma(2.0) - np.log(2.0))) < 0.03
    # тёплый старт QMLE даёт тот же оптимум
    m1 = fit_lin(Xq, None, np.log(yq), np.ones(len(yq)), [('base', None, None)], 'qmle_level', ylvl=yq)['base']
    m2 = fit_lin(Xq, None, np.log(yq), np.ones(len(yq)), [('base', None, None)], 'qmle_level', ylvl=yq, prev=m1)['base']
    assert abs(m1.a - m2.a) < 1e-9 and np.abs(m1.beta - m2.beta).max() < 1e-9 and m2.it <= 2
    # 5) расписание
    assert len(_pts(2018, CFG['refit']['monthly'])) == 12 and len(_pts(2018, CFG['refit']['quarterly'])) == 4
    cs = _pts(2018, CFG['refit']['quarterly'])
    a = _assign(pd.to_datetime(['2018-01-03', '2018-03-30', '2018-04-02', '2018-12-28']).values.astype('datetime64[ns]'), cs)
    assert list(a) == [0, 0, 1, 3]
    # 6) применимость
    for bad in ({'h': 5, 'leg': {'rls_lambda': 0.99, 'refit': 'monthly'}}, {'h': 5, 'boost': {'boost_refit': 'monthly'}},
                {'h': 5, 'leg': {'estimator': 'qmle_level'}, 'calib': {'update': 'daily'}}, {'h': 5, 'leg': {'type': 'HEAVY'}},
                {'h': 5, 'leg': {'type': 'HARX'}}, {'h': 5, 'boost': {'role': 'feature'}}, {'h': 5, 'mh': {'persist': 'OLS'}},
                {'h': 5, 'boost': {'role': 'residual', 'lib': 'both_avg'}}, {'h': 5, 'leg': {'refit': ['yearly', 'monthly']}}):
        try:
            check(bad)
            raise AssertionError(f'не поймано: {bad}')
        except NotApplicable:
            pass
    s = check({'h': 1, 'leg': {'type': 'HARX', 'regressors': 'S4'}})
    assert len(s['variants']) == 9 and s['variants'][0][0] == 'l0.01_a0'
    assert len(check({'h': 1, 'leg': {'type': 'HARX', 'regressors': 'S4', 'estimator': 'qmle_level'}})['variants']) == 1
    log('selftest OK: эл. сеть = harx.fit_en (9 точек), взвешенный МНК, RLS = рекурсия, QMLE (гамма/пуассон), расписание, применимость;',
        f'слой калибровки: {calib_layer()}' + (f' ({_LAYER.get("err")})' if calib_layer() == 'R' else ''))


# ================================================================ контроль и дым
def control(steps=tuple(range(1, MAXSTEP + 1))):
    """Дизайн по умолчанию: R_raw = нога L E/HARJ_emb_d0_h{h} (17 тикеров), R = HARJ_emb_d0 × V0_F (sm)."""
    A, harj = C.load_d0()
    E = C.REGEN / 'R2' / 'cache' / 'E'
    rows = []
    for h in C.HS:
        out, _ = run_one({'h': h}, A, harj, 42, steps)
        ref = pd.read_parquet(E / f'HARJ_emb_d0_h{h}_sdet_val.parquet')
        ref = ref[ref.step.isin(steps)]
        F = pd.read_csv(E / f'V0_F_h{h}.csv')
        F = F[(F.ryad == 'HARJ_emb_d0') & (F.s2 == 'sm')].set_index('step').F
        m = out['R_raw'].merge(ref, on=KEY, suffixes=('', '_r'))
        dr = float(np.abs(np.log(m.rv_pred / m.rv_pred_r)).max())
        dy = float(np.abs(m.rv_actual / m.rv_actual_r - 1).max())
        mc = out['R'].merge(ref.assign(rv_pred=ref.rv_pred * ref.step.map(F)), on=KEY, suffixes=('', '_r'))
        dc = float(np.abs(np.log(mc.rv_pred / mc.rv_pred_r)).max())
        ok = len(m) == len(ref) == len(out['R_raw']) and dr <= 1e-8 and dc <= 1e-8 and dy < 1e-12
        rows.append(dict(h=h, strok=len(m), max_dln_raw=dr, max_dln_calib=dc, max_rel_y=dy, sloy=calib_layer(), ok=bool(ok)))
        log(f'контроль h={h}: R_raw {dr:.1e}, R {dc:.1e}, цель {dy:.1e}', 'OK' if ok else 'ОШИБКА')
    return pd.DataFrame(rows)


def dym():
    D = CFG['dym']
    tick, steps, seed = tuple(D['tickers']), tuple(D['steps']), int(D['seed'])
    out = C.MAC_ROOT / CFG['out']['dym']
    out.mkdir(parents=True, exist_ok=True)
    bf = out / 'blok_sdvig_dym.csv'
    selftest()
    ctl = control()
    A, harj = C.load_d0(tick)
    for h in D['h']:   # дым-контроль: нога по умолчанию = r2_core._harj_ln на тех же 3 тикерах
        o, _ = run_one({'h': h}, A, harj, seed, steps)
        ref = C._smoke_series(A, harj, h, steps)
        m = o['R_raw'].merge(ref, on=KEY, suffixes=('', '_r'))
        dl = float(np.abs(np.log(m.rv_pred / m.rv_pred_r)).max())
        ctl = pd.concat([ctl, pd.DataFrame([dict(h=h, strok=len(m), max_dln_raw=dl, sloy='dym_core', ok=bool(len(m) == len(ref) and dl <= 1e-8))])])
    ctl.to_csv(out / 'kontrol.csv', index=False)
    summ, leak_rows, sdvig, par = [], [], [], []
    for h in D['h']:
        ref = run_one({'h': h}, A, harj, seed, steps)[0]['R']
        for spec in D['designs']:
            nm = spec['name']
            des = C.validate({'design': nm, 'h': h, 'seed': seed, **{k: v for k, v in spec.items() if k != 'name'}})
            t0 = time.time()
            try:
                check(des)
            except NotApplicable as e:
                summ.append(dict(h=h, design=nm, status='ne_primenim', note=str(e)))
                continue
            base, prows, lr = leak_bench(des, A, harj, seed, steps, D['cuts_mid'], D['refit_points'], CFG['guards']['leak']['plant_point'])
            shok = []
            for rn, V in base.items():
                sc = C.shift_check(V, A, h, f'R:{nm}:{rn}', des, block_file=bf)
                shok.append(sc['ok'])
                sdvig.append(dict(h=h, design=nm, ryad=rn, **{k: v for k, v in sc.items() if k.startswith('corr') or k in ('argmax', 'ok', 'n_strok')}))
                V.to_parquet(out / f'{nm}__{rn}_h{h}_s{seed}_val.parquet', index=False)
            main = 'R' if 'R' in base else sorted(x for x in base if x.startswith('R__'))[0]
            rr =base[main].merge(ref[KEY + ['rv_pred']], on=KEY, suffixes=('', '_b'))
            rr = np.log(rr.rv_pred / rr.rv_pred_b)
            leak_rows += [dict(h=h, design=nm, **r) for r in lr]
            P = params_frame(des, h, seed, prows).assign(design=nm)
            par.append(P)
            summ.append(dict(h=h, design=nm, status='ok', ryadov=len(base), strok=len(base[main]), dln_vs_base_mean=float(rr.mean()),
                             dln_vs_base_sd=float(rr.std()), leak_ok=all(r['ok'] for r in lr), n_leak=len(lr),
                             ogovorok=sum(bool(r.get('ogovorka')) for r in lr), shift_ok=all(shok), n_params=len(P),
                             sec=round(time.time() - t0, 1), sha_design=C.design_hash(des)[:16], slozhnost=C.complexity(des)))
            log(f'h={h} {nm}: рядов {len(base)}, детектор {"OK" if summ[-1]["leak_ok"] else "ОШИБКА"} ({len(lr)}), '
                f'сдвиг {"OK" if summ[-1]["shift_ok"] else "БЛОК"}, {summ[-1]["sec"]} с')
    S = pd.DataFrame(summ)
    S.to_csv(out / 'dym_summary.csv', index=False)
    pd.DataFrame(leak_rows).to_csv(out / 'leak.csv', index=False)
    pd.DataFrame(sdvig).to_csv(out / 'sdvig.csv', index=False)
    pd.concat(par, ignore_index=True).to_csv(out / 'params.csv', index=False)
    # механика конвейера: r2_core.run_phase -> кэш (временный корень), params.csv, сдвиг, DONE
    rp = []
    for h in D['h']:
        T = run([{'design': 'dym_base'}, {'design': 'dym_bqw', 'boost': {'role': 'residual', 'boost_refit': 'quarterly_warm'}},
                 {'design': 'dym_rls', 'leg': {'type': 'HARX', 'regressors': 'S1', 'rls_lambda': 0.995, 'enet': {'lambda': 0.01, 'alpha': 0.5}}}],
                h, (seed,), steps, workers=3, phase='C', root=out / 'cache', check_gate=False, block_file=bf, A=A, harj=harj)
        rp.append(T.assign(h=h))
    RP = pd.concat(rp, ignore_index=True)
    RP.to_csv(out / 'run_phase.csv', index=False)
    pf = sorted((out / 'cache').glob('*/params.csv'))
    ok = S[S.status == 'ok']
    log(f'дым: контуров {len(ok)}, детектор {int(ok.leak_ok.sum())}/{len(ok)}, сдвиг {int(ok.shift_ok.sum())}/{len(ok)}, '
        f'не применимо {int((S.status == "ne_primenim").sum())}, контроль {int(ctl.ok.sum())}/{len(ctl)}; run_phase: рядов {len(RP)}, '
        f'ошибок {int((RP.ryad == "ОШИБКА").sum()) if "ryad" in RP else "?"}, params.csv {len(pf)}')
    return S


def leak_cmd(designs, steps, tickers, seed):
    G = CFG['guards']['leak']
    A, harj = C.load_d0(tickers)
    rows = []
    for d in designs:
        des = C.validate({**d, 'seed': seed})
        _, _, lr = leak_bench(des, A, harj, seed, steps, G['cuts_mid'], G['refit_points'], G['plant_point'])
        rows += [dict(design=des.get('design'), h=int(des['h']), **r) for r in lr]
    R = pd.DataFrame(rows)
    print(R.to_string(index=False))
    assert R.ok.all(), 'детектор: ОШИБКА'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['selftest', 'control', 'dym', 'run', 'leak'])
    ap.add_argument('--designs')
    ap.add_argument('--h', type=int)
    ap.add_argument('--seeds', type=int, nargs='*', default=[42])
    ap.add_argument('--steps', type=int, nargs='*', default=list(range(1, MAXSTEP + 1)))
    ap.add_argument('--tickers', nargs='*', default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--phase', default='C')
    a = ap.parse_args()
    if a.mode == 'selftest':
        selftest()
    elif a.mode == 'control':
        R = control(tuple(a.steps))
        sys.exit(0 if R.ok.all() else 1)
    elif a.mode == 'dym':
        dym()
    elif a.mode == 'run':
        T = run(load_designs(a.designs), a.h, a.seeds, a.steps, a.workers, a.phase)
        print(T.to_string(index=False))
    else:
        leak_cmd(load_designs(a.designs), a.steps, a.tickers, a.seeds[0])


if __name__ == '__main__':
    main()
