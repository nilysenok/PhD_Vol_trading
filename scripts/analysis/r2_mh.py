#!/usr/bin/env python3
"""R2, агент M (протокол v3 docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md, раздел 6, строка «M»; yaml models/designs/r2/r2_M_mh.yaml).
Ядро — scripts/analysis/r2_core.py (агент S): validate / design_hash / cache_dir / write_series, load_d0, perturb_frame / plant_frame,
shift_check, leak_check, run_phase. Код первого раунда не меняется (рецепты — Д18, Д19, Д20 = x6 Д22, Д24.2, Д25.3, Д26.3).

Кластер 6 — схемы над готовым рядом h (нога L или роль B), ключи mh.*:
  scheme    direct | decomp[k] | iterated[k_iter] | joint_h | perlead (h=5)
  slope     наклон срочной структуры: ŷ_h = ŷ_k · exp(ŝ)
  persist   ln ŷ_h += ρ·r̂₁, r̂₁ = ln base_1 − ln leg_1; ρ: OLS | qlike | pos_neg | sqrtRQ
  reconcile MinT: узлы {RV̄₁, RV̄₂₋₅, RV̄₆₋₂₂} -> {RV̄₅, RV̄₂₂}, Σ — Ледуа–Вольф по прошлым OOS
  Порядок: scheme -> slope -> persist -> reconcile. На h=1 persist и reconcile — фиктивные (ряд не меняется).
Кластер 11 — нога τ × g (leg.struct, struct_split, tau_window, tau_extend): tau_midas | tau_rvi | tau_boost; BV + J.
Определения — в yaml (разделы mh, struct), записаны до счёта.

Окна (раздел 1): шаг k — val-год vy = 2015 + k, обучение — годы < vy, эмбарго по концу цели (день t+b тикера < 1 янв. vy).
ρ и Σ MinT на шаге t — по OOS-строкам шагов < t с тем же эмбарго («ρ только на окне шага»); шаг 1 — без поправки.
Ничего за годы ≥ 2022 не читается: матрица — r2_core.load_d0 (строки ≤ 2021), входные ряды — фильтр step ≤ 6 + assert.
test_locked/ не читается. Метрик модуль не считает (выбор — агент O).

python3 scripts/analysis/r2_mh.py selftest
python3 scripts/analysis/r2_mh.py control [--steps 1 2 3]       # HAR[1..h] модуля = нога L HARJ_emb_d0 (17 тикеров), допуск 1e-7
python3 scripts/analysis/r2_mh.py dym                           # 3 тикера × шаги 1–3 × seed 42 × h ∈ {1, 5, 22}: ряды, сдвиг, детектор
python3 scripts/analysis/r2_mh.py run --designs <f.yaml> --h 5 [--seeds 42 43 44] [--steps 1 .. 6] [--inputs <f.yaml>] [--workers N]
                                                                 # боевой счёт (сервер): r2_core.run_phase -> кэш, сдвиг, DONE
python3 scripts/analysis/r2_mh.py leak --designs <f.yaml> [--tickers ...] [--steps ...]"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
import argparse, hashlib, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import r2_core as C  # noqa: E402

YAML = REPO / 'models' / 'designs' / 'r2' / 'r2_M_mh.yaml'
CFG = yaml.safe_load(YAML.read_text())
REG = C.REGEN
MAXSTEP = int(CFG['max_step'])
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
KEY = ['date', 'ticker', 'step']
STEPS = tuple(range(1, MAXSTEP + 1))
FLOOR = float(CFG['struct']['floor'])
EPS = 1e-10
RHO_LO, RHO_HI = CFG['mh']['persist']['rho_bounds']
S_MINT = np.array([[1 / 5, 4 / 5, 0], [1 / 22, 4 / 22, 17 / 22], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float64')
MODULE = 'M'


class NotApplicable(Exception):
    """Сочетание ключей не применимо на этом h (дизайн отклоняется до счёта)."""


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def off(v):
    return v in (False, None, 'off')


# ================================================================ применимость (дизайн — полный, после C.validate)
def is_struct(d):
    return d['leg']['struct'] != 'single' or d['leg']['struct_split'] != 'rv_whole'


def check_mh(d):
    """-> множество нужных входов; NotApplicable — сочетание не применимо."""
    h, m = int(d['h']), d['mh']
    sch, need = m['scheme'], set()
    if h == 1:
        if sch != 'direct' or m['slope'] is True:
            raise NotApplicable(f'h=1: scheme={sch}, slope={m["slope"]} не применимы')
        return need  # persist / reconcile на h=1 — фиктивные
    if sch != 'decomp' and m['decomp_k'] is not None:
        raise NotApplicable('decomp_k без scheme=decomp')
    if sch != 'iterated' and m['iterated_k'] is not None:
        raise NotApplicable('iterated_k без scheme=iterated')
    if sch == 'decomp':
        k = m['decomp_k']
        if k not in CFG['mh']['decomp_k'][h]:
            raise NotApplicable(f'decomp_k={k} на h={h}')
        need.add('base_1' if k == 1 else 'base_5')
        if k == 1 and 'far' in (d.get('inputs') or {}):  # E1 (владелец 20:25): дальняя часть — готовый ряд RV̄₂₋ₕ из inputs
            need.add('far')
    elif sch == 'iterated':
        if m['iterated_k'] not in CFG['mh']['iterated_k'][h]:
            raise NotApplicable(f'iterated_k={m["iterated_k"]} на h={h}')
        need |= {'base_1', 'leg_1'}
    elif sch == 'perlead':
        if h != 5:
            raise NotApplicable('perlead — только h=5')
        need.add('base_1')
    if m['slope'] is True:
        if sch != 'direct':
            raise NotApplicable('slope — только при scheme=direct')
        need.add('base_5' if h == 22 else 'base_1')
    if not off(m['persist']):
        need |= {'base_1', 'leg_1'}
    if m['reconcile'] == 'MinT':
        need |= {'base_1', 'base_22' if h == 5 else 'base_5'}
    return need


def check_struct(d):
    lg, m = d['leg'], d['mh']
    if lg['struct'] == 'single':
        if lg['struct_split'] == 'rv_whole':
            raise NotApplicable('single + rv_whole — нога L (HARJ)')
        if int(lg['tau_window']) != 22 or lg['tau_extend'] != 'const':
            raise NotApplicable('tau_window / tau_extend без τ')
    if m['scheme'] != 'direct' or m['slope'] is True or not off(m['persist']) or not off(m['reconcile']):
        raise NotApplicable('нога кластера 11 и схема кластера 6 — разные вызовы (нога -> роль -> схема)')


# ================================================================ матрица d0 и подмодели шагов
class Ctx:
    """A — кадр r2_core.load_d0 (возможно, после perturb_frame / plant_frame). plant: None | 'look1' (входные ряды дня d+1) | 'noemb'."""

    def __init__(self, A, harj, steps=STEPS, plant=None):
        for k in steps:
            C.assert_dev_step(k)
        A = A.sort_values(['date', 'ticker'], kind='mergesort').reset_index(drop=True)
        assert A['date'].dt.year.max() <= C.MAX_YEAR
        self.A, self.harj, self.steps, self.plant = A, list(harj), tuple(steps), plant
        self.H = np.nan_to_num(np.log(A[self.harj].values.astype('float64').clip(1e-10)), nan=0, posinf=0, neginf=0)
        self.year = A['year'].values
        self.tk = A['ticker'].values.astype(str)
        self.key = pd.DataFrame({'date': A['date'].values, 'ticker': self.tk, 'ai': np.arange(len(A))})
        self.q = np.sqrt((A['n_bars'].astype('float64') * A['rkurt'].astype('float64') / 3).clip(lower=0)).clip(upper=10).fillna(1.0).values
        self._c = {}
        self.used = set()

    def tgt(self, a, b):
        """RV̄_{a..b}: среднее RV за [t+a, t+b] (дни тикера); после конца данных (> 2021) — NaN."""
        k = ('tgt', a, b)
        if k not in self._c:
            g = self.A.groupby('ticker')['rv']
            self._c[k] = (sum(g.shift(-j) for j in range(a, b + 1)) / (b - a + 1)).values.astype('float64')
        return self._c[k]

    def tend(self, b):
        k = ('tend', b)
        if k not in self._c:
            self._c[k] = self.A.groupby('ticker')['date'].shift(-b).dt.year.fillna(9999).values.astype(int)
        return self._c[k]

    def emb_ok(self, b, vy):
        return np.ones(len(self.A), bool) if self.plant == 'noemb' else (self.tend(b) < vy)

    def trmask(self, vy, b, y):
        return (self.year < vy) & np.isfinite(y) & self.emb_ok(b, vy)

    def rows(self, df):
        ai = df[['date', 'ticker']].merge(self.key, on=['date', 'ticker'], how='left')['ai']
        assert ai.notna().all(), f'строк ряда нет в матрице: {int(ai.isna().sum())}'
        return ai.values.astype(int)

    def lagmat(self, s, L):
        """(n, L): уровни дней t−L+1 … t тикера; недостающее начало — ближайшее известное."""
        g = pd.Series(s).groupby(self.tk)
        M = np.column_stack([g.shift(k).values for k in range(L - 1, -1, -1)]).astype('float64')
        for j in range(L - 2, -1, -1):
            M[:, j] = np.where(np.isnan(M[:, j]), M[:, j + 1], M[:, j])
        return M

    def _fit_steps(self, key, X, y, b):
        self.used.add(key)
        if key not in self._c:
            ln, co = {}, {}
            for s in self.steps:
                m = self.trmask(2015 + s, b, y)
                lr = LinearRegression().fit(X[m], y[m])
                ln[s], co[s] = lr.predict(X), np.r_[lr.intercept_, lr.coef_]
            self._c[key] = (ln, co)
        return self._c[key]

    def har_sub(self, a, b):
        """HAR-J d0 на ln RV̄_{a..b} (как r2_nogi.fam_harj, эмбарго по концу цели)."""
        return self._fit_steps(('har', a, b), self.H, np.log(self.tgt(a, b) + EPS), b)

    def slope_ln(self, h):
        kk = 5 if h == 22 else 1
        return self._fit_steps(('slope', h), self.H, np.log(self.tgt(1, h) + EPS) - np.log(self.tgt(1, kk) + EPS), h)

    def joint_ln(self, h):
        """Одна HAR на объединённой выборке h' ∈ {1, 5, 22}: регрессоры HAR-J, они же × ln h', фиктивные h' (Д24.2 в эконометрике)."""
        k = ('joint', h)
        self.used.add(k)
        if k not in self._c:
            n = len(self.H)
            Xh = lambda hh: np.column_stack([self.H, self.H * np.log(hh), np.full(n, float(hh == 5)), np.full(n, float(hh == 22))])
            ln, co = {}, {}
            for s in self.steps:
                Xs, ys = [], []
                for hh in (1, 5, 22):
                    y = np.log(self.tgt(1, hh) + EPS)
                    m = self.trmask(2015 + s, hh, y)
                    Xs.append(Xh(hh)[m]); ys.append(y[m])
                lr = LinearRegression().fit(np.vstack(Xs), np.concatenate(ys))
                ln[s], co[s] = lr.predict(Xh(h)), np.r_[lr.intercept_, lr.coef_]
            self._c[k] = (ln, co)
        return self._c[k]

    def at_steps(self, D, ai, step):
        out = np.full(len(ai), np.nan)
        for s in np.unique(step):
            m = step == s
            out[m] = D[int(s)][ai[m]]
        return out

    def coef_meta(self, meta):
        """Коэффициенты подмоделей шагов, которыми пользовался этот дизайн (детектор эмбарго, диагностика D)."""
        for k in sorted(self.used, key=str):
            for s, c in self._c[k][1].items():
                meta.setdefault(s, {})['_'.join(map(str, k))] = c


_D0 = {}


def load_A(tickers=None):
    k = tuple(tickers) if tickers else None
    if k not in _D0:
        _D0[k] = C.load_d0(tickers)
        C.check_d0(_D0[k][0])
    A, harj = _D0[k]
    return A.copy(), harj


# ================================================================ входные ряды (ноги L, роли B)
def read_input(spec, seed=42, h=None, tickers=None, steps=STEPS):
    """spec — путь (от data/regen, шаблоны {seed}, {h}) или вход дизайна r2_core: {file[, col]} | {design_sha, series}."""
    import pyarrow.parquet as pq
    if isinstance(spec, dict) and spec.get('design_sha'):
        X = C.load_series(spec['design_sha'], ryad=spec.get('series'), h=h, steps=steps)
        sd = X.seed.astype(str)
        X = X[sd == str(seed)] if (sd == str(seed)).any() else X[sd == sd.iloc[0]]
        d, p = X[COLS].copy(), C.CACHE_ROOT / spec['design_sha'][:16]
    else:
        rel, col = (spec['file'], spec.get('col')) if isinstance(spec, dict) else (spec, None)
        p = Path(str(rel).format(seed=seed, h=h))
        p = p if p.is_absolute() else REG / p
        cols = ['date', 'ticker', 'year', 'step', 'rv_actual', col or 'rv_pred']
        d = pq.read_table(p, columns=cols, filters=[('step', '<=', MAXSTEP)]).to_pandas()
        if col:
            d = d.rename(columns={col: 'rv_pred'})
    d['date'] = pd.to_datetime(d['date'])
    d['ticker'] = d['ticker'].astype(str)
    if tickers:
        d = d[d.ticker.isin(list(tickers))]
    assert d['year'].max() <= C.MAX_YEAR and d['step'].max() <= MAXSTEP, f'{p}: строки ≥ 2022'
    return d[d.step.isin(list(steps))].sort_values(KEY).reset_index(drop=True)[COLS], p


UPSTREAM = ('BG', 'B', 'R', 'P', 'L')  # модули цепочки до M (r2_S_driver.yaml chain_order), последний — первым


def chain_bases(design, h, seed, steps, tickers=None, root=None):
    """Ряды h предыдущего модуля цепочки в каталоге этого дизайна (r2_core.load_series): {имя ряда: кадр}. Нет — {}."""
    try:
        X = C.load_series(design, h=h, steps=steps, root=root)
    except FileNotFoundError:
        return {}, None
    if not len(X) or 'rv_actual' not in X:
        return {}, None
    X = X[X.modul.isin(UPSTREAM)]
    if not len(X):
        return {}, None
    last = next(m for m in UPSTREAM if m in set(X.modul))
    X = X[X.modul == last]
    sd = X.seed.astype(str)
    X = X[sd == str(seed)] if (sd == str(seed)).any() else X[sd == 'det']
    X = X.assign(date=pd.to_datetime(X.date), ticker=X.ticker.astype(str))
    if tickers:
        X = X[X.ticker.isin(list(tickers))]
    return {ry: g.sort_values(KEY).reset_index(drop=True)[COLS] for ry, g in X.groupby('ryad')}, last


def default_inputs(h, seed, tickers, steps, override=None, design=None):
    """base — ряд h дизайна; leg_1, base_1, base_5, base_22 — соседние горизонты; ref — строки и rv_actual (нога L HARJ_emb_d0_h{h}).
    Порядок источников: inputs дизайна (base_1, leg_1 — входят в хэш) > override (--inputs) > inputs_default yaml."""
    src = {**CFG['inputs_default'], **(override or {}), **((design or {}).get('inputs') or {})}
    I, files = {}, {}
    I['base'], files['base'] = read_input(src.get('base', src[f'base_{h}']), seed, h, tickers, steps)
    for nm in ('leg_1', 'base_1', 'base_5', 'base_22', 'ref'):
        I[nm], files[nm] = read_input(src[nm], seed, 1 if nm in ('leg_1', 'base_1') else h, tickers, steps)
    for nm, spec in ((design or {}).get('inputs') or {}).items():  # прочие входы дизайна (например, far у E1)
        if nm not in I:
            I[nm], files[nm] = read_input(spec, seed, h, tickers, steps)
    return I, files


def perturb_inputs(I, cut, seed):
    lo, hi = CFG['guards']['leak']['noise']
    out = {}
    for i, (k, d) in enumerate(sorted(I.items())):
        rng = np.random.default_rng(seed + 1000 + i)
        d = d.copy()
        fut = (d['date'] > pd.Timestamp(cut)).values
        d.loc[fut, 'rv_pred'] = d.loc[fut, 'rv_pred'].values * rng.uniform(lo, hi, int(fut.sum()))
        out[k] = d
    return out


def plant_inputs(I):
    """Подлог: входной ряд строки d := значение строки d+1 того же тикера и шага."""
    out = {}
    for k, d in I.items():
        d = d.sort_values(['ticker', 'step', 'date']).copy()
        d['rv_pred'] = d.groupby(['ticker', 'step'])['rv_pred'].shift(-1).fillna(d['rv_pred'])
        out[k] = d.sort_values(KEY).reset_index(drop=True)
    return out


# ================================================================ кластер 6
def _ols0(X, y):
    X = np.asarray(X, dtype='float64').reshape(len(y), -1)
    return np.zeros(X.shape[1]) if len(y) < 30 else np.linalg.lstsq(X, y, rcond=None)[0]


def _q_profile(y, lnf):
    x = y / np.exp(lnf)
    x = x / x.mean()
    return float(np.mean(x - np.log(x) - 1))


def persist(F, ctx, mode, h, meta):
    r = np.log(F['base_1'].values) - np.log(F['leg_1'].values)
    ai, step = F['ai'].values, F['step'].values
    y, q = ctx.tgt(1, h)[ai], ctx.q[ai]
    lnp = np.log(F['p'].values)
    out = lnp.copy()
    for s in sorted(np.unique(step)):
        vy = 2015 + s
        hist = (step < s) & np.isfinite(y) & (y > 1e-12) & np.isfinite(r) & ctx.emb_ok(h, vy)[ai]
        yh, rh, lh, qh = y[hist], r[hist], lnp[hist], q[hist]
        res = np.log(yh) - lh
        if hist.sum() < 30:
            par = np.zeros(2)
        elif mode == 'OLS':
            par = np.r_[_ols0(rh, res), 0.0]
        elif mode == 'qlike':
            par = np.r_[minimize_scalar(lambda c: _q_profile(yh, lh + c * rh), bounds=(RHO_LO, RHO_HI), method='bounded',
                                        options={'xatol': 1e-8}).x, 0.0]
        elif mode == 'pos_neg':
            pp, nn = rh > 0, rh < 0
            par = np.r_[_ols0(rh[pp], res[pp]), _ols0(rh[nn], res[nn])]
        elif mode == 'sqrtRQ':
            par = _ols0(np.column_stack([rh, rh * qh]), res)
        else:
            raise ValueError(mode)
        if mode != 'sqrtRQ':
            par = np.clip(par, RHO_LO, RHO_HI)
        m = step == s
        rho = np.where(r[m] > 0, par[0], par[1]) if mode == 'pos_neg' else (par[0] + par[1] * q[m] if mode == 'sqrtRQ' else par[0])
        out[m] = lnp[m] + rho * np.nan_to_num(r[m])
        meta.setdefault(s, {})[f'persist_{mode}'] = np.r_[par, hist.sum()]
    F['p'] = np.exp(out)


def iterate(hist, ln_f1, delta, c, J):
    """Путь f_1..f_J (уровни): f_1 = exp(ln_f1); ln f_j = c0 + c_d ln W_last + c_w ln mean5(W) + c_m ln mean22(W) + δ (d22_full.iterate)."""
    W = hist.copy()
    F = np.empty((len(W), J))
    F[:, 0] = np.exp(np.clip(ln_f1, np.log(FLOOR), np.log(0.1)))
    W = np.column_stack([W[:, 1:], F[:, 0]])
    for j in range(1, J):
        lf = c[0] + c[1] * np.log(W[:, -1]) + c[2] * np.log(W[:, -5:].mean(1)) + c[3] * np.log(W.mean(1)) + delta
        F[:, j] = np.exp(np.clip(lf, np.log(FLOOR), np.log(0.1)))
        W = np.column_stack([W[:, 1:], F[:, j]])
    return F


def scheme_iterated(F, ctx, h, k, meta):
    if 'dh' not in ctx._c:
        Hd = ctx.lagmat(ctx.A['rv_d'].astype('float64').clip(lower=FLOOR).values, 22)
        ctx._c['dh'] = (Hd, np.column_stack([np.log(Hd[:, -1]), np.log(Hd[:, -5:].mean(1)), np.log(Hd.mean(1))]))
    Hd, X = ctx._c['dh']
    y = np.log(ctx.tgt(1, 1) + EPS)
    kk = min(int(k), h)
    ai, step = F['ai'].values, F['step'].values
    far = np.exp(ctx.at_steps(ctx.har_sub(kk + 1, h)[0], ai, step)) if kk < h else np.zeros(len(F))
    it = np.empty(len(F))
    for s in sorted(np.unique(step)):
        m_tr = ctx.trmask(2015 + s, 1, y)
        c = np.linalg.lstsq(np.column_stack([np.ones(m_tr.sum()), X[m_tr]]), y[m_tr], rcond=None)[0]
        m = step == s
        a_ = ai[m]
        delta = np.log(F['leg_1'].values[m]) - (c[0] + X[a_] @ c[1:])
        P = iterate(Hd[a_], np.log(F['base_1'].values[m]), delta, c, kk)
        it[m] = (P.sum(1) + (h - kk) * far[m]) / h
        meta.setdefault(s, {})['iter_daily'] = c
    F['p'] = 0.5 * (it + F['p'].values)


def reconcile_mint(F, ctx, h, meta):
    ai, step = F['ai'].values, F['step'].values
    sub = lambda a, b: np.exp(ctx.at_steps(ctx.har_sub(a, b)[0], ai, step))
    p5 = F['p'].values if h == 5 else F['base_5'].values
    p22 = F['p'].values if h == 22 else F['base_22'].values
    Yh = np.column_stack([p5, p22, F['base_1'].values, sub(2, 5), sub(6, 22)])
    Ya = np.column_stack([ctx.tgt(1, 5), ctx.tgt(1, 22), ctx.tgt(1, 1), ctx.tgt(2, 5), ctx.tgt(6, 22)])[ai]
    row = 0 if h == 5 else 1
    out = F['p'].values.copy()
    for s in sorted(np.unique(step)):
        hist = (step < s) & np.isfinite(Ya).all(1) & np.isfinite(Yh).all(1) & ctx.emb_ok(22, 2015 + s)[ai]
        if hist.sum() < 50:
            meta.setdefault(s, {})['mint_n'] = np.array([hist.sum()])
            continue
        Sig = LedoitWolf().fit(Ya[hist] / Yh[hist] - 1).covariance_
        Si = np.linalg.inv(Sig)
        m = step == s
        Dinv = 1.0 / Yh[m]
        Winv = Si[None, :, :] * Dinv[:, :, None] * Dinv[:, None, :]
        M = np.einsum('ki,nkl,lj->nij', S_MINT, Winv, S_MINT)
        v = np.einsum('ki,nkl,nl->ni', S_MINT, Winv, Yh[m])
        b = np.linalg.solve(M, v[..., None])[..., 0]
        out[m] = np.clip(b @ S_MINT[row], 1e-10, None)
        G = np.linalg.solve(S_MINT.T @ Si @ S_MINT, S_MINT.T @ Si)  # веса при равных уровнях узлов (D = I): ỹ_h = w·ŷ
        meta.setdefault(s, {}).update(mint_w=S_MINT[row] @ G, mint_sigma=Sig.ravel(), mint_n=np.array([hist.sum()]))
    F['p'] = out


def apply_mh(design, ctx, I):
    """Кластер 6 над рядом I['base'] (h дизайна). -> (ряд COLS, meta {шаг: {имя: массив}})."""
    d = C.validate(design)
    h, m = int(d['h']), d['mh']
    need = check_mh(d)
    ctx.used = set()
    miss = [k for k in need if k not in I]
    if miss:
        raise ValueError(f'нет входов {miss}')
    F = I['base'][COLS].copy().sort_values(KEY).reset_index(drop=True)
    F = F[F.step.isin(ctx.steps)].reset_index(drop=True)
    F['ai'] = ctx.rows(F)
    for nm in sorted(need):
        F = F.merge(I[nm][KEY + ['rv_pred']].rename(columns={'rv_pred': nm}), on=KEY, how='left', validate='one_to_one')
        assert F[nm].notna().all(), f'вход {nm}: нет {int(F[nm].isna().sum())} строк'
    F['p'] = F['rv_pred'].astype('float64')
    meta = {}
    if h == 1:  # persist / reconcile — фиктивные
        return F[COLS].assign(rv_pred=F['p'].values), meta
    ai, step = F['ai'].values, F['step'].values
    sch = m['scheme']
    if sch == 'decomp' and 'far' in need:  # E1: (ŷ₁ + (h−1)·far₂₋ₕ)/h, far — вход дизайна (в хэше)
        F['p'] = (F['base_1'].values + (h - 1) * F['far'].values) / h
    elif sch == 'decomp':
        near, parts = ((F['base_1'].values, [(2, 5, 4)] + ([(6, 22, 17)] if h == 22 else [])) if m['decomp_k'] == 1
                       else (5 * F['base_5'].values, [(6, 22, 17)]))
        F['p'] = (near + sum(w * np.exp(ctx.at_steps(ctx.har_sub(a, b)[0], ai, step)) for a, b, w in parts)) / h
    elif sch == 'iterated':
        scheme_iterated(F, ctx, h, m['iterated_k'], meta)
    elif sch == 'joint_h':
        F['p'] = 0.5 * (np.exp(ctx.at_steps(ctx.joint_ln(h)[0], ai, step)) + F['p'].values)
    elif sch == 'perlead':
        F['p'] = (F['base_1'].values + sum(np.exp(ctx.at_steps(ctx.har_sub(j, j)[0], ai, step)) for j in range(2, 6))) / 5
    if m['slope'] is True:
        F['p'] = (F['base_5'] if h == 22 else F['base_1']).values * np.exp(ctx.at_steps(ctx.slope_ln(h)[0], ai, step))
    if not off(m['persist']):
        persist(F, ctx, m['persist'], h, meta)
    if m['reconcile'] == 'MinT':
        reconcile_mint(F, ctx, h, meta)
    ctx.coef_meta(meta)
    assert np.isfinite(F['p']).all() and (F['p'] > 0).all(), 'нечисловой прогноз'
    return F[COLS].assign(rv_pred=F['p'].values), meta


# ================================================================ кластер 11: нога τ × g
def _beta_w(W, om):
    k = np.arange(W)
    w = (1 - k / W) ** (om - 1)
    return w / w.sum()


def tau_ext(ctx, lntau, s, h, ext, meta):
    """ln τ̄ за [t+1, t+h] из τ_t: const — τ_t; AR1 — пул φ по окну шага с тикерными средними."""
    if ext == 'const':
        return lntau
    vy = 2015 + s
    win = ctx.year < vy
    mu = pd.Series(np.where(win, lntau, np.nan)).groupby(ctx.tk).transform('mean').fillna(np.nanmean(lntau[win])).values
    nxt = pd.Series(lntau).groupby(ctx.tk).shift(-1).values
    x, yv = lntau - mu, nxt - mu
    ok = win & ctx.emb_ok(1, vy) & np.isfinite(x) & np.isfinite(yv)
    phi = float(np.clip((x[ok] * yv[ok]).sum() / max((x[ok] ** 2).sum(), 1e-12), 0.0, 0.9999))
    meta.setdefault(s, {})['ar1_phi'] = np.array([phi])
    pw = phi ** np.arange(1, h + 1)
    return np.log(np.exp(mu[:, None] + x[:, None] * pw[None, :]).mean(1))


def _rvi(ctx):
    return pd.Series(ctx.A['close_RVI'].astype('float64').values).groupby(ctx.tk).ffill().values


def tau_rvi(ctx, Hz, W, s, meta):
    vy = 2015 + s
    x = np.log(_rvi(ctx) ** 2)
    flat = np.log(Hz[:, -W:].mean(1))
    ok = np.isfinite(x)
    D = pd.get_dummies(pd.Categorical(ctx.tk, categories=sorted(set(ctx.tk)))).values.astype('float64')
    X = np.column_stack([D, np.where(ok, x, 0.0)])
    m = (ctx.year < vy) & ok
    co = np.linalg.lstsq(X[m], flat[m], rcond=None)[0]
    meta.setdefault(s, {})['tau_rvi'] = co
    return np.where(ok, X @ co, flat)


def tau_boost(ctx, Zp, Hz, W, s, seed, meta):
    import lightgbm as lgb
    vy = 2015 + s
    imx = pd.Series(ctx.A['rv_IMOEX'].astype('float64').values).groupby(ctx.tk).ffill()
    imx22 = imx.groupby(ctx.tk).transform(lambda v: v.rolling(22, min_periods=1).mean()).values
    codes = pd.Categorical(ctx.tk, categories=sorted(set(ctx.tk))).codes.astype('float64')
    X = np.column_stack([np.log(Hz[:, -5:].mean(1)), np.log(Hz[:, -22:].mean(1)), np.log(Hz[:, -66:].mean(1)), np.log(Hz[:, -W:].mean(1)),
                         np.log(_rvi(ctx) ** 2), np.log(np.clip(imx22, 1e-12, None)), codes])
    g = pd.Series(Zp).groupby(ctx.tk)
    yt = np.log((sum(g.shift(-j) for j in range(1, W + 1)) / W).values + EPS)
    P = dict(CFG['struct']['lgbm'])
    nest = int(P.pop('n_estimators'))
    P.update(objective='regression', seed=int(seed))
    fit = lambda mask: lgb.train(P, lgb.Dataset(X[mask], yt[mask], categorical_feature=[6]), num_boost_round=nest)
    m_all = ctx.trmask(vy, W, yt)
    lntau = fit(m_all).predict(X)
    tend = ctx.tend(W)
    for Y in sorted(set(ctx.year[m_all])):  # кросс-фит строк обучения: год Y — модель без года Y и без целей, заходящих в Y
        mY = m_all & (ctx.year != Y) & (tend != Y)
        rY = m_all & (ctx.year == Y)
        if mY.sum() > 100 and rY.any():
            lntau[rY] = fit(mY).predict(X[rY])
    meta.setdefault(s, {})['tau_boost'] = np.array([m_all.sum(), float(np.mean(lntau[m_all]))])
    return lntau


def struct_leg(design, ctx, ref):
    """Нога кластера 11 на строках ref (val шагов ctx.steps). -> (ряд COLS, lnall: date, ticker, year, step, part, rv_pred, emb_ok, meta)."""
    d = C.validate(design)
    check_struct(d)
    h, lg = int(d['h']), d['leg']
    st, split, W, ext = lg['struct'], lg['struct_split'], int(lg['tau_window']), lg['tau_extend']
    A, tk = ctx.A, ctx.tk
    rv = A['rv'].astype('float64').values
    jmp = A['jump'].astype('float64').clip(lower=0).values
    Z = rv if split == 'rv_whole' else np.clip(rv - jmp, 0, None)
    Zp = pd.Series(np.where(Z > 0, Z, np.nan)).groupby(tk).ffill().fillna(FLOOR).clip(lower=FLOOR).values
    Hz = ctx.lagmat(Zp, max(66, W))
    zd, zw, zm = np.log(Hz[:, -1]), np.log(Hz[:, -5:].mean(1)), np.log(Hz[:, -22:].mean(1))
    Jx = ctx.H[:, 3:]  # скачки HAR-J
    g = pd.Series(Z).groupby(tk)
    y = np.log(sum(g.shift(-j) for j in range(1, h + 1)).values / h + EPS)
    meta, lnp = {}, {}

    def fit_g(s, lntau):
        m = ctx.trmask(2015 + s, h, y)
        lnbar = tau_ext(ctx, lntau, s, h, ext, meta)
        X = np.column_stack([zd - lntau, zw - lntau, zm - lntau, Jx])
        lr = LinearRegression().fit(X[m], (y - lnbar)[m])
        pr = lnbar + lr.predict(X)
        return pr, float(np.mean((y[m] - pr[m]) ** 2)), np.r_[lr.intercept_, lr.coef_]

    for s in ctx.steps:
        if st == 'single':
            m = ctx.trmask(2015 + s, h, y)
            X = np.column_stack([zd, zw, zm, Jx])
            lr = LinearRegression().fit(X[m], y[m])
            lnp[s] = lr.predict(X)
            meta.setdefault(s, {})['single'] = np.r_[lr.intercept_, lr.coef_]
        elif st == 'tau_midas':
            best = None
            for om in CFG['struct']['omega_grid']:
                pr, mse, co = fit_g(s, np.log(Hz[:, -W:] @ _beta_w(W, om)[::-1]))
                if best is None or mse < best[1]:
                    best = (pr, mse, co, om)
            lnp[s] = best[0]
            meta.setdefault(s, {})['midas'] = np.r_[best[3], best[2]]
        elif st == 'tau_rvi':
            lnp[s], _, co = fit_g(s, tau_rvi(ctx, Hz, W, s, meta))
            meta.setdefault(s, {})['g'] = co
        elif st == 'tau_boost':
            lnp[s], _, co = fit_g(s, tau_boost(ctx, Zp, Hz, W, s, d['seed'], meta))
            meta.setdefault(s, {})['g'] = co
        else:
            raise ValueError(st)
    pred = {s: np.exp(lnp[s]) for s in ctx.steps}
    if split == 'bv_plus_jump':
        Jh = ctx.lagmat(jmp, 22)
        XJ = np.column_stack([Jh[:, -1], Jh[:, -5:].mean(1), Jh.mean(1)])
        yj = (sum(pd.Series(jmp).groupby(tk).shift(-j) for j in range(1, h + 1)) / h).values
        for s in ctx.steps:
            m = ctx.trmask(2015 + s, h, yj)
            lr = LinearRegression().fit(XJ[m], yj[m])
            pred[s] = pred[s] + np.clip(lr.predict(XJ), 0, None)
            meta.setdefault(s, {})['jump'] = np.r_[lr.intercept_, lr.coef_]
    R = ref[COLS].copy()
    R = R[R.step.isin(ctx.steps)].sort_values(KEY).reset_index(drop=True)
    ai, step = ctx.rows(R), R['step'].values
    R['rv_pred'] = np.clip(ctx.at_steps(pred, ai, step), 1e-10, None)
    assert np.isfinite(R['rv_pred']).all()
    la = []  # lnall (формат ног L): tr — годы < vy с целью, va — строки ref шага; emb_ok — конец цели < 1 янв. vy
    for s in ctx.steps:
        vy = 2015 + s
        for part, idx in (('tr', np.where((ctx.year < vy) & np.isfinite(y))[0]), ('va', ai[step == s])):
            la.append(pd.DataFrame({'date': A['date'].values[idx], 'ticker': tk[idx], 'year': ctx.year[idx], 'step': s, 'part': part,
                                    'rv_pred': np.clip(pred[s][idx], 1e-10, None), 'emb_ok': ctx.tend(h)[idx] < vy}))
    return R[COLS], pd.concat(la, ignore_index=True), meta


# ================================================================ один дизайн; детектор
def run_one(design, ctx, I):
    d = C.validate(design)
    if is_struct(d):
        out, la, meta = struct_leg(d, ctx, I['ref'])
        return out, meta, la
    out, meta = apply_mh(d, ctx, I)
    return out, meta, None


def fingerprint(meta, steps):
    h = hashlib.sha256()
    for s in sorted(k for k in meta if k in steps):
        for nm in sorted(meta[s]):
            h.update(nm.encode())
            h.update(np.ascontiguousarray(np.asarray(meta[s][nm], dtype='float64')).tobytes())
    return h.hexdigest()


def _maxdl(a, b, cut, after=False):
    m = a.merge(b, on=KEY, suffixes=('', '_p'))
    m = m[(m.date > pd.Timestamp(cut)) if after else (m.date <= pd.Timestamp(cut))]
    return float(np.abs(np.log(m.rv_pred) - np.log(m.rv_pred_p)).max()) if len(m) else float('nan')


class LeakBench:
    """Детектор для схем кластера 6: шум после среза — и в матрице (r2_core.perturb_frame), и во входных рядах (ноги L, роли B);
    середина года — Δ = 0 до среза; 31.12 — параметры шагов с vy ≤ год + 1 совпадают (эмбарго); подлоги look1 (матрица и входы
    дня d+1, r2_core.plant_frame) и noemb (без эмбарго) пойманы. Ноги кластера 11 — r2_core.leak_check (ряды на всех строках, lnall)."""

    def __init__(self, tickers, steps):
        L = CFG['guards']['leak']
        self.mid, self.ye, self.steps, self.tickers = L['cuts_mid'], L['cut_year_end'], tuple(steps), tickers
        A, self.harj = load_A(tickers)
        self.A = A
        self.ctx = {'base': Ctx(A, self.harj, steps)}
        self.seeds = {c: 20261026 + i for i, c in enumerate(self.mid)}
        self.seeds[self.ye] = 20261099
        for c, sd in self.seeds.items():
            self.ctx[c] = Ctx(C.perturb_frame(A, c, sd), self.harj, steps)
        self.ctx['look1'] = Ctx(C.plant_frame(C.perturb_frame(A, self.mid[0], self.seeds[self.mid[0]])), self.harj, steps, plant='look1')
        self.ctx['look1_0'] = Ctx(C.plant_frame(A), self.harj, steps, plant='look1')
        self.ctx['noemb'] = Ctx(C.perturb_frame(A, self.ye, self.seeds[self.ye]), self.harj, steps, plant='noemb')
        self.ctx['noemb_0'] = Ctx(A, self.harj, steps, plant='noemb')

    def check_mh(self, design, I):
        base, mb, _ = run_one(design, self.ctx['base'], I)
        rows, tol = [], CFG['guards']['leak']['tol']
        for c in self.mid:
            o, _, _ = run_one(design, self.ctx[c], perturb_inputs(I, c, self.seeds[c]))
            dl, da = _maxdl(base, o, c), _maxdl(base, o, c, after=True)
            rows.append(dict(test=f'srez {c}', max_dln_do=dl, max_dln_posle=da, ok=bool(dl <= tol)))
        st = [s for s in self.steps if 2015 + s <= pd.Timestamp(self.ye).year + 1]
        _, mp, _ = run_one(design, self.ctx[self.ye], perturb_inputs(I, self.ye, self.seeds[self.ye]))
        same = fingerprint(mb, st) == fingerprint(mp, st)
        rows.append(dict(test=f'embargo {self.ye}, parametry shagov {st}', max_dln_do=0.0 if same else 1.0, ok=bool(same)))
        trivial = not mb  # прямой ряд без оценок (direct, h=1): подлогу нечего ловить
        o0, _, _ = run_one(design, self.ctx['look1_0'], plant_inputs(I))  # подлог после шума (как plant_frame(perturb_frame(A)))
        o1, _, _ = run_one(design, self.ctx['look1'], plant_inputs(perturb_inputs(I, self.mid[0], self.seeds[self.mid[0]])))
        dl = _maxdl(o0, o1, self.mid[0])
        rows.append(dict(test='podlog look1', max_dln_do=dl, ok=bool(trivial or dl > 0)))
        _, m0, _ = run_one(design, self.ctx['noemb_0'], I)
        _, m1, _ = run_one(design, self.ctx['noemb'], perturb_inputs(I, self.ye, self.seeds[self.ye]))
        caught = fingerprint(m0, st) != fingerprint(m1, st)
        rows.append(dict(test='podlog noemb', max_dln_do=float(caught), ok=bool(trivial or caught)))
        return base, rows

    def check_struct(self, design, I, name):
        def build(A):
            out, _, la = run_one(design, Ctx(A, self.harj, self.steps), I)
            return {f'{name}_val': out, f'{name}_all': la}
        R = C.leak_check(build, self.A, f'M:{name}', cuts=tuple(self.mid) + (self.ye,))
        base, _, _ = run_one(design, self.ctx['base'], I)
        rows = [dict(test=f'core {r["srez"]} {r["ryad"]}', max_dln_do=r['max_dln_do'], max_dln_posle=r['max_dln_posle'],
                     podlog=r.get('podlog_max_dln_do', np.nan), ok=bool(r['ok'])) for r in R.to_dict('records')]
        # noemb: без эмбарго граница года должна «протечь» (детектор видит)
        nb = lambda A: run_one(design, Ctx(A, self.harj, self.steps, plant='noemb'), I)[2]
        a0, a1 = nb(self.A), nb(C.perturb_frame(self.A, self.ye, self.seeds[self.ye]))
        on = ['date', 'ticker', 'step', 'part']
        m = a0.merge(a1, on=on, suffixes=('', '_p'))
        kmax = pd.Timestamp(self.ye).year - 2015 + 1
        m = m[(m.date <= pd.Timestamp(self.ye)) & (m.step <= kmax)]
        dl = float(np.abs(np.log(m.rv_pred) - np.log(m.rv_pred_p)).max())
        rows.append(dict(test='podlog noemb', max_dln_do=dl, ok=bool(dl > 0)))
        return base, rows


# ================================================================ параметры по шагам (единый формат для D)
NODES = ('RV5', 'RV22', 'RV1', 'RV2_5', 'RV6_22')
PNAMES = {  # имя массива meta -> имена элементов (иначе — имя[i])
    'persist_OLS': ('rho', None, 'n_hist'), 'persist_qlike': ('rho', None, 'n_hist'),
    'persist_pos_neg': ('rho_pos', 'rho_neg', 'n_hist'), 'persist_sqrtRQ': ('rho0', 'rho_q', 'n_hist'),
    'mint_w': tuple(f'mint_w_{n}' for n in NODES), 'mint_n': ('mint_n_hist',), 'ar1_phi': ('tau_ar1_phi',),
    'tau_boost': ('tau_boost_n_train', 'tau_boost_mean_lntau'),
}


def params_rows(design_sha, h, seed, meta):
    """meta {шаг: {имя: массив}} -> строки design_sha, h, step, param, value (+ seed: ρ и Σ зависят от seed входных рядов)."""
    rows = []
    for s in sorted(meta):
        for nm, v in sorted(meta[s].items()):
            v = np.ravel(np.asarray(v, dtype='float64'))
            names = PNAMES.get(nm)
            if nm == 'mint_sigma':
                k = int(round(np.sqrt(len(v))))
                names = tuple(f'mint_sigma_{NODES[i]}_{NODES[j]}' for i in range(k) for j in range(k))
            elif nm == 'midas':
                names = ('tau_midas_omega',) + tuple(f'g_coef[{i}]' for i in range(len(v) - 1))
            for i, x in enumerate(v):
                p = names[i] if names and i < len(names) else f'{nm}[{i}]'
                if p is not None:
                    rows.append(dict(design_sha=design_sha, h=int(h), step=int(s), param=p, value=float(x), seed=seed))
    return rows


PARAMS_FILE = f'params_{MODULE}.csv'  # решение дирижёра: у каждого модуля свой файл; чужие params_<модуль>.csv M не читает и не пишет


def write_params(design, h, seed, meta, path, prefix=''):
    """R2/cache/<hash>/params_M.csv: свои строки (h, seed) заменяются; flock + os.replace (несколько seed одного дизайна в пуле).
    Читается только свой файл. prefix — имя входного ряда, если у предыдущего модуля их несколько."""
    import fcntl
    assert Path(path).name == PARAMS_FILE, f'M пишет только {PARAMS_FILE}'
    metas = meta if isinstance(meta, list) else [(prefix, meta)]  # [(префикс, meta), ...]
    rows = [dict(r, param=f'{pf}:{r["param"]}' if pf else r['param']) for pf, m in metas
            for r in params_rows(C.design_hash(design)[:16], h, seed, m)]
    new = pd.DataFrame(rows, columns=['design_sha', 'h', 'step', 'param', 'value', 'seed'])
    with open(str(path) + '.lock', 'a') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        old = pd.read_csv(path) if path.exists() else new.iloc[:0]
        old = old[~((old.h == int(h)) & (old.seed.astype(str) == str(seed)))]
        tmp = path.with_name(f'.{PARAMS_FILE}.{os.getpid()}.tmp')
        pd.concat([old, new], ignore_index=True).sort_values(['h', 'seed', 'step', 'param']).to_csv(tmp, index=False)
        os.replace(tmp, path)
        fcntl.flock(lk, fcntl.LOCK_UN)


# ================================================================ run_phase ядра
def phase_fn(design, seed, steps):
    """fn для r2_core.run_phase: SHARED['A'], SHARED['harj'] (load_d0 до пула), SHARED['M_inputs'] — {h: {вход: путь}} (необязательно)."""
    S = C.SHARED
    h = int(design['h'])
    root = S.get('M_root')
    ctx = Ctx(S['A'], S['harj'], steps)
    tk = os.environ.get('R2_TICKERS')  # дым драйвера: строки 3 тикеров (матрица и модели — на всех 17)
    tk = tk.split(',') if tk else None
    I, files = default_inputs(h, seed, tk, steps, (S.get('M_inputs') or {}).get(h), design)
    res, metas = {}, []
    bases, src = ({}, None) if is_struct(design) else chain_bases(design, h, seed, steps, tickers=tk, root=root)
    if not bases:  # нога кл. 11 или в каталоге нет рядов предыдущего модуля — база из inputs yaml
        out, meta, la = run_one(design, ctx, I)
        res['M'] = out
        metas.append(('', meta))
        if la is not None:
            C.write_series(design, 'M', la, h, seed, part='lnall', module=MODULE, root=root)
    else:  # контракт драйвера: база — ряды последнего модуля цепочки до M в каталоге этого дизайна
        files['base'] = f'cache/{C.design_hash(design)[:16]}: модуль {src}, ряды {sorted(bases)}'
        for ry, b in sorted(bases.items()):
            out, meta, _ = run_one(design, ctx, {**I, 'base': b})
            res[f'M_{ry}'] = out
            metas.append((ry if len(bases) > 1 else '', meta))
    cd = C.cache_dir(design, root=root)
    write_params(design, h, seed, metas, C.params_path(design, MODULE, root=root))  # r2_core (PR #201): <кэш>/<hash16>/params_M.csv

    def ident(p):
        p = Path(p) if not isinstance(p, str) else p
        return (str(p.relative_to(REG)) if str(p).startswith(str(REG)) else str(p), C.sha_file(p) if isinstance(p, Path) and p.is_file() else '')
    pd.DataFrame([dict(vhod=k, fayl=ident(p)[0], sha256=ident(p)[1]) for k, p in sorted(files.items())]).to_csv(
        cd / f'M_h{h}_s{seed}_vhody.csv', index=False)
    return res


# ================================================================ режимы
def selftest():
    rng = np.random.default_rng(1)
    b = rng.uniform(1e-4, 1e-3, (50, 3))
    Y = b @ S_MINT.T
    Si = np.linalg.inv(np.diag([1.0, 2.0, 0.5, 0.7, 1.5]))

    def mint(Yh):
        Wi = Si[None] / Yh[:, :, None] / Yh[:, None, :]
        M = np.einsum('ki,nkl,lj->nij', S_MINT, Wi, S_MINT)
        v = np.einsum('ki,nkl,nl->ni', S_MINT, Wi, Yh)
        return np.linalg.solve(M, v[..., None])[..., 0]
    assert np.abs(mint(Y) / b - 1).max() < 1e-10, 'MinT меняет согласованный вход'
    yt = mint(Y * rng.uniform(0.8, 1.2, Y.shape)) @ S_MINT.T
    assert np.abs(yt[:, 0] - (yt[:, 2] + 4 * yt[:, 3]) / 5).max() < 1e-15
    assert np.abs(yt[:, 1] - (yt[:, 2] + 4 * yt[:, 3] + 17 * yt[:, 4]) / 22).max() < 1e-15
    r = rng.normal(0, 0.3, 20000)
    lnp = rng.normal(-7, 0.5, 20000)
    y = np.exp(lnp + 0.6 * r + rng.normal(0, 0.4, 20000))
    assert abs(_ols0(r, np.log(y) - lnp)[0] - 0.6) < 0.02
    rq = minimize_scalar(lambda c: _q_profile(y, lnp + c * r), bounds=(RHO_LO, RHO_HI), method='bounded').x
    assert abs(rq - 0.6) < 0.05, rq
    for W in (22, 44, 66):
        for om in CFG['struct']['omega_grid']:
            w = _beta_w(W, om)
            assert abs(w.sum() - 1) < 1e-12 and (np.diff(w) <= 1e-15).all()
    for na in ({'h': 1, 'mh': {'scheme': 'decomp', 'decomp_k': 1}}, {'h': 5, 'mh': {'scheme': 'iterated', 'iterated_k': 10}},
               {'h': 22, 'mh': {'scheme': 'perlead'}}, {'h': 5, 'mh': {'scheme': 'decomp', 'decomp_k': 5}},
               {'h': 22, 'mh': {'scheme': 'joint_h', 'slope': 'on'}}, {'h': 5, 'mh': {'decomp_k': 1}}):
        try:
            check_mh(C.validate(na))
            raise AssertionError(f'не поймано: {na}')
        except NotApplicable:
            pass
    for na in ({'h': 5, 'leg': {'struct': 'single'}}, {'h': 5, 'leg': {'struct': 'single', 'struct_split': 'bv_plus_jump', 'tau_window': 44}},
               {'h': 5, 'leg': {'struct': 'tau_rvi'}, 'mh': {'persist': 'OLS'}}):
        try:
            check_struct(C.validate(na))
            raise AssertionError(f'не поймано: {na}')
        except NotApplicable:
            pass
    assert check_mh(C.validate({'h': 1, 'mh': {'persist': 'qlike', 'reconcile': 'MinT'}})) == set()
    assert check_mh(C.validate({'h': 22, 'mh': {'persist': 'sqrtRQ', 'reconcile': 'MinT'}})) == {'base_1', 'leg_1', 'base_5'}
    try:
        C.validate({'h': 5, 'mh': {'persist': 'xx'}})
        raise AssertionError('схема не поймала persist=xx')
    except C.DesignError:
        pass
    log('selftest OK: MinT (согласованность), ρ (OLS, QLIKE), веса MIDAS, применимость, схема через r2_core')


def control(steps):
    """HAR[1..h] модуля (d0, эмбарго по концу цели) = нога L HARJ_emb_d0_h{h} на 17 тикерах."""
    A, harj = load_A(None)
    ctx = Ctx(A, harj, steps)
    rows = []
    for h in (1, 5, 22):
        ref, _ = read_input(CFG['inputs_default']['ref'], 42, h, None, steps)
        ai = ctx.rows(ref)
        dl = float(np.abs(ctx.at_steps(ctx.har_sub(1, h)[0], ai, ref.step.values) - np.log(ref.rv_pred.values)).max())
        y = ctx.tgt(1, h)[ai]
        ok = np.isfinite(y)
        dy = float(np.abs(y[ok] / ref.rv_actual.values[ok] - 1).max())
        rows.append(dict(h=h, rows=len(ref), max_dln=dl, max_rel_target=dy, ok=bool(dl <= CFG['guards']['control']['tol'] and dy < 1e-9)))
        log(f'контроль h={h}: max|Δ ln| = {dl:.2e}, цель {dy:.2e}', 'OK' if rows[-1]['ok'] else 'ОШИБКА')
    return pd.DataFrame(rows)


def dym():
    D = CFG['dym']
    tick, steps, seed = D['tickers'], tuple(D['steps']), int(D['seed'])
    out = C.MAC_ROOT / CFG['out']['dym']
    out.mkdir(parents=True, exist_ok=True)
    selftest()
    ctl = control(steps)
    ctl.to_csv(out / 'kontrol.csv', index=False)
    bench = LeakBench(tick, steps)
    summ, leak_rows, sdvig, par_rows = [], [], [], []
    for h in D['h']:
        I, _ = default_inputs(h, seed, tick, steps)
        for spec in D['designs']:
            if spec.get('only_h') and h not in spec['only_h']:
                continue
            nm = spec['name']
            des = C.validate({'design': nm, 'h': h, 'seed': seed, **{k: v for k, v in spec.items() if k in ('mh', 'leg')}})
            t0 = time.time()
            try:
                (check_struct if is_struct(des) else check_mh)(des)
            except NotApplicable as e:
                summ.append(dict(h=h, design=nm, status='ne_primenim', note=str(e)))
                continue
            base, lr = (bench.check_struct(des, I, nm) if is_struct(des) else bench.check_mh(des, I))
            sc = C.shift_check(base, bench.A, h, f'M:{nm}', des, block_file=out / 'blok_sdvig_dym.csv')
            base.to_parquet(out / f'{nm}_h{h}_s{seed}_val.parquet', index=False)
            rr = base.merge(I['base'][KEY + ['rv_pred']], on=KEY, suffixes=('', '_in'))
            rr = np.log(rr.rv_pred / rr.rv_pred_in)
            summ.append(dict(h=h, design=nm, status='ok', rows=len(base), dln_vs_base_mean=float(rr.mean()), dln_vs_base_sd=float(rr.std()),
                             fiktivnyy=bool(h == 1 and not is_struct(des)), leak_ok=all(r['ok'] for r in lr), shift_ok=sc['ok'],
                             shift_argmax=sc['argmax'], sec=round(time.time() - t0, 1), sha_design=C.design_hash(des)[:16],
                             slozhnost=C.complexity(des)))
            leak_rows += [dict(h=h, design=nm, **r) for r in lr]
            par_rows += [dict(design=nm, **r) for r in params_rows(C.design_hash(des)[:16], h, seed, run_one(des, bench.ctx['base'], I)[1])]
            sdvig.append(dict(h=h, design=nm, **{k: v for k, v in sc.items() if k.startswith('corr') or k in ('argmax', 'ok', 'n_strok')}))
            log(f'h={h} {nm}: строк {len(base)}, детектор {"OK" if summ[-1]["leak_ok"] else "ОШИБКА"}, '
                f'сдвиг argmax={sc["argmax"]} {"OK" if sc["ok"] else "БЛОК"}, {summ[-1]["sec"]} с')
    S = pd.DataFrame(summ)
    S.to_csv(out / 'dym_summary.csv', index=False)
    pd.DataFrame(leak_rows).to_csv(out / 'leak.csv', index=False)
    pd.DataFrame(par_rows).to_csv(out / PARAMS_FILE, index=False)  # формат D (design_sha, h, step, param, value, seed) + имя дизайна
    pd.DataFrame(sdvig).to_csv(out / 'sdvig.csv', index=False)
    ok = S[S.status == 'ok']
    log(f'дым: контуров {len(ok)}, детектор {int(ok.leak_ok.sum())}/{len(ok)}, сдвиг {int((ok.shift_ok == True).sum())}/{len(ok)}, '
        f'не применимо {int((S.status == "ne_primenim").sum())}, контроль {int(ctl.ok.sum())}/{len(ctl)}')
    return S


def load_designs(p):
    Y = yaml.safe_load(Path(p).read_text())
    return Y['designs'] if isinstance(Y, dict) and 'designs' in Y else (Y if isinstance(Y, list) else [Y])


def run(designs, h, seeds, steps, inputs=None, workers=None, phase='E', root=None, check_gate=True, block_file=None):
    """Боевой счёт (сервер) через r2_core.run_phase: ворота, кэш, запись, сдвиг, DONE. Дизайны — только применимые.
    root / check_gate=False / block_file — только для проверки механики вне боевого кэша."""
    A, harj = load_A(None)
    C.share(A=A, harj=harj, M_inputs=(yaml.safe_load(Path(inputs).read_text()) if inputs else None), M_root=root)
    ok = []
    for d in load_designs(designs):
        full = C.validate({**d, 'h': h})
        try:
            (check_struct if is_struct(full) else check_mh)(full)
            ok.append(full)
        except NotApplicable as e:
            log(f'не применим: {full.get("design")} — {e}')
    return C.run_phase(phase, h, ok, phase_fn, seeds=tuple(seeds), steps=tuple(steps), workers=workers, module=MODULE, root=root,
                       block_file=block_file or os.environ.get('R2_BLOCK_FILE'), check_gate=check_gate)


def contour(d):
    """Ключи, от которых зависит код M: один детектор на контур (h, mh, leg.struct*, входы h=1)."""
    lg = d['leg']
    return repr((int(d['h']), sorted(d['mh'].items()), lg['struct'], lg['struct_split'], lg['tau_window'], lg['tau_extend'],
                 sorted((d.get('inputs') or {}).keys())))


def leak_cmd(designs, steps, tickers, seed, out=None):
    """Детектор на каждом новом контуре файла дизайнов (драйвер: один раз на (модуль, h) до боевого счёта). По умолчанию —
    тикеры и шаги дыма (срезы детектора — внутри шагов 1–3). Код выхода ≠ 0 — провал."""
    tickers = tickers or CFG['dym']['tickers']
    steps = tuple(steps or CFG['dym']['steps'])
    bench = LeakBench(tickers, steps)
    rows, seen = [], set()
    for d in load_designs(designs):
        des = C.validate({**d, 'seed': seed})
        try:
            (check_struct if is_struct(des) else check_mh)(des)
        except NotApplicable:
            continue
        k = contour(des)
        if k in seen:
            continue
        seen.add(k)
        h = int(des['h'])
        I, _ = default_inputs(h, seed, tickers, steps, design=des)
        _, lr = bench.check_struct(des, I, des['design'] or 'M') if is_struct(des) else bench.check_mh(des, I)
        rows += [dict(design=des.get('design'), h=h, **r) for r in lr]
    R = pd.DataFrame(rows)
    if out:
        R.to_csv(out, index=False)
    log(f'детектор M: контуров {len(seen)}, проверок {len(R)}, провалов {int((~R.ok).sum()) if len(R) else 0}')
    if len(R) and not R.ok.all():
        print(R[~R.ok].to_string(index=False))
        sys.exit(1)


# ================================================================ E1 (поручение владельца 20:25): разложение по горизонту, вне драйвера
E1_DIR = REG / 'R2' / 'M' / 'E1'
E1_HYB1 = 'R2/cache/VYR/F_d0_s2emb_h1_s{seed}_val.parquet'          # hyb₁: финал R1 h=1, d0, эмбарго s²
E1_ETA = 'R2/cache/VYR/ETA_b_d0_s2emb_h{h}_sdet_val.parquet'        # В2: HAR-J d0 + s² sm с эмбарго
E1_F = 'R2/cache/VYR/F_d0_s2emb_h{h}_s{seed}_val.parquet'           # опора F_R1_d0
SEEDS10 = tuple(range(42, 52))


def har_corr(ctx, S, h, a, b, rows, s2emb='on'):
    """HAR-J d0 (эмбарго обучения) на RV̄_{a..b} на строках rows (val шагов ctx.steps) с поправкой s² sm кода L
    (r2_calib.calibrate, calib.s2_embargo = s2emb) — тот же код, что ETA_b_d0_s2emb. -> (кадр COLS, сырой кадр)."""
    import r2_calib as LC
    R = rows[KEY + ['year']].copy()
    ai = ctx.rows(R)
    R['rv_actual'] = ctx.tgt(a, b)[ai]  # цель подгоризонта; после 2021 — NaN (в пул s² шагов ≤ 6 не входит)
    R['rv_pred'] = np.exp(ctx.at_steps(ctx.har_sub(a, b)[0], ai, R.step.values))
    Vc, _ = LC.calibrate({'h': h, 'calib': {'s2_embargo': s2emb}}, R[COLS], S)
    out = R[KEY].merge(Vc[KEY + ['rv_pred']], on=KEY, how='left')
    assert out.rv_pred.notna().all()
    return R[COLS].assign(rv_pred=out.rv_pred.values), R[COLS]


def e1_series(ctx, S, h, hyb1, rows, s2emb='on'):
    """E1: h=5 — (hyb₁ + 4·HAR₂₋₅)/5; h=22 — (hyb₁ + 21·HAR₂₋₂₂)/22; HAR — с поправкой s² sm (эмбарго)."""
    far, raw = har_corr(ctx, S, h, 2, h, rows, s2emb)
    m = rows[COLS].merge(hyb1[KEY + ['rv_pred']].rename(columns={'rv_pred': 'h1'}), on=KEY, how='left') \
        .merge(far[KEY + ['rv_pred']].rename(columns={'rv_pred': 'far'}), on=KEY, how='left')
    assert m.h1.notna().all() and m.far.notna().all(), 'E1: нет строк hyb₁ или HAR₂₋ₕ'
    fac = (far.rv_pred / raw.rv_pred).groupby(far.step).agg(['min', 'max'])
    assert ((fac['max'] - fac['min']) < 1e-9).all(), 'поправка s² — скаляр шага'
    return m[COLS].assign(rv_pred=(m.h1.values + (h - 1) * m.far.values) / h), far, fac['min']


def e1_leak(A, harj, S, h, rows, hyb1):
    """Детектор на ряде E1 (seed 42): середина года — Δ = 0 до среза; 31.12 — поправка s² шагов vy ≤ год + 1 совпадает (эмбарго);
    подлоги look1 (матрица и hyb₁ дня d+1) и noemb (без эмбарго обучения и s²) пойманы."""
    L = CFG['guards']['leak']
    mk = lambda A_, I_, plant=None, emb='on': e1_series(Ctx(A_, harj, STEPS, plant=plant), S, h, I_, rows, emb)
    base, _, f0 = mk(A, hyb1)
    out = []
    for i, c in enumerate(L['cuts_mid']):
        o, _, _ = mk(C.perturb_frame(A, c, 20261026 + i), perturb_inputs({'b': hyb1}, c, 20261026 + i)['b'])
        out.append(dict(test=f'srez {c}', max_dln_do=_maxdl(base, o, c), ok=bool(_maxdl(base, o, c) == 0.0)))
    ye = L['cut_year_end']
    kmax = pd.Timestamp(ye).year - 2015 + 1
    _, _, f1 = mk(C.perturb_frame(A, ye, 20261099), perturb_inputs({'b': hyb1}, ye, 20261099)['b'])
    same = bool((f0.loc[:kmax] == f1.loc[:kmax]).all())
    out.append(dict(test=f'embargo {ye}: s² шагов ≤ {kmax}', max_dln_do=float(np.abs(np.log(f0.loc[:kmax] / f1.loc[:kmax])).max()), ok=same))
    c = L['cuts_mid'][0]
    o0, _, _ = mk(C.plant_frame(A), plant_inputs({'b': hyb1})['b'], plant='look1')
    o1, _, _ = mk(C.plant_frame(C.perturb_frame(A, c, 20261026)), plant_inputs(perturb_inputs({'b': hyb1}, c, 20261026))['b'], plant='look1')
    dl = _maxdl(o0, o1, c)
    out.append(dict(test='podlog look1', max_dln_do=dl, ok=bool(dl > 0)))
    _, _, g0 = mk(A, hyb1, plant='noemb', emb='off')
    _, _, g1 = mk(C.perturb_frame(A, ye, 20261099), perturb_inputs({'b': hyb1}, ye, 20261099)['b'], plant='noemb', emb='off')
    dl = float(np.abs(np.log(g0.loc[:kmax] / g1.loc[:kmax])).max())
    out.append(dict(test='podlog noemb', max_dln_do=dl, ok=bool(dl > 0)))
    return out


def _row(nm, h, r, sig=None, opora=False):
    m = r['moshchnost']
    return dict(h=h, ryad=nm, opora=opora, t_srednyaya=r['t_mean'], skladok_t_ge_1=r['n_t_ge'],
                **{f't_{int(y)}': t for y, t in zip(r['folds'].Y, r['folds'].t)}, hit=r['hit'],
                tikery=f"{r['n_tikerov_vyigrysh']}/{r['n_tikerov']}", konc=r['konc'], c3_vse=r['c3_vse'], vyigrysh_pct=r['vyigrysh_pct'],
                P_p_lt_005=m['P_p_lt_005'], t_ozhid=m['t_ozhid'], moshchnost=m.get('kategoriya', m.get('kat', '')),
                vhod=r['vhod']['proshyol'], vhod_net=r['vhod']['net'],
                **({} if sig is None else dict(s2_d_mean=sig['d_mean'], s2_porog=sig['thr'], s2_proshyol=sig['passed'],
                                               s2_luchshe_seed=sig['n_better'])))


def e1(seeds=SEEDS10):
    import r2_calib as LC
    import r2_gates as GT
    E1_DIR.mkdir(parents=True, exist_ok=True)
    A, harj = load_A(None)
    ctx = Ctx(A, harj, STEPS)
    M = LC.N.mat_L()
    rows, leak_rows, kon, sdv = [], [], [], []
    for h in (5, 22):
        S = LC.state_frame(M, h)
        Eb, _ = read_input(E1_ETA, h=h)
        # контроль: тот же путь на RV̄₁₋ₕ даёт эталон В2 ETA_b_d0_s2emb
        c1, _ = har_corr(ctx, S, h, 1, h, Eb)
        mm = c1.merge(Eb[KEY + ['rv_pred']], on=KEY, suffixes=('', '_E'))
        kon.append(dict(h=h, proverka='HAR₁₋ₕ + s² sm emb = ETA_b_d0_s2emb', max_rel=float(np.abs(mm.rv_pred / mm.rv_pred_E - 1).max())))
        log(f'контроль h={h}: max rel = {kon[-1]["max_rel"]:.2e}')
        hyb = {s: read_input(E1_HYB1, s, 1)[0] for s in seeds}
        V = {}
        for s in seeds:
            V[s], far, fac = e1_series(ctx, S, h, hyb[s], Eb)
        far_f = E1_DIR / f'HARc_2_{h}_h{h}_sdet_val.parquet'
        far.to_parquet(far_f, index=False)
        # кэш по схеме: дизайн decomp k=1 + входы base_1 (hyb₁) и far (HAR₂₋ₕ с s²) — оба в хэше
        des = C.with_inputs({'design': f'E1_decomp_h{h}', 'h': h, 'mh': {'scheme': 'decomp', 'decomp_k': 1}},
                            {'base_1': {'file': E1_HYB1}, 'far': {'file': str(far_f.relative_to(REG))}})
        for s in seeds:
            C.write_series(des, 'M_E1', V[s], h, s, module=MODULE, note='E1 владелец 20:25: (hyb1 + (h-1) HAR_2_h)/h')
            C.done_path(des, MODULE, h, s).write_text(time.strftime('%F %T'))
        write_params(des, h, 'det', {int(k): {'far_s2_F': np.array([v])} for k, v in fac.items()}, C.params_path(des, MODULE))
        sc = C.shift_check(V[42], A, h, 'M:E1', des)
        sdv.append(dict(h=h, **{k: v for k, v in sc.items() if k.startswith('corr') or k in ('argmax', 'ok')}))
        log(f'сдвиг E1 h={h}: argmax={sc["argmax"]} {"OK" if sc["ok"] else "БЛОК"}')
        lr = e1_leak(A, harj, S, h, Eb, hyb[42])
        leak_rows += [dict(h=h, **r) for r in lr]
        log(f'детектор E1 h={h}: ' + ', '.join(f'{r["test"]} {"OK" if r["ok"] else "ОШИБКА"}' for r in lr))
        r = GT.ocenka({'E1': V[42]}, Eb, h)
        Fs = {s: read_input(E1_F, s, h)[0] for s in seeds}
        sig = GT.sigma2_F(V, h, seeds=list(seeds), F=Fs)
        rows.append(_row(f'E1 h={h}', h, r, sig) | dict(design_sha=C.design_hash(des)[:16]))
        op = GT.opora_ocenka(h, Eb, F=Fs, seeds=list(seeds))
        rows.append(_row('F_R1_d0 (опора)', h, op, opora=True))
        r['folds'].assign(h=h, ryad='E1').to_csv(E1_DIR / f'folds_E1_h{h}.csv', index=False)
    T = pd.DataFrame(rows)
    T.to_csv(E1_DIR / 'e1_tablica.csv', index=False)
    pd.DataFrame(leak_rows).to_csv(E1_DIR / 'e1_leak.csv', index=False)
    pd.DataFrame(kon).to_csv(E1_DIR / 'e1_kontrol.csv', index=False)
    pd.DataFrame(sdv).to_csv(E1_DIR / 'e1_sdvig.csv', index=False)
    pd.set_option('display.width', 250)
    print(T.drop(columns=[c for c in T.columns if c.startswith('t_20')]).to_string(index=False))
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['selftest', 'control', 'dym', 'run', 'leak', 'e1'])
    ap.add_argument('--designs')
    ap.add_argument('--inputs')
    ap.add_argument('--h', type=int)
    ap.add_argument('--phase', default='E')
    ap.add_argument('--steps', type=int, nargs='+')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tickers', nargs='+')
    ap.add_argument('--workers', type=int)
    a = ap.parse_args()
    if a.cmd == 'selftest':
        selftest()
    elif a.cmd == 'control':
        print(control(tuple(a.steps or STEPS)).to_string(index=False))
    elif a.cmd == 'dym':
        dym()
    elif a.cmd == 'run':
        print(run(a.designs, a.h, a.seeds, a.steps or list(STEPS), a.inputs, a.workers, a.phase).to_string(index=False))
    elif a.cmd == 'e1':
        e1()
    elif a.cmd == 'leak':
        leak_cmd(a.designs, a.steps, a.tickers, a.seed)


if __name__ == '__main__':
    main()
