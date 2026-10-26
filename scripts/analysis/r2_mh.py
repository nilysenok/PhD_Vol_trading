#!/usr/bin/env python3
"""R2, агент M (протокол v3 docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md, раздел 6, строка «M»; yaml models/designs/r2/r2_M_mh.yaml).

Кластер 6 — многогоризонтные схемы над готовым рядом h (нога L или роль B), ключи mh.*:
  scheme   direct | decomp[k] | iterated[k_iter] | joint_h | perlead (h=5)       — R1: Д18, Д20 (x6 Д22), Д24.2, Д26.3
  slope    наклон срочной структуры ŷ_h = ŷ_k · exp(ŝ)                            — R1: Д25.3
  persist  ln ŷ_h += ρ·r̂₁, r̂₁ = ln base_1 − ln leg_1; ρ: OLS | qlike | pos_neg | sqrtRQ — R1: Д19.1, x3 Д22
  reconcile MinT: узлы {RV̄₁, RV̄₂₋₅, RV̄₆₋₂₂} -> {RV̄₅, RV̄₂₂}, Σ — Ледуа–Вольф по прошлым OOS
  Порядок: scheme -> slope -> persist -> reconcile. На h=1 persist и reconcile — фиктивные (ряд не меняется).
Кластер 11 — нога τ × g (ключи leg.struct, struct_split, tau_window, tau_extend): tau_midas | tau_rvi | tau_boost; BV + J.
Определения — в yaml (раздел mh / struct); они записаны до счёта.

Окна (раздел 1 протокола): шаг k — val-год vy = 2015 + k, обучение — годы < vy, эмбарго по концу цели (строка t+b тикера < 1 янв. vy).
Всё, что оценивается по прошлым ошибкам (ρ, Σ MinT), на шаге t — по OOS-строкам шагов < t с тем же эмбарго; шаг 1 — без поправки.
Ничего за годы >= 2022 не читается: features_pit_v3 — фильтр date <= 2021-12-31 при чтении + assert; входные ряды — step <= 6 + assert.
test_locked/ не читается. Метрик модуль не считает (выбор — агент O).

Ядро r2_core (агент S): если модуль есть — берутся его validate_design / design_hash / cache_dir; иначе — локальная реализация по
шапке schema_v1.yaml (до PR агента S).

python3 scripts/analysis/r2_mh.py selftest
python3 scripts/analysis/r2_mh.py control  [--steps 1 2 3]          # HAR[1..h] модуля = нога L HARJ_emb_d0 (17 тикеров), допуск 1e-7
python3 scripts/analysis/r2_mh.py dym                               # дым: 3 тикера × шаги 1–3 × seed 42 × h ∈ {1, 5, 22} + сдвиг + детектор
python3 scripts/analysis/r2_mh.py run --designs <file.yaml> [--steps 1..6] [--seed 42] [--inputs <file.yaml>]   # боевой счёт (сервер)
python3 scripts/analysis/r2_mh.py leak --designs <file.yaml> [--tickers ...] [--steps ...]"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
import argparse, copy, hashlib, json, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
YAML = REPO / 'models' / 'designs' / 'r2' / 'r2_M_mh.yaml'
SCHEMA_F = REPO / 'models' / 'designs' / 'r2' / 'schema_v1.yaml'
CFG = yaml.safe_load(YAML.read_text())
SCHEMA = yaml.safe_load(SCHEMA_F.read_text())
ROOT = Path(os.environ.get('PHD_ROOT', CFG['root']))
REG = ROOT / 'data' / 'regen'
FEAT = ROOT / CFG['features']
MAXD = pd.Timestamp(CFG['max_date'])
MAXSTEP = int(CFG['max_step'])
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
KEY = ['date', 'ticker', 'step']
STEPS = tuple(range(1, MAXSTEP + 1))
FLOOR = float(CFG['struct']['floor'])
EPS = 1e-10
RHO_LO, RHO_HI = CFG['mh']['persist']['rho_bounds']
S_MINT = np.array([[1 / 5, 4 / 5, 0], [1 / 22, 4 / 22, 17 / 22], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype='float64')

try:  # ядро агента S (появится в его PR)
    import r2_core as CORE
except Exception:  # noqa: BLE001
    CORE = None


class NotApplicable(Exception):
    """Сочетание ключей не применимо на этом h (дизайн отклоняется до счёта)."""


def now():
    return time.strftime('%F %T')


def log(*a):
    print(now(), *a, flush=True)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def git_head():
    try:
        return subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ''


# ================================================================ схема: дополнение default, валидация, канонический хэш (fallback ядра)
def _is_leaf(v):
    return isinstance(v, dict) and bool({'values', 'default', 'type', 'fixed'} & set(v))


def _defaults(node):
    out = {}
    for k, v in node.items():
        if k == 'schema':
            continue
        if _is_leaf(v):
            if 'default' in v:
                out[k] = copy.deepcopy(v['default'])
            elif 'values' in v and v.get('fixed'):
                out[k] = copy.deepcopy(v['values'][0])
            elif v.get('fixed'):
                out[k] = {a: b for a, b in v.items() if a not in ('fixed', 'note')}
        elif isinstance(v, dict):
            out[k] = _defaults(v)
    return out


def _merge(a, b):
    r = copy.deepcopy(a)
    for k, v in (b or {}).items():
        r[k] = _merge(r[k], v) if isinstance(v, dict) and isinstance(r.get(k), dict) else copy.deepcopy(v)
    return r


def full_design(d):
    return _merge(_defaults(SCHEMA), d)


def validate_design(d):
    """Значение вне списка схемы — ошибка (до счёта). Проверяются все ключи дизайна, которые есть в схеме; неизвестный ключ — ошибка."""
    if CORE is not None and hasattr(CORE, 'validate_design'):
        return CORE.validate_design(d)

    def walk(node, sch, path):
        for k, v in node.items():
            if k in ('design', 'schema'):
                continue
            if k not in sch:
                raise ValueError(f'ключа {path}{k} нет в схеме v1')
            s = sch[k]
            if _is_leaf(s):
                vals = s.get('values')
                if vals is None:
                    continue
                if isinstance(v, list) and k == 'features':
                    bad = [x for x in v if x not in vals]
                    if bad or 'G0' not in v:
                        raise ValueError(f'{path}{k}: {v} вне списка или без G0')
                elif v not in vals and not (isinstance(v, list) and all(x in vals for x in v)):
                    raise ValueError(f'{path}{k} = {v!r} вне списка {vals}')
            elif isinstance(v, dict):
                walk(v, s, f'{path}{k}.')
    walk(d, SCHEMA, '')
    if 'h' not in d:
        raise ValueError('в дизайне нет h')
    return True


def _canon(x):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        return ('%.15g' % float(x)) if not float(x).is_integer() else str(int(x))
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    return x


def design_hash(d):
    if CORE is not None and hasattr(CORE, 'design_hash'):
        return CORE.design_hash(d)
    f = full_design(d)
    f.pop('design', None)
    f['schema'] = 'v1'
    s = yaml.safe_dump(_canon(f), sort_keys=True, allow_unicode=True, default_flow_style=False)
    return sha_bytes(s.encode())


def cache_dir(d):
    if CORE is not None and hasattr(CORE, 'cache_dir'):
        return Path(CORE.cache_dir(d))
    return ROOT / CFG['out']['cache'] / design_hash(d)[:16]


# ================================================================ применимость
def is_struct(d):
    lg = d['leg']
    return lg['struct'] != 'single' or lg['struct_split'] != 'rv_whole'


def check_mh(d):
    """Возвращает список нужных входов; NotApplicable — сочетание не применимо."""
    h, m = int(d['h']), d['mh']
    sch, need = m['scheme'], set()
    if h == 1:
        if sch != 'direct' or m['slope'] == 'on':
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
        need |= {'base_1'} if k == 1 else {'base_5'}
    elif sch == 'iterated':
        k = m['iterated_k']
        if k not in CFG['mh']['iterated_k'][h]:
            raise NotApplicable(f'iterated_k={k} на h={h}')
        need |= {'base_1', 'leg_1'}
    elif sch == 'perlead':
        if h != 5:
            raise NotApplicable('perlead — только h=5')
        need.add('base_1')
    if m['slope'] == 'on':
        if sch != 'direct':
            raise NotApplicable('slope — только при scheme=direct')
        need.add('base_5' if h == 22 else 'base_1')
    if m['persist'] != 'off':
        need |= {'base_1', 'leg_1'}
    if m['reconcile'] == 'MinT':
        need |= {'base_1', 'base_22' if h == 5 else 'base_5'}
    return need


def check_struct(d):
    lg = d['leg']
    if lg['struct'] == 'single':
        if lg['struct_split'] == 'rv_whole':
            raise NotApplicable('single + rv_whole — нога L (HARJ)')
        if lg['tau_window'] != 22 or lg['tau_extend'] != 'const':
            raise NotApplicable('tau_window / tau_extend без τ')
    if any(v != 'off' and v is not None and v != 'direct' for v in (d['mh']['persist'], d['mh']['reconcile'], d['mh']['scheme'])):
        raise NotApplicable('нога кластера 11 и схема кластера 6 — разные вызовы (нога -> роль -> схема); ядро вызывает их по очереди')


# ================================================================ матрица признаков (<= 2021) и подмодели шагов
def _feat_cols():
    import pyarrow.parquet as pq
    names = pq.read_schema(FEAT).names
    jump = [c for c in names if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower()]
    return jump, ['date', 'ticker', 'rv', 'bv', 'rkurt', 'n_bars', 'close_RVI', 'rv_IMOEX'] + jump


_RAW = {}


def read_features(tickers=None):
    """features_pit_v3, только date <= 2021-12-31 (фильтр при чтении) и нужные столбцы."""
    k = tuple(tickers) if tickers else None
    if k not in _RAW:
        jump, cols = _feat_cols()
        flt = [('date', '<=', MAXD)] + ([('ticker', 'in', list(tickers))] if tickers else [])
        A = pd.read_parquet(FEAT, columns=cols, filters=flt)
        A['date'] = pd.to_datetime(A['date'])
        assert A['date'].max() <= MAXD and A['date'].dt.year.max() <= 2021, 'прочитаны строки >= 2022'
        _RAW[k] = (A, jump)
    A, jump = _RAW[k]
    return A.copy(), jump


def perturb_frame(A, cut, seed, cols=None):
    """Всё числовое после среза × U(lo; hi) (детектор «будущее не влияет»)."""
    lo, hi = CFG['guards']['leak']['noise']
    rng = np.random.default_rng(seed)
    B = A.copy()
    fut = (B['date'] > pd.Timestamp(cut)).values
    for c in (cols or [c for c in B.columns if c not in ('date', 'ticker', 'year', 'step') and pd.api.types.is_numeric_dtype(B[c])]):
        B.loc[fut, c] = B.loc[fut, c].values * rng.uniform(lo, hi, fut.sum())
    return B


class Ctx:
    """Матрица d0 (данные по день d включительно) и подмодели шагов. plant: None | 'look1' (регрессоры и входы дня d+1) | 'noemb'."""

    def __init__(self, tickers=None, steps=STEPS, perturb=None, plant=None, seed=42):
        assert max(steps) <= MAXSTEP
        A, self.jump = read_features(tickers)
        if perturb is not None:
            A = perturb_frame(A, *perturb)
        A = A.sort_values(['date', 'ticker'], kind='mergesort').reset_index(drop=True)
        A['year'] = A['date'].dt.year
        tk = A['ticker']
        rv = A['rv'].astype('float64')
        rvp = rv.where(rv > 0).groupby(tk).ffill().fillna(rv)
        A['rv_d'] = rvp
        A['rv_w'] = rvp.groupby(tk).transform(lambda s: s.rolling(5, min_periods=1).mean())
        A['rv_m'] = rvp.groupby(tk).transform(lambda s: s.rolling(22, min_periods=1).mean())
        if plant == 'look1':
            A['rv_d'] = A.groupby('ticker')['rv_d'].shift(-1).fillna(A['rv_d'])
        self.A, self.steps, self.plant, self.seed = A, tuple(steps), plant, seed
        self.harj = ['rv_d', 'rv_w', 'rv_m'] + [c for c in self.jump if c not in ('rv_d', 'rv_w', 'rv_m')]
        self.H = np.nan_to_num(np.log(A[self.harj].values.astype('float64').clip(1e-10)), nan=0, posinf=0, neginf=0)
        self.year = A['year'].values
        self.key = pd.DataFrame({'date': A['date'].values, 'ticker': A['ticker'].values.astype(str), 'ai': np.arange(len(A))})
        self.q = np.sqrt((A['n_bars'].astype('float64') * A['rkurt'].astype('float64') / 3).clip(lower=0)).clip(upper=10).fillna(1.0).values
        self._c = {}

    # ---- цели и эмбарго
    def tgt(self, a, b):
        k = ('tgt', a, b)
        if k not in self._c:
            g = self.A.groupby('ticker')['rv']
            self._c[k] = (sum(g.shift(-j) for j in range(a, b + 1)) / (b - a + 1)).values.astype('float64')
        return self._c[k]

    def tend(self, b):
        """Год дня t+b тикера (конец окна цели); нет дня (конец данных) — 9999."""
        k = ('tend', b)
        if k not in self._c:
            d = self.A.groupby('ticker')['date'].shift(-b)
            self._c[k] = d.dt.year.fillna(9999).values.astype(int)
        return self._c[k]

    def emb_ok(self, b, vy):
        return np.ones(len(self.A), bool) if self.plant == 'noemb' else (self.tend(b) < vy)

    def trmask(self, vy, b, y):
        return (self.year < vy) & np.isfinite(y) & self.emb_ok(b, vy)

    def rows(self, df):
        ai = df[['date', 'ticker']].merge(self.key, on=['date', 'ticker'], how='left')['ai']
        assert ai.notna().all(), f'строк ряда нет в матрице: {ai.isna().sum()}'
        return ai.values.astype(int)

    def lagmat(self, s, L):
        """(n, L): уровни дней t−L+1 … t тикера; недостающее начало — ближайшее известное."""
        g = pd.Series(s).groupby(self.A['ticker'].values)
        M = np.column_stack([g.shift(k).values for k in range(L - 1, -1, -1)]).astype('float64')
        for j in range(L - 2, -1, -1):
            M[:, j] = np.where(np.isnan(M[:, j]), M[:, j + 1], M[:, j])
        return M

    # ---- HAR-J d0 на цель RV̄_{a..b}
    def har_sub(self, a, b):
        k = ('har', a, b)
        if k not in self._c:
            y = np.log(self.tgt(a, b) + EPS)
            ln, co = {}, {}
            for s in self.steps:
                m = self.trmask(2015 + s, b, y)
                lr = LinearRegression().fit(self.H[m], y[m])
                ln[s], co[s] = lr.predict(self.H), np.r_[lr.intercept_, lr.coef_]
            self._c[k] = (ln, co)
        return self._c[k]

    def at_steps(self, D, ai, step):
        out = np.full(len(ai), np.nan)
        for s in np.unique(step):
            m = step == s
            out[m] = D[int(s)][ai[m]]
        return out


# ================================================================ входы
def read_series(rel, seed=42, h=None, tickers=None, steps=STEPS):
    p = REG / rel.format(seed=seed, h=h)
    import pyarrow.parquet as pq
    flt = [('step', '<=', MAXSTEP)] + ([('ticker', 'in', list(tickers))] if tickers else [])
    d = pq.read_table(p, columns=COLS, filters=flt).to_pandas()
    d['date'] = pd.to_datetime(d['date'])
    assert d['year'].max() <= 2021 and d['step'].max() <= MAXSTEP, f'{p}: строки >= 2022'
    d = d[d.step.isin(steps)]
    return d.sort_values(KEY).reset_index(drop=True), p


def default_inputs(h, seed, tickers, steps, need=('leg_1', 'base_1', 'base_5', 'base_22')):
    I, files = {}, {}
    src = CFG['inputs_default']
    if h != 1:
        I['base'], files['base'] = read_series(src[f'base_{h}'], seed, h, tickers, steps)
    else:
        I['base'], files['base'] = read_series(src['base_1'], seed, 1, tickers, steps)
    for nm in need:
        I[nm], files[nm] = read_series(src[nm], seed, h, tickers, steps)
    I['ref'], files['ref'] = read_series(src['ref'], seed, h, tickers, steps)
    return I, files


def perturb_inputs(I, cut, seed):
    out = {}
    for i, (k, d) in enumerate(sorted(I.items())):
        out[k] = perturb_frame(d, cut, seed + 1000 + i, cols=['rv_pred'])
    return out


# ================================================================ кластер 6
def _ols0(X, y):
    X = np.atleast_2d(X.T).T
    if len(y) < 30:
        return np.zeros(X.shape[1])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _q_profile(y, lnf):
    f = np.exp(lnf)
    x = y / f
    x = x / x.mean()
    return float(np.mean(x - np.log(x) - 1))


def persist(F, ctx, mode, h, meta):
    r = (np.log(F['base_1'].values) - np.log(F['leg_1'].values))
    if ctx.plant == 'look1':  # подлог: r̂₁ дня d+1
        r = pd.Series(r).groupby(F['ticker'].values).shift(-1).fillna(pd.Series(r)).values
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
        par = np.clip(par, RHO_LO, RHO_HI) if mode != 'sqrtRQ' else par
        m = step == s
        if mode == 'pos_neg':
            rho = np.where(r[m] > 0, par[0], par[1])
        elif mode == 'sqrtRQ':
            rho = par[0] + par[1] * q[m]
        else:
            rho = par[0]
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


def daily_hist(ctx):
    if 'dh' not in ctx._c:
        rv = ctx.A['rv_d'].astype('float64').clip(lower=FLOOR).values
        Hd = ctx.lagmat(rv, 22)
        X = np.column_stack([np.log(Hd[:, -1]), np.log(Hd[:, -5:].mean(1)), np.log(Hd.mean(1))])
        ctx._c['dh'] = (Hd, X)
    return ctx._c['dh']


def scheme_iterated(F, ctx, h, k, meta):
    Hd, X = daily_hist(ctx)
    y = np.log(ctx.tgt(1, 1) + EPS)
    kk = min(int(k), h)
    ai, step = F['ai'].values, F['step'].values
    far = np.exp(ctx.at_steps(ctx.har_sub(kk + 1, h)[0], ai, step)) if kk < h else 0.0
    it = np.empty(len(F))
    for s in sorted(np.unique(step)):
        m_tr = ctx.trmask(2015 + s, 1, y)
        c = np.linalg.lstsq(np.column_stack([np.ones(m_tr.sum()), X[m_tr]]), y[m_tr], rcond=None)[0]
        m = step == s
        a_ = ai[m]
        base = c[0] + X[a_] @ c[1:]
        delta = np.log(F['leg_1'].values[m]) - base
        P = iterate(Hd[a_], np.log(F['base_1'].values[m]), delta, c, kk)
        fr = far[m] if kk < h else 0.0
        it[m] = (P.sum(1) + (h - kk) * fr) / h
        meta.setdefault(s, {})['iter_c'] = c
    F['p'] = 0.5 * (it + F['p'].values)


def joint_ln(ctx, h):
    k = ('joint', h)
    if k not in ctx._c:
        HS = (1, 5, 22)

        def X_of(hh):
            n = len(ctx.H)
            return np.column_stack([ctx.H, ctx.H * np.log(hh), np.full(n, float(hh == 5)), np.full(n, float(hh == 22))])
        out, co = {}, {}
        for s in ctx.steps:
            Xs, ys = [], []
            for hh in HS:
                y = np.log(ctx.tgt(1, hh) + EPS)
                m = ctx.trmask(2015 + s, hh, y)
                Xs.append(X_of(hh)[m]); ys.append(y[m])
            lr = LinearRegression().fit(np.vstack(Xs), np.concatenate(ys))
            out[s], co[s] = lr.predict(X_of(h)), np.r_[lr.intercept_, lr.coef_]
        ctx._c[k] = (out, co)
    return ctx._c[k]


def slope_ln(ctx, h):
    k = ('slope', h)
    if k not in ctx._c:
        kk = 5 if h == 22 else 1
        y = np.log(ctx.tgt(1, h) + EPS) - np.log(ctx.tgt(1, kk) + EPS)
        out, co = {}, {}
        for s in ctx.steps:
            m = ctx.trmask(2015 + s, h, y)
            lr = LinearRegression().fit(ctx.H[m], y[m])
            out[s], co[s] = lr.predict(ctx.H), np.r_[lr.intercept_, lr.coef_]
        ctx._c[k] = (out, co)
    return ctx._c[k]


def reconcile_mint(F, ctx, h, meta):
    ai, step = F['ai'].values, F['step'].values
    sub = lambda a, b: np.exp(ctx.at_steps(ctx.har_sub(a, b)[0], ai, step))
    p5 = F['p'].values if h == 5 else F['base_5'].values
    p22 = F['p'].values if h == 22 else F['base_22'].values
    Yh = np.column_stack([p5, p22, F['base_1'].values, sub(2, 5), sub(6, 22)])
    Ya = np.column_stack([ctx.tgt(1, 5), ctx.tgt(1, 22), ctx.tgt(1, 1), ctx.tgt(2, 5), ctx.tgt(6, 22)])[ai]
    row = 0 if h == 5 else 1
    out = F['p'].values.copy()
    S = S_MINT
    for s in sorted(np.unique(step)):
        hist = (step < s) & np.isfinite(Ya).all(1) & np.isfinite(Yh).all(1) & ctx.emb_ok(22, 2015 + s)[ai]
        if hist.sum() < 50:
            meta.setdefault(s, {})['mint_n'] = np.array([hist.sum()])
            continue
        E = Ya[hist] / Yh[hist] - 1
        Sig = LedoitWolf().fit(E).covariance_
        Si = np.linalg.inv(Sig)
        m = step == s
        Dinv = 1.0 / Yh[m]
        Winv = Si[None, :, :] * Dinv[:, :, None] * Dinv[:, None, :]
        M = np.einsum('ki,nkl,lj->nij', S, Winv, S)
        v = np.einsum('ki,nkl,nl->ni', S, Winv, Yh[m])
        b = np.linalg.solve(M, v[..., None])[..., 0]
        out[m] = np.clip(b @ S[row], 1e-10, None)
        meta.setdefault(s, {})['mint_sigma'] = np.r_[Sig.ravel(), hist.sum()]
    F['p'] = out


def apply_mh(design, ctx, I):
    """Кластер 6 над рядом I['base'] (h дизайна). Возвращает (ряд COLS, meta {шаг: {имя: массив}})."""
    d = full_design(design)
    h, m = int(d['h']), d['mh']
    need = check_mh(d)
    miss = [k for k in need if k not in I]
    if miss:
        raise ValueError(f'нет входов {miss}')
    F = I['base'][COLS].copy().sort_values(KEY).reset_index(drop=True)
    F['ai'] = ctx.rows(F)
    for nm in need:
        x = I[nm][KEY + ['rv_pred']].rename(columns={'rv_pred': nm})
        F = F.merge(x, on=KEY, how='left', validate='one_to_one')
        assert F[nm].notna().all(), f'вход {nm}: нет {F[nm].isna().sum()} строк'
    F['p'] = F['rv_pred'].astype('float64')
    meta = {}
    if h == 1:  # persist / reconcile — фиктивные
        return F[COLS].assign(rv_pred=F['p']), meta
    ai, step = F['ai'].values, F['step'].values
    sch = m['scheme']
    if sch == 'decomp':
        if m['decomp_k'] == 1:
            near, parts = F['base_1'].values, [(2, 5, 4)] + ([(6, 22, 17)] if h == 22 else [])
        else:
            near, parts = 5 * F['base_5'].values, [(6, 22, 17)]
        F['p'] = (near + sum(w * np.exp(ctx.at_steps(ctx.har_sub(a, b)[0], ai, step)) for a, b, w in parts)) / h
    elif sch == 'iterated':
        scheme_iterated(F, ctx, h, m['iterated_k'], meta)
    elif sch == 'joint_h':
        F['p'] = 0.5 * (np.exp(ctx.at_steps(joint_ln(ctx, h)[0], ai, step)) + F['p'].values)
    elif sch == 'perlead':
        F['p'] = (F['base_1'].values + sum(np.exp(ctx.at_steps(ctx.har_sub(j, j)[0], ai, step)) for j in range(2, 6))) / 5
    if m['slope'] == 'on':
        F['p'] = (F['base_5'] if h == 22 else F['base_1']).values * np.exp(ctx.at_steps(slope_ln(ctx, h)[0], ai, step))
    if m['persist'] != 'off':
        persist(F, ctx, m['persist'], h, meta)
    if m['reconcile'] == 'MinT':
        reconcile_mint(F, ctx, h, meta)
    for nm, D in (('har', None),):
        pass
    for k, v in ctx._c.items():  # коэффициенты подмоделей шагов — в meta (детектор эмбарго, диагностика D)
        if isinstance(v, tuple) and len(v) == 2 and isinstance(v[1], dict):
            for s, c in v[1].items():
                meta.setdefault(s, {})['_'.join(map(str, k))] = c
    assert np.isfinite(F['p']).all() and (F['p'] > 0).all(), 'нечисловой прогноз'
    return F[COLS].assign(rv_pred=F['p'].values), meta


# ================================================================ кластер 11: нога τ × g
def _beta_w(W, om):
    k = np.arange(W)
    w = (1 - k / W) ** (om - 1)
    return w / w.sum()


def struct_leg(design, ctx, ref):
    """Нога кластера 11 на строках ref (val шагов ctx.steps). Возвращает (ряд COLS, lnall, meta)."""
    d = full_design(design)
    check_struct(d)
    h, lg = int(d['h']), d['leg']
    st, split, W, ext = lg['struct'], lg['struct_split'], int(lg['tau_window']), lg['tau_extend']
    A, n = ctx.A, len(ctx.A)
    rv = A['rv'].astype('float64').values
    jmp = A['jump'].astype('float64').clip(lower=0).values
    Z = rv if split == 'rv_whole' else np.clip(rv - jmp, 0, None)
    tk = A['ticker'].values
    Zp = pd.Series(np.where(Z > 0, Z, np.nan)).groupby(tk).ffill().fillna(FLOOR).clip(lower=FLOOR).values
    if ctx.plant == 'look1':
        Zp = pd.Series(Zp).groupby(tk).shift(-1).fillna(pd.Series(Zp)).values
    L = max(66, W)
    Hz = ctx.lagmat(Zp, L)
    zd, zw, zm = np.log(Hz[:, -1]), np.log(Hz[:, -5:].mean(1)), np.log(Hz[:, -22:].mean(1))
    Jx = ctx.H[:, 3:]
    g = pd.Series(Z).groupby(tk)
    y = np.log(sum(g.shift(-j) for j in range(1, h + 1)).values / h + EPS)
    meta, ln_pred = {}, {}

    def fit_g(s, lntau):
        m = ctx.trmask(2015 + s, h, y)
        lnbar = tau_ext(ctx, lntau, s, h, ext, meta)
        X = np.column_stack([zd - lntau, zw - lntau, zm - lntau, Jx])
        yg = y - lnbar
        lr = LinearRegression().fit(X[m], yg[m])
        pr = lnbar + lr.predict(X)
        return pr, float(np.mean((y[m] - pr[m]) ** 2)), np.r_[lr.intercept_, lr.coef_]

    for s in ctx.steps:
        if st == 'single':
            m = ctx.trmask(2015 + s, h, y)
            X = np.column_stack([zd, zw, zm, Jx])
            lr = LinearRegression().fit(X[m], y[m])
            ln_pred[s] = lr.predict(X)
            meta.setdefault(s, {})['single'] = np.r_[lr.intercept_, lr.coef_]
            continue
        if st == 'tau_midas':
            best = None
            for om in CFG['struct']['omega_grid']:
                lntau = np.log(Hz[:, -W:] @ _beta_w(W, om)[::-1])
                pr, mse, co = fit_g(s, lntau)
                if best is None or mse < best[1]:
                    best = (pr, mse, co, om)
            ln_pred[s] = best[0]
            meta.setdefault(s, {})['midas'] = np.r_[best[3], best[2]]
        elif st == 'tau_rvi':
            lntau = tau_rvi(ctx, Hz, W, s, meta)
            ln_pred[s], _, co = fit_g(s, lntau)
            meta.setdefault(s, {})['g'] = co
        elif st == 'tau_boost':
            lntau = tau_boost(ctx, Zp, Hz, W, s, int(d['seed']), meta)
            ln_pred[s], _, co = fit_g(s, lntau)
            meta.setdefault(s, {})['g'] = co
        else:
            raise ValueError(st)
    pred = {s: np.exp(ln_pred[s]) for s in ctx.steps}
    if split == 'bv_plus_jump':
        Jh = ctx.lagmat(jmp, 22)
        XJ = np.column_stack([Jh[:, -1], Jh[:, -5:].mean(1), Jh.mean(1)])
        gj = pd.Series(jmp).groupby(tk)
        yj = (sum(gj.shift(-j) for j in range(1, h + 1)) / h).values
        for s in ctx.steps:
            m = ctx.trmask(2015 + s, h, yj)
            lr = LinearRegression().fit(XJ[m], yj[m])
            pred[s] = pred[s] + np.clip(lr.predict(XJ), 0, None)
            meta.setdefault(s, {})['jump'] = np.r_[lr.intercept_, lr.coef_]
    R = ref[COLS].copy().sort_values(KEY).reset_index(drop=True)
    ai, step = ctx.rows(R), R['step'].values
    R['rv_pred'] = np.clip(ctx.at_steps(pred, ai, step), 1e-10, None)
    assert np.isfinite(R['rv_pred']).all()
    # lnall (формат ног L): tr — годы < vy с целью, va — строки ref шага; emb_ok — конец цели < 1 янв. vy
    la = []
    for s in ctx.steps:
        vy = 2015 + s
        tr = np.where((ctx.year < vy) & np.isfinite(y))[0]
        va = ai[step == s]
        for part, idx in (('tr', tr), ('va', va)):
            la.append(pd.DataFrame({'date': A['date'].values[idx], 'ticker': tk[idx].astype(str), 'step': s, 'part': part,
                                    'base_ln': np.log(np.clip(pred[s][idx], 1e-10, None)), 'emb_ok': ctx.tend(h)[idx] < vy}))
    return R[COLS], pd.concat(la, ignore_index=True), meta


def tau_ext(ctx, lntau, s, h, ext, meta):
    """ln τ̄ за [t+1, t+h] из τ_t: const — τ_t; AR1 — пул φ по окну шага с тикерными средними."""
    if ext == 'const':
        return lntau
    vy = 2015 + s
    tk = ctx.A['ticker'].values
    m = (ctx.year < vy) & ctx.emb_ok(1, vy)
    mu = pd.Series(np.where(ctx.year < vy, lntau, np.nan)).groupby(tk).transform('mean').fillna(np.nanmean(lntau[ctx.year < vy])).values
    nxt = pd.Series(lntau).groupby(tk).shift(-1).values
    x, yv = lntau - mu, nxt - mu
    ok = m & np.isfinite(x) & np.isfinite(yv)
    phi = float(np.clip((x[ok] * yv[ok]).sum() / max((x[ok] ** 2).sum(), 1e-12), 0.0, 0.9999))
    meta.setdefault(s, {})['ar1'] = np.array([phi])
    pw = phi ** np.arange(1, h + 1)
    return np.log(np.exp(mu[:, None] + x[:, None] * pw[None, :]).mean(1))


def tau_rvi(ctx, Hz, W, s, meta):
    A, vy = ctx.A, 2015 + s
    tk = A['ticker'].values
    rvi = pd.Series(A['close_RVI'].astype('float64').values).groupby(tk).ffill().values
    x = np.log(rvi ** 2)
    flat = np.log(Hz[:, -W:].mean(1))
    ok = np.isfinite(x)
    D = pd.get_dummies(pd.Categorical(tk, categories=sorted(set(tk)))).values.astype('float64')
    X = np.column_stack([D, np.where(ok, x, 0.0)])
    m = (ctx.year < vy) & ok
    co = np.linalg.lstsq(X[m], flat[m], rcond=None)[0]
    meta.setdefault(s, {})['rvi'] = co
    return np.where(ok, X @ co, flat)


def tau_boost(ctx, Zp, Hz, W, s, seed, meta):
    import lightgbm as lgb
    A, vy = ctx.A, 2015 + s
    tk = A['ticker'].values
    rvi = pd.Series(A['close_RVI'].astype('float64').values).groupby(tk).ffill().values
    imx = pd.Series(A['rv_IMOEX'].astype('float64').values).groupby(tk).ffill()
    imx22 = imx.groupby(tk).transform(lambda v: v.rolling(22, min_periods=1).mean()).values
    codes = pd.Categorical(tk, categories=sorted(set(tk))).codes.astype('float64')
    X = np.column_stack([np.log(Hz[:, -5:].mean(1)), np.log(Hz[:, -22:].mean(1)), np.log(Hz[:, -66:].mean(1)), np.log(Hz[:, -W:].mean(1)),
                         np.log(rvi ** 2), np.log(np.clip(imx22, 1e-12, None)), codes])
    g = pd.Series(Zp).groupby(tk)
    yt = np.log((sum(g.shift(-j) for j in range(1, W + 1)) / W).values + EPS)
    P = dict(CFG['struct']['lgbm'])
    nest = int(P.pop('n_estimators'))
    P.update(objective='regression', seed=seed)

    def fit(mask):
        ds = lgb.Dataset(X[mask], yt[mask], categorical_feature=[6], free_raw_data=False)
        return lgb.train(P, ds, num_boost_round=nest)
    m_all = ctx.trmask(vy, W, yt)
    bst = fit(m_all)
    lntau = bst.predict(X)
    tend = ctx.tend(W)
    for Y in sorted(set(ctx.year[m_all])):  # кросс-фит на строках обучения: год Y — модель без года Y и без целей, заходящих в Y
        mY = m_all & (ctx.year != Y) & (tend != Y)
        rowsY = m_all & (ctx.year == Y)
        if mY.sum() > 100 and rowsY.any():
            lntau[rowsY] = fit(mY).predict(X[rowsY])
    meta.setdefault(s, {})['boost'] = np.array([m_all.sum(), float(np.mean(lntau[m_all]))])
    return lntau


# ================================================================ охрана: сдвиг (п. 2.6) и детектор
def shift_check(out, ctx):
    """Спирмен ln ŷ_d с ln RV_{d+s}, s ∈ {−2, −1, 0, +1}, val 2017–2021, по тикерам, медиана. argmax должен быть 0."""
    g = CFG['guards']['shift_check']
    d = out[(out.year >= g['years'][0]) & (out.year <= g['years'][1])].copy()
    if d.empty:
        return {'ok': None, 'argmax': None}
    ai = ctx.rows(d)
    gr = ctx.A.groupby('ticker')['rv']
    res = {}
    for s in g['s']:
        rvs = gr.shift(-s).values[ai]
        cs = []
        for t, x in d.assign(_r=rvs).groupby('ticker'):
            ok = (x._r > 0) & np.isfinite(x._r)
            if ok.sum() > 20:
                cs.append(spearmanr(np.log(x.rv_pred[ok]), np.log(x._r[ok])).correlation)
        res[s] = float(np.median(cs)) if cs else np.nan
    am = max(res, key=lambda k: res[k])
    return {'ok': am == g['need_argmax'], 'argmax': am, **{f'c{s:+d}': v for s, v in res.items()}}


def fingerprint(meta, steps):
    h = hashlib.sha256()
    for s in sorted(k for k in meta if k in steps):
        for nm in sorted(meta[s]):
            h.update(nm.encode()); h.update(np.ascontiguousarray(np.asarray(meta[s][nm], dtype='float64')).tobytes())
    return h.hexdigest()


def run_one(design, ctx, I):
    d = full_design(design)
    if is_struct(d):
        out, _, meta = struct_leg(d, ctx, I['ref'])
    else:
        out, meta = apply_mh(d, ctx, I)
    return out, meta


def _maxdl(a, b, cut):
    m = a.merge(b, on=KEY, suffixes=('', '_p'))
    m = m[m.date <= pd.Timestamp(cut)]
    return float(np.abs(np.log(m.rv_pred) - np.log(m.rv_pred_p)).max()) if len(m) else np.nan


class LeakBench:
    """Контексты детектора на один срез (тикеры, шаги): основа, шум после средних срезов, шум после 31.12, подлоги."""

    def __init__(self, tickers, steps, seed=42):
        L = CFG['guards']['leak']
        self.mid, self.ye = L['cuts_mid'], L['cut_year_end']
        self.ctx = {'base': Ctx(tickers, steps, seed=seed)}
        for i, c in enumerate(self.mid):
            self.ctx[c] = Ctx(tickers, steps, perturb=(c, 20261026 + i), seed=seed)
        self.ctx[self.ye] = Ctx(tickers, steps, perturb=(self.ye, 20261099), seed=seed)
        self.ctx['look1'] = Ctx(tickers, steps, perturb=(self.mid[0], 20261026), plant='look1', seed=seed)
        self.ctx['noemb'] = Ctx(tickers, steps, perturb=(self.ye, 20261099), plant='noemb', seed=seed)

    def check(self, design, I):
        d = full_design(design)
        base, mb = run_one(d, self.ctx['base'], I)
        rows = []
        for i, c in enumerate(self.mid):
            o, _ = run_one(d, self.ctx[c], perturb_inputs(I, c, 20261026 + i))
            dl = _maxdl(base, o, c)
            rows.append(dict(test=f'srez {c}', delta=dl, ok=bool(dl <= CFG['guards']['leak']['tol'])))
        yv = pd.Timestamp(self.ye).year + 1
        st_ok = [s for s in self.ctx['base'].steps if 2015 + s <= yv]
        o, mp = run_one(d, self.ctx[self.ye], perturb_inputs(I, self.ye, 20261099))
        same = fingerprint(mb, st_ok) == fingerprint(mp, st_ok)
        rows.append(dict(test=f'embargo {self.ye} (param. shagov {st_ok})', delta=0.0 if same else 1.0, ok=bool(same)))
        trivial = not mb  # прямой ряд без оценок: подлогу нечего ловить
        o, _ = run_one(d, self.ctx['look1'], perturb_inputs(I, self.mid[0], 20261026))
        dl = _maxdl(base, o, self.mid[0])
        rows.append(dict(test='podlog look1', delta=dl, ok=bool(trivial or dl > 0)))
        o, mp = run_one(d, self.ctx['noemb'], perturb_inputs(I, self.ye, 20261099))
        caught = fingerprint(mb, st_ok) != fingerprint(mp, st_ok)
        rows.append(dict(test='podlog noemb', delta=float(caught), ok=bool(trivial or caught)))
        return base, rows


# ================================================================ запись
def write_series(df, path, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(['step', 'date', 'ticker']).reset_index(drop=True)[COLS]
    assert df['year'].max() <= 2021
    df.to_parquet(path, index=False)
    return sha_file(path)


def write_meta(meta, path):
    rows = [dict(step=s, name=nm, values=json.dumps([float(x) for x in np.ravel(v)])) for s in sorted(meta) for nm, v in sorted(meta[s].items())]
    pd.DataFrame(rows).to_csv(path, index=False)


def index_row(f, ryad, h, seed, steps, rows, sha, dh, note):
    return dict(fayl=str(Path(f).relative_to(REG)) if str(f).startswith(str(REG)) else str(f), ryad=ryad, h=h, seed=seed,
                shagi=f'{min(steps)}-{max(steps)}', stroki=rows, sha256=sha, sha_koda=sha_file(__file__)[:16],
                yaml=f'{YAML.relative_to(REPO)}@{sha_file(YAML)[:16]}', sha_design=dh[:16], git=git_head(), vremya=now(), primechanie=note)


def append_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode='a', header=not path.exists(), index=False)


# ================================================================ режимы
def selftest():
    # MinT: согласованный вход не меняется; несогласованный — становится согласованным
    rng = np.random.default_rng(1)
    b = rng.uniform(1e-4, 1e-3, (50, 3))
    Y = b @ S_MINT.T
    Si = np.linalg.inv(np.diag([1.0, 2.0, 0.5, 0.7, 1.5]))
    Dinv = 1.0 / Y
    Winv = Si[None] * Dinv[:, :, None] * Dinv[:, None, :]
    M = np.einsum('ki,nkl,lj->nij', S_MINT, Winv, S_MINT)
    v = np.einsum('ki,nkl,nl->ni', S_MINT, Winv, Y)
    bt = np.linalg.solve(M, v[..., None])[..., 0]
    assert np.abs(bt / b - 1).max() < 1e-10, 'MinT меняет согласованный вход'
    Y2 = Y * rng.uniform(0.8, 1.2, Y.shape)
    v = np.einsum('ki,nkl,nl->ni', S_MINT, Si[None] / Y2[:, :, None] / Y2[:, None, :], Y2)
    M = np.einsum('ki,nkl,lj->nij', S_MINT, Si[None] / Y2[:, :, None] / Y2[:, None, :], S_MINT)
    bt = np.linalg.solve(M, v[..., None])[..., 0]
    yt = bt @ S_MINT.T
    assert np.abs(yt[:, 0] - (yt[:, 2] + 4 * yt[:, 3]) / 5).max() < 1e-15 and np.abs(yt[:, 1] - (yt[:, 2] + 4 * yt[:, 3] + 17 * yt[:, 4]) / 22).max() < 1e-15
    # ρ: OLS и QLIKE восстанавливают подложенный ρ
    r = rng.normal(0, 0.3, 20000)
    lnp = rng.normal(-7, 0.5, 20000)
    y = np.exp(lnp + 0.6 * r + rng.normal(0, 0.4, 20000))
    assert abs(_ols0(r, np.log(y) - lnp)[0] - 0.6) < 0.02
    rq = minimize_scalar(lambda c: _q_profile(y, lnp + c * r), bounds=(RHO_LO, RHO_HI), method='bounded').x
    assert abs(rq - 0.6) < 0.05, rq
    # веса MIDAS
    for W in (22, 44, 66):
        for om in CFG['struct']['omega_grid']:
            w = _beta_w(W, om)
            assert abs(w.sum() - 1) < 1e-12 and (np.diff(w) <= 1e-15).all()
    # схема: дополнение, валидация, хэш
    d = {'design': 'x', 'h': 5, 'mh': {'persist': 'qlike'}}
    validate_design(d)
    assert design_hash(d) == design_hash({'design': 'y', 'h': 5, 'mh': {'persist': 'qlike', 'reconcile': 'off'}})
    assert design_hash(d) != design_hash({'h': 5, 'mh': {'persist': 'OLS'}})
    for bad in ({'h': 5, 'mh': {'persist': 'xx'}}, {'h': 7}, {'h': 5, 'mh': {'nope': 1}}):
        try:
            validate_design(bad)
            raise AssertionError(f'не поймано: {bad}')
        except ValueError:
            pass
    for na in ({'h': 1, 'mh': {'scheme': 'decomp', 'decomp_k': 1}}, {'h': 5, 'mh': {'scheme': 'iterated', 'iterated_k': 10}},
               {'h': 22, 'mh': {'scheme': 'perlead'}}, {'h': 5, 'mh': {'scheme': 'decomp', 'decomp_k': 5}},
               {'h': 22, 'mh': {'scheme': 'joint_h', 'slope': 'on'}}):
        try:
            check_mh(full_design(na))
            raise AssertionError(f'не поймано: {na}')
        except NotApplicable:
            pass
    assert check_mh(full_design({'h': 1, 'mh': {'persist': 'qlike', 'reconcile': 'MinT'}})) == set()
    log('selftest OK (MinT, ρ, MIDAS, схема, применимость)', 'ядро:', 'r2_core' if CORE else 'fallback')


def control(steps):
    """HAR[1..h] модуля (d0, эмбарго по концу цели) = нога L HARJ_emb_d0_h{h} на 17 тикерах."""
    ctx = Ctx(None, steps)
    rows = []
    for h in (1, 5, 22):
        ref, p = read_series(CFG['inputs_default']['ref'], 42, h, None, steps)
        ai = ctx.rows(ref)
        ln = ctx.at_steps(ctx.har_sub(1, h)[0], ai, ref.step.values)
        dl = float(np.abs(ln - np.log(ref.rv_pred.values)).max())
        y = ctx.tgt(1, h)[ai]
        ok = np.isfinite(y)
        dy = float(np.abs(y[ok] / ref.rv_actual.values[ok] - 1).max())
        rows.append(dict(h=h, rows=len(ref), max_dln=dl, max_rel_target=dy, ok=dl <= CFG['guards']['control']['tol'] and dy < 1e-9))
        log(f'контроль h={h}: max|Δ ln| = {dl:.2e}, цель {dy:.2e}', 'OK' if rows[-1]['ok'] else 'ОШИБКА')
    return pd.DataFrame(rows)


def dym():
    D = CFG['dym']
    tick, steps, seed = D['tickers'], tuple(D['steps']), int(D['seed'])
    out = ROOT / CFG['out']['dym']
    out.mkdir(parents=True, exist_ok=True)
    selftest()
    ctl = control(steps)
    ctl.to_csv(out / 'kontrol.csv', index=False)
    bench = LeakBench(tick, steps, seed)
    summ, leak_rows, sdvig = [], [], []
    for h in D['h']:
        I, files = default_inputs(h, seed, tick, steps)
        for spec in D['designs']:
            if spec.get('only_h') and h not in spec['only_h']:
                continue
            des = {'design': spec['name'], 'h': h, 'seed': seed, **{k: v for k, v in spec.items() if k in ('mh', 'leg')}}
            t0 = time.time()
            try:
                validate_design(des)
                dfull = full_design(des)
                (check_struct if is_struct(dfull) else check_mh)(dfull)
            except NotApplicable as e:
                summ.append(dict(h=h, design=spec['name'], status='ne_primenim', note=str(e)))
                continue
            base, lr = bench.check(des, I)
            sc = shift_check(base, bench.ctx['base'])
            sha = write_series(base, out / f'{spec["name"]}_h{h}_s{seed}_val.parquet')
            ratio = base.merge(I['base'][KEY + ['rv_pred']], on=KEY, suffixes=('', '_in'))
            rr = np.log(ratio.rv_pred / ratio.rv_pred_in)
            fict = h == 1 and not is_struct(dfull)
            summ.append(dict(h=h, design=spec['name'], status='ok', rows=len(base), nan=int(base.rv_pred.isna().sum()),
                             dln_vs_input_mean=float(rr.mean()), dln_vs_input_sd=float(rr.std()), fiktivnyy=fict,
                             leak_ok=all(r['ok'] for r in lr), shift_ok=sc['ok'], shift_argmax=sc['argmax'], sec=round(time.time() - t0, 1),
                             sha_design=design_hash(des)[:16], sha_file=sha[:16]))
            for r in lr:
                leak_rows.append(dict(h=h, design=spec['name'], **r))
            sdvig.append(dict(h=h, design=spec['name'], **sc))
            log(f'h={h} {spec["name"]}: строк {len(base)}, детектор {"OK" if summ[-1]["leak_ok"] else "ОШИБКА"}, '
                f'сдвиг argmax={sc["argmax"]} {"OK" if sc["ok"] else "БЛОК"}, {summ[-1]["sec"]} с')
    pd.DataFrame(summ).to_csv(out / 'dym_summary.csv', index=False)
    pd.DataFrame(leak_rows).to_csv(out / 'leak.csv', index=False)
    pd.DataFrame(sdvig).to_csv(out / 'sdvig.csv', index=False)
    S = pd.DataFrame(summ)
    ok = S[S.status == 'ok']
    log(f'дым: {len(ok)} контуров, детектор {int(ok.leak_ok.sum())}/{len(ok)}, сдвиг {int((ok.shift_ok == True).sum())}/{len(ok)}, '
        f'не применимо {int((S.status == "ne_primenim").sum())}, контроль {int(ctl.ok.sum())}/{len(ctl)}')


def load_designs(p):
    Y = yaml.safe_load(Path(p).read_text())
    return Y['designs'] if isinstance(Y, dict) and 'designs' in Y else (Y if isinstance(Y, list) else [Y])


def run(designs, steps, seed, tickers=None, inputs_yaml=None):
    """Боевой счёт (сервер): ряд дизайна -> R2/cache/<sha16>/M_h{h}_s{seed}_val.parquet; сдвиг (блок -> R2/O/blok_sdvig.csv); INDEX_M."""
    ctx = Ctx(tickers, steps, seed=seed)
    extra = yaml.safe_load(Path(inputs_yaml).read_text()) if inputs_yaml else {}
    idx = []
    for des in load_designs(designs):
        des = {**des, 'seed': des.get('seed', seed)}
        validate_design(des)
        d = full_design(des)
        h = int(d['h'])
        (check_struct if is_struct(d) else check_mh)(d)
        I, files = default_inputs(h, seed, tickers, steps)
        for k, rel in (extra.get(h, {}) or extra.get(str(h), {}) or {}).items():
            I[k], files[k] = read_series(rel, seed, h, tickers, steps)
        dh = design_hash(des)
        cd = cache_dir(des)
        if is_struct(d):
            out, la, meta = struct_leg(d, ctx, I['ref'])
            la.to_parquet(cd / f'M_h{h}_lnall.parquet', index=False) if cd.mkdir(parents=True, exist_ok=True) is None else None
        else:
            out, meta = apply_mh(d, ctx, I)
        sc = shift_check(out, ctx)
        f = cd / f'M_h{h}_s{seed}_val.parquet'
        sha = write_series(out, f)
        write_meta(meta, cd / f'M_h{h}_s{seed}_meta.csv')
        (cd / 'design.yaml').write_text(yaml.safe_dump(d, sort_keys=True, allow_unicode=True))
        if not sc['ok']:
            append_csv([dict(ryad=f'M:{des.get("design", "")}', h=h, sha_design=dh[:16], **sc, vremya=now())], ROOT / CFG['out']['blok_sdvig'])
            log(f'БЛОК по сдвигу: {des.get("design")} h={h} argmax={sc["argmax"]} — доклад дирижёру')
        idx.append(index_row(f, des.get('design', dh[:16]), h, seed, steps, len(out), sha, dh,
                             ('BLOK_sdvig ' if not sc['ok'] else '') + 'vhody: ' + ', '.join(f'{k}={Path(v).name}' for k, v in sorted(files.items()))))
        log(f'{des.get("design")} h={h}: {len(out)} строк -> {f.relative_to(ROOT)}')
    append_csv(idx, ROOT / CFG['out']['index'])


def leak_cmd(designs, steps, tickers, seed):
    bench = LeakBench(tickers, steps, seed)
    rows = []
    for des in load_designs(designs):
        h = int(des['h'])
        I, _ = default_inputs(h, seed, tickers, steps)
        _, lr = bench.check({**des, 'seed': seed}, I)
        rows += [dict(design=des.get('design'), h=h, **r) for r in lr]
    R = pd.DataFrame(rows)
    print(R.to_string(index=False))
    assert R.ok.all(), 'детектор: ОШИБКА'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['selftest', 'control', 'dym', 'run', 'leak'])
    ap.add_argument('--designs')
    ap.add_argument('--inputs')
    ap.add_argument('--steps', type=int, nargs='+', default=list(STEPS))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tickers', nargs='+')
    a = ap.parse_args()
    if a.cmd == 'selftest':
        selftest()
    elif a.cmd == 'control':
        print(control(tuple(a.steps)).to_string(index=False))
    elif a.cmd == 'dym':
        dym()
    elif a.cmd == 'run':
        run(a.designs, tuple(a.steps), a.seed, a.tickers, a.inputs)
    elif a.cmd == 'leak':
        leak_cmd(a.designs, tuple(a.steps), a.tickers, a.seed)


if __name__ == '__main__':
    main()
