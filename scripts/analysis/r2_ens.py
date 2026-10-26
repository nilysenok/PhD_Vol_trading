#!/usr/bin/env python3
"""R2, агент E: ансамбли (кластеры 7, 8) и смесь экспертов (кластер 13). Протокол v3 docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md
(раздел 6, строка «E»); схема models/designs/r2/schema_v1.yaml (ключи ens.*, leg.moe*); параметры — models/designs/r2/r2_E_ens.yaml.

Ряд — val-формат: date, ticker, year, step, rv_actual, rv_pred (сырой, без s²); шаг k — val-год vy = 2015 + k, обучение — годы < vy с
эмбарго h (конец окна цели < 1 янв. vy). Всё, что оценивается по прошлым ошибкам (κ в ec, веса leg_ens, стек, калибровка привратника),
на шаге t берётся по OOS-строкам шагов < t с концом цели < 1 янв. vy. Годы >= 2022 не читаются (фильтр при чтении и assert).

Уровень ноги (выход — ряд ноги для ролей B / смеси):
  window_ens  — нога (HAR-J / HAR-X МНК) на трёх окнах: расширяющееся, скользящее 3 и 5 лет; среднее уровней
  ticker_bag  — нога-пул без одного тикера × 10 (16 из 17), среднее уровней
  leg_ens     — equal | median | qlike_shrunk_05 | qlike_shrunk_075: w = λ/K + (1−λ)·w*, w* — мин. QLIKE на симплексе по прошлым OOS
  moe (leg.moe k2|k3, moe_gate logit|boost, moe_calib isotonic|none, moe_hard off|0.7) — эксперты HAR-X по режимам с усадкой к общему,
                привратник по признакам дня t, изотонная калибровка на прошлых OOS привратника, мягкая смесь (порог 0,7 — жёсткая)
Уровень итога (после ролей B / многогоризонтных M), порядок: seeds → lib_agg/agg → uncert_shrink → stack → ec → range_guard
  agg mean|median по seed × lib; lib_agg avg | best_by_LOYO (варианты lib=…); uncert_shrink w = 1/(1+Var_seed/τ) к ноге (варианты tau=…);
  stack nested_oos — NNLS в логах на прошлых OOS; ec m5|m10 — Д23.1 (d23_ec.ebar_from_e): ln ŷ += κ_h·ē_m; range_guard — [min, max]
  ln цели тикера на обучении шага.

Обвязка на каждом контуре: проверка сдвига (п. 2.6: Спирмен ln ŷ_d с ln RV_{d+s}, s ∈ {−2..+1}, 2017–2021, медиана по тикерам,
argmax ≠ 0 — блок) и детектор утечек (как Д26: возмущение всего после среза; Δ на строках <= среза = 0; подлог пойман).

Ядро — r2_core.py (агент S): validate / design_hash / get (схема и хэш), load_d0 / train_mask (d0, эмбарго), write_series / run_phase
(кэш, пул, DONE), shift_check (п. 2.6, блок в R2/O/blok_sdvig.csv), leak_check / perturb_frame / plant_frame (детектор контуров
с переобучением). Детектор контуров уровня итога (на рядах) — свой: ядро возмущает матрицу, а эти контуры читают готовые ряды.
Функция фазы для r2_core.run_phase — fn_E(design, seed, steps) (данные — r2_core.SHARED: E_M, по желанию E_inputs, E_tickers).

python3 scripts/analysis/r2_ens.py selftest
python3 scripts/analysis/r2_ens.py dym                        # Мак: 3 тикера × шаги 1–3 × seed 42 × h 1, 5, 22 (+ run_phase дыма)
python3 scripts/analysis/r2_ens.py leak [--h 1 5 22] [--full] # детектор (по умолчанию — объём дыма)
python3 scripts/analysis/r2_ens.py run --design <yaml> [--h 5] [--dym]   # сервер: дизайн → r2_core.run_phase('E', …) → кэш"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
import argparse, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import r2_core as C  # noqa: E402  ядро агента S

CFG = yaml.safe_load((REPO / 'models' / 'designs' / 'r2' / 'r2_E_ens.yaml').read_text())
R = Path(os.environ['R2_DATA']) if os.environ.get('R2_DATA') else C.REGEN
OUT = R / 'R2' / 'E'
MAXY = int(CFG['max_year'])
MAXSTEP = MAXY - 2015
STEPS = tuple(CFG['steps'])
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
K = ['date', 'ticker']
KEY = ['date', 'ticker', 'step']
SMAP = {'S1': ['ln_rvi2'], 'S2': ['ln_rvi2', 'ln_rvm_imoex']}
SMAP['S3'] = SMAP['S2'] + ['lhar_1', 'lhar_5', 'lhar_22', 'pc1']
SMAP['S4'] = SMAP['S3'] + ['harq', 'ln_rsv_pos', 'ln_rsv_neg']
SMAP['S5'] = SMAP['S4'] + ['ln_rv66', 'ln_rv132']
SEEDS = {int(k): list(v) for k, v in CFG['ens']['seeds'].items()}
TAUS = tuple(float(t) for t in CFG['ens']['uncert_shrink']['tau'])
LAM = {k: float(v['lam']) for k, v in CFG['ens']['leg_ens'].items() if isinstance(v, dict) and 'lam' in v}
WINS = {'expanding': None, 'rolling3y': 3, 'rolling5y': 5}


def now():
    return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S'], capture_output=True, text=True).stdout.strip()


def qlike(y, p):
    r = y / np.clip(p, 1e-12, None)
    return float(np.mean(r - np.log(r) - 1))


def q_years(df, lo=2017, hi=MAXY):
    d = df[(df.year >= lo) & (df.year <= hi) & (df.rv_actual > 1e-12)].dropna(subset=['rv_pred'])
    return {int(y): qlike(g.rv_actual.values, g.rv_pred.values) for y, g in d.groupby('year')}


# ================================================================ схема и хэш
MOE_KEYS = ('moe', 'moe_gate', 'moe_calib', 'moe_hard')
ENS_KEYS = tuple(p[1] for p in C.load_schema().leaves if p[0] == 'ens')


def canon_hash(design):
    """SHA-256 канонического yaml — r2_core.design_hash (единое правило с O)."""
    return C.design_hash(design)


def onoff(v):
    """Схема читается YAML 1.1: off/on — False/True; r2_core.validate нормализует дизайн так же. Для модуля — строки 'off'/'on'."""
    return {False: 'off', True: 'on'}.get(v, v) if isinstance(v, bool) else v


def ens_cfg(design):
    """r2_core.validate (неизвестный ключ, значение вне списка — DesignError до счёта) -> ключи ens.* и leg.moe*.
    Скаляр-сетка (ключ списком) у ens.* / leg.moe* не поддерживается — выбор внутри складки здесь только по вариантам (tau, lib).
    λ = 0 (веса без усадки) — ошибка."""
    full = C.validate(design)
    out = {k: onoff(C.get(full, f'ens.{k}')) for k in ENS_KEYS}
    out.update({k: onoff(C.get(full, f'leg.{k}')) for k in MOE_KEYS})
    for k, v in out.items():
        if isinstance(v, list):
            raise C.DesignError(f'{k}: сетка внутри складки для ключей E не предусмотрена')
    if out['leg_ens'].startswith('qlike_shrunk') and LAM.get(out['leg_ens'], 0) <= 0:
        raise C.DesignError('веса без усадки запрещены')
    return out


# ================================================================ чтение
def read_val(path, tickers=None, steps=None):
    """Val-ряд: только шаги <= 6 (фильтр pyarrow) и годы <= 2021 (assert)."""
    d = pd.read_parquet(path, columns=COLS, filters=[('step', '<=', MAXSTEP)])
    d['date'] = pd.to_datetime(d.date)
    d['year'] = d.date.dt.year.astype(int)
    d['ticker'] = d.ticker.astype(str)
    assert d.step.max() <= MAXSTEP and d.year.max() <= MAXY, f'{path}: год > {MAXY}'
    assert (d.step == d.year - 2015).all(), f'{path}: не val-формат'
    if tickers:
        d = d[d.ticker.isin(tickers)]
    if steps:
        d = d[d.step.isin(steps)]
    return d.sort_values(KEY).reset_index(drop=True)[COLS]


def inp(name, h, seed=42, lib=None, **kw):
    s = CFG['inputs'][name]
    if isinstance(s, dict):
        s = s[lib]
    return read_val(R / s.format(h=h, seed=seed), **kw)


def load_d0():
    """r2_core.load_d0: матрица d0 (rv_d = RV дня d), строки <= 2021, _tend{h}; проверка info_set = d."""
    A, harj = C.load_d0()
    C.check_d0(A)
    assert A.date.dt.year.max() <= MAXY
    return A, harj


def prep(A, harj):
    """Матрица модуля: A (d0, возможно возмущённая/с подлогом ядра) + слагаемые HAR-X (harx.TERMS, без ln_sig1) + ln HAR-J."""
    import harx as HX
    A = A.copy()
    A['date'] = pd.to_datetime(A.date)
    A['year'] = A.date.dt.year
    HX._SIG.clear(); HX.GH_HARJ = harj
    X = HX.build_terms(A, sorted(set(sum(SMAP.values(), []))))
    H = np.nan_to_num(np.log(A[harj].values.clip(1e-10)), nan=0, posinf=0, neginf=0)
    return dict(A=A, harj=harj, X=X, H=H)


def trmask(M, h, vy, plant=False):
    """r2_core.train_mask (эмбарго h); plant — подлог «обучение видит val-год» (для детектора контуров на рядах)."""
    A = M['A']
    if plant:
        return (A.year.values <= vy) & A[f'rv_target_h{h}'].notna().values
    return C.train_mask(A, h, vy)


def leg_terms(design, h):
    lg = design.get('leg') or {}
    if lg.get('type', 'HARJ') == 'HARJ':
        return []
    r = lg.get('regressors', 'none')
    return SMAP[r] if r in SMAP else [t for t in SMAP['S5' if h == 22 else 'S4']]


def _z(M, terms, tr):
    if not terms:
        return None
    X = M['X'][terms]
    return X.fillna(X.loc[tr].mean()).fillna(0.0).values.astype('float64')


def fit_ln(M, h, terms, tr):
    """ln-прогноз ноги (HAR-J / HAR-X МНК, пул) на всех строках M; подгонка на строках tr."""
    import harx as HX
    y = np.log(M['A'][f'rv_target_h{h}'].values + 1e-10)
    H, Z = M['H'], _z(M, terms, tr)
    if Z is None:
        return LinearRegression().fit(H[tr], y[tr]).predict(H)
    return HX.fit_en(H[tr], Z[tr], y[tr], 0.0, 1.0)[0](H, Z)


def _val(M, h, k, pred, tickers=None):
    A = M['A']; vy = 2015 + k; t = f'rv_target_h{h}'
    sel = ((A.year == vy) & A[t].notna()).values
    if tickers:
        sel &= A.ticker.isin(tickers).values
    return pd.DataFrame({'date': A.date.values[sel], 'ticker': A.ticker.values[sel].astype(str), 'year': vy, 'step': k,
                         'rv_actual': A[t].values[sel], 'rv_pred': np.clip(pred[sel], 1e-10, None)})


def _fin(V):
    return pd.concat(V, ignore_index=True).sort_values(KEY).reset_index(drop=True)[COLS]


# ================================================================ кластер 8: уровень ноги (переобучение)
def window_ens(M, h, terms, steps=STEPS, tickers=None, plant=False):
    V, P = [], []
    for k in steps:
        vy = 2015 + k; base = trmask(M, h, vy, plant)
        ps = []
        for nm, w in WINS.items():
            tr = base & (M['A'].year.values >= vy - w) if w else base
            ps.append(np.exp(fit_ln(M, h, terms, tr)))
            P.append(dict(step=k, okno=nm, n=int(tr.sum())))
        V.append(_val(M, h, k, np.mean(ps, axis=0), tickers))
    return _fin(V), pd.DataFrame(P)


def bag_drop(M, h):
    tk = sorted(M['A'].ticker.unique())
    cfg = CFG['ens']['ticker_bag']
    return list(np.random.default_rng(20261026 + h).choice(tk, size=int(cfg['n_bags']), replace=False))


def ticker_bag(M, h, terms, steps=STEPS, tickers=None, plant=False):
    drop = bag_drop(M, h); V = []
    tkv = M['A'].ticker.values
    for k in steps:
        base = trmask(M, h, 2015 + k, plant)
        V.append(_val(M, h, k, np.mean([np.exp(fit_ln(M, h, terms, base & (tkv != d))) for d in drop], axis=0), tickers))
    return _fin(V), pd.DataFrame({'bag': range(len(drop)), 'bez_tikera': drop})


# ================================================================ кластер 13: смесь экспертов
def _gate(kind, seed):
    if kind == 'boost':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05, min_child_samples=50, random_state=seed,
                              n_jobs=1, verbose=-1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def _auc(lab, P, nk):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(lab)) < nk:
        return np.nan
    return float(roc_auc_score(lab, P[:, 1]) if nk == 2 else roc_auc_score(lab, P, multi_class='ovr', average='macro'))


def moe(M, h, terms, mode, gate='logit', calib='none', hard='off', seed=42, steps=STEPS, tickers=None, plant=False):
    """-> (ряд смеси, таблица шагов: пороги, AUC привратника на отложенном годе). mode k2|k3."""
    from sklearn.isotonic import IsotonicRegression
    nk = {'k2': 2, 'k3': 3}[mode]; kap = float(CFG['moe']['kappa_shrink'])
    A = M['A']; t = f'rv_target_h{h}'
    lny = np.log(A[t].values + 1e-10); has = A[t].notna().values
    past, V, info = [], [], []
    for k in sorted(steps):
        vy = 2015 + k; tr = trmask(M, h, vy, plant)
        Zk = _z(M, terms, tr)  # пропуски — средним по обучению шага
        G = np.column_stack([M['H']] + ([Zk] if Zk is not None else []))
        med = pd.Series(lny[tr]).groupby(A.ticker.values[tr]).median()
        z = lny - A.ticker.map(med).values
        cuts = [0.0] if nk == 2 else list(np.quantile(z[tr], [1 / 3, 2 / 3]))
        lab = np.digitize(z, cuts)
        ln0 = fit_ln(M, h, terms, tr)
        lnr = [(1 - kap) * ln0 + kap * fit_ln(M, h, terms, tr & (lab == r)) for r in range(nk)]
        clf = _gate(gate, seed).fit(G[tr], lab[tr])
        P = clf.predict_proba(G)
        va = (A.year.values == vy) & has
        Pc = P.copy()
        if calib == 'isotonic' and past:
            pp = pd.concat(past, ignore_index=True)
            pp = pp[pp.fd < np.datetime64(f'{vy}-01-01')] if not plant else pp
            if len(pp) > 50:
                for r in range(nk):
                    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(pp[f'p{r}'].values, (pp.lab.values == r) * 1.0)
                    Pc[:, r] = iso.predict(P[:, r])
                s = Pc.sum(1, keepdims=True)
                Pc = np.where(s > 1e-12, Pc / np.where(s > 1e-12, s, 1), 1.0 / nk)
        lv = np.exp(np.column_stack(lnr))
        pred = (Pc * lv).sum(1)
        if hard != 'off':
            j = Pc.argmax(1); top = Pc.max(1) >= float(hard)
            pred = np.where(top, lv[np.arange(len(lv)), j], pred)
        past.append(pd.DataFrame({**{f'p{r}': P[va, r] for r in range(nk)}, 'lab': lab[va], 'fd': pd.to_datetime(A[f'_tend{h}']).values[va]}))
        V.append(_val(M, h, k, pred, tickers))
        info.append(dict(step=k, year=vy, cuts=';'.join(f'{c:.4f}' for c in cuts), auc_raw=_auc(lab[va], P[va], nk),
                         auc_calib=_auc(lab[va], Pc[va], nk), dolya_hard=float((Pc[va].max(1) >= float(hard)).mean()) if hard != 'off' else 0.0))
    return _fin(V), pd.DataFrame(info)


# ================================================================ кластеры 7, 8: уровень ряда (без переобучения)
def align(dfs):
    """Общие строки (date, ticker, step) рядов -> (рамка COLS без rv_pred, матрица прогнозов n × K). rv_actual обязаны совпадать."""
    m = dfs[0][COLS].rename(columns={'rv_pred': 'p0'})
    for i, d in enumerate(dfs[1:], 1):
        m = m.merge(d[KEY + ['rv_actual', 'rv_pred']].rename(columns={'rv_pred': f'p{i}', 'rv_actual': f'a{i}'}), on=KEY)
        ok = np.isclose(m[f'a{i}'].values, m.rv_actual.values, rtol=1e-6, atol=1e-14) | (m.rv_actual.isna() & m[f'a{i}'].isna())
        assert ok.all(), f'ряд {i}: другая цель'
        m = m.drop(columns=f'a{i}')
    m = m.sort_values(KEY).reset_index(drop=True)
    return m[['date', 'ticker', 'year', 'step', 'rv_actual']], m[[f'p{i}' for i in range(len(dfs))]].values


def _out(base, p):
    o = base.copy(); o['rv_pred'] = np.clip(p, 1e-10, None)
    return o[COLS]


def fd_of(base, h):
    b = base.sort_values(['ticker', 'date'])
    return b.groupby('ticker')['date'].shift(-h).reindex(base.index).values


def past_mask(base, fd, k, plant=False):
    vy = 2015 + k
    if plant:
        return (base.step.values <= k)
    return (base.step.values < k) & (fd < np.datetime64(f'{vy}-01-01'))


def agg(dfs, how='mean'):
    base, P = align(dfs)
    return _out(base, np.median(P, 1) if how == 'median' else P.mean(1))


def uncert_shrink(dfs, leg, tau, how='mean'):
    """ln ŷ = ln ŷ_ноги + w·(ln ŷ_анс − ln ŷ_ноги), w = 1/(1 + Var_seed/τ), Var — по ln компонент (ddof = 0)."""
    base, P = align(list(dfs) + [leg])
    C, L = np.log(np.clip(P[:, :-1], 1e-12, None)), np.log(np.clip(P[:, -1], 1e-12, None))
    ens = np.log(np.clip(np.median(P[:, :-1], 1) if how == 'median' else P[:, :-1].mean(1), 1e-12, None))
    w = 1.0 / (1.0 + C.var(1) / tau)
    return _out(base, np.exp(L + w * (ens - L))), w


def _w_qlike(P, y):
    from scipy.optimize import minimize
    Kn = P.shape[1]

    def f(th):
        w = np.exp(np.r_[0.0, th]); w /= w.sum()
        return qlike(y, P @ w)
    r = minimize(f, np.zeros(Kn - 1), method='L-BFGS-B')
    w = np.exp(np.r_[0.0, r.x]); return w / w.sum()


def leg_ens(dfs, how, h, plant=False):
    """equal | median | qlike_shrunk_*: w = λ/K + (1−λ)·w*; w* по прошлым OOS (шаги < t, конец цели < 1 янв. vy); шаг 1 — равные."""
    base, P = align(dfs)
    if how == 'equal':
        return _out(base, P.mean(1)), None
    if how == 'median':
        return _out(base, np.median(P, 1)), None
    lam = LAM[how]; Kn = P.shape[1]; fd = fd_of(base, h)
    pred = np.full(len(base), np.nan); W = []
    for k in sorted(base.step.unique()):
        cur = (base.step.values == k)
        pm = past_mask(base, fd, k, plant) & (base.rv_actual.values > 1e-12)
        ws = _w_qlike(P[pm], base.rv_actual.values[pm]) if pm.sum() > 50 else np.full(Kn, 1.0 / Kn)
        w = lam / Kn + (1 - lam) * ws
        pred[cur] = P[cur] @ w
        W.append(dict(step=int(k), n=int(pm.sum()), **{f'w{i}': float(x) for i, x in enumerate(w)}))
    return _out(base, pred), pd.DataFrame(W)


def stack(dfs, h, plant=False):
    """ln y = c + Σ b_k ln ŷ_k, b ≥ 0 (NNLS на центрированных), только прошлые OOS; шаг 1 или b ≡ 0 — среднее уровней."""
    from scipy.optimize import nnls
    base, P = align(dfs); L = np.log(np.clip(P, 1e-12, None)); fd = fd_of(base, h)
    pred = P.mean(1).copy(); W = []
    y = np.log(np.clip(base.rv_actual.values, 1e-12, None))
    for k in sorted(base.step.unique()):
        cur = base.step.values == k
        pm = past_mask(base, fd, k, plant) & (base.rv_actual.values > 1e-12)
        if pm.sum() > 50:
            mx, my = L[pm].mean(0), y[pm].mean()
            b, _ = nnls(L[pm] - mx, y[pm] - my)
            if b.sum() > 0:
                pred[cur] = np.exp(my - mx @ b + L[cur] @ b)
                W.append(dict(step=int(k), c=float(my - mx @ b), **{f'b{i}': float(x) for i, x in enumerate(b)}))
    return _out(base, pred), pd.DataFrame(W)


def ec(base_df, src1, h, m, plant=False):
    """Д23.1: e(s) = ln RV_{s+1} − ln ŷ¹(s) ряда src1 (h = 1); ē_m(t) — d23_ec.ebar_from_e (среднее e(t−1..t−m) по тикеру);
    ln ŷ_h(t) += κ_h·ē_m(t), κ_h — МНК без константы по OOS шагов < t с концом цели < 1 янв. vy (<= 50 строк — 0)."""
    import d23_ec as D23
    a = base_df.sort_values(['ticker', 'date']).reset_index(drop=True)
    s = a[K].merge(src1[K + ['rv_actual', 'rv_pred']], on=K, how='left')
    e = pd.Series(np.log(s.rv_actual.values + 1e-10) - np.log(np.clip(s.rv_pred.values, 1e-10, None)), index=a.index)
    if plant:  # подлог: ē(t) включает e(t) (её цель — RV_{t+1})
        e = e.groupby(a.ticker.values, sort=False).shift(-1)
    x = D23.ebar_from_e(a, e.values, ms=(m,))[f'ebar{m}'].values
    fd = fd_of(a, h); ln = np.log(np.clip(a.rv_pred.values, 1e-10, None))
    r = np.log(a.rv_actual.values + 1e-10) - ln
    ok = np.isfinite(r) & np.isfinite(x) & (a.rv_actual.values > 0)
    out = ln.copy(); KP = []
    for k in sorted(a.step.unique()):
        cur = a.step.values == k
        tr = ok & past_mask(a, fd, k, plant)
        kap = float(np.linalg.lstsq(x[tr, None], r[tr], rcond=None)[0][0]) if tr.sum() > 50 else 0.0
        out[cur] = ln[cur] + kap * x[cur]
        KP.append(dict(step=int(k), kappa=kap, n=int(tr.sum())))
    o = a.copy(); o['rv_pred'] = np.exp(out)
    return o.sort_values(KEY).reset_index(drop=True)[COLS], pd.DataFrame(KP)


def range_guard(df, M, h, plant=False):
    """ln ŷ шага k обрезается в [min, max] ln цели тикера на обучении шага (годы < vy, эмбарго h)."""
    A = M['A']; t = f'rv_target_h{h}'
    o = df.copy(); lnp = np.log(np.clip(o.rv_pred.values, 1e-12, None)); B = []
    for k in sorted(o.step.unique()):
        tr = trmask(M, h, 2015 + k, plant)
        g = pd.Series(np.log(A[t].values[tr] + 1e-10)).groupby(A.ticker.values[tr]).agg(['min', 'max'])
        cur = (o.step.values == k)
        lo = o.ticker[cur].map(g['min']).values; hi = o.ticker[cur].map(g['max']).values
        lnp[cur] = np.clip(lnp[cur], np.where(np.isfinite(lo), lo, -np.inf), np.where(np.isfinite(hi), hi, np.inf))
        B.append(g.assign(step=int(k)).reset_index().rename(columns={'index': 'ticker'}))
    o['rv_pred'] = np.exp(lnp)
    return o[COLS], pd.concat(B, ignore_index=True)


# ================================================================ дизайн → ряды
def post(cfg, h, hyb, leg=None, legs=(), src1=None, M=None):
    """Уровень итога. hyb: {(lib, seed): ряд}; leg — ряд ноги дизайна (uncert_shrink); legs — ряды для стека; src1 — ряд h=1 (ec).
    -> {вариант: ряд}. Порядок: seeds → lib_agg/agg → uncert_shrink → stack → ec → range_guard."""
    seeds = SEEDS[int(cfg['seeds'])]
    sel = {kk: v for kk, v in hyb.items() if kk[1] in seeds}
    assert sel, 'нет компонент с нужными seed'
    libs = sorted({kk[0] for kk in sel})
    groups = {'': list(sel.values())} if (cfg['lib_agg'] == 'avg' or len(libs) == 1) else \
        {f'lib={lb}': [v for kk, v in sel.items() if kk[0] == lb] for lb in libs}
    out = {}
    for gname, comps in groups.items():
        if cfg['uncert_shrink'] == 'on':
            assert leg is not None, 'uncert_shrink: нужен ряд ноги'
            cur = {f'{gname}|tau={t:g}'.lstrip('|'): uncert_shrink(comps, leg, t, cfg['agg'])[0] for t in TAUS}
        else:
            cur = {gname: agg(comps, cfg['agg'])}
        for nm in list(cur):
            d = cur[nm]
            if cfg['stack'] == 'nested_oos':
                d = stack([d] + list(legs), h)[0]
            if cfg['ec'] != 'off':
                d = ec(d, src1, h, int(cfg['ec'][1:]))[0]
            if cfg['range_guard'] == 'on':
                d = range_guard(d, M, h)[0]
            out[nm or 'ens'] = d
    return out


def leg_level(cfg, design, h, M, legs=(), seed=42, steps=STEPS, tickers=None):
    """Уровень ноги -> {имя: (ряд, таблица)}: window_ens, ticker_bag, leg_ens, moe (ключи, отличные от default)."""
    terms = leg_terms(design, h); out = {}
    if cfg['window_ens'] == 'on':
        out['window_ens'] = window_ens(M, h, terms, steps, tickers)
    if cfg['ticker_bag'] == 'on':
        out['ticker_bag'] = ticker_bag(M, h, terms, steps, tickers)
    if cfg['leg_ens'] != 'single':
        out[f'leg_ens={cfg["leg_ens"]}'] = leg_ens(list(legs), cfg['leg_ens'], h)
    if cfg['moe'] != 'off':
        out[f'moe={cfg["moe"]}'] = moe(M, h, terms, cfg['moe'], cfg['moe_gate'], cfg['moe_calib'], cfg['moe_hard'], seed, steps, tickers)
    return out


# ================================================================ проверка сдвига (п. 2.6)
def sdvig(df, M, h, name, design=None, block_file=None):
    """r2_core.shift_check (п. 2.6) на матрице d0; блок — строка в block_file (по умолчанию R2/O/blok_sdvig.csv)."""
    r = C.shift_check(df, M['A'], h, name, design, block_file=block_file)
    r['blok'] = not r['ok']
    return r


# ================================================================ детектор утечек
def tmul(dfs, rng):
    """Общий множитель цели U(0,5; 2) на (date, ticker): у рядов с одной целью возмущённая цель совпадает."""
    k = pd.concat([d[K] for d in dfs]).drop_duplicates().sort_values(K).reset_index(drop=True)
    return k.assign(_m=rng.uniform(0.5, 2.0, len(k)))


def perturb_series(df, cut, h, rng, tm=None):
    """Всё после среза × U(0,5; 2): rv_pred и rv_actual строк > среза и rv_actual строк <= среза, чьё окно цели заходит за срез."""
    tm = tmul([df], rng) if tm is None else tm
    o = df.copy(); fd = fd_of(o, h)
    aft = (o.date > cut).values
    tgt = aft | ~(fd <= np.datetime64(cut))
    m = o[K].merge(tm, on=K, how='left')['_m'].values
    o.loc[aft, 'rv_pred'] = o.rv_pred.values[aft] * rng.uniform(0.5, 2.0, aft.sum())
    o.loc[tgt, 'rv_actual'] = o.rv_actual.values[tgt] * m[tgt]
    return o


def _dd(a, b, cut):
    m = a.merge(b, on=KEY, suffixes=('', '_p'))
    d = np.abs(np.log(m.rv_pred.values) - np.log(m.rv_pred_p.values)); p = (m.date <= cut).values
    return (float(d[p].max()) if p.any() else 0.0), (float(d[~p].max()) if (~p).any() else 0.0)


def _plant_shift(df):
    o = df.sort_values(['ticker', 'date']).copy()
    o['rv_pred'] = o.groupby('ticker')['rv_pred'].shift(-1).fillna(o.rv_pred)
    return o.sort_values(KEY).reset_index(drop=True)


def leak_posthoc(h, comps, leg, src1, M, Mp, cut, rng):
    """Контуры уровня итога на рядах: исходные → возмущённые после среза; подлог — свой у каждого контура."""
    tm = tmul(list(comps) + [leg, src1], rng)
    pc = [perturb_series(c, cut, h, rng, tm) for c in comps]
    pl = perturb_series(leg, cut, h, rng, tm)
    ps = perturb_series(src1, cut, 1, rng, tm)
    T = {
        'agg_mean': (lambda C, L, S, MM, pl_: agg([_plant_shift(C[0])] + C[1:] if pl_ else C, 'mean')),
        'agg_median': (lambda C, L, S, MM, pl_: agg([_plant_shift(C[0])] + C[1:] if pl_ else C, 'median')),
        'uncert_shrink': (lambda C, L, S, MM, pl_: uncert_shrink([_plant_shift(C[0])] + C[1:] if pl_ else C, L, 0.01)[0]),
        'leg_ens_qlike_shrunk_05': (lambda C, L, S, MM, pl_: leg_ens(C + [L], 'qlike_shrunk_05', h, pl_)[0]),
        'stack': (lambda C, L, S, MM, pl_: stack(C + [L], h, pl_)[0]),
        'ec_m5': (lambda C, L, S, MM, pl_: ec(agg(C), S, h, 5, pl_)[0]),
        'range_guard': (lambda C, L, S, MM, pl_: range_guard(agg(C), MM, h, pl_)),
    }
    kmax = cut.year - 2015
    rows = []
    for nm, f in T.items():
        a0, b0 = f(comps, leg, src1, M, False), f(pc, pl, ps, Mp, False)
        a1, b1 = f(comps, leg, src1, M, True), f(pc, pl, ps, Mp, True)
        p0 = p1 = 0.0
        if isinstance(a0, tuple):  # range_guard: граница не всегда активна — сверяются и границы шагов <= года среза
            bd = lambda x, y: float(np.abs(x[0][x[0].step <= kmax][['min', 'max']].values - y[0][y[0].step <= kmax][['min', 'max']].values).max())
            p0, p1 = bd(a0[1:], b0[1:]), bd(a1[1:], b1[1:])
            a0, b0, a1, b1 = a0[0], b0[0], a1[0], b1[0]
        d0, f0 = _dd(a0, b0, cut); d1, _ = _dd(a1, b1, cut)
        rows.append(dict(h=h, kontur=nm, srez=str(cut.date()), max_d_do_sreza=max(d0, p0), max_d_posle=f0,
                         podlog_max_d_do_sreza=max(d1, p1), ok=bool(d0 == 0.0 and p0 == 0.0 and max(d1, p1) > 0)))
    return rows


def leak_refit(h, A, harj, cuts, design, steps, tickers, seed):
    """Контуры уровня ноги (переобучение) — r2_core.leak_check: perturb_frame после среза, подлог plant_frame («вход d := d+1»)."""
    terms = leg_terms(design, h)

    def build(AA):
        MM = prep(AA, harj)
        return {'window_ens': window_ens(MM, h, terms, steps, tickers)[0],
                'ticker_bag': ticker_bag(MM, h, terms, steps, tickers)[0],
                'moe_k2_logit_iso': moe(MM, h, terms, 'k2', 'logit', 'isotonic', 'off', 42, steps, tickers)[0],
                'moe_k3_boost_hard': moe(MM, h, terms, 'k3', 'boost', 'none', 0.7, 42, steps, tickers)[0]}
    L = C.leak_check(build, A, f'E_h{h}', cuts=cuts, seed=seed)
    return [dict(h=h, kontur=r.ryad, srez=r.srez, max_d_do_sreza=r.max_dln_do, max_d_posle=r.max_dln_posle,
                 podlog_max_d_do_sreza=r.get('podlog_max_dln_do', np.nan), ok=bool(r.ok)) for _, r in L.iterrows()]


# ================================================================ дым и входы дыма
DYM_DESIGN = {1: {'leg': {'type': 'HARX', 'regressors': 'S4'}}, 5: {'leg': {'type': 'HARX', 'regressors': 'S3'}},
              22: {'leg': {'type': 'HARJ'}}}


VARIANTS = [  # (имя, отличие от default) — по одному ключу, плюс сочетания для h = 5 (две lib)
    ('base', {}), ('agg=median', {'agg': 'median'}), ('lib_agg=avg', {'lib_agg': 'avg'}),
    ('uncert_shrink', {'uncert_shrink': 'on', 'lib_agg': 'avg'}), ('ec=m5', {'ec': 'm5'}), ('ec=m10', {'ec': 'm10'}),
    ('range_guard', {'range_guard': 'on'}), ('stack', {'stack': 'nested_oos'}),
]
LEG_VARIANTS = [('window_ens', {'window_ens': 'on'}), ('ticker_bag', {'ticker_bag': 'on'}),
                ('leg_ens=equal', {'leg_ens': 'equal'}), ('leg_ens=median', {'leg_ens': 'median'}),
                ('leg_ens=qlike_shrunk_05', {'leg_ens': 'qlike_shrunk_05'}), ('leg_ens=qlike_shrunk_075', {'leg_ens': 'qlike_shrunk_075'}),
                ('moe=k2|logit', {'moe': 'k2'}), ('moe=k2|logit|isotonic', {'moe': 'k2', 'moe_calib': 'isotonic'}),
                ('moe=k3|boost|isotonic|0.7', {'moe': 'k3', 'moe_gate': 'boost', 'moe_calib': 'isotonic', 'moe_hard': 0.7})]


def _cfg(diff, h):
    d = dict(DYM_DESIGN[h]); d['h'] = h
    e = {k: v for k, v in diff.items() if k in ENS_KEYS}
    if e:  # пустой словарь r2_core.validate считает листом «ens»
        d['ens'] = e
    d['leg'] = dict(d['leg'], **{k: v for k, v in diff.items() if k in MOE_KEYS})
    return d, ens_cfg(d)


def fname(nm):
    return nm.replace('|', '_').replace('=', '-')


def fn_E(design, seed, steps):
    """Функция фазы для r2_core.run_phase: полный дизайн -> {ряд: DataFrame}. Данные — r2_core.SHARED: E_M (prep), E_tickers (дым);
    E_inputs(design, h, seeds, steps, tickers) -> (hyb {(lib, seed): ряд}, legs [ряды], src1 ряд h=1) — от ядра/оркестрации;
    без него — inputs r2_E_ens.yaml (ряды R2/cache до кэша по хэшу)."""
    cfg = ens_cfg(design); h = int(design['h']); M = C.SHARED['E_M']; tk = C.SHARED.get('E_tickers')
    seeds = SEEDS[int(cfg['seeds'])]
    if C.SHARED.get('E_seeds_only'):
        seeds = [s for s in seeds if s in C.SHARED['E_seeds_only']]
    getter = C.SHARED.get('E_inputs') or yaml_inputs
    hyb, legs, src1 = getter(design, h, seeds, list(steps), tk)
    lref = legs[2] if h != 22 else legs[0]
    res = {f'E_{fname(k)}': v for k, v in post(cfg, h, hyb, lref, legs, src1, M).items()}
    res.update({f'E_{fname(k)}': v[0] for k, v in leg_level(cfg, design, h, M, legs, int(seed), list(steps), tk).items()})
    return res


def yaml_inputs(design, h, seeds, steps, tickers):
    """Входы по inputs r2_E_ens.yaml: гибрид — F_d0 (h = 1, 22) или F5_D12_h5 xgb + lgbm (h = 5); ноги L — HARJ, HEAVY, HARX_ols."""
    kw = dict(tickers=tickers, steps=steps)
    hyb = {}
    for s in seeds:
        hyb.update({(lb, s): inp('boost_h5', h, s, lb, **kw) for lb in ('xgboost', 'lightgbm')} if h == 5 else
                   {('xgboost', s): inp('F_d0', h, s, **kw)})
    legs = [inp(n, h, **kw) for n in ('leg_HARJ', 'leg_HEAVY', 'leg_HARX')]
    src1 = inp('F_d0', 1, seeds[0], **kw) if h != 1 else hyb[sorted(hyb)[0]]
    return hyb, legs, src1


def dym():
    c = CFG['dym']; tk, st, seed = list(c['tickers']), list(c['steps']), int(c['seed'])
    od = OUT / 'dym'; od.mkdir(parents=True, exist_ok=True)
    blk = od / 'blok_sdvig_dym.csv'
    t0 = time.time(); A, harj = load_d0(); M = prep(A, harj)
    Q, SD, AUC = [], [], []

    def rec(h, name, df, ref, design):
        df.to_parquet(od / f'{fname(name)}_h{h}_s{seed}_val.parquet', index=False)
        qy, qr = q_years(df), q_years(ref)
        Q.append(dict(h=h, ryad=name, hash=canon_hash(design)[:16], n=len(df), **{f'q{y}': qy.get(y) for y in sorted(qy)},
                      **{f'otn{y}': qy[y] / qr[y] for y in sorted(qy) if y in qr}))
        SD.append(sdvig(df, M, h, name, design, blk))

    for h in CFG['dym']['h']:
        hyb, legs, src1 = yaml_inputs(None, h, [seed], st, tk)
        ref = agg(list(hyb.values()))
        for nm, diff in VARIANTS:
            d, cfg = _cfg(diff, h)
            for vn, df in post(cfg, h, hyb, leg=legs[2] if h != 22 else legs[0], legs=legs, src1=src1, M=M).items():
                rec(h, f'{nm}|{vn}' if vn != 'ens' else nm, df, ref, d)
        lref = legs[2] if h != 22 else legs[0]
        for nm, diff in LEG_VARIANTS:
            d, cfg = _cfg(diff, h)
            for vn, (df, tab) in leg_level(cfg, d, h, M, legs, seed, st, tk).items():
                rec(h, nm, df, lref, d)
                if vn.startswith('moe') and tab is not None:
                    AUC.append(tab.assign(h=h, ryad=nm))
        print(f'h={h} готов, {time.time() - t0:.0f} с', flush=True)
    Q = pd.DataFrame(Q); S = pd.DataFrame(SD); AU = pd.concat(AUC, ignore_index=True)
    Q.to_csv(od / 'dym_qlike.csv', index=False); S.to_csv(od / 'dym_sdvig.csv', index=False); AU.to_csv(od / 'dym_moe_auc.csv', index=False)
    print(Q.round(4).to_string()); print(S.round(3).to_string()); print(AU.round(3).to_string())
    P = run(CFG['dym']['design'], hs=CFG['dym']['h'], dym_scope=True, A=A, harj=harj, M=M)  # r2_core.run_phase на дыме
    L = leak(hs=CFG['dym']['h'], full=False, A=A, harj=harj)
    ok = bool(L.ok.all()) and not S.blok.any() and bool(P.sdvig_ok.all()) and 'ОШИБКА' not in set(P.ryad)
    print(f'дым: рядов {len(Q)} + run_phase {len(P)}, блок сдвига {int(S.blok.sum())}, детектор {"OK" if L.ok.all() else "ОШИБКА"}, '
          f'{time.time() - t0:.0f} с')
    return ok


def leak(hs=(1, 5, 22), full=False, A=None, harj=None):
    """Детектор: контуры с переобучением — r2_core.leak_check; контуры на рядах — свой (возмущение рядов и матрицы после среза)."""
    c = CFG['dym']; lc = CFG['leak']
    tk = None if full else list(c['tickers']); st = list(STEPS) if full else list(c['steps']); seed = int(c['seed'])
    if A is None:
        A, harj = load_d0()
    M = prep(A, harj)
    cuts = [x for x in lc['cuts_mid'] + lc['cuts_year_end'] if pd.Timestamp(x).year - 2015 <= max(st)]
    rows = []
    for h in hs:
        rows += leak_refit(h, A, harj, cuts, DYM_DESIGN[h], st, tk, int(lc['rng']) + h)
        hyb, legs, src1 = yaml_inputs(None, h, [seed], st, tk)
        comps = list(hyb.values()) + ([legs[1]] if len(hyb) == 1 else [])  # ≥ 2 компоненты для agg/uncert
        for i, cs in enumerate(lc['cuts_mid']):
            cut = pd.Timestamp(cs)
            if cut.year - 2015 > max(st):
                continue
            rng = np.random.default_rng(int(lc['rng']) + 100 * h + i)
            Mp = prep(C.perturb_frame(A, cut, int(lc['rng']) + 1000 + h), harj)
            rows += leak_posthoc(h, comps, legs[0], src1, M, Mp, cut, rng)
        print(f'детектор h={h} готов', flush=True)
    L = pd.DataFrame(rows); (OUT / 'dym').mkdir(parents=True, exist_ok=True)
    L.to_csv(OUT / ('leak_E.csv' if full else 'dym' + os.sep + 'dym_leak_E.csv'), index=False)
    print(L.to_string()); print('ВСЕ OK' if L.ok.all() else 'ЕСТЬ ОШИБКИ')
    return L


# ================================================================ сервер: дизайн → r2_core.run_phase → кэш
def run(path, hs=None, dym_scope=False, A=None, harj=None, M=None, seeds=None, workers=None):
    """Дизайн yaml -> r2_core.run_phase('E', h, …, fn_E): запись в кэш по хэшу, shift_check каждого ряда, DONE.
    dym_scope — объём дыма: 3 тикера × шаги 1–3 × seed 42, корень кэша R2/E/dym/cache, без ворот, блок — в R2/E/dym/."""
    design = dict(path) if isinstance(path, dict) else yaml.safe_load(Path(path).read_text())
    if A is None:
        A, harj = load_d0()
    M = M or prep(A, harj)
    c = CFG['dym']
    C.share(A=A, E_M=M, E_tickers=list(c['tickers']) if dym_scope else None,
            E_seeds_only=[int(c['seed'])] if dym_scope else None)
    st = list(c['steps']) if dym_scope else list(STEPS)
    root = (OUT / 'dym' / 'cache') if dym_scope else None
    T = []
    for h in (hs or [design['h']]):
        d = dict(design, h=int(h))
        T.append(C.run_phase('E', int(h), [d], fn_E, seeds=tuple(seeds or (int(d.get('seed', 42)),)), steps=st, workers=workers or 1,
                             module='E', root=root, block_file=(OUT / 'dym' / 'blok_sdvig_dym.csv') if dym_scope else None,
                             check_gate=not dym_scope))
    P = pd.concat(T, ignore_index=True)
    print(P.to_string())
    return P


# ================================================================ самопроверка (синтетика)
def selftest():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2016-01-01', '2018-12-31')
    tks = ['AAA', 'BBB']
    base = pd.DataFrame([(d, t) for d in dates for t in tks], columns=K)
    base['year'] = base.date.dt.year; base['step'] = base.year - 2015
    lv = np.exp(rng.normal(-8, 0.5, len(base)))
    base['rv_actual'] = lv
    comps = [base.assign(rv_pred=lv * np.exp(rng.normal(0, 0.3, len(base)))) for _ in range(3)]
    P = np.column_stack([c.rv_pred for c in comps])
    a = agg(comps, 'mean'); assert np.allclose(a.rv_pred, P.mean(1))
    a = agg(comps, 'median'); assert np.allclose(a.rv_pred, np.median(P, 1))
    leg = base.assign(rv_pred=lv * 1.1)
    u, w = uncert_shrink(comps, leg, 0.01)
    lnP = np.log(P); exp_w = 1 / (1 + lnP.var(1) / 0.01)
    assert np.allclose(w, exp_w) and np.allclose(np.log(u.rv_pred), np.log(lv * 1.1) + exp_w * (np.log(P.mean(1)) - np.log(lv * 1.1)))
    le, W = leg_ens(comps, 'qlike_shrunk_05', 1)
    wc = W[[c for c in W if c.startswith('w')]].values
    assert np.allclose(wc.sum(1), 1) and (wc >= 0.5 / 3 - 1e-12).all() and np.allclose(wc[0], 1 / 3)
    for bad in ({'h': 1, 'ens': {'leg_ens': 'qlike_raw'}}, {'h': 1, 'ens': {'agg': 'trimmed'}}, {'h': 1, 'ens': {'agg': ['mean', 'median']}},
                {'h': 1, 'ens': {'tau': 0.01}}):
        try:
            ens_cfg(bad); raise AssertionError(bad)
        except C.DesignError:
            pass
    cf = ens_cfg({'h': 1, 'leg': {'moe': 'k2', 'moe_hard': 0.7}, 'ens': {'range_guard': 'on', 'ec': 'off'}})
    assert ens_cfg({'h': 1})['agg'] == 'mean' and ens_cfg({'h': 1})['window_ens'] == 'off'
    assert cf['moe_hard'] == 0.7 and cf['range_guard'] == 'on' and cf['ec'] == 'off' and cf['moe'] == 'k2'
    h1 = canon_hash({'design': 'a', 'h': 1, 'ens': {'agg': 'median', 'ec': 'm5'}})
    h2 = canon_hash({'ens': {'ec': 'm5', 'agg': 'median'}, 'h': 1, 'design': 'b'})
    h3 = canon_hash({'h': 1, 'ens': {'agg': 'median', 'ec': 'm5', 'seeds': 1}})  # seeds = default
    assert h1 == h2 == h3 and h1 != canon_hash({'h': 1, 'ens': {'agg': 'median'}})
    # ec: ручной расчёт ē и κ (шаг 2 — по 2016 с концом цели < 2017)
    src = comps[0]; out, KP = ec(comps[1], src, 1, 5)
    s = src.sort_values(['ticker', 'date']).reset_index(drop=True)
    e = np.log(s.rv_actual + 1e-10) - np.log(s.rv_pred)
    eb = e.groupby(s.ticker).shift(1).groupby(s.ticker).transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(0)
    b = comps[1].sort_values(['ticker', 'date']).reset_index(drop=True)
    r = np.log(b.rv_actual + 1e-10) - np.log(b.rv_pred)
    fd = b.groupby('ticker').date.shift(-1)
    tr = (b.step == 1) & (fd < pd.Timestamp('2017-01-01'))
    kap = float((eb[tr] * r[tr]).sum() / (eb[tr] ** 2).sum())
    assert abs(KP.set_index('step').kappa[2] - kap) < 1e-10 and KP.set_index('step').kappa[1] == 0.0
    o = out.sort_values(['ticker', 'date']).reset_index(drop=True)
    m2 = (o.step == 2).values
    assert np.allclose(np.log(o.rv_pred[m2]), np.log(b.rv_pred[m2]) + kap * eb[m2])
    # детектор на синтетике: Δ = 0 до среза, подлог пойман
    cut = pd.Timestamp('2017-06-30'); r2 = np.random.default_rng(1)
    tm = tmul(comps, r2)
    pc = [perturb_series(c, cut, 1, r2, tm) for c in comps]
    for f in (lambda C, p: leg_ens(C, 'qlike_shrunk_075', 1, p)[0], lambda C, p: stack(C, 1, p)[0],
              lambda C, p: ec(agg(C[1:]), C[0], 1, 10, p)[0]):
        d0, _ = _dd(f(comps, False), f(pc, False), cut); d1, _ = _dd(f(comps, True), f(pc, True), cut)
        assert d0 == 0.0 and d1 > 0, (d0, d1)
    # сдвиг: прогноз = RV_d (s = 0) — argmax 0; прогноз = RV_{d+1} — +1 (блок)
    import tempfile
    Mx = {'A': base[K].assign(rv=lv)}
    sh = base.assign(rv_pred=lv)
    with tempfile.TemporaryDirectory() as td:
        assert sdvig(sh, Mx, 1, 's0', block_file=Path(td) / 'b.csv')['argmax'] == 0
        nx = sh.copy(); nx['rv_pred'] = nx.groupby('ticker').rv_pred.shift(-1).fillna(1e-4)
        assert sdvig(nx, Mx, 1, 's+1', block_file=Path(td) / 'b.csv')['blok'] and (Path(td) / 'b.csv').exists()
    print('selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['selftest', 'dym', 'leak', 'run'])
    ap.add_argument('--h', type=int, nargs='*', default=None)
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--design')
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--dym', action='store_true', help='run: объём дыма, кэш в R2/E/dym/cache/, без ворот')
    a = ap.parse_args()
    if a.mode == 'selftest':
        selftest()
    elif a.mode == 'dym':
        sys.exit(0 if dym() else 1)
    elif a.mode == 'leak':
        sys.exit(0 if leak(tuple(a.h or (1, 5, 22)), a.full).ok.all() else 1)
    else:
        P = run(a.design, a.h, a.dym, seeds=a.seeds, workers=a.workers)
        sys.exit(0 if 'ОШИБКА' not in set(P.ryad) else 1)


if __name__ == '__main__':
    main()
