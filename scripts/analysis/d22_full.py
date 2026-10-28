#!/usr/bin/env python3
"""Д22 «Полный гибрид v3», слои 3–4, оценка, абляции, детектор утечек (h ∈ {5, 22}). Слои 1–2 — d22_harxp.py, d22_h1.py.
Слой 3: ln ŷ_h = Σ β_j x_j, все x_j — OOS (файлы шагов слоёв 1–2; год Y берётся из val-файла шага k = Y − 2015, тест — из тест-файла шага):
  x1 = ln HAR-X⁺_h;  x6 = ln итерированного hyb₁ на h дней;  x2 = ln[(k·S_k + (h−k)·HX⁺_{k+1..h}) / h], S_k — среднее итерированного hyb₁ за
  дни 1..k, HX⁺_{2..5}, HX⁺_{6..22} — HAR-X⁺ подгоризонтов (h=22: HX⁺_{2..22} = (4·HX⁺_{2..5} + 17·HX⁺_{6..22})/21); k ∈ {1, 5} для h=22
  (по валидации t−1), k = 1 для h=5 (k = 5 совпадает с x6);  x3 = r̂₁⁺, r̂₁⁻, r̂₁·q (q = √(n·rkurt/3) дня t — √RQ/RV);  x4 = m̂_t;  x5 = d̂ = 2p − 1;
  x7 (h=22, решение владельца 01:40) = ln Д16.5 (среднее ln XGB и ln LGBM того же seed).
  Итерированный hyb₁: день 1 = hyb₁(t); дни j ≥ 2 — дневной HAR с текущим днём (ln RV(s+1) ~ ln RV(s), ln RV_5(s), ln RV_22(s), окно шага,
  эмбарго) по ряду RV, где неизвестные дни заменены прогнозами, плюс смещение δ_t = ln HAR-X⁺₁(t) − HAR_дн(t) (вклад слагаемых X в t,
  держится постоянным).
  Объединитель: на OOS-строках лет < Y (эмбарго: последний день горизонта < Y; тест шага k — годы ≤ Y_val шага k):
  min mean((ln y − Σβx)²) + 0,01·λ·‖β − β₀‖²_sd, β₀ = (β1 = 1, остальные 0) («HAR_h» = x1 = ln HAR-X⁺_h; отдельного HAR-J в объединителе нет), отклонения в единицах СКО вклада, β1 + β2 + β6 (+ β7) = 1, β ≥ 0 в этой группе; x3–x5 без знака; сдвиг к β1 = 1;
  λ ∈ {0,1; 1; 10; 100} и k — по QLIKE года Y−1 (обучение на годах < Y−1); первый год без истории — λ = 1, k = 1; год 2016 — ŷ = x1.
Слой 4: защита Д14 (ln RV_m(t) вне [1; 99] перцентиля окна шага -> ŷ = x1), винзоризация итогового ln ŷ по [0,5; 99,5] перцентилю
  ln RV_h окна шага, поправка s² (Б0.1, как harx.add_bias); усадки и cap нет.
python3 scripts/analysis/d22_full.py --run [--seeds 42 43 44 0] | --ablate | --leak | --export"""
import argparse, os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harx as HX
import d22_harxp as P

R = HX.REGEN
OUT = R / 'D22'
LOCK = R / 'test_locked'
CODE = {5: 'sm', 22: 'smrob'}
LAMS = (0.1, 1.0, 10.0, 100.0)  # решение владельца 02:25: + 100
USE_X9 = True
NO_X5_H22 = True  # решение владельца 02:25: классификатор x5 на h=22 исключён из объединителя
KEY = ['date', 'ticker', 'step']
GROUP = ('x1', 'x2', 'x6', 'x7', 'x9')
# x9 — прошедшие кандидаты Д24 (yaml в final3/kandidaty, решение владельца 02:15): h=5 — D24_6_rgarch, h=22 — D24_6_d18r; в группе с суммой 1
X9 = {5: 'D24_6_rgarch', 22: 'D24_6_d18r'}
X9_STEP10 = {5: 'D24_6/step10', 22: None}
FREE0 = ('r1p', 'r1n', 'r1q', 'm1', 'd')
FREE = FREE0 + ('e5', 'e10', 'e22')  # x8 (Д23.1, решение владельца 01:55): ē₅ / ē₁₀ (m по валидации t−1), h=22 — ещё ē₂₂
EBAR = R / 'D23_ebar.parquet'
ALLX = GROUP + FREE
_A = {}
RV_FLOOR = 1e-6
CENTER = False
PEN = 0.01  # масштаб штрафа: λ·PEN·Σ(β_j·sd_j)² в единицах дисперсии ln (λ = 10 -> вес 0,1)


def matrix():
    if 'A' not in _A:
        A, harj = P.load()
        A = A.sort_values(['ticker', 'date']).reset_index(drop=True)
        rv = A['rv'].astype('float64').clip(lower=RV_FLOOR)  # пол 1e-6 (дни без сделок: ln RV = −23 ломает итерацию)
        g = rv.groupby(A['ticker'])
        hist = np.column_stack([g.shift(s).values for s in range(21, -1, -1)])  # RV(t−21..t)
        for j in range(hist.shape[1] - 2, -1, -1):  # начало ряда: недостающее прошлое — ближайшее известное
            hist[:, j] = np.where(np.isnan(hist[:, j]), hist[:, j + 1], hist[:, j])
        A['_ld'] = np.log(hist[:, -1]); A['_lw'] = np.log(hist[:, -5:].mean(1)); A['_lm'] = np.log(hist.mean(1))
        A['_q'] = np.sqrt((A['n_bars'].astype('float64') * A['rkurt'].astype('float64') / 3).clip(lower=0)).clip(upper=10)
        A['_lrvm'] = np.log(A['rv_m'].astype('float64').clip(lower=1e-12))
        _A.update(A=A, harj=harj, hist=hist)
    return _A['A'], _A['hist']


def har_daily_coefs(A, step, test=False):
    """Дневной HAR с текущим днём на окне шага: годы < vy (эмбарго tend_1 < vy)."""
    vy = 2015 + step
    m = (A.year < vy) & A['rv_target_h1'].notna() & (A['_tend_1'] < vy)
    X = np.column_stack([np.ones(m.sum()), A.loc[m, ['_ld', '_lw', '_lm']].values])
    y = np.log(A.loc[m, 'rv_target_h1'].values + 1e-10)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def iterate(hist, ln_f1, delta, c, J=22):
    """Пути f_1..f_J (уровни): f_1 = exp(ln_f1); ln f_j = c0 + c_d ln W_last + c_w ln mean5(W) + c_m ln mean22(W) + δ."""
    W = hist.copy()
    F = np.empty((len(W), J))
    F[:, 0] = np.exp(np.clip(ln_f1, np.log(RV_FLOOR), np.log(0.1)))
    W = np.column_stack([W[:, 1:], F[:, 0]])
    for j in range(1, J):
        lf = c[0] + c[1] * np.log(W[:, -1]) + c[2] * np.log(W[:, -5:].mean(1)) + c[3] * np.log(W.mean(1)) + delta
        F[:, j] = np.exp(np.clip(lf, np.log(RV_FLOOR), np.log(0.1)))
        W = np.column_stack([W[:, 1:], F[:, j]])
    return F


# ---------------------------------------------------------------- входы OOS
def _rd(name, kind, lockdir):
    return pd.read_parquet((OUT / f'{name}_val.parquet') if kind == 'val' else (lockdir / f'{name}_test.parquet'))


def load_inputs(kind, base='HXP', h1src='L2_h1', clf='clf'):
    """Словарь OOS-таблиц (date, ticker, step, ...) для kind ∈ {val, test}. base — префикс слоя 1 (HXP | HJ); h1src — файл слоя 2."""
    lk1 = LOCK / ('D22_HXP' if base == 'HXP' else 'D22_HJ')
    I = {}
    for tg in ('1', '5', '22', '2_5', '6_22'):
        d = _rd(f'{base}_{tg}', kind, lk1)
        I[tg] = d[KEY + ['year', 'rv_actual', 'rv_pred']]
    lk2 = LOCK / 'D22_h1'
    I['h1'] = _rd(h1src, kind, lk2) if (OUT / f'{h1src}_val.parquet').exists() else None
    for h in (5, 22):
        p = OUT / f'{clf}_h{h}_val.parquet'
        I[f'clf{h}'] = _rd(f'{clf}_h{h}', kind, lk2) if p.exists() else None
    # x7: Д16.5 h=22 (yaml: regen data/regen, test_dir test_locked/D16_5_h22; шаг 10 — final2/step10)
    fr = []
    for m in ('xgb', 'lgbm'):
        for sfx in ('', '_seeds4344'):
            f = (R / f'D16_5_h22_{m}_h22{sfx}_val.parquet') if kind == 'val' else (LOCK / 'D16_5_h22' / f'D16_5_h22_{m}_h22{sfx}_test.parquet')
            fr.append(pd.read_parquet(f).assign(model=m))
        f10 = (R / 'final2' / 'step10' / f'D16_5_h22_{m}_h22_s10_val.parquet') if kind == 'val' else (LOCK / 'final2' / 'step10' / f'D16_5_h22_{m}_h22_s10_test.parquet')
        if f10.exists():
            fr.append(pd.read_parquet(f10).assign(model=m))
    x7 = pd.concat(fr, ignore_index=True)
    x7['ln'] = np.log(x7.rv_pred)
    x7 = x7.groupby(KEY + ['seed'], as_index=False)['ln'].mean()  # среднее ln XGB и ln LGBM
    x7 = pd.concat([x7, x7.groupby(KEY, as_index=False)['ln'].mean().assign(seed=0)], ignore_index=True)
    I['x7'] = x7.rename(columns={'ln': 'x7'})
    for h, b in X9.items():  # x9: среднее ln XGB и ln LGBM кандидата Д24 (один прогноз, без seed); шаг 10 — если есть
        fr = []
        for m in ('xgb', 'lgbm'):
            f = (R / f'{b}_{m}_h{h}_val.parquet') if kind == 'val' else (LOCK / b / f'{b}_{m}_h{h}_test.parquet')
            if f.exists():
                fr.append(pd.read_parquet(f)[KEY + ['rv_pred']].assign(model=m))
            if X9_STEP10[h]:
                f10 = (R / X9_STEP10[h] / f'{b}_{m}_h{h}_s10_val.parquet') if kind == 'val' else (LOCK / b / 'step10' / f'{b}_{m}_h{h}_s10_test.parquet')
                if f10.exists():
                    d10 = pd.read_parquet(f10)
                    d10 = d10[d10.seed == 42] if 'seed' in d10 else d10
                    fr.append(d10[KEY + ['rv_pred']].assign(model=m))
        if fr:
            x9 = pd.concat(fr, ignore_index=True); x9['x9'] = np.log(x9.rv_pred)
            I[f'x9_{h}'] = x9.groupby(KEY, as_index=False)['x9'].mean()
        else:
            I[f'x9_{h}'] = None
    if EBAR.exists():  # x8 (Д23.1) — файл агента Д23 (один на val и тест: ē по датам; на тесте 2026 нет -> 0)
        E = pd.read_parquet(EBAR)
        E['date'] = pd.to_datetime(E['date'])
        I['ebar'] = E[['date', 'ticker', 'ebar5', 'ebar10', 'ebar22_h22']].rename(columns={'ebar5': 'e5', 'ebar10': 'e10', 'ebar22_h22': 'e22'})
    else:
        I['ebar'] = None
    return I


def assemble(h, kind, seed, I, A, hist, no_h1=False):
    """Строки шагов 1–10 kind-файлов с регрессорами. no_h1=True — абляция «минус слой 2»: hyb₁ := HAR-X⁺₁ (r̂₁ = m̂ = d̂ = 0)."""
    tg = str(h)
    d = I[tg].rename(columns={'rv_pred': 'p_h'})
    d = d.merge(I['1'][KEY + ['rv_pred']].rename(columns={'rv_pred': 'p_1'}), on=KEY, how='left')
    d = d.merge(I['2_5'][KEY + ['rv_pred']].rename(columns={'rv_pred': 'p_25'}), on=KEY, how='left')
    d = d.merge(I['6_22'][KEY + ['rv_pred']].rename(columns={'rv_pred': 'p_622'}), on=KEY, how='left')
    if no_h1 or I['h1'] is None:
        d['ln_hyb1'] = np.log(d.p_1); d['r1'] = 0.0; d['m1'] = 0.0; d['d'] = 0.0
    else:
        hh = I['h1'][I['h1'].seed == seed][KEY + ['ln_hyb1', 'r1', 'm1']]
        d = d.merge(hh, on=KEY, how='left')
        c = I[f'clf{h}']
        if c is not None:
            cc = c[c.seed == seed][KEY + ['p']] if seed != 0 else c.groupby(KEY, as_index=False)['p'].mean()
            d = d.merge(cc, on=KEY, how='left'); d['d'] = 2 * d.pop('p') - 1
        else:
            d['d'] = 0.0
    if h == 22:
        x7 = I['x7'][I['x7'].seed == seed][KEY + ['x7']]
        d = d.merge(x7, on=KEY, how='left')
    key = A[['date', 'ticker']].reset_index().rename(columns={'index': 'ai'})
    d = d.merge(key, on=['date', 'ticker'], how='left', validate='many_to_one')
    ai = d['ai'].values
    d['y'] = np.log(d.rv_actual.clip(lower=1e-10))
    d['x1'] = np.log(d.p_h)
    d['r1'] = d['r1'].fillna(0.0); d['m1'] = d['m1'].fillna(0.0); d['d'] = d['d'].fillna(0.0)
    d['ln_hyb1'] = d['ln_hyb1'].fillna(np.log(d.p_1))
    d['r1p'] = d.r1.clip(lower=0); d['r1n'] = d.r1.clip(upper=0); d['r1q'] = d.r1 * A['_q'].values[ai]
    d['_lrvm'] = A['_lrvm'].values[ai]
    d['_tend'] = A[f'_tend_{tg}'].values[ai]
    S = {}
    for st in sorted(d.step.unique()):
        m = (d.step == st).values
        c = har_daily_coefs(A, st)
        a_ = ai[m]
        base = c[0] + c[1] * A['_ld'].values[a_] + c[2] * A['_lw'].values[a_] + c[3] * A['_lm'].values[a_]
        delta = np.log(d.loc[m, 'p_1'].values) - base
        F = iterate(hist[a_], d.loc[m, 'ln_hyb1'].values, delta, c, J=h)
        for k in (1, 5, h):
            S.setdefault(k, np.full(len(d), np.nan))[m] = F[:, :k].mean(1)
    d['x6'] = np.log(S[h])
    if h == 5:
        d['x2_k1'] = np.log((S[1] + 4 * d.p_25) / 5)
    else:
        d['x2_k1'] = np.log((S[1] + 4 * d.p_25 + 17 * d.p_622) / 22)
        d['x2_k5'] = np.log((5 * S[5] + 17 * d.p_622) / 22)
    if h != 22:
        d['x7'] = np.nan
    if I.get(f'x9_{h}') is not None:  # нет прогноза Д24 на строке (шаг 10 у h=22) -> x9 = x1 (вес уходит в x1)
        d = d.merge(I[f'x9_{h}'], on=KEY, how='left')
        d['x9'] = d['x9'].fillna(d['x1'])
    else:
        d['x9'] = d['x1']
    if I.get('ebar') is not None:  # x8 (Д23.1): ē по OOS-ошибкам h=1 за m дней (и ē₂₂ для h=22)
        d = d.merge(I['ebar'], on=['date', 'ticker'], how='left')
        for c in ('e5', 'e10', 'e22'):
            d[c] = d[c].fillna(0.0)
    return d


# ---------------------------------------------------------------- объединитель
def fit_comb(T, xs, lam, obj='ridge'):
    """T — обучающие строки; xs — список используемых регрессоров (первый — 'x1'). Возвращает словарь β."""
    grp = [x for x in xs if x in GROUP and x != 'x1']
    fr = [x for x in xs if x in FREE]
    z = (T.y - T.x1).values
    U = np.column_stack([(T[g] - T.x1).values for g in grp] + [T[f].values for f in fr]) if (grp or fr) else np.zeros((len(T), 0))
    if U.shape[1] == 0:
        return {'x1': 1.0}
    n = len(z)
    sd = U.std(0); sd[sd < 1e-12] = 1.0
    ng = len(grp)
    if obj == 'qlike':  # Д24.12: прямая минимизация QLIKE(y, exp(x1 + Uβ)) + тот же штраф
        P_ = lam * PEN * sd ** 2

        def fun(b):
            e = z - U @ b
            return float(np.mean(np.exp(e) - e - 1) + b @ (P_ * b))

        def jac(b):
            e = z - U @ b
            return (1 - np.exp(e)) @ U / n + 2 * P_ * b
    else:
        G_ = U.T @ U / n + lam * PEN * np.diag(sd ** 2)
        b_ = U.T @ z / n
        fun = lambda b: float(b @ G_ @ b - 2 * b @ b_)
        jac = lambda b: 2 * (G_ @ b - b_)
    cons = [{'type': 'ineq', 'fun': lambda b: 1 - b[:ng].sum(), 'jac': lambda b: -np.r_[np.ones(ng), np.zeros(len(b) - ng)]}] if ng else []
    bnds = [(0, 1)] * ng + [(None, None)] * len(fr)
    r = minimize(fun, np.zeros(U.shape[1]), jac=jac, bounds=bnds, constraints=cons, method='SLSQP', options={'maxiter': 500, 'ftol': 1e-12})
    b = r.x
    out = {g: float(b[i]) for i, g in enumerate(grp)}
    out.update({f: float(b[ng + i]) for i, f in enumerate(fr)})
    out['x1'] = float(1 - b[:ng].sum())
    return out


def _cen(T, D, ref):
    mu = ref[[c for c in FREE if c in ref]].mean()
    return T.assign(**{c: T[c] - mu[c] for c in mu.index}), D.assign(**{c: D[c] - mu[c] for c in mu.index})


def predict(D, beta):
    return sum(beta[x] * D[x].values for x in beta)


def qlike_ln(y_lvl, lnp, F=1.0):
    r = y_lvl / (np.exp(lnp) * F)
    return float(np.mean(r - np.log(r) - 1))


def xs_for(h, k, drop=(), em=0):
    xs = ['x1', f'x2_k{k}', 'x6'] + (['x7'] if h == 22 else []) + (['x9'] if USE_X9 else []) + [f for f in FREE0 if not (h == 22 and NO_X5_H22 and f == 'd')] \
        + ([f'e{em}'] + (['e22'] if h == 22 else []) if em else [])
    out = []
    for x in xs:
        nm = 'x2' if x.startswith('x2') else x
        grpname = {'r1p': 'x3', 'r1n': 'x3', 'r1q': 'x3', 'm1': 'x4', 'd': 'x5', 'e5': 'x8', 'e10': 'x8', 'e22': 'x8'}.get(nm, nm)
        if nm in drop or grpname in drop:
            continue
        out.append(x)
    return out


def _ren(beta):
    return {('x2' if k.startswith('x2') else k): v for k, v in beta.items()}


def layer3(h, V, Te, drop=(), fixed=None, obj='ridge', center=False):
    """V — val-строки (шаги 1–10), Te — тест-строки. Возвращает (lnhat_val, lnhat_test, таблица β по шагам)."""
    ks = (1, 5) if (h == 22 and 'x2' not in drop) else (1,)
    ems = (5, 10) if ('e5' in V.columns and 'x8' not in drop) else (0,)
    ok = lambda D: D[D.rv_actual > 1e-12]
    lv, lt = np.full(len(V), np.nan), np.full(len(Te), np.nan)
    rows = []
    for st in range(1, 11):
        Y = 2015 + st
        mv, mt = (V.step == st).values, (Te.step == st).values
        for kind, msk, ymax in (('val', mv, Y), ('test', mt, Y + 1)):
            if not msk.any():
                continue
            Tr = ok(V[(V.year < ymax) & (V._tend < ymax)]).dropna(subset=['x1'])
            D = V[msk] if kind == 'val' else Te[msk]
            if len(Tr) == 0:
                beta, lam, k, em = {'x1': 1.0}, np.nan, 1, 0
            else:
                In = ok(V[(V.year < ymax - 1) & (V._tend < ymax - 1)])
                Ev = ok(V[V.year == ymax - 1])
                if center:  # свободные регрессоры — отклонения от среднего своего обучающего окна (без роли свободного члена)
                    In, Ev = _cen(In, Ev, In)
                    Tr, D = _cen(Tr, D, Tr)
                if fixed is not None:
                    lam, k = fixed; em = ems[-1]
                elif len(In) == 0:
                    lam, k, em = 1.0, 1, ems[-1]
                else:
                    best = None
                    for k_, em_ in [(a, b) for a in ks for b in ems]:
                        for lam_ in LAMS:
                            xs = xs_for(h, k_, drop, em_)
                            b = fit_comb(In.dropna(subset=xs), xs, lam_, obj)
                            res = (In.y - predict(In, b)).values
                            q = qlike_ln(Ev.rv_actual.values, predict(Ev, b), np.exp(np.nanvar(res) / 2) if obj == 'ridge' else 1.0)
                            if best is None or q < best[0]:
                                best = (q, lam_, k_, em_)
                    _, lam, k, em = best
                xs = xs_for(h, k, drop, em)
                beta = fit_comb(Tr.dropna(subset=xs), xs, lam, obj)
            D = D.fillna({x: 0.0 for x in FREE})
            D = D.assign(**{g: D[g].fillna(D['x1']) for g in GROUP if g in D and g != 'x1'})  # нет компоненты на строке -> её вес на x1
            pr = predict(D, beta)
            (lv if kind == 'val' else lt)[msk] = pr
            rows.append(dict(h=h, step=st, kind=kind, year=Y if kind == 'val' else Y + 1, lam=lam, k=k, m_ebar=em, n_train=len(Tr), **_ren(beta)))
    return lv, lt, pd.DataFrame(rows)


def layer4(h, D, lnhat, A, no_guard=False, no_wins=False):
    """Защита Д14 и винзоризация по окну шага (годы < Y_val шага; для теста — годы < Y_test шага, как обучение шага)."""
    out = lnhat.copy()
    tgt = np.log(A[f'rv_target_h{h}'].clip(lower=1e-10))
    for st in sorted(D.step.unique()):
        m = (D.step == st).values
        win = A.year < (2015 + st)  # окно шага k: годы < vy (как обучение моделей шага; на тесте шага k — то же окно)
        if not no_guard:
            lo, hi = np.nanpercentile(A.loc[win, '_lrvm'], [1, 99])
            g = (D.loc[m, '_lrvm'] < lo) | (D.loc[m, '_lrvm'] > hi)
            out[m] = np.where(g.values, D.loc[m, 'x1'].values, out[m])
        if not no_wins:
            lo, hi = np.nanpercentile(tgt[win & A[f'rv_target_h{h}'].notna()], [0.5, 99.5])
            out[m] = np.clip(out[m], lo, hi)
    return out


def frames(h, V, Te, lv, lt):
    fv = V[['date', 'ticker', 'year', 'step', 'rv_actual']].assign(rv_pred=np.exp(lv))
    ft = Te[['date', 'ticker', 'year', 'step', 'rv_actual']].assign(rv_pred=np.exp(lt))
    return HX.add_bias(fv.reset_index(drop=True), ft.reset_index(drop=True))


def year_q(fv, code, lo=2, hi=6):
    q = HX.dev_table(fv, f'rv_pred_{code}' if code != 'raw' else 'rv_pred', lo, hi)
    return {2015 + s: v for s, v in q.items()}


def pipeline(h, seed, I_v, I_t, drop=(), no_h1=False, no_guard=False, no_wins=False, fixed=None, obj='ridge', center=None):
    center = CENTER if center is None else center
    A, hist = matrix()
    V = assemble(h, 'val', seed, I_v, A, hist, no_h1)
    Te = assemble(h, 'test', seed, I_t, A, hist, no_h1)
    if 'L3' in drop:
        lv, lt, B = V.x1.values.copy(), Te.x1.values.copy(), pd.DataFrame()
    else:
        lv, lt, B = layer3(h, V, Te, drop, fixed, obj, center)
    lv, lt = layer4(h, V, lv, A, no_guard, no_wins), layer4(h, Te, lt, A, no_guard, no_wins)
    fv, ft = frames(h, V, Te, lv, lt)
    return fv, ft, B, V


# ---------------------------------------------------------------- эталоны и финалисты (шаги ≤ 6)
def refs(h):
    A, harj = HX.load(P.FEAT)
    hv, ht, _ = HX.har_steps(h, A, harj)
    hv, _ = HX.add_bias(hv, ht)
    out = {'HAR-J с поправкой': hv[hv.step <= 6]}
    leg = lambda df: HX.add_bias(df[df.step <= 6].groupby(KEY + ['year'], as_index=False).agg(rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean')),
                                 df.iloc[:0][KEY + ['year', 'rv_actual', 'rv_pred']].copy())[0]
    rd = lambda n: pd.read_parquet(R / n)
    if h == 5:
        out['D12_h5 (wrap2)'] = leg(pd.concat([rd('D12_h5_xgb_h5_val.parquet'), rd('D12_h5_lgbm_h5_val.parquet')]))
    if h == 22:
        out['Д16.5 (x7, wrap2 seed 42)'] = leg(pd.concat([rd('D16_5_h22_xgb_h22_val.parquet'), rd('D16_5_h22_lgbm_h22_val.parquet')]))
    if h in X9 and (R / f'{X9[h]}_xgb_h{h}_val.parquet').exists():
        out[f'{X9[h]} (x9, Д24)'] = leg(pd.concat([rd(f'{X9[h]}_xgb_h{h}_val.parquet'), rd(f'{X9[h]}_lgbm_h{h}_val.parquet')]))
    out['HAR-X⁺ с поправкой (слой 1)'] = pd.read_parquet(OUT / f'HXP_{h}_val.parquet').query('step <= 6')
    out['HAR-X (Д16) с поправкой'] = pd.read_parquet(R / f'HARX_h{h}_val.parquet').query('step <= 6')
    return out


def summarize(name, fv, code):
    q = year_q(fv, code)
    return dict(model=name, **{str(y): round(v, 4) for y, v in q.items()}, m1720=round(np.mean([q[y] for y in range(2017, 2021)]), 4), y2021=round(q[2021], 4))


def run(seeds, hs=(5, 22)):
    I_v, I_t = load_inputs('val'), load_inputs('test')
    rows, betas = [], []
    for h in hs:
        code = CODE[h]
        for nm, d in refs(h).items():
            rows.append(dict(h=h, seed='—', **summarize(nm, d, code)))
        for sd in seeds:
            fv, ft, B, _ = pipeline(h, sd, I_v, I_t)
            tag = f'D22_h{h}'
            sfx = '' if sd == 42 else f'_s{sd}'
            fv.to_parquet(OUT / f'{tag}{sfx}_final_val.parquet', index=False)
            (LOCK / tag).mkdir(parents=True, exist_ok=True)
            ft.to_parquet(LOCK / tag / f'{tag}{sfx}_final_test.parquet', index=False)
            B.assign(seed=sd).to_csv(OUT / f'{tag}{sfx}_beta.csv', index=False)
            betas.append(B.assign(seed=sd))
            rows.append(dict(h=h, seed=sd, **summarize('Д22', fv, code)))
            print(rows[-1], flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(OUT / 'd22_eval.csv', index=False)
    print(S.to_string())
    return S, pd.concat(betas)


ABL = {  # имя: аргументы pipeline
    'Д22 (полный)': {},
    'минус слой 1 (HAR-J вместо HAR-X⁺)': {'base': 'HJ'},
    'минус слой 2 (hyb₁ := HAR-X⁺₁, без r̂₁, m̂, d̂)': {'no_h1': True},
    'минус слой 3 (ŷ = x1)': {'drop': ('L3',)},
    'минус слой 4: без s²': {'code': 'raw'},
    'минус слой 4: без защиты Д14': {'no_guard': True},
    'минус слой 4: без винзоризации': {'no_wins': True},
    'минус x2': {'drop': ('x2',)}, 'минус x3': {'drop': ('x3',)}, 'минус x4': {'drop': ('x4',)},
    'минус x5': {'drop': ('x5',)}, 'минус x6': {'drop': ('x6',)}, 'минус x7': {'drop': ('x7',)}, 'минус x8': {'drop': ('x8',)}, 'минус x9 (Д24)': {'drop': ('x9',)},
}


def ablate(seeds=(42, 43, 44), hs=(5, 22)):
    cache = {}
    rows = []
    for h in hs:
        for nm, kw in ABL.items():
            if (nm == 'минус x7' and h != 22) or (nm == 'минус x8' and not EBAR.exists()) or (nm == 'минус x5' and h == 22 and NO_X5_H22):
                continue
            kw = dict(kw)
            base = kw.pop('base', 'HXP'); code = kw.pop('code', CODE[h])
            if base not in cache:
                cache[base] = (load_inputs('val', base, 'L2_h1' if base == 'HXP' else 'abl_HJ_L2_h1', 'clf' if base == 'HXP' else 'abl_HJ_clf'),
                               load_inputs('test', base, 'L2_h1' if base == 'HXP' else 'abl_HJ_L2_h1', 'clf' if base == 'HXP' else 'abl_HJ_clf'))
            Iv, It = cache[base]
            qs = []
            for sd in seeds:
                fv, _, _, _ = pipeline(h, sd, Iv, It, **kw)
                qs.append(year_q(fv, code))
            q = {y: np.mean([x[y] for x in qs]) for y in qs[0]}
            per = [np.mean([x[y] for y in range(2017, 2021)]) for x in qs]
            rows.append(dict(h=h, variant=nm, **{str(y): round(v, 4) for y, v in q.items()}, m1720=round(np.mean(per), 4),
                             y2021=round(q[2021], 4), **{f's{sd}_1720': round(p, 4) for sd, p in zip(seeds, per)}))
            print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    full = T[T.variant == 'Д22 (полный)'].set_index('h')
    T['вклад_1720_%'] = [round(100 * (r.m1720 / full.loc[r.h, 'm1720'] - 1), 2) for r in T.itertuples()]
    T['вклад_2021_%'] = [round(100 * (r.y2021 / full.loc[r.h, 'y2021'] - 1), 2) for r in T.itertuples()]
    T.to_csv(OUT / 'd22_ablation.csv', index=False)
    print(T.to_string())
    return T


# ---------------------------------------------------------------- детектор «будущее не влияет» на контуре слоёв 3–4
def leak(cuts=('2019-06-30', '2021-06-30'), h_list=(5, 22), seed=42, plant=False):
    """Строки date > t во всех OOS-входах (x_j, цели, классификатор, Д16.5) и в матрице (RV, RV_m, цели) заменяются шумом; прогноз слоёв 3–4
    (до s²; s² шага k оценивается по годам < Y и проверяется отдельно) на строках date ≤ t должен совпасть. plant=True — самопроверка:
    в x1 подмешивается ln RV(t+1) (детектор обязан поймать)."""
    rng = np.random.default_rng(7)
    A0, hist0 = matrix()
    rows = []
    for h in h_list:
        Iv, It = load_inputs('val'), load_inputs('test')
        for t in cuts:
            t = pd.Timestamp(t)
            outs = []
            for pert in (False, True):
                A = A0.copy(); hist = hist0.copy()
                Iv2 = {k: (v.copy() if v is not None else None) for k, v in Iv.items()}
                It2 = {k: (v.copy() if v is not None else None) for k, v in It.items()}
                if pert:
                    fut = (A.date > t).values
                    for c in ('rv', 'rv_m', 'rv_target_h1', f'rv_target_h{h}', '_ld', '_lw', '_lm', '_q', '_lrvm', 'rv_target_mean_h5', 'rv_target_mean_h22'):
                        A.loc[fut, c] = A.loc[fut, c].values * np.exp(rng.normal(0, 1, fut.sum())) if not c.startswith('_l') else A.loc[fut, c].values + rng.normal(0, 1, fut.sum())
                    hist[fut] = hist[fut] * np.exp(rng.normal(0, 1, hist[fut].shape))
                    for I in (Iv2, It2):
                        for k, v in I.items():
                            if v is None:
                                continue
                            f = (v.date > t).values
                            for c in v.columns:
                                if c in ('date', 'ticker', 'step', 'seed', 'year') or not np.issubdtype(v[c].dtype, np.number):
                                    continue
                                v.loc[f, c] = v.loc[f, c].values * np.exp(rng.normal(0, 0.5, f.sum())) if c in ('rv_actual', 'rv_pred', 'p') else v.loc[f, c].values + rng.normal(0, 0.5, f.sum())
                _A_saved = dict(_A); _A.update(A=A, hist=hist)
                V = assemble(h, 'val', seed, Iv2, A, hist); Te = assemble(h, 'test', seed, It2, A, hist)
                if plant:
                    nxt = A.groupby('ticker')['rv'].shift(-1)
                    key = A[['date', 'ticker']].assign(nx=np.log(nxt.clip(lower=1e-10)).values)
                    V = V.merge(key, on=['date', 'ticker'], how='left'); V['x6'] = V['x6'] + 0.5 * V['nx'].fillna(0); V.pop('nx')
                lv, lt, _ = layer3(h, V, Te)
                lv, lt = layer4(h, V, lv, A), layer4(h, Te, lt, A)
                _A.clear(); _A.update(_A_saved)
                outs.append((V[['date', 'ticker', 'step']].assign(p=lv), Te[['date', 'ticker', 'step']].assign(p=lt)))
            for i, kind in ((0, 'val'), (1, 'test')):
                a, b = outs[0][i], outs[1][i]
                m = (a.date <= t).values
                diff = np.abs(a.p.values[m] - b.p.values[m])
                rows.append(dict(h=h, t=str(t.date()), kind=kind, n=int(m.sum()), mismatch=int((diff > 1e-9).sum()), max_abs=float(np.nanmax(diff)) if m.any() else 0.0,
                                 changed_after=int((np.abs(a.p.values[~m] - b.p.values[~m]) > 1e-9).sum()), plant=plant))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def eval_h1(code='sm'):
    """Слой 2 (h=1): QLIKE по годам (поправка sm, своя s² модели), по seed 42/43/44: D22_h1 (XGB, LGBM, среднее ln = hyb₁) против финалистов
    D12_h1_A_d812 и D16_h1_d812x (XGB, точка ранга 1) и эталона HAR-J с поправкой. Только шаги ≤ 6."""
    def per_seed(prefix, m):
        a = pd.read_parquet(R / f'{prefix}_{m}_h1_val.parquet').assign(seed=42)
        b = pd.read_parquet(R / f'{prefix}_{m}_h1_seeds4344_val.parquet')
        d = pd.concat([a, b], ignore_index=True)
        return {sd: g[g.step <= 6][KEY + ['year', 'rv_actual', 'rv_pred']].reset_index(drop=True) for sd, g in d.groupby('seed')}
    bias = lambda d: HX.add_bias(d.copy(), d.iloc[:0].copy())[0]
    rows = []
    A, harj = HX.load(P.FEAT)
    hv, ht, _ = HX.har_steps(1, A, harj); hv, _ = HX.add_bias(hv, ht)
    rows.append(dict(seed='—', **summarize('HAR-J с поправкой', hv[hv.step <= 6], code)))
    rows.append(dict(seed='—', **summarize('HAR-X⁺₁ (слой 1)', pd.read_parquet(OUT / 'HXP_1_val.parquet').query('step <= 6'), code)))
    X, Lg = per_seed('D22_h1', 'xgb'), per_seed('D22_h1', 'lgbm')
    for sd in (42, 43, 44):
        w = X[sd].merge(Lg[sd][KEY + ['rv_pred']], on=KEY, suffixes=('', '_l'))
        w['rv_pred'] = np.exp((np.log(w.rv_pred) + np.log(w.rv_pred_l)) / 2)
        for nm, d in (('D22_h1 hyb₁ (среднее ln XGB, LGBM)', w.drop(columns='rv_pred_l')), ('D22_h1 XGB', X[sd]), ('D22_h1 LGBM', Lg[sd])):
            rows.append(dict(seed=sd, **summarize(nm, bias(d), code)))
        for pref in ('D12_h1_A_d812', 'D16_h1_d812x'):
            rows.append(dict(seed=sd, **summarize(f'{pref} XGB', bias(per_seed(pref, 'xgb')[sd]), code)))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / 'd22_eval_h1.csv', index=False)
    print(S.to_string())
    return S


def cmp_obj(hs=(5, 22), seeds=(42, 43, 44)):
    """Д24.12: объединитель слоя 3 — ridge (основной) против прямой минимизации QLIKE (те же ограничения, штраф, λ и k по t−1), а также
    справочно — свободные регрессоры, центрированные по окну. QLIKE с поправкой s² кода h и без неё (raw); среднее по 3 seed."""
    I_v, I_t = load_inputs('val'), load_inputs('test')
    rows = []
    for h in hs:
        for nm, kw in (('ridge (основной)', {}), ('QLIKE (Д24.12)', {'obj': 'qlike'}), ('ridge, центрирование x3–x5, x8', {'center': True}),
                       ('QLIKE, центрирование', {'obj': 'qlike', 'center': True})):
            for code in (CODE[h], 'raw'):
                qs = []
                for sd in seeds:
                    fv, _, _, _ = pipeline(h, sd, I_v, I_t, **kw)
                    qs.append(year_q(fv, code))
                per = [np.mean([q[y] for y in range(2017, 2021)]) for q in qs]
                rows.append(dict(h=h, variant=nm, poprav=code, **{str(y): round(np.mean([q[y] for q in qs]), 4) for y in range(2017, 2022)},
                                 m1720=round(np.mean(per), 4), **{f's{sd}_1720': round(p, 4) for sd, p in zip(seeds, per)}))
                print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'd22_obj_cmp.csv', index=False)
    print(T.to_string())
    return T


def abl_h1(code='sm'):
    """Абляции слоя 2 (h=1, hyb₁ = среднее ln XGB и ln LGBM, точки D22_h1 без нового перебора, шаги 1–6, seed 42/43/44)."""
    def hyb(d):
        d = d[d.step <= 6].copy(); d['ln'] = np.log(d.rv_pred)
        w = d.pivot_table(index=KEY + ['year', 'seed', 'rv_actual'], columns='model', values='ln').reset_index()
        w['rv_pred'] = np.exp((w.xgb + w.lgbm) / 2)
        return {sd: HX.add_bias(g[KEY + ['year', 'rv_actual', 'rv_pred']].reset_index(drop=True), g.iloc[:0][KEY + ['year', 'rv_actual', 'rv_pred']].copy())[0]
                for sd, g in w.groupby('seed')}
    full = []
    for m in ('xgb', 'lgbm'):
        a = pd.read_parquet(R / f'D22_h1_{m}_h1_val.parquet').assign(seed=42, model=m)
        b = pd.read_parquet(R / f'D22_h1_{m}_h1_seeds4344_val.parquet').assign(model=m)
        full += [a, b]
    V = {'Д22 слой 2 (полный)': pd.concat(full, ignore_index=True)}
    for v, nm in (('HJ', 'минус слой 1 (основа Д8.12 — HAR-J)'), ('nofeat', 'минус календарь/RRV/гэп'), ('nowins', 'минус винзоризация цели')):
        V[nm] = pd.read_parquet(OUT / f'abl_{v}_h1_val.parquet')
    rows = []
    for nm, d in V.items():
        qs = {sd: year_q(f, code) for sd, f in hyb(d).items()}
        per = {sd: np.mean([q[y] for y in range(2017, 2021)]) for sd, q in qs.items()}
        rows.append(dict(h=1, variant=nm, **{str(y): round(np.mean([q[y] for q in qs.values()]), 4) for y in range(2017, 2022)},
                         m1720=round(np.mean(list(per.values())), 4), **{f's{sd}_1720': round(p, 4) for sd, p in per.items()}))
    T = pd.DataFrame(rows)
    T['вклад_1720_%'] = (100 * (T.m1720 / T.m1720.iloc[0] - 1)).round(2)
    T['вклад_2021_%'] = (100 * (T['2021'] / T['2021'].iloc[0] - 1)).round(2)
    T.to_csv(OUT / 'd22_ablation_h1.csv', index=False)
    print(T.to_string())
    return T


def export(hs=(5, 22)):
    """Файлы кандидата в формате gibrid_v17 / final3 (одиночная модель в обоих слотах): val шагов 1–9 — data/regen/D22_h{h}_{m}_h{h}[_seeds4344]_val,
    тест шагов 1–9 — test_locked/D22_h{h}/, шаг 10 — data/regen/D22/step10/ и test_locked/D22_h{h}/step10/. rv_pred — после слоёв 3–4 без s²
    (поправку делает гибрид: B-*). Метрики теста не считаются."""
    I_v, I_t = load_inputs('val'), load_inputs('test')
    cols = ['date', 'ticker', 'year', 'step', 'seed', 'rank', 'rv_actual', 'rv_pred']
    for h in hs:
        tag = f'D22_h{h}'
        V, T = [], []
        for sd in (42, 43, 44):
            fv, ft, _, _ = pipeline(h, sd, I_v, I_t)
            V.append(fv.assign(seed=sd, rank=1)[cols]); T.append(ft.assign(seed=sd, rank=1)[cols])
        V, T = pd.concat(V, ignore_index=True), pd.concat(T, ignore_index=True)
        lk = LOCK / tag; (lk / 'step10').mkdir(parents=True, exist_ok=True); (OUT / 'step10').mkdir(parents=True, exist_ok=True)
        for m in ('xgb', 'lgbm'):
            for sfx, sel in (('', [42]), ('_seeds4344', [43, 44])):
                V[(V.step <= 9) & V.seed.isin(sel)].to_parquet(R / f'{tag}_{m}_h{h}{sfx}_val.parquet', index=False)
                T[(T.step <= 9) & T.seed.isin(sel)].to_parquet(lk / f'{tag}_{m}_h{h}{sfx}_test.parquet', index=False)
            V[V.step == 10].to_parquet(OUT / 'step10' / f'{tag}_{m}_h{h}_s10_val.parquet', index=False)
            T[T.step == 10].to_parquet(lk / 'step10' / f'{tag}_{m}_h{h}_s10_test.parquet', index=False)
        print(f'{tag}: val {len(V)} строк, тест {len(T)} строк (метрики теста не считались)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true'); ap.add_argument('--ablate', action='store_true'); ap.add_argument('--leak', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='*', default=[42, 43, 44, 0]); ap.add_argument('--hs', type=int, nargs='*', default=[5, 22])
    ap.add_argument('--export', action='store_true'); ap.add_argument('--h1', action='store_true')
    a = ap.parse_args()
    if a.h1:
        eval_h1()
    if a.export:
        export(a.hs)
    if a.run:
        run(a.seeds, a.hs)
    if a.ablate:
        ablate(hs=a.hs)
    if a.leak:
        r = pd.concat([leak(h_list=a.hs), leak(h_list=a.hs[:1], cuts=('2019-06-30',), plant=True)])
        r.to_csv(OUT / 'leak_contour.csv', index=False)
        print(r.to_string())


if __name__ == '__main__':
    main()
