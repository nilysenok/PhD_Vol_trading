#!/usr/bin/env python3
"""R2, агент Э (протокол v2, docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md, строка «Э»): ноги уровня 1 без бустинга, h ∈ {1, 5, 22},
val шагов 1–6 (val-год vy = 2015 + k, обучение — годы < vy), seed нет (файлы _sdet). Код первого раунда только импортируется.

Четыре версии каждой ноги (суффикс имени ряда):
  ''        — определение первого раунда: HAR-регрессоры features_pit_v3 (rv_d = RV_{d−1}, rv_w/rv_m — окна, кончающиеся d−1), без эмбарго
  _emb      — эмбарго по концу цели (решение дирижёра 26.10): строки обучения, у которых окно цели [t+1, t+h] заходит в val-год шага
              (дата строки t+h тикера >= 1 янв. vy), не обучают; у HAR-X «внутренняя» точка — и внутренняя валидация с тем же эмбарго
  _d0       — «с RV дня d» (PR #143): rv_d = RV_d, rv_w = среднее RV_{d−4..d}, rv_m = среднее RV_{d−21..d} (нулевой RV — последним
              положительным RV тикера); jump и jump_annualized — уже день d; слагаемые HAR-X, построенные из rv_d / rv_m (ln_rvm_mkt, harq,
              ln_rv66/132) и ln_sig1 (отклонение от HAR-J₁ той же матрицы) пересчитываются из этой матрицы
  _emb_d0   — оба; **основная версия до решения владельца** (колонка osnovnaya в индексе)
Семейства (имя ряда без суффикса версии):
  HARJ                     HAR-J (как gibrid_har.har_forecasts)
  HARX_<точка>             HAR-J + harx.default_terms(h, True), Фриш–Во + эластичная сеть (harx.fit_en); 46 точек: ols, inner, l<λ>_a<α>
  HARXsub_plus|minus_<гр>  HAR-J + одна группа / HAR-X без группы, OLS; группы RVI, rynok, LHAR, PC1, sig1 (h=5/22), D17
  HARXreg                  режимный HAR-X (Д24.4 RX, d24_4.design/fit_pred_factory); в R1 он с эмбарго — его повторяет HARXreg_emb
  HEAVY                    HEAVY (Д24.6, d24_6.fit_forecast): целей в обучении нет (QMLE на данных < vy), RV дня d уже на входе —
                           все четыре версии совпадают, файлы пишутся под всеми именами
Выход (data/regen/R2/cache/E/):
  <ряд>_h{h}_sdet_val.parquet — date, ticker, year, step, rv_actual, rv_pred (сырой, без s²)
  <ряд>_h{h}_lnall.parquet    — date, ticker, step, part (tr/va), base_ln, emb_ok (образец data/regen/D17/harx_base_h1.parquet): ln-прогноз
      модели шага k; tr — годы < vy с целью (внутри выборки, как HAR-J в Д8.12), va — год vy; emb_ok — строка проходит эмбарго шага
      (дата t+h < 1 янв. vy). Для рядов ролей: HARJ, 8 точек HAR-X (ols, inner, λ ∈ {1e-3, 1e-2, 0,1} × α ∈ {0,5; 1}), HARXsub, HARXreg, HEAVY.
  V0_F_h{h}.csv, V0_q_h{h}.csv — множитель Б0.1 (val шага t — s² шага t−1, шаг 1 — F = 1, как harx.add_bias), s² ∈ {bez, ord, rob, sm,
      smrob}; QLIKE 2017–2021 (шаги 2–6) по ряду × s²; V0_versii.csv — четыре версии рядом.
  INDEX: data/regen/R2/cache/INDEX_E.csv.
Годы >= 2022 не строятся и не читаются как метрики: шаги 7–10 не считаются.
Режимы: zamer | det [--hs] [--workers <= 2] | leak [--workers <= 2]
  leak — «будущее не влияет»: всё после среза (входы и цели, чей горизонт заходит за срез) × U(0,5; 2). Срезы в середине года: val и lnall
  моделей с vy <= года среза на строках <= среза совпадают точно; подлог (rv_d := значение дня d+1, у HEAVY — RM дня d+1) пойман.
  Срезы на границе года (31.12): у версий _emb совпадает ещё и lnall модели следующего года; у версий без эмбарго он меняется (утечка видна)."""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import sys, time, hashlib, argparse, tempfile, subprocess, warnings, multiprocessing as mp
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harx as HX
import gibrid_har as GH

DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
REG = DATA / 'regen'
CACHE = REG / 'R2' / 'cache' / 'E'
INDEX = REG / 'R2' / 'cache' / 'INDEX_E.csv'
YAML = HERE.parents[1] / 'models' / 'designs' / 'r2' / 'r2_E_nogi.yaml'
FEAT = REG / 'features_pit_v3.parquet'
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
KEY = ['date', 'ticker', 'step']
HS = (1, 5, 22)
STEPS = range(1, 7)
TOL = 1e-7
S2 = ('bez', 'ord', 'rob', 'sm', 'smrob')
VERSIONS = {'': (False, False), '_emb': (True, False), '_d0': (False, True), '_emb_d0': (True, True)}
OSN = '_emb_d0'
GRID = [('ols', 0.0, 1.0), ('inner', None, None)] + [(f'l{al:g}_a{l1:g}', al, l1) for l1 in HX.L1S for al in HX.ALPHAS]
ROLE_HARX = {'ols', 'inner', 'l0.001_a0.5', 'l0.001_a1', 'l0.01_a0.5', 'l0.01_a1', 'l0.1_a0.5', 'l0.1_a1'}
assert len(GRID) == 46 and ROLE_HARX <= {g[0] for g in GRID}


def split_name(nm):
    for v in ('_emb_d0', '_emb', '_d0'):
        if nm.endswith(v):
            return nm[:-len(v)], v
    return nm, ''


def family(nm):
    b = split_name(nm)[0]
    return 'HARXsub' if b.startswith('HARXsub') else ('HARX' if b.startswith('HARX_') else b)


def groups(h):
    g = {'RVI': ['ln_rvi2'], 'rynok': ['ln_rvm_imoex', 'ln_rvm_mkt'], 'LHAR': ['lhar_1', 'lhar_5', 'lhar_22'], 'PC1': ['pc1']}
    if h != 1:
        g['sig1'] = ['ln_sig1']
    g['D17'] = list(HX.D17_TERMS[h])
    assert sorted(sum(g.values(), [])) == sorted(HX.default_terms(h, True))
    return g


def subsets(h):
    full, g = HX.default_terms(h, True), groups(h)
    out = [(f'plus_{k}', v) for k, v in g.items()] + [(f'minus_{k}', [t for t in full if t not in v]) for k, v in g.items()]
    assert len(out) <= 12
    return out


# ================================================================ матрица
def load(feat=FEAT, d0=False, plant=False):
    """harx.load (gibrid_har.load_pit); d0 — HAR-регрессоры с RV дня d; plant — подложенная утечка rv_d := значение строки d+1 тикера."""
    HX._SIG.clear()
    A, harj = HX.load(feat)
    g = A.groupby('ticker')
    if d0:
        rvp = A['rv'].where(A['rv'] > 0).groupby(A['ticker']).ffill().fillna(A['rv'])
        gp = rvp.groupby(A['ticker'])
        A['rv_d'] = rvp
        A['rv_w'] = gp.transform(lambda s: s.rolling(5, min_periods=1).mean())
        A['rv_m'] = gp.transform(lambda s: s.rolling(22, min_periods=1).mean())
    if plant:
        A['rv_d'] = A.groupby('ticker')['rv_d'].shift(-1).fillna(A['rv_d'])
    for h in HS:  # год последнего дня окна цели (строка t+h тикера)
        A[f'_tendy{h}'] = pd.to_datetime(g['date'].shift(-h)).dt.year
    return A, harj


def trmask(A, h, vy, emb):
    t = f'rv_target_h{h}'
    m = (A.year.values < vy) & A[t].notna().values
    return m & (A[f'_tendy{h}'].values < vy) if emb else m


def _val(A, h, k, ln):
    t = f'rv_target_h{h}'; vy = 2015 + k
    sel = ((A.year == vy) & A[t].notna()).values
    return pd.DataFrame({'date': pd.to_datetime(A.date.values[sel]), 'ticker': A.ticker.values[sel].astype(str), 'year': vy, 'step': k,
                         'rv_actual': A[t].values[sel], 'rv_pred': np.clip(np.exp(ln[sel]), 1e-10, None)})


def _lnw(A, h, lns):
    """Широкий ln: date, ticker, year, ok, tendy, ln_s{k} (только строки year <= vy_k)."""
    d = pd.DataFrame({'date': pd.to_datetime(A.date.values), 'ticker': A.ticker.values.astype(str), 'year': A.year.values,
                      'ok': A[f'rv_target_h{h}'].notna().values, 'tendy': A[f'_tendy{h}'].values})
    for k, ln in sorted(lns.items()):
        d[f'ln_s{k}'] = np.where(A.year.values <= 2015 + k, ln, np.nan)
    return d[d.year <= 2015 + max(lns)].sort_values(['date', 'ticker']).reset_index(drop=True)


def to_parts(W):
    """Широкий ln -> date, ticker, step, part (tr/va), base_ln, emb_ok."""
    out = []
    for c in [c for c in W.columns if c.startswith('ln_s')]:
        k = int(c[4:]); vy = 2015 + k
        for part, m in (('tr', (W.year < vy) & W.ok), ('va', (W.year == vy) & W.ok)):
            d = W[m]
            out.append(pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'step': k, 'part': part, 'base_ln': d[c].values,
                                     'emb_ok': (d.tendy < vy).values}))
    return pd.concat(out, ignore_index=True).sort_values(['step', 'part', 'date', 'ticker'], ascending=True).reset_index(drop=True)


def _fin(V):
    return pd.concat(V).sort_values(['step', 'date', 'ticker']).reset_index(drop=True)[COLS]


# ================================================================ семейства: (A, harj, h, emb) -> {ряд: (V, W | None)}
def fam_harj(A, harj, h, emb, steps=STEPS):
    t = f'rv_target_h{h}'
    H = GH.clean(np.log(A[harj].values.clip(1e-10)))
    V, L = [], {}
    for k in steps:
        tp = trmask(A, h, 2015 + k, emb)
        ln = LinearRegression().fit(H[tp], np.log(A[t].values[tp] + 1e-10)).predict(H)
        V.append(_val(A, h, k, ln)); L[k] = ln
    return {'HARJ': (_fin(V), _lnw(A, h, L))}


def _inner_emb(A, h, k, H, Xt, emb):
    """Внутренняя валидация шага (как harx.forecast_step, INNER = 1) с эмбарго: обучение — trmask(vy−1), оценка — строки года vy−1, чьё
    окно цели кончается до vy; затем подгонка на trmask(vy). Возвращает ln на всех строках."""
    t = f'rv_target_h{h}'; vy = 2015 + k
    y = np.log(A[t].values + 1e-10)
    i_tr = trmask(A, h, vy - 1, emb)
    i_va = (A.year.values == vy - 1) & A[t].notna().values & (A[f'_tendy{h}'].values < vy)
    Ztr = Xt.fillna(Xt.loc[i_tr].mean()).fillna(0.0).values.astype('float64')
    res = []
    for l1, al in [(l1, al) for l1 in HX.L1S for al in HX.ALPHAS] + [(0.0, 0.0)]:
        p, _ = HX.fit_en(H[i_tr], Ztr[i_tr], y[i_tr], al, l1 if al > 0 else 1.0)
        res.append((al, l1, float(np.mean((y[i_va] - p(H[i_va], Ztr[i_va])) ** 2))))
    al, l1, _ = min(res, key=lambda r: r[2])
    tp = trmask(A, h, vy, emb)
    Z = Xt.fillna(Xt.loc[tp].mean()).fillna(0.0).values.astype('float64')
    return HX.fit_en(H[tp], Z[tp], y[tp], al, l1 if al > 0 else 1.0)[0](H, Z)


def _harx_points(A, harj, h, emb, terms, X, points, steps, prefix, role):
    t = f'rv_target_h{h}'
    H = GH.clean(np.log(A[harj].values.clip(1e-10)))
    Xt = X[terms]
    acc = {p[0]: ([], {}) for p in points}
    for k in steps:
        tp = trmask(A, h, 2015 + k, emb)
        Z = Xt.fillna(Xt.loc[tp].mean()).fillna(0.0).values.astype('float64')
        y = np.log(A[t].values[tp] + 1e-10)
        for nm, al, l1 in points:
            if nm == 'inner':
                ln = _inner_emb(A, h, k, H, Xt, True) if emb else HX.forecast_step(A, h, k, harj, terms, X, all_rows=True)[2].values
            else:
                ln = HX.fit_en(H[tp], Z[tp], y, al, l1)[0](H, Z)
            acc[nm][0].append(_val(A, h, k, ln)); acc[nm][1][k] = ln
    return {f'{prefix}{nm}': (_fin(V), _lnw(A, h, L) if role(nm) else None) for nm, (V, L) in acc.items()}


def fam_harx(A, harj, h, emb, steps=STEPS, points=GRID, X=None):
    terms = HX.default_terms(h, True)
    X = HX.build_terms(A, terms) if X is None else X
    return _harx_points(A, harj, h, emb, terms, X, points, steps, 'HARX_', lambda nm: nm in ROLE_HARX)


def fam_sub(A, harj, h, emb, steps=STEPS, only=None, X=None):
    X = HX.build_terms(A, HX.default_terms(h, True)) if X is None else X
    out = {}
    for nm, terms in subsets(h):
        if only is None or nm in only:
            r = _harx_points(A, harj, h, emb, terms, X, [('ols', 0.0, 1.0)], steps, f'HARXsub_{nm}', lambda _: True)
            out[f'HARXsub_{nm}'] = r[f'HARXsub_{nm}ols']
    return out


def fam_reg(A, harj, h, emb, steps=STEPS):
    """Д24.4 RX на матрице A; эмбарго — d24b_lib.train_mask (дата строки t+h < 1 янв. vy), без эмбарго — годы < vy с целью."""
    import d24_4 as R4, d24b_lib as L4
    B = A.sort_values(['ticker', 'date']).reset_index(drop=True)
    g = B.groupby('ticker', sort=False)
    B = pd.concat([B, pd.DataFrame({f'_fd{j}': g['date'].shift(-j) for j in HS}, index=B.index)], axis=1)
    HX._SIG.clear()
    X, Z, I, _ = R4.design(B, harj, h, 'RX')
    fp = R4.fit_pred_factory(B, h, X, Z, I)
    allr = np.ones(len(B), bool)
    V, L = [], {}
    for k in steps:
        vy = 2015 + k
        tr = L4.train_mask(B, h, vy) if emb else ((B.year.values < vy) & L4.target(B, h).notna().values)
        ln = fp(k, vy, tr)[0](allr)
        V.append(_val(B, h, k, ln)); L[k] = ln
    return {'HARXreg': (_fin(V), _lnw(B, h, L))}


def fam_heavy(A, h, steps=STEPS, plant=False):
    """HEAVY как d24_6.main (close и rv той же матрицы; RV дня d на входе фильтра); plant=True — RM дня d+1."""
    import d24_6 as G
    a = A[['date', 'ticker', 'close', 'rv', f'rv_target_h{h}', f'_tendy{h}']].copy()
    a['date'] = pd.to_datetime(a.date); a['year'] = a.date.dt.year
    V, W = [], []
    for tk in sorted(a.ticker.unique()):
        gg = G.prep(a, tk)
        if plant:
            gg['xx'] = gg['xx'].shift(-1).fillna(gg['xx'].iloc[-1])
        lns = {}
        for k in steps:
            vy = 2015 + k
            F, _ = G.fit_forecast(gg.rr.values, gg.xx.values, int((gg.year < vy).sum()))['heavy']
            ln = np.log(np.clip(F[h] / 1e4, 1e-10, None))
            sel = ((gg.year == vy) & gg[f'rv_target_h{h}'].notna()).values
            V.append(pd.DataFrame({'date': gg.date.values[sel], 'ticker': tk, 'year': vy, 'step': k,
                                   'rv_actual': gg[f'rv_target_h{h}'].values[sel], 'rv_pred': np.exp(ln[sel])}))
            lns[k] = ln
        W.append(_lnw(gg, h, lns))
    return {'HEAVY': (_fin(V), pd.concat(W).sort_values(['date', 'ticker']).reset_index(drop=True))}


FAMS = ('HARJ', 'HEAVY', 'HARXreg', 'HARXsub', 'HARX')


def build_h(h, feat=FEAT, plant=False, light=False, versions=tuple(VERSIONS)):
    """Все ряды горизонта h во всех версиях. light=True (детектор) — три точки HAR-X и два подмножества."""
    out = {}
    mats = {d0: load(feat, d0=d0, plant=plant) for d0 in sorted({VERSIONS[v][1] for v in versions})}
    pts = [g for g in GRID if g[0] in ('ols', 'inner', 'l0.01_a0.5')] if light else GRID
    only = ('plus_RVI', 'minus_D17') if light else None
    hv = fam_heavy(load(feat)[0], h, plant=plant)['HEAVY']  # HEAVY: одинаков во всех версиях
    Xs = {}
    for d0, (A, harj) in mats.items():
        HX._SIG.clear(); HX.GH_HARJ = harj  # ln_sig1 кэшируется в harx._SIG по первой матрице — сброс на каждую
        Xs[d0] = HX.build_terms(A, HX.default_terms(h, True))
    for v in versions:
        emb, d0 = VERSIONS[v]
        A, harj = mats[d0]
        HX._SIG.clear()
        S = {}
        S.update(fam_harj(A, harj, h, emb))
        S['HEAVY'] = hv
        S.update(fam_reg(A, harj, h, emb))
        S.update(fam_sub(A, harj, h, emb, only=only, X=Xs[d0]))
        S.update(fam_harx(A, harj, h, emb, points=pts, X=Xs[d0]))
        out.update({f'{k}{v}': val for k, val in S.items()})
    return out


# ================================================================ V0
def qlike(y, p):
    r = y / np.clip(p, 1e-12, None)
    return float(np.mean(r - np.log(r) - 1))


def v0_tables(series, h):
    Fr, Qr = [], []
    for nm, (V, _) in series.items():
        b, v_ = split_name(nm)
        v = V[V.rv_actual > 1e-12]
        E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
        meta = dict(ryad=nm, noga=b, versiya=v_.lstrip('_') or 'r1', semeystvo=family(nm), h=h)
        for code in S2:
            F = {st: (1.0 if (code == 'bez' or st == 1) else float(np.exp(HX.s2_est(code, E, st - 1) / 2))) for st in STEPS}
            Fr += [dict(meta, s2=code, step=st, F=f) for st, f in F.items()]
            for st, g in v[v.step.between(2, 6)].groupby('step'):
                Qr.append(dict(meta, s2=code, year=2015 + int(st), q=qlike(g.rv_actual.values, g.rv_pred.values * F[st])))
    return pd.DataFrame(Fr), pd.DataFrame(Qr)


# ================================================================ сверка
def _cmp(V, r, nm, h, src, col='rv_pred'):
    r = r[r.step.between(1, 6)].copy(); r['date'] = pd.to_datetime(r.date)
    m = V.merge(r[KEY + ['rv_actual', col]].rename(columns={col: 'rv_pred'}), on=KEY, how='outer', suffixes=('', '_r1'), indicator=True)
    b = m._merge == 'both'
    d = float(np.abs(m.rv_pred[b] - m.rv_pred_r1[b]).max())
    da = float(np.nanmax(np.abs(m.rv_actual[b] - m.rv_actual_r1[b])))
    nout = int((~b).sum())
    return dict(ryad=nm, h=h, etalon=src, n=int(b.sum()), n_vne_pary=nout, max_abs=d, max_abs_rv_actual=da,
                status='OK' if (nout == 0 and d <= TOL and da <= TOL) else 'РАСХОЖДЕНИЕ')


def sverka(S, F, h):
    rows = []
    A, harj = HX.load(FEAT)
    rows.append(_cmp(S['HARJ'][0], GH.har_forecasts(h, A, harj)[0], 'HARJ', h, 'gibrid_har.har_forecasts'))
    r = pd.read_parquet(REG / f'HARX_h{h}_val.parquet')
    rows.append(_cmp(S['HARX_inner'][0], r, 'HARX_inner', h, f'HARX_h{h}_val'))
    V = S['HARX_inner'][0]
    for code in S2[1:]:
        f = F[(F.ryad == 'HARX_inner') & (F.s2 == code)].set_index('step').F
        rows.append(_cmp(V.assign(rv_pred=V.rv_pred * V.step.map(f)), r, f'HARX_inner×F_{code}', h, f'HARX_h{h}_val.rv_pred_{code}', f'rv_pred_{code}'))
    rows.append(_cmp(S['HARXreg_emb'][0], pd.read_parquet(REG / 'D24b' / 'd24_4' / f'RX_h{h}_val.parquet'), 'HARXreg_emb', h, f'D24b/d24_4/RX_h{h}_val'))
    rows.append(_cmp(S['HEAVY'][0], pd.read_parquet(REG / f'GARCH_svheavy_h{h}_val.parquet'), 'HEAVY', h, f'GARCH_svheavy_h{h}_val'))
    # lnall в формате tr/va против D17/harx_base_h{h} (Д17.1, тот же harx.forecast_step) на шагах 1-6
    P = to_parts(S['HARX_inner'][1]); D = pd.read_parquet(REG / 'D17' / f'harx_base_h{h}.parquet')
    D = D[D.step.between(1, 6) & D.part.isin(['tr', 'va'])]
    m = P.merge(D, on=['date', 'ticker', 'step', 'part'], how='outer', suffixes=('', '_r1'), indicator=True)
    b = m._merge == 'both'; d = float(np.abs(m.base_ln[b] - m.base_ln_r1[b]).max())
    rows.append(dict(ryad='HARX_inner lnall', h=h, etalon=f'D17/harx_base_h{h} (tr, va)', n=int(b.sum()), n_vne_pary=int((~b).sum()), max_abs=d,
                     max_abs_rv_actual=np.nan, status='OK' if (b.all() and d <= TOL) else 'РАСХОЖДЕНИЕ'))
    # d0: rv_d версии d0 в строке d = rv_d первого раунда в строке d+1 (то же RV дня d); нулевые RV (2025) — заменены
    A0, _ = load(FEAT, d0=True)
    nx = A.groupby('ticker')['rv_d'].shift(-1)
    mm = nx.notna() & (A['rv'] > 0)
    d = float(np.abs(A0.loc[mm, 'rv_d'] - nx[mm]).max())
    rows.append(dict(ryad='матрица d0: rv_d(d) = rv_d_R1(d+1)', h=h, etalon='features_pit_v3', n=int(mm.sum()), n_vne_pary=0, max_abs=d,
                     max_abs_rv_actual=np.nan, status='OK' if d <= TOL else 'РАСХОЖДЕНИЕ'))
    return rows


# ================================================================ запись
def sha256(p):
    hh = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            hh.update(b)
    return hh.hexdigest()


def _git():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ''


def write(S, h):
    CACHE.mkdir(parents=True, exist_ok=True)
    code, ys, now, git = sha256(Path(__file__))[:16], (sha256(YAML)[:16] if YAML.exists() else ''), datetime.now().strftime('%Y-%m-%d %H:%M:%S'), _git()
    rows = []
    for nm, (V, W) in S.items():
        b, v = split_name(nm)
        assert V.step.max() <= 6 and V.year.max() <= 2021
        items = [(CACHE / f'{nm}_h{h}_sdet_val.parquet', V, 'val')]
        if W is not None:
            assert W.year.max() <= 2021
            items.append((CACHE / f'{nm}_h{h}_lnall.parquet', to_parts(W), 'lnall'))
        for fp, d, kind in items:
            d.to_parquet(fp, index=False)
            rows.append(dict(file=str(fp.relative_to(REG)), ryad=nm, noga=b, versiya=v.lstrip('_') or 'r1', emb=VERSIONS[v][0], d0=VERSIONS[v][1],
                             osnovnaya=(v == OSN), semeystvo=family(nm), h=h, seed='det', vid=kind, shagi='1-6', stroki=len(d), sha256=sha256(fp),
                             sha_koda=code, git=git, yaml=ys, vremya=now,
                             primechanie='HEAVY: все версии совпадают (целей в обучении нет, RV дня d на входе)' if b == 'HEAVY' else ''))
    return rows


def write_index(rows):
    new = pd.DataFrame(rows)
    old = pd.read_csv(INDEX) if INDEX.exists() else new.iloc[:0]
    old = old[~old.file.isin(new.file) & old.file.map(lambda f: (REG / f).exists())]
    pd.concat([old, new], ignore_index=True).sort_values('file').to_csv(INDEX, index=False)


def _job(h):
    t0 = time.time()
    S = build_h(h)
    F, Q = v0_tables(S, h)
    return h, S, F, Q, sverka(S, F, h), time.time() - t0


def versii_table(Q):
    """QLIKE 2017–2020 / 2021 / 2017–2021 четырёх версий рядом (s² = все), разности к версии первого раунда."""
    Q = Q.copy()
    agg = Q.groupby(['h', 'semeystvo', 'noga', 's2', 'versiya']).apply(
        lambda g: pd.Series({'q1720': g[g.year <= 2020].q.mean(), 'q2021': g[g.year == 2021].q.mean(), 'q1721': g.q.mean()})).reset_index()
    w = agg.pivot_table(index=['h', 'semeystvo', 'noga', 's2'], columns='versiya', values=['q1720', 'q2021', 'q1721'])
    w.columns = [f'{a}_{b}' for a, b in w.columns]
    w = w.reset_index()
    for v in ('emb', 'd0', 'emb_d0'):
        w[f'd1721_{v}_pct'] = 100 * (w[f'q1721_{v}'] / w['q1721_r1'] - 1)
    return w


def det(hs=HS, workers=2):
    t0 = time.time()
    with mp.get_context('fork').Pool(min(workers, len(hs), 2)) as pool:
        res = pool.map(_job, hs, chunksize=1)
    sv, idx, QQ = [], [], []
    CACHE.mkdir(parents=True, exist_ok=True)
    for h, S, F, Q, s, dt in res:
        sv += s
        for r in s:
            print(f"h={h} {r['ryad']}: {r['status']} max|Δ|={r['max_abs']:.1e} (строк {r['n']}, вне пары {r['n_vne_pary']})", flush=True)
        pd.DataFrame(sv).to_csv(CACHE / 'sverka_E.csv', index=False)
        bad = [r['ryad'] for r in s if r['status'] != 'OK']
        if bad:
            raise SystemExit(f'стоп: h={h} расхождение: {bad}')
        idx += write(S, h)
        F.to_csv(CACHE / f'V0_F_h{h}.csv', index=False); Q.to_csv(CACHE / f'V0_q_h{h}.csv', index=False); QQ.append(Q)
        print(f'h={h}: рядов {len(S)}, {dt:.0f} с', flush=True)
    write_index(idx)
    W = versii_table(pd.concat(QQ)); W.to_csv(CACHE / 'V0_versii.csv', index=False)
    print(f'det готов: {len(idx)} файлов, {time.time() - t0:.0f} с')


# ================================================================ детектор
def _perturb(feat, cut, rng, out):
    B = pd.read_parquet(feat)
    dt = pd.to_datetime(B['date'])
    fut = (dt > cut).values
    num = [c for c in B.columns if c != 'year' and pd.api.types.is_float_dtype(B[c])]
    B.loc[fut, num] = B.loc[fut, num].values * rng.uniform(0.5, 2.0, size=(fut.sum(), len(num)))
    past = B.assign(_dt=dt).sort_values(['ticker', '_dt']).loc[lambda d: d._dt <= cut]
    for c in [c for c in num if 'target' in c]:  # цели, чей горизонт (n строк) заходит за срез
        ix = past.groupby('ticker').tail(int(c.split('_h')[-1])).index
        B.loc[ix, c] = B.loc[ix, c].values * rng.uniform(0.5, 2.0, size=len(ix))
    B.to_parquet(out, index=False)


def _dd(a, b, cut, kmax):
    """max |Δ| на строках <= cut (val и lnall моделей шагов <= kmax) и max |Δ| val после cut."""
    past, fut = 0.0, 0.0
    for nm in a:
        m = a[nm][0].merge(b[nm][0], on=KEY, suffixes=('', '_p'))
        d = np.abs(m.rv_pred - m.rv_pred_p).values; p = (m.date <= cut).values
        past = max(past, float(d[p].max()) if p.any() else 0.0); fut = max(fut, float(d[~p].max()) if (~p).any() else 0.0)
        if a[nm][1] is not None:
            m = a[nm][1].merge(b[nm][1], on=['date', 'ticker'], suffixes=('', '_p')); m = m[m.date <= cut]
            for k in range(1, min(kmax, 6) + 1):
                past = max(past, float(np.nanmax(np.abs(m[f'ln_s{k}'] - m[f'ln_s{k}_p']).values)))
    return past, fut


CUTS_MID = ('2018-06-29', '2020-03-02', '2021-06-30')
CUTS_YE = ('2017-12-31', '2019-12-31', '2020-12-31')


def _leak_h(h):
    rng = np.random.default_rng(20261026 + h); rows = []
    with tempfile.TemporaryDirectory() as td:
        b0, b1 = build_h(h, light=True), build_h(h, plant=True, light=True)
        for c in CUTS_MID + CUTS_YE:
            cut = pd.Timestamp(c); ye = c in CUTS_YE
            tmp = Path(td) / f'{c}.parquet'; _perturb(FEAT, cut, rng, tmp)
            p0 = build_h(h, tmp, light=True)
            p1 = None if ye else build_h(h, tmp, plant=True, light=True)
            kmax = cut.year - 2015 + (1 if ye else 0)
            for v in VERSIONS:
                for fam in FAMS:
                    sel = lambda D: {k: x for k, x in D.items() if family(k) == fam and split_name(k)[1] == v}
                    past0, fut0 = _dd(sel(b0), sel(p0), cut, kmax)
                    r = dict(h=h, versiya=v.lstrip('_') or 'r1', semeystvo=fam, srez=c, tip='граница года' if ye else 'середина года',
                             max_d_do_sreza=past0, max_d_posle_sreza=fut0)
                    if ye:  # проверка эмбарго: у _emb Δ = 0; у версий без эмбарго Δ > 0 ожидаема (утечка), кроме HEAVY
                        r['ok'] = bool(past0 == 0.0) if (VERSIONS[v][0] or fam == 'HEAVY') else True
                        r['utechka_vidna'] = bool(past0 > 0)
                    else:
                        past1, _ = _dd(sel(b1), sel(p1), cut, kmax)
                        r['podlog_max_d_do_sreza'] = past1
                        r['ok'] = bool(past0 == 0.0 and fut0 > 0 and past1 > 0)
                    rows.append(r)
            print(h, c, 'готов', flush=True)
    return rows


def leak(hs=HS, workers=2):
    with mp.get_context('fork').Pool(min(workers, len(hs), 2)) as pool:
        rows = sum(pool.map(_leak_h, hs, chunksize=1), [])
    R = pd.DataFrame(rows); CACHE.mkdir(parents=True, exist_ok=True); R.to_csv(CACHE / 'leak_E.csv', index=False)
    print(R.groupby(['tip', 'versiya', 'semeystvo']).agg(ok=('ok', 'all'), d_do=('max_d_do_sreza', 'max')).to_string())
    print('ВСЕ OK' if R.ok.all() else 'ЕСТЬ ОШИБКИ'); return bool(R.ok.all())


def zamer():
    t = time.time(); build_h(5, light=True, versions=('_emb_d0',))
    s = time.time() - t
    Z = pd.DataFrame([dict(edinica='h × версия, лёгкий набор (3 точки HAR-X, 2 подмножества, все шаги)', sek=round(s, 1),
                           prognoz_polnyy_min=round(s * 3 * 4 * 8 / 60, 1))])
    CACHE.mkdir(parents=True, exist_ok=True); Z.to_csv(CACHE / 'zamer_E.csv', index=False); print(Z.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['zamer', 'det', 'leak'])
    ap.add_argument('--hs', type=int, nargs='*', default=list(HS))
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    if a.mode == 'zamer':
        zamer()
    elif a.mode == 'det':
        det(tuple(a.hs), min(a.workers, 2))
    else:
        sys.exit(0 if leak(tuple(a.hs), min(a.workers, 2)) else 1)


if __name__ == '__main__':
    main()
