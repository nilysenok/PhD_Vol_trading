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
import sys, io, time, hashlib, argparse, tempfile, subprocess, warnings, multiprocessing as mp
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


# ======================================================================================================================================
# L (протокол v3, раздел 6, строка «L»): ноги кластера 2 и эконометрические части кластеров 11 (τ), 14 (панель), 15 (refit, RLS, QMLE)
# по схеме models/designs/r2/schema_v1.yaml. Всё — в версии d0 с эмбарго (load(d0=True) и обучение только на строках, у которых окно
# цели кончается до точки переобучения). Интерфейс для ядра r2_core (агент S; пока — по протоколу, раздел 5):
#   run_leg(design, M=None, steps=..., tickers=None) -> {вариант: (V, W)}
#     design — дизайн по схеме v1 (вложенный dict; недостающие ключи — default схемы, complete());
#     вариант — значения ключей-списков (сетка λ, α; κ пула; число режимов; окно τ), выбирается ядром/O внутри складки (п. 2.1);
#     V — val-строки: date, ticker, year, step, rv_actual, rv_pred (сырой, без калибровки — её делает r2_calib.py);
#     W — ln-прогноз модели шага k (годовая подгонка на 1 янв. vy) на всех строках year <= vy_k (вход ролей B; to_parts() -> tr/va).
#   M — матрица mat_L(feat) (один раз в родителе, дальше fork).
# Определения, которых нет в схеме (❓ — умолчание до ответа, записано в r2_L_nogi.yaml):
#   HARX S1..S5 — harx.TERMS: S1 = ln_rvi2; S2 = + ln_rvm_imoex; S3 = + lhar_1/5/22 + pc1; S4 = + harq + ln_rsv_pos/neg; S5 = + ln_rv66/132.
#     Сетка эластичной сети λ ∈ {0,01; 0,1; 1} × α ∈ {0; 0,5; 1} (harx.fit_en: HAR-члены без штрафа, Фриш–Во) — 9 вариантов.
#   HARX_regime — [HAR, S] + взаимодействия с индикаторами режима: 2 режима — RVI > медианы за 252 прошлых дня (Д24.4); 3 — терцили
#     RVI относительно 252 прошлых дней (два индикатора). HARX_pooled — β_i = (1−κ)β_pool + κβ_i (κ — список). robust = Huber —
#     HuberRegressor вместо МНК (у HARX с S — без штрафа). estimator = qmle_level — QMLE в уровнях: min Σ (y·e^{−xβ} + xβ).
#   MIDAS_RV — ln y = a + b·ln Σ_{j<66} w_j(θ) RV_{d−j}, веса Beta(1, θ), θ ∈ {1, 2, 3, 5, 8, 12, 20, 30} — по МНК на обучении шага.
#   HAR_perlead — отдельная HAR-модель на ln RV_{d+j}, j = 1..h; ŷ = среднее exp(прогнозов) (h=1 = HARJ).
#   RVI_anchor — Д25.1 (а): ŷ = v·k_i, v = (RVI/100)²/252, k_i — медиана RV_h/v по 252 строкам с эмбарго h; нет k — HAR-J шага.
#   factor — рынок × идиосинкразия: ln ŷ = HAR рынка (ln m_d, m_w, m_m; рынок — leg.panel_market) + пул-МНК отклонения тикера
#     ln y − ln y_рынка на ln(rv/m)_{d,w,m} с интерцептом тикера.
#   struct (кластер 11): tau_midas — τ_d = Σ_{j<W} w_j RV_{d−j}, Beta(1, 5), W = tau_window; tau_rvi — τ_d = якорь RVI (h=22);
#     нога моделирует ln(y/τ_ext) на HAR от ln(RV/τ_d); tau_extend const — τ_ext = τ_d, AR1 — ln τ среднее-возвратный AR(1) на обучении;
#     tau_boost — бустинг, агент B (здесь — NotImplementedError). struct_split = bv_plus_jump — HAR-CJ d0 (BV и скачок раздельно).
#   panel (кластер 14): P_i = α_i + β_i·ln ŷ_рынка (α, β — скользящая регрессия ln RV_i на ln m за panel_beta_window дней по день d);
#     factor — ln ŷ = P_i; factor_shrink_s — ln ŷ = (1−s)·ln ŷ_ноги + s·P_i, s = 0,25 / 0,5; _state — s = 0,5 при RVI выше медианы, иначе 0,25.
#   refit (кластер 15): yearly — подгонка на 1 янв. vy; quarterly / monthly — заново на начало квартала / месяца года vy по строкам,
#     чьё окно цели кончилось до этой даты; rls_lambda < 1 — ежедневное переобучение с весами λ^(возраст в торговых днях) (RLS с забыванием).
#   data: recency_halflife (веса 0,5^(возраст/полураспад)), winsor (1/99 по тикеру на обучении), history_start (дата из ctx) — поддержаны;
#     target ≠ ln_mean — NotImplementedError (кластер 1, не L).
SCHEMA = HERE.parents[1] / 'models' / 'designs' / 'r2' / 'schema_v1.yaml'
LDIR = REG / 'R2' / 'L'
DYM_TK = ('SBER', 'LKOH', 'AFLT')
DYM_STEPS = range(1, 4)
S_TERMS = {'S1': ['ln_rvi2'], 'S2': ['ln_rvi2', 'ln_rvm_imoex']}
S_TERMS['S3'] = S_TERMS['S2'] + ['lhar_1', 'lhar_5', 'lhar_22', 'pc1']
S_TERMS['S4'] = S_TERMS['S3'] + ['harq', 'ln_rsv_pos', 'ln_rsv_neg']
S_TERMS['S5'] = S_TERMS['S4'] + ['ln_rv66', 'ln_rv132']
ENET_LAM, ENET_ALPHA = (0.01, 0.1, 1.0), (0.0, 0.5, 1.0)
MIDAS_TH, MIDAS_K = (1, 2, 3, 5, 8, 12, 20, 30), 66
CJ_COLS = ['c_d', 'c_w', 'c_m', 'j_d', 'j_w', 'j_m']
RV_COLS = ('rv_d', 'rv_w', 'rv_m', 'c_d', 'c_w', 'c_m')


# ---------------------------------------------------------------- схема
def _onoff(x):
    """YAML 1.1 читает off/on как false/true — в схеме это строки-значения переключателей."""
    if isinstance(x, bool):
        return 'on' if x else 'off'
    if isinstance(x, list):
        return [_onoff(v) for v in x]
    if isinstance(x, dict):
        return {k: _onoff(v) for k, v in x.items()}
    return x


def _core():
    import r2_core as RC
    return RC


def complete(design):
    """Полный дизайн через r2_core.validate (default схемы, нормализация, DesignError), затем on/off — строками (внутреннее правило L)."""
    return _onoff(_core().validate(design))


# ---------------------------------------------------------------- матрица d0
def mat_L(feat=FEAT, plant=False, tickers=None):
    """r2_core.load_d0 (d0, строки <= 2021, _tend{h}) + всё, что нужно ногам L (mat_from). plant — подлог r2_core.plant_frame
    (все входы строки d := строки d+1). tickers — срез строк ПОСЛЕ построения рыночных рядов."""
    RC = _core()
    A, harj = RC.load_d0(feat=feat)
    if plant:
        A = RC.plant_frame(A)
    return mat_from(A, harj, tickers)


def mat_from(A, harj, tickers=None):
    """Колонки ног L по уже загруженной (возможно, испорченной детектором) матрице d0. Всё строится только из прошлого и дня d."""
    A = A.copy()
    A['date'] = pd.to_datetime(A['date']); A['year'] = A['date'].dt.year
    A = A.sort_values(['date', 'ticker']).reset_index(drop=True)
    g = A.groupby('ticker')
    for h in HS:
        A[f'_tend{h}'] = pd.to_datetime(g['date'].shift(-h)) if f'_tend{h}' not in A else pd.to_datetime(A[f'_tend{h}'])
        A[f'_tendy{h}'] = A[f'_tend{h}'].dt.year
    rvp = A['rv'].where(A['rv'] > 0).groupby(A['ticker']).ffill().fillna(A['rv']).clip(lower=1e-10)
    A['_rvp'] = rvp
    bvp = A['bv'].where(A['bv'] > 0).groupby(A['ticker']).ffill().fillna(A['bv'])  # HAR-CJ d0 (как vyr_spravochnye.load_cj)
    for src, p in ((bvp, 'c'), (A['jump'], 'j')):
        A[p + '_d'] = src
        for k, n in (('w', 5), ('m', 22)):
            A[p + '_' + k] = src.groupby(A['ticker']).transform(lambda v: v.rolling(n, min_periods=1).mean())
    HX._SIG.clear(); HX.GH_HARJ = harj
    X = HX.build_terms(A, S_TERMS['S5'])
    for c in X.columns:
        A['S_' + c] = X[c].values
    # режимы RVI (по датам, только прошлое): медиана и терцили за 252 прошлых дня
    dd = A.groupby('date')['close_RVI'].first().sort_index().astype(float).ffill()
    past = dd.shift(1).rolling(252, min_periods=126)
    reg = pd.DataFrame({'I2': (dd > past.median()).astype(float), 'I3a': (dd > past.quantile(1 / 3)).astype(float),
                        'I3b': (dd > past.quantile(2 / 3)).astype(float)}).where(past.count().ge(126), 0.0)
    A = A.merge(reg, left_on='date', right_index=True, how='left')
    # рынок: RV_IMOEX (по дате) и RV_eq17 (среднее rv 17 тикеров дня)
    mk = pd.DataFrame({'m_RV_IMOEX': A.groupby('date')['rv_IMOEX'].first(), 'm_RV_eq17': A.groupby('date')['_rvp'].mean()}).sort_index()
    mk['m_RV_IMOEX'] = mk['m_RV_IMOEX'].where(mk['m_RV_IMOEX'] > 0).ffill().fillna(mk['m_RV_eq17'])  # только прошлое (без bfill)
    A = A.merge(mk, left_on='date', right_index=True, how='left')
    A['lnrvi2'] = A['S_ln_rvi2']
    A['vov'] = np.log(A['_rvp']).groupby(A['ticker']).transform(lambda s: s.rolling(22, min_periods=11).std())
    A = A.sort_values(['date', 'ticker']).reset_index(drop=True)
    if tickers is not None:
        A = A[A.ticker.isin(tickers)].reset_index(drop=True)
    return {'A': A, 'harj': harj, 'mk': mk}


def target_col(h):
    return f'rv_target_h{h}'


# ---------------------------------------------------------------- линейное ядро
def _fit_en_w(H, Z, y, al, l1, w):
    """harx.fit_en с весами наблюдений (w=None — сам harx.fit_en)."""
    if w is None:
        p, co = HX.fit_en(H, Z, y, al, l1)
        p.b = np.concatenate([co['b'], co['g_raw']])
        return p
    from sklearn.linear_model import ElasticNet
    ws = w / w.sum()
    mu = ws @ Z; sd = np.sqrt(ws @ (Z - mu) ** 2)
    keep = sd > 1e-12
    Zs = np.zeros_like(Z); Zs[:, keep] = (Z[:, keep] - mu[keep]) / sd[keep]
    Hc = np.column_stack([np.ones(len(H)), H]); sw = np.sqrt(w)[:, None]
    B = np.linalg.lstsq(Hc * sw, np.column_stack([y, Zs]) * sw, rcond=None)[0]
    R = np.column_stack([y, Zs]) - Hc @ B
    g = np.zeros(Z.shape[1])
    if keep.any():
        en = ElasticNet(alpha=al, l1_ratio=l1, fit_intercept=False, max_iter=20000, tol=1e-7, precompute=True)
        en.fit(R[:, 1:][:, keep], R[:, 0], sample_weight=w * len(w) / w.sum()); g[keep] = en.coef_
    b = B[:, 0] - B[:, 1:] @ g
    f = lambda H2, Z2: np.column_stack([np.ones(len(H2)), H2]) @ b + np.where(keep, (Z2 - mu) / np.where(keep, sd, 1), 0) @ g
    f.b = np.concatenate([b, np.where(keep, g / np.where(keep, sd, 1), 0.0)])
    return f


def _fit_lin(kind, M, y, w=None, ylvl=None):
    """kind: ols | huber | qmle. M — регрессоры (без константы). -> predict(M2) в логах."""
    Mc = np.column_stack([np.ones(len(M)), M])
    if kind == 'ols':
        sw = np.ones(len(y)) if w is None else np.sqrt(w)
        b = np.linalg.lstsq(Mc * sw[:, None], y * sw, rcond=None)[0]
    elif kind == 'huber':
        from sklearn.linear_model import HuberRegressor
        mu, sd = M.mean(0), M.std(0); sd[sd < 1e-12] = 1
        r = HuberRegressor(max_iter=500).fit((M - mu) / sd, y, sample_weight=w)
        b = np.concatenate([[r.intercept_ - (r.coef_ / sd) @ mu], r.coef_ / sd])
    elif kind == 'qmle':
        from scipy.optimize import minimize
        yl = np.exp(y) if ylvl is None else ylvl
        ww = np.ones(len(y)) if w is None else w
        b0 = np.linalg.lstsq(Mc, y, rcond=None)[0]
        f = lambda b: float(ww @ (yl * np.exp(-(Mc @ b)) + Mc @ b)) / len(y)
        gr = lambda b: Mc.T @ (ww * (1 - yl * np.exp(-(Mc @ b)))) / len(y)
        b = minimize(f, b0, jac=gr, method='L-BFGS-B').x
    else:
        raise ValueError(kind)
    f = lambda M2: np.column_stack([np.ones(len(M2)), M2]) @ b
    f.b = b
    return f


def _rec(ctx, step, name, value):
    """Параметр шага в общий формат D (design_sha, h, step, param, value) — пишет write_params."""
    if ctx.get('_par') is not None and value is not None and np.isfinite(float(value)):
        ctx['_par'].append((int(step), str(name), float(value)))


def _kind(leg):
    return 'qmle' if leg['estimator'] == 'qmle_level' else ('huber' if leg['robust'] == 'Huber' else 'ols')


def _base_H(A, leg):
    cols = CJ_COLS if leg['struct_split'] == 'bv_plus_jump' else None
    return cols


def _weights(A, rows, cutoff, data):
    hl = data.get('recency_halflife', 'inf')
    if hl in ('inf', None, float('inf')):
        return None
    yrs = float(str(hl).rstrip('y'))
    age = (np.datetime64(cutoff) - A['date'].values[rows]) / np.timedelta64(1, 'D') / 365.25
    return 0.5 ** (age / yrs)


def _train_rows(A, h, cutoff, data, ctx, extra=None):
    t = target_col(h)
    m = (A['date'].values < np.datetime64(cutoff)) & A[t].notna().values & (A[f'_tend{h}'].values < np.datetime64(cutoff))
    if ctx.get('history_start_date') is not None:
        m &= A['date'].values >= np.datetime64(ctx['history_start_date'])
    if extra is not None:
        m &= extra
    return m


def _y(A, h, rows, data):
    y = np.log(A[target_col(h)].values[rows] + 1e-10)
    if data.get('winsor') == 'p1_99_by_ticker':
        tk = A['ticker'].values[rows]
        s = pd.Series(y)
        lo = s.groupby(tk).transform(lambda v: v.quantile(0.01)); hi = s.groupby(tk).transform(lambda v: v.quantile(0.99))
        y = s.clip(lo, hi).values
    return y


def _cutoffs(A, vy, leg):
    """Точки переобучения в году vy: (дата, маска строк прогноза)."""
    yr = A['year'].values == vy
    dates = np.sort(A.loc[yr, 'date'].unique())
    if leg['rls_lambda'] != 1:
        cs = list(pd.to_datetime(dates))
    elif leg['refit'] == 'yearly':
        cs = [pd.Timestamp(f'{vy}-01-01')]
    else:
        n = 4 if leg['refit'] == 'quarterly' else 12
        cs = [pd.Timestamp(year=vy, month=1 + 12 // n * i, day=1) for i in range(n)]
    out = []
    for i, c in enumerate(cs):
        nxt = cs[i + 1] if i + 1 < len(cs) else pd.Timestamp(f'{vy + 1}-01-01')
        out.append((c, yr & (A['date'].values >= np.datetime64(c)) & (A['date'].values < np.datetime64(nxt))))
    return out


def _rls_w(A, rows, cutoff, lam, didx):
    if lam == 1:
        return None
    age = didx[pd.Timestamp(cutoff)] - A['_di'].values[rows]
    return lam ** age.astype(float)


# ---------------------------------------------------------------- τ (кластер 11)
def _beta_w(n, th1, th2):
    x = (np.arange(n) + 0.5) / n
    w = x ** (th1 - 1) * (1 - x) ** (th2 - 1)
    return w / w.sum()


def _lagsum(A, col, w):
    """Σ_j w_j·col_{d−j} по тикеру (только прошлое и день d)."""
    out = np.full(len(A), np.nan)
    for _, ix in A.groupby('ticker').indices.items():
        v = A[col].values[ix]
        c = np.convolve(v, w, mode='full')[:len(v)]
        c[:len(w) - 1] = np.nan
        out[ix] = c
    return out


def _tau(A, h, leg, ctx):
    if leg['struct'] == 'single':
        return None
    if leg['struct'] == 'tau_boost':
        raise NotImplementedError('struct = tau_boost — бустинг-компонента τ, агент B')
    if leg['struct'] == 'tau_midas':
        lt = np.log(np.clip(_lagsum(A, '_rvp', _beta_w(int(leg['tau_window']), 1, 5)), 1e-10, None))
    else:  # tau_rvi: якорь RVI h=22
        lt = np.log(np.clip(_rvi_anchor(A, 22, ctx)[0], 1e-10, None))
    return pd.Series(lt, index=A.index).groupby(A['ticker']).ffill().values


def _tau_ext(A, h, lt, leg, tr_cut, ctx=None):
    """ln τ_ext для всех строк при обучении до tr_cut: const — ln τ_d; AR1 — среднее по j=1..h итераций AR(1)."""
    if leg['tau_extend'] == 'const':
        return lt
    nxt = pd.Series(lt).groupby(A['ticker'].values).shift(-1).values
    m = np.isfinite(lt) & np.isfinite(nxt) & (A['_tend1'].values < np.datetime64(tr_cut))
    b = np.linalg.lstsq(np.column_stack([np.ones(m.sum()), lt[m]]), nxt[m], rcond=None)[0]
    phi = min(max(b[1], 0.0), 0.999); mu = b[0] / (1 - phi) if phi < 0.999 else float(np.nanmean(lt[m]))
    if ctx is not None:
        ctx['_ar1'] = {'tau_ar1_phi': phi, 'tau_ar1_mu': mu}
    return mu + (lt - mu) * np.mean(phi ** np.arange(1, h + 1))


# ---------------------------------------------------------------- RVI-якорь, MIDAS, рынок
def _rvi_anchor(A, h, ctx):
    import d25_1 as D25
    B = A[['date', 'ticker', 'close_RVI', f'rv_target_mean_h{h}']].copy()
    L = D25.rvi_leg(B, h, lag=h)  # k — эмбарго h; подлог детектора — r2_core.plant_frame (RVI дня d+1)
    m = A[['date', 'ticker']].merge(L, on=['date', 'ticker'], how='left')
    return m['leg'].values, m['k'].values


def _midas_cols(A):
    out = {}
    for th in MIDAS_TH:
        out[th] = np.log(np.clip(_lagsum(A, '_rvp', _beta_w(MIDAS_K, 1, th)), 1e-10, None))
    return out


def _market(A, mk, h, col, cutoff, steps_rows):
    """HAR рынка по датам: ln mean_{d+1..d+h} m на ln m_d, m_w, m_m; обучение — даты, чья цель кончилась до cutoff. -> ln ŷ_рынка по строкам A."""
    m = mk[col].astype(float)
    lm = np.log(m.clip(lower=1e-10))
    Xd = pd.DataFrame({'d': lm, 'w': np.log(m.rolling(5, min_periods=1).mean()), 'mm': np.log(m.rolling(22, min_periods=1).mean())})
    fut = sum(m.shift(-j) for j in range(1, h + 1)) / h
    yd = np.log(fut.clip(lower=1e-10))
    dts = m.index.values
    tend = pd.Series(dts, index=m.index).shift(-h).values
    tr = np.isfinite(yd.values) & (tend < np.datetime64(cutoff)) & np.isfinite(Xd.values).all(1)
    b = np.linalg.lstsq(np.column_stack([np.ones(tr.sum()), Xd.values[tr]]), yd.values[tr], rcond=None)[0]
    pr = pd.Series(np.column_stack([np.ones(len(Xd)), Xd.fillna(0).values]) @ b, index=m.index)
    return pr.reindex(A['date']).values, Xd, yd


# ---------------------------------------------------------------- ноги
def _lin_mats(A, h, leg, harj):
    """H (HAR или HAR-CJ, лог), Zdf (слагаемые S, DataFrame или None)."""
    cols = CJ_COLS if leg['struct_split'] == 'bv_plus_jump' else list(harj)
    H = GH.clean(np.log(A[cols].values.astype('float64').clip(1e-10)))
    rvi = [i for i, c in enumerate(cols) if c in RV_COLS]
    Zdf = None
    if leg['type'] in ('HARX', 'HARX_regime', 'HARX_pooled', 'HAR_perlead') and leg['regressors'] != 'none':
        Zdf = A[['S_' + c for c in S_TERMS[leg['regressors']]]]
    names = list(cols) + ([] if Zdf is None else [c[2:] for c in Zdf.columns])
    return H, rvi, Zdf, names


def grid_keys_L(full):
    """Сетки внутри складки, которые относятся к ноге этого дизайна (r2_core.variants(keys=...))."""
    leg = _onoff(full)['leg']
    k = []
    if leg['type'] == 'HARX' and leg['regressors'] != 'none' and _kind(leg) == 'ols':
        k += ['leg.enet.lambda', 'leg.enet.alpha']
    if leg['type'] == 'HARX_pooled':
        k.append('leg.pooled_kappa')
    if leg['type'] == 'HARX_regime':
        k.append('leg.regime_count')
    if leg['struct'] == 'tau_midas':
        k.append('leg.tau_window')
    return k


def _scal(x):
    return x[0] if isinstance(x, list) else x


def _variants(full):
    """-> [(тег r2_core.variants, конкретный leg, var)]; ключи-списки, не относящиеся к ноге, берутся первым значением."""
    out = []
    for tag, _, d in _core().variants(full, keys=grid_keys_L(full)):
        leg = dict(_onoff(d)['leg'])
        for k in ('pooled_kappa', 'regime_count', 'tau_window', 'rls_lambda'):
            leg[k] = _scal(leg[k])
        e = leg['enet'] if isinstance(leg['enet'], dict) else {'lambda': ENET_LAM, 'alpha': ENET_ALPHA}
        var = {'lam': float(_scal(e['lambda'])), 'alpha': float(_scal(e['alpha'])), 'pooled_kappa': leg['pooled_kappa'],
               'regime_count': leg['regime_count'], 'tau_window': leg['tau_window']}
        out.append((tag, leg, var))
    return out


def _predict_linear(A, h, leg, var, harj, data, ctx, tr, pm, lt_ext, didx, cutoff):
    """Одна подгонка линейной ноги на строках tr; ln-прогноз на строках pm (маски)."""
    H, rvi, Zdf, names = _lin_mats(A, h, leg, harj)
    if lt_ext is not None:
        H = H.copy(); H[:, rvi] = H[:, rvi] - np.nan_to_num(ctx['_lt'])[:, None]  # HAR от ln(RV/τ_d)
    Z = None if Zdf is None else Zdf.fillna(Zdf[tr].mean()).fillna(0.0).values.astype('float64')
    y = _y(A, h, tr, data) - (lt_ext[tr] if lt_ext is not None else 0)
    w = _weights(A, tr, cutoff, data)
    wr = _rls_w(A, tr, cutoff, leg['rls_lambda'], didx)
    w = wr if w is None else (w if wr is None else w * wr)
    kind = _kind(leg)
    off = lt_ext[pm] if lt_ext is not None else 0
    t = leg['type']
    bn = ['const'] + names
    if t in ('HARJ', 'HARX') and Z is not None and kind == 'ols':
        p = _fit_en_w(H[tr], Z[tr], y, var['lam'], var['alpha'], w)
        ctx['_lastb'] = dict(zip(bn, p.b))
        return p(H[pm], Z[pm]) + off
    M = H if Z is None else np.column_stack([H, Z])
    if t == 'HARX_regime':
        I = A[['I2']].values if var.get('regime_count', 2) == 2 else A[['I3a', 'I3b']].values
        M = np.column_stack([M] + [np.column_stack([I[:, j], I[:, j:j + 1] * M]) for j in range(I.shape[1])])
        bn = bn + sum([[f'I{j}'] + [f'I{j}*{n}' for n in names] for j in range(I.shape[1])], [])
    ylvl = np.exp(y)
    if t == 'HARX_pooled':
        kap = float(var.get('pooled_kappa', 1))
        f0 = _fit_lin(kind, M[tr], y, w, ylvl)
        ctx['_lastb'] = {f'pool_{k}': v for k, v in zip(bn, f0.b)}
        out = f0(M[pm])
        tk_tr, tk_pm = A['ticker'].values[tr], A['ticker'].values[pm]
        for tk in np.unique(tk_pm):
            mi = tk_tr == tk
            if kap > 0 and mi.sum() > M.shape[1] + 10:
                fi = _fit_lin(kind, M[tr][mi], y[mi], None if w is None else w[mi], ylvl[mi])
                pj = tk_pm == tk
                out[pj] = (1 - kap) * out[pj] + kap * fi(M[pm][pj])
        return out + off
    f = _fit_lin(kind, M[tr], y, w, ylvl)
    ctx['_lastb'] = dict(zip(bn, f.b))
    return f(M[pm]) + off


def _run_linear(A, h, leg, var, harj, data, ctx, steps, didx):
    t = target_col(h)
    lt = _tau(A, h, dict(leg, tau_window=var.get('tau_window', leg['tau_window'])), ctx)
    V, Wl = [], {}
    for k in steps:
        vy = 2015 + k
        lnv = np.full(len(A), np.nan)
        for c, pm in _cutoffs(A, vy, leg):
            pm = pm & A[t].notna().values
            if not pm.any():
                continue
            tr = _train_rows(A, h, c, data, ctx)
            lte = None
            if lt is not None:
                ctx['_lt'] = lt; lte = _tau_ext(A, h, lt, leg, c)
                tr &= np.isfinite(lte) & np.isfinite(lt)
            lnv[pm] = _predict_linear(A, h, leg, var, harj, data, ctx, tr, pm, lte, didx, c)
        V.append(_val(A, h, k, lnv))
        c0 = pd.Timestamp(f'{vy}-01-01'); tr0 = _train_rows(A, h, c0, data, ctx)
        lte0 = None
        if lt is not None:
            ctx['_lt'] = lt; lte0 = _tau_ext(A, h, lt, leg, c0, ctx); tr0 &= np.isfinite(lte0) & np.isfinite(lt)
        allr = (A['year'].values <= vy) & (lt is None or np.isfinite(lte0))
        ln0 = np.full(len(A), np.nan)
        ln0[allr] = _predict_linear(A, h, dict(leg, rls_lambda=1), var, harj, data, ctx, tr0, allr, lte0, didx, c0)
        Wl[k] = ln0
        # параметры шага (годовая подгонка на 1 янв. vy): коэффициенты, точка сетки, κ пула, режимы, τ
        for n, v in ctx.get('_lastb', {}).items():
            _rec(ctx, k, f'b_{n}', v)
        if leg['type'] == 'HARX' and leg['regressors'] != 'none' and _kind(leg) == 'ols':
            _rec(ctx, k, 'lambda', var['lam']); _rec(ctx, k, 'alpha', var['alpha'])
        if leg['type'] == 'HARX_pooled':
            _rec(ctx, k, 'pooled_kappa', var['pooled_kappa'])
        if leg['type'] == 'HARX_regime':
            _rec(ctx, k, 'regime_count', var['regime_count'])
        if lt is not None:
            _rec(ctx, k, 'tau_window', var['tau_window'])
            for n, v in ctx.get('_ar1', {}).items():
                _rec(ctx, k, n, v)
        _rec(ctx, k, 'n_train', int(tr0.sum()))
    return _fin(V), _lnw(A, h, Wl)


def _run_perlead(A, h, leg, var, harj, data, ctx, steps, didx):
    if h == 1:
        return _run_linear(A, h, dict(leg, type='HARJ' if leg['regressors'] == 'none' else 'HARX'), var, harj, data, ctx, steps, didx)
    leads = {j: A.groupby('ticker')['_rvp'].shift(-j).values for j in range(1, h + 1)}
    V, Wl = [], {}
    tcol = target_col(h)
    for k in steps:
        vy = 2015 + k
        acc_v = np.zeros(len(A)); acc_w = np.zeros(len(A))
        c0 = pd.Timestamp(f'{vy}-01-01')
        allr = A['year'].values <= vy
        for j, lead in leads.items():
            B = A.assign(**{tcol: lead})
            B[tcol] = B[tcol].where(A[tcol].notna())
            lnv = np.full(len(A), np.nan)
            for c, pm in _cutoffs(A, vy, leg):
                pm = pm & A[tcol].notna().values
                if pm.any():
                    tr = _train_rows(B, h, c, data, ctx)
                    lnv[pm] = _predict_linear(B, h, dict(leg, type='HARJ' if leg['regressors'] == 'none' else 'HARX_pooled', pooled_kappa=0),
                                              dict(var, pooled_kappa=0), harj, data, ctx, tr, pm, None, didx, c)
            acc_v += np.exp(np.nan_to_num(lnv, nan=-50))
            tr0 = _train_rows(B, h, c0, data, ctx)
            l0 = np.full(len(A), np.nan)
            l0[allr] = _predict_linear(B, h, dict(leg, rls_lambda=1, type='HARJ' if leg['regressors'] == 'none' else 'HARX_pooled', pooled_kappa=0),
                                       dict(var, pooled_kappa=0), harj, data, ctx, tr0, allr, None, didx, c0)
            acc_w += np.exp(np.nan_to_num(l0, nan=-50))
        V.append(_val(A, h, k, np.log(acc_v / h)))
        Wl[k] = np.where(allr, np.log(acc_w / h), np.nan)
    return _fin(V), _lnw(A, h, Wl)


def _run_midas(A, h, leg, var, harj, data, ctx, steps, didx):
    if '_midas' not in ctx:
        ctx['_midas'] = _midas_cols(A)
    MC = ctx['_midas']; t = target_col(h)
    V, Wl = [], {}
    for k in steps:
        vy = 2015 + k
        lnv = np.full(len(A), np.nan)

        def fit(c):
            tr = _train_rows(A, h, c, data, ctx) & np.isfinite(MC[MIDAS_TH[0]]) & np.isfinite(MC[MIDAS_TH[-1]])
            y = _y(A, h, tr, data); w = _weights(A, tr, c, data)
            best = None
            for th in MIDAS_TH:
                f = _fit_lin(_kind(leg), MC[th][tr, None], y, w)
                sse = float(np.mean((y - f(MC[th][tr, None])) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, th, f)
            return best
        for c, pm in _cutoffs(A, vy, leg):
            pm = pm & A[t].notna().values
            if pm.any():
                _, th, f = fit(c)
                lnv[pm] = f(MC[th][pm, None])
        V.append(_val(A, h, k, lnv))
        _, th, f = fit(pd.Timestamp(f'{vy}-01-01'))
        Wl[k] = np.where(A['year'].values <= vy, f(np.nan_to_num(MC[th])[:, None]), np.nan)
        _rec(ctx, k, 'midas_theta', th); _rec(ctx, k, 'b_const', f.b[0]); _rec(ctx, k, 'b_ln_midas', f.b[1])
    return _fin(V), _lnw(A, h, Wl)


def _run_anchor(A, h, leg, var, harj, data, ctx, steps, didx):
    leg_v, kk = _rvi_anchor(A, h, ctx)
    hv = _run_linear(A, h, dict(leg, type='HARJ', regressors='none', struct='single'), {}, harj, data, {}, steps, didx)
    for k in steps:
        sel = (A['year'].values == 2015 + k)
        _rec(ctx, k, 'anchor_k_median', float(np.nanmedian(kk[sel])) if np.isfinite(kk[sel]).any() else np.nan)
        _rec(ctx, k, 'anchor_dolya_bez_k', float(np.mean(~np.isfinite(kk[sel]))))
    V = hv[0].merge(pd.DataFrame({'date': A['date'], 'ticker': A['ticker'], 'anc': leg_v}), on=['date', 'ticker'], how='left')
    V['rv_pred'] = V['anc'].fillna(V['rv_pred'])
    W = hv[1].copy()
    an = pd.DataFrame({'date': A['date'], 'ticker': A['ticker'].astype(str), 'a': np.log(np.clip(leg_v, 1e-10, None))})
    a = W[['date', 'ticker']].merge(an, on=['date', 'ticker'], how='left')['a'].values
    for c in [c for c in W.columns if c.startswith('ln_s')]:
        W[c] = np.where(np.isfinite(a) & W[c].notna().values, a, W[c].values)
    return V[COLS], W


def _run_factor(A, h, leg, var, harj, data, ctx, steps, didx, mk):
    col = 'm_' + leg['panel_market']; t = target_col(h)
    lm = np.log(A[col].values)
    X = np.column_stack([np.log(A['_rvp'].values) - lm,
                         np.log(A.groupby('ticker')['_rvp'].transform(lambda s: s.rolling(5, min_periods=1).mean()).values) - np.log(A['m5_' + col].values),
                         np.log(A.groupby('ticker')['_rvp'].transform(lambda s: s.rolling(22, min_periods=1).mean()).values) - np.log(A['m22_' + col].values)])
    D = pd.get_dummies(A['ticker']).values.astype(float)
    V, Wl = [], {}
    for k in steps:
        vy = 2015 + k
        lnv = np.full(len(A), np.nan)

        def fit(c, pm):
            mp, Xd, yd = _market(A, mk, h, col, c, None)
            ym = yd.reindex(A['date']).values
            tr = _train_rows(A, h, c, data, ctx) & np.isfinite(ym)
            y = _y(A, h, tr, data) - ym[tr]
            M = np.column_stack([X, D[:, 1:]])
            f = _fit_lin(_kind(leg), M[tr], y, _weights(A, tr, c, data))
            return mp[pm] + f(M[pm])
        for c, pm in _cutoffs(A, vy, leg):
            pm = pm & A[t].notna().values
            if pm.any():
                lnv[pm] = fit(c, pm)
        V.append(_val(A, h, k, lnv))
        allr = A['year'].values <= vy
        l0 = np.full(len(A), np.nan); l0[allr] = fit(pd.Timestamp(f'{vy}-01-01'), allr)
        Wl[k] = l0
    return _fin(V), _lnw(A, h, Wl)


def _run_garch(A, h, leg, var, harj, data, ctx, steps, didx):
    import d24_6 as G
    m = 'heavy' if leg['type'] == 'HEAVY' else 'rgarch'
    a = A[['date', 'ticker', 'close', 'rv', target_col(h), f'_tendy{h}', 'year']].copy()
    V, W = [], []
    for tk in sorted(a.ticker.unique()):
        gg = G.prep(a, tk)
        lns = {}
        for k in steps:
            vy = 2015 + k
            F, par = G.fit_forecast(gg.rr.values, gg.xx.values, int((gg.year < vy).sum()))[m]
            for pn, pv in par.items():
                if isinstance(pv, (int, float, np.floating)):
                    _rec(ctx, k, f'{tk}.{pn}', pv)
            ln = np.log(np.clip(F[h] / 1e4, 1e-10, None))
            sel = ((gg.year == vy) & gg[target_col(h)].notna()).values
            V.append(pd.DataFrame({'date': gg.date.values[sel], 'ticker': tk, 'year': vy, 'step': k,
                                   'rv_actual': gg[target_col(h)].values[sel], 'rv_pred': np.exp(ln[sel])}))
            lns[k] = ln
        W.append(_lnw(gg, h, lns))
    return _fin(V), pd.concat(W).sort_values(['date', 'ticker']).reset_index(drop=True)


def _panel(A, h, leg, V, W, mk, data, ctx, steps):
    """Кластер 14: P_i = α_i + β_i·ln ŷ_рынка; смесь с ногой по leg.panel."""
    if leg['panel'] == 'off':
        return V, W
    col = 'm_' + leg['panel_market']; n = int(leg['panel_beta_window'])
    x = np.log(A[col].values); y = np.log(A['_rvp'].values)
    df = pd.DataFrame({'tk': A['ticker'].values, 'x': x, 'y': y, 'xy': x * y, 'xx': x * x})
    r = df.groupby('tk')[['x', 'y', 'xy', 'xx']].transform(lambda s: s.rolling(n, min_periods=n // 2).mean())
    beta = ((r.xy - r.x * r.y) / (r.xx - r.x ** 2)).values
    alpha = (r.y - beta * r.x).values
    s = {'factor': 1.0, 'factor_shrink_025': 0.25, 'factor_shrink_05': 0.5}.get(leg['panel'])
    sv = np.where(A['I2'].values > 0, 0.5, 0.25) if s is None else np.full(len(A), s)
    key = pd.DataFrame({'date': A['date'], 'ticker': A['ticker'], 'alpha': alpha, 'beta': beta, 'sv': sv})
    outV, W2 = [], W.copy()
    for k in steps:
        vy = 2015 + k
        mp = _market(A, mk, h, col, pd.Timestamp(f'{vy}-01-01'), None)[0]
        P = key.assign(P=alpha + beta * mp)
        v = V[V.step == k].merge(P, on=['date', 'ticker'], how='left')
        lnl = np.log(v.rv_pred.values)
        Pk = np.where(np.isfinite(v.P.values), v.P.values, lnl)
        v['rv_pred'] = np.exp((1 - v.sv.values) * lnl + v.sv.values * Pk)
        outV.append(v[COLS])
        _rec(ctx, k, 'panel_beta_median', float(np.nanmedian(v.beta.values)) if np.isfinite(v.beta.values).any() else np.nan)
        _rec(ctx, k, 'panel_s_mean', float(np.mean(v.sv.values)))
        wm = W2[['date', 'ticker']].merge(P, on=['date', 'ticker'], how='left')
        Pw = np.where(np.isfinite(wm.P.values), wm.P.values, W2[f'ln_s{k}'].values)
        W2[f'ln_s{k}'] = (1 - wm.sv.values) * W2[f'ln_s{k}'].values + wm.sv.values * Pw
    return _fin(outV), W2


def _prep(M):
    A, harj, mk = M['A'], M['harj'], M['mk']
    if 'm5_m_RV_IMOEX' not in A:
        for col in ('m_RV_IMOEX', 'm_RV_eq17'):
            mm = mk[col].astype(float)
            A['m5_' + col] = mm.rolling(5, min_periods=1).mean().reindex(A['date']).values
            A['m22_' + col] = mm.rolling(22, min_periods=1).mean().reindex(A['date']).values
    if '_di' not in A:
        ud = np.sort(A['date'].unique()); A['_di'] = np.searchsorted(ud, A['date'].values)
    if '_didx' not in M:
        M['_didx'] = {pd.Timestamp(x): i for i, x in enumerate(np.sort(A['date'].unique()))}
    return A, harj, mk, M['_didx']


def fit_predict(design, train_mask, predict_mask, M=None, tag=None):
    """Нога «обучить на маске -> прогноз» (для E: window_ens, ticker_bag, смесь экспертов).
    design — дизайн по схеме v1 (нужны h, leg.*, data.*); если в нём есть сетки ноги — tag выбирает вариант (тег r2_core.variants).
    train_mask, predict_mask — булевы маски по строкам M['A'] (M — mat_from/mat_L; по умолчанию r2_core.SHARED['M']).
      Эмбарго и info_set — ответственность вызывающего: строки обучения должны иметь окно цели до первой даты прогноза
      (r2_core.train_mask / _tend{h}); здесь маска только пересекается с «цель есть».
    -> pd.Series ln-прогноза (без калибровки) с индексом строк A[predict_mask].
    Все ноги кластера 2 (HARJ, HARX, HARX_regime, HARX_pooled, MIDAS_RV, HAR_perlead, RVI_anchor, factor, HEAVY, RealizedGARCH),
    struct (τ), struct_split, robust, estimator, data.recency/winsor, panel. refit/RLS не применяются: одна подгонка на маске
    (веса давности — от первой даты прогноза). HEAVY / RealizedGARCH: целей в обучении нет, поэтому QMLE — по всем строкам тикера
    до его первой даты прогноза (как run_leg); обучение должно начинаться с начала истории тикера, иначе ValueError."""
    d = complete(design)
    h, data = int(d['h']), d['data']
    M = _core().SHARED['M'] if M is None else M
    A, harj, mk, didx = _prep(M)
    vs = _variants(d)
    if tag is None and len(vs) > 1:
        raise ValueError(f'в дизайне сетка ноги — нужен tag из {[v[0] for v in vs]}')
    tg, leg, var = next(v for v in vs if tag is None or v[0] == tag)
    leg = dict(leg, rls_lambda=1)
    t = leg['type']; tc = target_col(h)
    tr = np.asarray(train_mask, bool) & A[tc].notna().values
    pm = np.asarray(predict_mask, bool)
    c = pd.Timestamp(A['date'].values[pm].min())
    ctx = {}
    ln = np.full(len(A), np.nan)
    if t in ('HARJ', 'HARX', 'HARX_regime', 'HARX_pooled'):
        lt = _tau(A, h, dict(leg, tau_window=var['tau_window']), ctx)
        lte = None
        if lt is not None:
            ctx['_lt'] = lt; lte = _tau_ext(A, h, lt, leg, c); tr = tr & np.isfinite(lte) & np.isfinite(lt)
        ln[pm] = _predict_linear(A, h, leg, var, harj, data, ctx, tr, pm, lte, didx, c)
    elif t == 'HAR_perlead':
        if h == 1:
            ln[pm] = _predict_linear(A, h, dict(leg, type='HARJ' if leg['regressors'] == 'none' else 'HARX'), var, harj, data, ctx, tr, pm, None, didx, c)
        else:
            acc = np.zeros(int(pm.sum()))
            for j in range(1, h + 1):
                B = A.assign(**{tc: A.groupby('ticker')['_rvp'].shift(-j).where(A[tc].notna())})
                acc += np.exp(_predict_linear(B, h, dict(leg, type='HARJ' if leg['regressors'] == 'none' else 'HARX_pooled', pooled_kappa=0),
                                              dict(var, pooled_kappa=0), harj, data, ctx, tr & B[tc].notna().values, pm, None, didx, c))
            ln[pm] = np.log(acc / h)
    elif t == 'MIDAS_RV':
        MC = _midas_cols(A); y = _y(A, h, tr, data); w = _weights(A, tr, c, data)
        best = None
        for th in MIDAS_TH:
            ok = np.isfinite(MC[th][tr])
            f = _fit_lin(_kind(leg), MC[th][tr][ok, None], y[ok], None if w is None else w[ok])
            sse = float(np.mean((y[ok] - f(MC[th][tr][ok, None])) ** 2))
            if best is None or sse < best[0]:
                best = (sse, th, f)
        ln[pm] = best[2](np.nan_to_num(MC[best[1]][pm])[:, None])
    elif t == 'RVI_anchor':
        a, _ = _rvi_anchor(A, h, ctx)
        hj = _predict_linear(A, h, dict(leg, type='HARJ', regressors='none', struct='single'), var, harj, data, ctx, tr, pm, None, didx, c)
        ln[pm] = np.where(np.isfinite(a[pm]) & (a[pm] > 0), np.log(np.clip(a[pm], 1e-10, None)), hj)
    elif t == 'factor':
        col = 'm_' + leg['panel_market']
        mp, _, yd = _market(A, mk, h, col, c, None)
        ym = yd.reindex(A['date']).values
        lm = np.log(A[col].values)
        g = A.groupby('ticker')['_rvp']
        X = np.column_stack([np.log(A['_rvp'].values) - lm,
                             np.log(g.transform(lambda s: s.rolling(5, min_periods=1).mean()).values) - np.log(A['m5_' + col].values),
                             np.log(g.transform(lambda s: s.rolling(22, min_periods=1).mean()).values) - np.log(A['m22_' + col].values),
                             pd.get_dummies(A['ticker']).values.astype(float)[:, 1:]])
        trf = tr & np.isfinite(ym)
        f = _fit_lin(_kind(leg), X[trf], _y(A, h, trf, data) - ym[trf], _weights(A, trf, c, data))
        ln[pm] = mp[pm] + f(X[pm])
    elif t in ('HEAVY', 'RealizedGARCH'):
        import d24_6 as G
        m = 'heavy' if t == 'HEAVY' else 'rgarch'
        a = A[['date', 'ticker', 'close', 'rv', tc, 'year']].assign(_i=np.arange(len(A)), _tr=tr, _pm=pm)
        for tk in sorted(a.loc[pm, 'ticker'].unique()):
            gg = G.prep(a, tk)
            if gg['_tr'].any() and not gg['_tr'].values[0]:
                raise ValueError(f'{t}: обучение {tk} не с начала истории (QMLE на расширяющемся окне)')
            n = int((gg['date'].values < gg.loc[gg['_pm'], 'date'].min()).sum())  # целей нет: QMLE — все строки до первой даты прогноза
            F, _ = G.fit_forecast(gg.rr.values, gg.xx.values, n)[m]
            sel = gg['_pm'].values
            ln[gg['_i'].values[sel]] = np.log(np.clip(F[h][sel] / 1e4, 1e-10, None))
    else:
        raise ValueError(f'leg.type = {t}')
    if leg['panel'] != 'off':
        col = 'm_' + leg['panel_market']; n = int(leg['panel_beta_window'])
        x = np.log(A[col].values); yv = np.log(A['_rvp'].values)
        df = pd.DataFrame({'tk': A['ticker'].values, 'x': x, 'y': yv, 'xy': x * yv, 'xx': x * x})
        r = df.groupby('tk')[['x', 'y', 'xy', 'xx']].transform(lambda s: s.rolling(n, min_periods=n // 2).mean())
        beta = ((r.xy - r.x * r.y) / (r.xx - r.x ** 2)).values; alpha = (r.y - beta * r.x).values
        s = {'factor': 1.0, 'factor_shrink_025': 0.25, 'factor_shrink_05': 0.5}.get(leg['panel'])
        sv = np.where(A['I2'].values > 0, 0.5, 0.25) if s is None else np.full(len(A), s)
        P = alpha + beta * _market(A, mk, h, col, c, None)[0]
        P = np.where(np.isfinite(P), P, ln)
        ln[pm] = (1 - sv[pm]) * ln[pm] + sv[pm] * P[pm]
    return pd.Series(ln[pm], index=A.index[pm], name=f'ln_{tg}')


def run_leg(design, M=None, steps=STEPS, tickers=None, feat=FEAT, ctx=None, params=None):
    """Нога по дизайну (схема v1) -> {вариант: (V, W)}. V — сырой прогноз val-строк шагов steps; W — ln модели шага на строках year <= vy.
    params — список: сюда добавляются (тег, шаг, параметр, значение) — коэффициенты годовой подгонки, точка сетки, κ, θ MIDAS,
    параметры HEAVY/RG по тикерам, k якоря RVI, β панели (формат D: write_params)."""
    d = complete(design)
    h, leg, data = int(d['h']), d['leg'], d['data']
    if data['target'] != 'ln_mean':
        raise NotImplementedError('data.target ≠ ln_mean — кластер 1, не L')
    ctx = dict(ctx or {})
    M = mat_L(feat, tickers=tickers) if M is None else M
    A, harj, mk, didx = _prep(M)
    t = leg['type']
    fn = {'HARJ': _run_linear, 'HARX': _run_linear, 'HARX_regime': _run_linear, 'HARX_pooled': _run_linear, 'HAR_perlead': _run_perlead,
          'MIDAS_RV': _run_midas, 'RVI_anchor': _run_anchor, 'HEAVY': _run_garch, 'RealizedGARCH': _run_garch}.get(t)
    for k in steps:
        _core().assert_dev_step(k)
    out = {}
    for tag, lg, var in _variants(d):
        c = dict(ctx, _par=[])
        if t == 'factor':
            V, W = _run_factor(A, h, lg, var, harj, data, c, steps, didx, mk)
        else:
            V, W = fn(A, h, lg, var, harj, data, c, steps, didx)
        out[tag] = _panel(A, h, lg, V, W, mk, data, c, steps)
        if params is not None:
            params += [(tag, s, p, v) for s, p, v in c['_par']]
    return out


def series_name(tag, kind='leg'):
    """Имя ряда в кэше дизайна: leg (вариант base) или leg__<тег r2_core.variants>."""
    return kind if tag == 'base' else f'{kind}__{tag}'


def write_params(design, h, rows, kind='leg', root=None):
    """Параметры по шагам в формате D: R2/cache/<hash>/params.csv — design_sha, h, step, param, value. param = <ряд>:<имя>.
    Запись под flock: строки с теми же (h, step, param) заменяются (повтор задачи идемпотентен)."""
    import fcntl
    RC = _core()
    root = RC.SHARED.get('L_root') if root is None else root
    p = RC.cache_dir(design, root=root) / 'params.csv'
    sha = RC.design_hash(design)
    new = pd.DataFrame([dict(design_sha=sha, h=int(h), step=int(s), param=f'{series_name(tag, kind)}:{n}', value=float(v))
                        for tag, s, n, v in rows if v is not None and np.isfinite(float(v))],
                       columns=['design_sha', 'h', 'step', 'param', 'value'])
    with open(p, 'a+', encoding='utf-8') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        txt = f.read()
        old = pd.read_csv(io.StringIO(txt)) if txt.strip() else new.iloc[:0]
        key = lambda d: d.h.astype(str) + '|' + d.step.astype(str) + '|' + d.param
        out = pd.concat([old[~key(old).isin(set(key(new)))], new], ignore_index=True).sort_values(['h', 'param', 'step'])
        f.seek(0); f.truncate()
        out.to_csv(f, index=False)
        fcntl.flock(f, fcntl.LOCK_UN)
    return p


def fn_leg(design, seed, steps):
    """Модуль для r2_core.run_phase: {ряд: V} по всем вариантам сетки ноги (seed не влияет). Данные — r2_core.SHARED['M'] (mat_from).
    Параметры по шагам — write_params (params.csv каталога дизайна)."""
    RC = _core()
    M = RC.SHARED['M']
    h = int(design['h'])
    par = []
    res = run_leg(design, M=M, steps=steps, params=par)
    write_params(design, h, par)
    cd = RC.cache_dir(design, root=RC.SHARED.get('L_root'))
    for tag, (_, W) in res.items():  # ln для ролей B: <ряд>_h{h}_lnall.parquet (tr/va, emb_ok), не ряд прогноза
        P = to_parts(W); assert P.date.dt.year.max() <= 2021
        tmp = cd / f'.{series_name(tag)}_h{h}_lnall.parquet.tmp{os.getpid()}'
        P.to_parquet(tmp, index=False); os.replace(tmp, cd / f'{series_name(tag)}_h{h}_lnall.parquet')
    return {series_name(tag): V for tag, (V, _) in res.items()}


def designs_L(h):
    """Пространство ног кластера 2 на h (уровень C, роль none / вход ролей B). Сетки — внутри складки (r2_core.variants)."""
    D = {'HARJ': {}, 'HARJ_huber': {'leg': {'robust': 'Huber'}}, 'HARJ_qmle': {'leg': {'estimator': 'qmle_level'}}}
    for s in ('S1', 'S2', 'S3', 'S4') + (('S5',) if h == 22 else ()):
        D[f'HARX_{s}'] = {'leg': {'type': 'HARX', 'regressors': s}}
    for s in ('S3', 'S4'):
        D[f'HARX_{s}_huber'] = {'leg': {'type': 'HARX', 'regressors': s, 'robust': 'Huber'}}
    D['HARX_S4_qmle'] = {'leg': {'type': 'HARX', 'regressors': 'S4', 'estimator': 'qmle_level'}}
    for s in ('none', 'S1', 'S4'):
        D[f'regime_{s}'] = {'leg': {'type': 'HARX_regime', 'regressors': s, 'regime_count': [2, 3]}}
        D[f'pooled_{s}'] = {'leg': {'type': 'HARX_pooled', 'regressors': s, 'pooled_kappa': [0, 0.5, 1]}}
    D.update({'HEAVY': {'leg': {'type': 'HEAVY'}}, 'RG': {'leg': {'type': 'RealizedGARCH'}}, 'MIDAS': {'leg': {'type': 'MIDAS_RV'}},
              'RVI_anchor': {'leg': {'type': 'RVI_anchor'}}, 'factor_imoex': {'leg': {'type': 'factor', 'panel_market': 'RV_IMOEX'}},
              'factor_eq17': {'leg': {'type': 'factor', 'panel_market': 'RV_eq17'}}})
    if h > 1:
        D['perlead'] = {'leg': {'type': 'HAR_perlead'}}
        D['perlead_S3'] = {'leg': {'type': 'HAR_perlead', 'regressors': 'S3'}}
    return {k: dict(v, h=h, design=f'L_{k}_h{h}') for k, v in D.items()}


def schet(hs=HS, workers=10):
    """Боевой счёт ног кластера 2 (решение 16:25: эконометрика — сразу): 17 тикеров × шаги 1–6, кэш r2_core (хэш дизайна), shift_check,
    params.csv, lnall для ролей. Список дизайнов — models/designs/r2/r2_L_designs_v1.yaml (коммит до счёта)."""
    import yaml
    RC = _core()
    A, harj = RC.load_d0(); RC.check_d0(A)
    M = mat_from(A, harj)
    RC.share(M=M, A=M['A'], harj=harj, L_root=None)
    lst = RC.REPO / 'models' / 'designs' / 'r2' / 'r2_L_designs_v1.yaml'
    allD = {h: designs_L(h) for h in hs}
    txt = yaml.safe_dump({'schema': 'v1', 'designs': {f'h{h}': list(allD[h].values()) for h in hs}}, allow_unicode=True, sort_keys=False)
    if lst.exists() and lst.read_text() != txt:
        raise SystemExit(f'{lst} отличается от designs_L — список меняется только новым файлом (v2)')
    lst.write_text(txt)
    T = []
    for h in hs:
        t = RC.run_phase('C', h, list(allD[h].values()), fn_leg, seeds=(42,), steps=STEPS, workers=workers, module='L', check_gate=False)
        t['h'] = h; T.append(t)
    T = pd.concat(T, ignore_index=True)
    (LDIR).mkdir(parents=True, exist_ok=True); T.to_csv(LDIR / 'schet_L_nogi.csv', index=False)
    RC.merge_index()
    print(f"счёт ног: рядов {int((T.ryad != 'ОШИБКА').sum())}, ошибок {int((T.ryad == 'ОШИБКА').sum())}, "
          f"сдвиг не OK {int((T.get('sdvig_ok') == False).sum())}")
    return T


def leg_ln(design, M, steps=STEPS):
    """ln-прогнозы ноги для ролей B: {тег: DataFrame date, ticker, step, part (tr/va), base_ln, emb_ok} (to_parts W; эмбарго — emb_ok,
    обучение роли шага k — part = tr и emb_ok). В run_phase не пишется (не ряд прогноза): B вызывает это в своём модуле."""
    return {tag: to_parts(W) for tag, (_, W) in run_leg(design, M=M, steps=steps).items()}


def leak_build(design, steps, tickers=None):
    """build(A) для r2_core.leak_check: val (part = val) и ln-прогнозы моделей шагов на всех строках year <= vy (part = all)."""
    def build(A):
        M = mat_from(A, SH_HARJ['harj'], tickers)
        out = {}
        for tag, (V, W) in run_leg(design, M=M, steps=steps).items():
            fr = [V[['date', 'ticker', 'step', 'rv_pred']].assign(part='val')]
            for c in [c for c in W.columns if c.startswith('ln_s')]:
                w = W[W[c].notna()]
                fr.append(pd.DataFrame({'date': w.date.values, 'ticker': w.ticker.values, 'step': int(c[4:]), 'part': 'all',
                                        'rv_pred': np.exp(w[c].values)}))
            out[series_name(tag)] = pd.concat(fr, ignore_index=True)
        return out
    return build


SH_HARJ = {}


# ---------------------------------------------------------------- дым: проверка сдвига и детектор — r2_core (shift_check, leak_check)
DYM_DESIGNS = {  # дым: ключевые ветки кода (каждая нога и каждый переключатель кластеров 2, 11, 14, 15 хотя бы раз)
    'HARJ': {}, 'HARX_S4': {'leg': {'type': 'HARX', 'regressors': 'S4'}},
    'HARX_S5_l01': {'leg': {'type': 'HARX', 'regressors': 'S5', 'enet': {'lambda': [0.1], 'alpha': [0.5]}}},
    'HARX_S3_huber': {'leg': {'type': 'HARX', 'regressors': 'S3', 'robust': 'Huber'}},
    'HARX_S2_qmle': {'leg': {'type': 'HARX', 'regressors': 'S2', 'estimator': 'qmle_level'}},
    'regime2': {'leg': {'type': 'HARX_regime', 'regressors': 'S1'}}, 'regime3': {'leg': {'type': 'HARX_regime', 'regime_count': 3}},
    'pooled': {'leg': {'type': 'HARX_pooled', 'pooled_kappa': [0, 0.5, 1]}},
    'HEAVY': {'leg': {'type': 'HEAVY'}}, 'RG': {'leg': {'type': 'RealizedGARCH'}}, 'MIDAS': {'leg': {'type': 'MIDAS_RV'}},
    'perlead': {'leg': {'type': 'HAR_perlead'}}, 'RVI_anchor': {'leg': {'type': 'RVI_anchor'}},
    'factor_eq17': {'leg': {'type': 'factor', 'panel_market': 'RV_eq17'}}, 'factor_imoex': {'leg': {'type': 'factor'}},
    'tau_midas_AR1': {'leg': {'struct': 'tau_midas', 'tau_window': [22, 66], 'tau_extend': 'AR1'}},
    'tau_rvi': {'leg': {'struct': 'tau_rvi'}}, 'cj': {'leg': {'struct_split': 'bv_plus_jump'}},
    'panel_shrink_state': {'leg': {'panel': 'factor_shrink_state', 'panel_beta_window': 126}},
    'panel_factor': {'leg': {'panel': 'factor'}},
    'refit_monthly': {'leg': {'refit': 'monthly'}}, 'refit_q_S1': {'leg': {'type': 'HARX', 'regressors': 'S1', 'refit': 'quarterly',
                                                                           'enet': {'lambda': [0.01], 'alpha': [1.0]}}},
    'rls_099': {'leg': {'rls_lambda': 0.99}},
    'recency_winsor': {'data': {'recency_halflife': '4y', 'winsor': 'p1_99_by_ticker'}},
}


DYM_CUTS = ('2017-06-30', '2018-06-29', '2016-12-31', '2017-12-31')


def _dym_leak_task(t):
    RC = _core()
    A3 = RC.SHARED['A3']
    R = RC.leak_check(leak_build(dict(t['design'], h=t['h']), DYM_STEPS), A3, f"L:{t['name']}", cuts=DYM_CUTS)
    R['dizayn'], R['h'] = t['name'], t['h']
    return R


def dym(hs=HS, names=None, workers=2):
    """Дым L на ядре r2_core: SBER, LKOH, AFLT × шаги 1–3 × seed 42 × h. run_phase (валидация, хэш, кэш во временном корне,
    shift_check каждого ряда) + leak_check (срезы 2017-06-30, 2018-06-29 — середина года с подлогом; 2016-12-31, 2017-12-31 — эмбарго)."""
    RC = _core()
    out = LDIR / 'dym'; out.mkdir(parents=True, exist_ok=True)
    A, harj = RC.load_d0(); RC.check_d0(A)
    SH_HARJ['harj'] = harj
    M3 = mat_from(A, harj, DYM_TK)
    A3 = A[A.ticker.isin(DYM_TK)].reset_index(drop=True)
    RC.share(M=M3, A=M3['A'], A3=A3, harj=harj)
    blok = out / 'blok_sdvig_dym.csv'
    if blok.exists():
        blok.unlink()
    sel = {k: v for k, v in DYM_DESIGNS.items() if not names or k in names}
    T, L = [], []
    with tempfile.TemporaryDirectory() as root:
        RC.share(L_root=root)
        for h in hs:
            designs = [dict(des, h=h, design=f'L_dym_{nm}') for nm, des in sel.items()]
            t = RC.run_phase('A', h, designs, fn_leg, seeds=(42,), steps=DYM_STEPS, workers=workers, module='L', root=root,
                             block_file=blok, check_gate=False)
            t['h'] = h; T.append(t)
        # params.csv (формат D) — по одному на дизайн
        P = []
        for h in hs:
            for nm, des in sel.items():
                p = RC.cache_dir(RC.validate(dict(des, h=h, design=f'L_dym_{nm}')), root=root) / 'params.csv'
                if p.exists():
                    q = pd.read_csv(p); P.append(dict(dizayn=nm, h=h, strok=len(q), shagov=q.step.nunique(), parametrov=q.param.nunique(),
                                                     kolonki=','.join(q.columns)))
        pd.DataFrame(P).to_csv(out / 'dym_L_params.csv', index=False)
        RC.share(L_root=None)
    # fit_predict (API для E) = run_leg на годовой подгонке: обучение r2_core.train_mask(vy), прогноз — val-год
    FP = []
    for nm in ('HARJ', 'HARX_S5_l01', 'regime3', 'pooled', 'MIDAS', 'perlead', 'RVI_anchor', 'factor_imoex', 'HEAVY', 'tau_midas_AR1', 'cj',
               'panel_shrink_state', 'HARX_S2_qmle'):
        if nm not in sel:
            continue
        for h in hs:
            des = dict(sel[nm], h=h)
            R = run_leg(des, M=M3, steps=DYM_STEPS)
            for tag, (V, _) in R.items():
                for k in DYM_STEPS:
                    vy = 2015 + k
                    s = fit_predict(des, RC.train_mask(M3['A'], h, vy), RC.val_mask(M3['A'], h, vy), M=M3, tag=tag)
                    v = V[V.step == k]
                    m = pd.DataFrame({'date': M3['A']['date'].values[s.index], 'ticker': M3['A']['ticker'].values[s.index], 'fp': s.values}).merge(v, on=['date', 'ticker'])
                    FP.append(dict(dizayn=nm, h=h, tag=tag, step=k, n=len(m), max_dln=float(np.abs(m.fp - np.log(m.rv_pred)).max())))
    FP = pd.DataFrame(FP); FP['ok'] = FP.max_dln <= 1e-8; FP.to_csv(out / 'dym_L_fit_predict.csv', index=False)
    print(f'fit_predict = run_leg: {int(FP.ok.sum())}/{len(FP)}; max |Δ ln| = {FP.max_dln.max():.1e}')
    tasks = [dict(name=nm, design=des, h=h, cost=1) for nm, des in sel.items() for h in hs]
    for r in RC.run_pool(_dym_leak_task, tasks, workers, label='дым L: детектор'):
        if r['ok']:
            L.append(r['result'])
        else:
            L.append(pd.DataFrame([dict(dizayn=r['task']['name'], h=r['task']['h'], ok=False, oshibka=r['error'].strip().splitlines()[-1])]))
    T = pd.concat(T, ignore_index=True); L = pd.concat(L, ignore_index=True)
    T.to_csv(out / 'dym_L_nogi_faza.csv', index=False); L.to_csv(out / 'dym_L_nogi_leak.csv', index=False)
    oshib = int((T.ryad == 'ОШИБКА').sum()) if 'ryad' in T else 0
    print(f"дым ног: рядов {len(T) - oshib}, ошибок {oshib}, сдвиг не OK {int((T.sdvig_ok == False).sum())}; "
          f"детектор: проверок {len(L)}, не OK {int((~L.ok.astype(bool)).sum())}")
    return T, L


def sverka_F():
    """Ноги L строятся на своей матрице d0 (r2_nogi.load(d0=True)). Сверка с признаками F (features_r2_v2, PR #170; только строки <= 2021):
    G1 (ln RVI², ln RV IMOEX, ln RV рынка, PC1), G4 ln BV, G9 ln(rv_d/rv_w), ln(rv_w/rv_m) — то же ли это."""
    RC = _core()
    A, harj = RC.load_d0()
    M = mat_from(A, harj)
    B = M['A']
    HX._SIG.clear(); HX.GH_HARJ = harj
    mine = pd.DataFrame({'date': B['date'], 'ticker': B['ticker'], 'g1_ln_rvi2': B['S_ln_rvi2'], 'g1_ln_rvm_imoex': B['S_ln_rvm_imoex'],
                         'g1_ln_rvm_mkt': HX.TERMS['ln_rvm_mkt'](B).values, 'g1_pc1': B['S_pc1'], 'g4_ln_bv': np.log(B['bv'].clip(lower=1e-12)),
                         'g9_ln_d_w': np.log(B['rv_d'] / B['rv_w']), 'g9_ln_w_m': np.log(B['rv_w'] / B['rv_m'])})
    F = pd.read_parquet(REG / 'R2' / 'features' / 'features_r2_v2.parquet', columns=list(mine.columns),
                        filters=[('date', '<=', pd.Timestamp('2021-12-31'))])
    F['date'] = pd.to_datetime(F['date'])
    m = mine.merge(F, on=['date', 'ticker'], suffixes=('', '_F'))
    rows = []
    for c in mine.columns[2:]:
        a, b = m[c].values.astype(float), m[c + '_F'].values.astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        rows.append(dict(priznak=c, n=int(ok.sum()), max_abs=float(np.abs(a[ok] - b[ok]).max()), corr=float(np.corrcoef(a[ok], b[ok])[0, 1]),
                         dolya_sovp_1e9=float(np.mean(np.abs(a[ok] - b[ok]) <= 1e-9))))
        print(rows[-1], flush=True)
    R = pd.DataFrame(rows); (LDIR / 'dym').mkdir(parents=True, exist_ok=True); R.to_csv(LDIR / 'dym' / 'sverka_F.csv', index=False)
    return R


def kontrol_L():
    """Код L на полной матрице, шаги 1–6 = ряды Э (d0 + эмбарго): HARJ = HARJ_emb_d0, HEAVY = HEAVY_emb_d0, HARX без S = HARJ_emb_d0;
    W (lnall) HARJ = HARJ_emb_d0_lnall (tr/va); |Δ| <= 1e-7, иначе стоп."""
    M = mat_L(); rows = []
    for h in HS:
        for nm, des, ref in (('HARJ', {}, 'HARJ_emb_d0'), ('HEAVY', {'leg': {'type': 'HEAVY'}}, 'HEAVY_emb_d0'),
                             ('HARX none', {'leg': {'type': 'HARX'}}, 'HARJ_emb_d0')):
            V, W = run_leg(dict(des, h=h), M=M)['base']
            r = pd.read_parquet(CACHE / f'{ref}_h{h}_sdet_val.parquet')
            rows.append(_cmp(V, r, nm, h, ref))
            if nm == 'HARJ':
                P = to_parts(W); Rl = pd.read_parquet(CACHE / f'{ref}_h{h}_lnall.parquet')
                m = P.merge(Rl, on=['date', 'ticker', 'step', 'part'], how='outer', suffixes=('', '_r'), indicator=True)
                b = m._merge == 'both'; d = float(np.abs(m.base_ln[b] - m.base_ln_r[b]).max())
                rows.append(dict(ryad='HARJ lnall', h=h, etalon=f'{ref}_lnall', n=int(b.sum()), n_vne_pary=int((~b).sum()), max_abs=d,
                                 status='OK' if (b.all() and d <= TOL) else 'РАСХОЖДЕНИЕ'))
            print(rows[-1], flush=True)
    R = pd.DataFrame(rows); (LDIR / 'dym').mkdir(parents=True, exist_ok=True); R.to_csv(LDIR / 'dym' / 'kontrol_L.csv', index=False)
    return bool((R.status == 'OK').all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['zamer', 'det', 'leak', 'dym', 'kontrol', 'sverka_F', 'schet', 'spisok'])
    ap.add_argument('--names', nargs='*', default=None)
    ap.add_argument('--hs', type=int, nargs='*', default=list(HS))
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    if a.mode == 'zamer':
        zamer()
    elif a.mode == 'det':
        det(tuple(a.hs), min(a.workers, 2))
    elif a.mode == 'dym':
        dym(tuple(a.hs), a.names, min(a.workers, 2))
    elif a.mode == 'kontrol':
        sys.exit(0 if kontrol_L() else 1)
    elif a.mode == 'sverka_F':
        sverka_F()
    elif a.mode == 'spisok':  # только записать список дизайнов (для коммита до счёта)
        import yaml
        RC = _core()
        allD = {h: designs_L(h) for h in a.hs}
        for h in a.hs:
            for d in allD[h].values():
                RC.validate(d)
        (RC.REPO / 'models' / 'designs' / 'r2' / 'r2_L_designs_v1.yaml').write_text(
            yaml.safe_dump({'schema': 'v1', 'designs': {f'h{h}': list(allD[h].values()) for h in a.hs}}, allow_unicode=True, sort_keys=False))
        print({h: len(allD[h]) for h in a.hs})
    elif a.mode == 'schet':
        schet(tuple(a.hs), a.workers)
    else:
        sys.exit(0 if leak(tuple(a.hs), min(a.workers, 2)) else 1)


if __name__ == '__main__':
    main()
