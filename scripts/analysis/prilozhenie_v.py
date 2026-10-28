#!/usr/bin/env python3
"""Приложение В: устойчивость финалистов третьего круга к составу тикеров (16 без LSRG; 20 = 17 + TGKA, MSNG, TRMK), только разработочный
набор: OOS QLIKE val 2017–2021 (шаги 2–6), поправка Б0.1 с s² шага t−1. Подтверждающий набор (val 2022–2024, шаги 7–9) и тестовые годы
не трогаются: шаги 7–9 не обучаются, тестовая часть шага подменяется одной строкой val-года (прогнозы на тестовые годы не строятся).

Финалы (FINAL_DESIGN_chernovik.md): h=1 — D22_h1 top5/V0_XGB/B-sm/L0 (слой 1 HAR-X⁺ d22_harxp + XGB Д8.12 на HAR-X⁺, d22_h1);
h=5 — D12_h5 wrap2/V0_XGB/B-sm/L0 (V9 + Д8.12 + Д7 + Huber; слот XGB = среднее LGBM и XGB по RV); h=22 — HAR-J с поправкой B-sm.
Протокол на составе: те же гиперпараметры, что у финала (D22_h1: топ-5 точек ЛГК шага, D22_h1_grid; D12_h5: точка optuna шага,
D12_h5_selected_h5.csv), те же исключения признаков (D22_h1 / Bv2b), бины — окно шага 1 состава, слой 1 и HAR-J переоцениваются на составе.
Эталон — HAR-J (features_pit_v3) с той же поправкой B-sm на том же составе. Общие файлы конвейера не меняются: подмены l2_lib — в процессе.

Режимы:
  data          признаки 20 тикеров: features_pit_v3_28 (Д12.1) + RRV (d172_rrv_features.new_columns) -> features_pit_v3_rrv_20.parquet;
                ночной гэп (d13_lib.intraday_one по data/iss_check/bars_10m) -> intraday1850_20.parquet; сверка строк 17 с исходными, SHA-256
  h1 --comp C   D22_h1 на составе C ∈ {16, 17, 20} -> data/regen/prilozhenie_V/D22_h1_c{C}_xgb_h1_top5_val.parquet (шаги 1–6)
  h5 --comp C   D12_h5 на составе C -> D12_h5_c{C}_{xgb,lgbm}_h5_val.parquet (шаги 1–6)
  otsenka       QLIKE по годам × состав × h (финал и HAR-J с поправкой), контроль 17 -> prilozhenie_V/qlike_*.csv, kontrol_17.csv
python3 scripts/analysis/prilozhenie_v.py data | h1 --comp 16 --workers 5 | h5 --comp 20 | otsenka"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, hashlib, json, sys, time, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MAIN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
R = MAIN / 'data' / 'regen'
OUT = R / 'prilozhenie_V'
CORE17 = ['AFLT', 'ALRS', 'HYDR', 'IRAO', 'LKOH', 'LSRG', 'MGNT', 'MOEX', 'MTLR', 'MTSS', 'NVTK', 'OGKB', 'PHOR', 'RTKM', 'SBER', 'TATN', 'VTBR']
NEW3 = ['MSNG', 'TGKA', 'TRMK']
COMPS = {'16': [t for t in CORE17 if t != 'LSRG'], '17': CORE17, '20': sorted(CORE17 + NEW3)}
COMMON16 = COMPS['16']
STEPS = range(1, 7)  # разработочный набор: шаги 1–6 (val 2016–2021); шаг 1 — только s² для шага 2
DEV = (2, 3, 4, 5, 6)
F_V3 = R / 'features_pit_v3.parquet'
F_V3_28 = R / 'features_pit_v3_28.parquet'
F_RRV = R / 'features_pit_v3_rrv.parquet'
F_RRV20 = OUT / 'features_pit_v3_rrv_20.parquet'
F_D7 = R / 'features_pit_v3_d7_h5.parquet'
F_D7_28 = R / 'features_pit_v3_28_d7_h5.parquet'
INTRA17 = R / 'D13' / 'intraday1850.parquet'
INTRA20 = OUT / 'intraday1850_20.parquet'
RRV_RAW = MAIN / 'data' / 'iss_check' / 'rrv_1850.parquet'
BARS = MAIN / 'data' / 'iss_check' / 'bars_10m'
D22_CFG = {'harf': True, 'wins': True}
D12_CFG = {'v9': True, 'harf': True, 'huber': True}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def now():
    import subprocess
    return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S'], capture_output=True, text=True).stdout.strip()


def same_bytes(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == object or b.dtype == object:
        return bool(pd.Series(a).equals(pd.Series(b)))
    return a.dtype == b.dtype and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------- данные 20 тикеров
def build_data():
    import d172_rrv_features as F172
    import d13_lib as D13
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    # (1) v3 + RRV для 20: строки и колонки v3 — из features_pit_v3_28 (тот же m0_pit_lib.assemble, строки 17 побайтно = v3)
    v28 = pd.read_parquet(F_V3_28)
    v3cols = list(pd.read_parquet(F_V3).columns)
    sub = v28[v28.ticker.isin(COMPS['20'])][v3cols].reset_index(drop=True)
    nc = F172.new_columns(sub[['date', 'ticker']], F172.load_rrv(RRV_RAW))
    out = sub.copy()
    for c in F172.NEW:
        out[c] = nc[c].values
    ref = pd.read_parquet(F_RRV)
    assert list(out.columns) == list(ref.columns), 'порядок колонок отличается от features_pit_v3_rrv'
    k = ['date', 'ticker']
    o17 = out[out.ticker.isin(CORE17)].sort_values(k).reset_index(drop=True)
    r17 = ref.sort_values(k).reset_index(drop=True)
    bad = [c for c in ref.columns if not same_bytes(o17[c].values, r17[c].values)]
    log.append(f'features_pit_v3_rrv_20: строк {len(out)} (17 основных {len(o17)}, новых {int(out.ticker.isin(NEW3).sum())}), колонок {out.shape[1]}; '
               f'строки 17 против features_pit_v3_rrv побайтно во всех {len(ref.columns)} колонках: {"PASS" if not bad and len(o17) == len(r17) else "FAIL " + str(bad[:5])}')
    assert not bad and len(o17) == len(r17)
    out.to_parquet(F_RRV20, index=False)
    log.append(f'NaN в RRV у новых тикеров: ' + ', '.join(f'{c} {out.loc[out.ticker.isin(NEW3), c].isna().mean():.4f}' for c in F172.FEAT))
    # (2) ночной гэп: d13_lib.intraday_one на барах data/iss_check/bars_10m (f_intraday не вызывается — он перезаписал бы общий кэш)
    I17 = pd.read_parquet(INTRA17)
    rec = pd.concat([D13.intraday_one(pd.read_parquet(BARS / f'{tk}.parquet'), tk) for tk in COMPS['20']], ignore_index=True)
    rec = rec[I17.columns]
    m = I17.merge(rec, on=['date', 'ticker'], how='outer', suffixes=('', '_r'), indicator=True)
    both = m[m._merge == 'both']
    eq = {c: bool(np.array_equal(both[c].values, both[c + '_r'].values, equal_nan=True)) for c in D13.INTRA_COLS}
    log.append(f'intraday1850 пересчёт 17 тикеров против кэша D13: ключей общих {len(both)}, только в кэше {int((m._merge == "left_only").sum())}, '
               f'только в пересчёте (17) {int(((m._merge == "right_only") & m.ticker.isin(CORE17)).sum())}; равенство колонок: {eq}')
    I20 = pd.concat([I17, rec[rec.ticker.isin(NEW3)]], ignore_index=True)
    I20.to_parquet(INTRA20, index=False)
    log.append(f'intraday1850_20: строк {len(I20)}, у новых {int(I20.ticker.isin(NEW3).sum())}; доля NaN gap_on у новых '
               f'{I20.loc[I20.ticker.isin(NEW3), "gap_on"].isna().mean():.4f}')
    # (3) Д7 h=5 для 20: features_pit_v3_28_d7_h5 (Д12.1); сверка строк 17 с features_pit_v3_d7_h5
    d7 = pd.read_parquet(F_D7)
    d28 = pd.read_parquet(F_D7_28)
    a = d28[d28.ticker.isin(CORE17)][d7.columns].sort_values(k).reset_index(drop=True)
    b = d7.sort_values(k).reset_index(drop=True)
    bad7 = [c for c in d7.columns if not same_bytes(a[c].values, b[c].values)]
    log.append(f'features_pit_v3_28_d7_h5: строки 17 против features_pit_v3_d7_h5 побайтно во всех {len(d7.columns)} колонках: '
               f'{"PASS" if not bad7 and len(a) == len(b) else "FAIL " + str(bad7[:5])}')
    hs = {p.name: sha(p) for p in (F_V3, F_V3_28, F_RRV, F_RRV20, F_D7, F_D7_28, INTRA17, INTRA20, RRV_RAW, R / 'features_pit_v3.config.json',
                                   R / 'features_pit_v3_rrv.config.json')}
    json.dump(dict(time=now(), sha256=hs, log=log), open(OUT / 'data_manifest.json', 'w'), ensure_ascii=False, indent=1)
    for s in log:
        print(s, flush=True)
    for n, v in hs.items():
        print(f'SHA-256 {n} {v}', flush=True)


# ---------------------------------------------------------------- общая часть l2_lib: без тестовых лет
def patch_split(L):
    """Тестовая часть шага -> первая строка val-года (тестовые годы не читаются моделью, прогнозы на них не строятся)."""
    s0 = L.split

    def split_dev(A, h, step):
        assert step <= 6, 'Приложение В: только разработочный набор (шаги 1–6)'
        tp, vi, td, vy, ty = s0(A, h, step)
        return tp, vi, vi.iloc[:1], vy, ty
    L.split = split_dev


G = {}


def _fit_job(a):
    import l2_lib as L
    model, step, p, tag, extra = a
    D = L.step_data(G['A'], G['cols'], G['h'], step, model, G['cfg'], G['btag'])
    ref = G['ref'].get(model) if step > 1 else None
    pv, _, bi, _, _ = L.fit(model, p, D, G['cfg'], seed=42, ref=ref)
    d = D['parts']['va']
    return pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': D['vy'], 'step': step, 'model': model, **extra,
                         'best_iteration': bi, 'rv_actual': d[L.tcol(G['h'])].values, 'rv_pred': np.clip(np.exp(pv), 1e-10, None)})


def run_pool(jobs, workers):
    with mp.get_context('fork').Pool(workers) as P:
        return P.map(_fit_job, jobs, chunksize=1)


# ---------------------------------------------------------------- h=1: D22_h1
def run_h1(comp, workers):
    import l2_lib as L
    import d22_h1 as D22  # подмены l2_lib (har_ln_x, build_features_x) — функции модуля, патчим вручную
    import d22_harxp as P
    import harx as HX
    from tune_L2 import grid, SPACEB
    t0 = time.time()
    tk = COMPS[comp]
    feat = F_RRV20 if comp == '20' else F_RRV
    P.FEAT, P.INTRA, P.OUT = feat, (INTRA20 if comp == '20' else INTRA17), OUT  # P.OUT — только на случай временных файлов
    # слой 1: HAR-X⁺ цели h=1 на составе, шаги 1–6
    A1, harj = HX.load(feat)
    A1 = A1[A1.ticker.isin(tk)].reset_index(drop=True)
    A1 = P.add_targets(A1)
    names = P.terms_for('1')
    X = P.build_terms(A1, names)
    ln = A1[['date', 'ticker']].copy()
    vals, logs = [], []
    for k in STEPS:
        v, _, la, _ = P.forecast_step(A1, '1', k, harj, names, X, logs)
        ln[f'hxp_1_{k}'] = la.values
        vals.append(v)
    ln.to_parquet(OUT / f'HXP_c{comp}_lnall.parquet', index=False)
    pd.concat(vals).to_parquet(OUT / f'HXP_c{comp}_1_val.parquet', index=False)
    pd.DataFrame(logs).to_csv(OUT / f'HXP_c{comp}_coefs.csv', index=False)
    print(f'[{now()}] c{comp}: слой 1 готов ({time.time() - t0:.0f} c)', flush=True)
    # слой 2: XGB Д8.12 на HAR-X⁺, топ-5 точек шага (точки D22_h1), seed 42, шаги 1–6
    L.FEAT, L.FEAT_CFG = feat, R / 'features_pit_v3_rrv.config.json'
    L.har_ln = D22.har_ln_x
    patch_split(L)
    A, FC = L.load_base()
    A = A[A.ticker.isin(tk)].reset_index(drop=True)
    A = A.merge(ln, on=['date', 'ticker'], how='left', validate='one_to_one')
    assert A['hxp_1_1'].notna().all()
    A2, cols, _, _ = D22.build_features_x(A, FC, 1, D22_CFG)
    GR = pd.concat([grid(SPACEB, 'xgb', 40), grid(SPACEB, 'xgb', 100, 100)]).set_index('point')
    gcsv = pd.read_csv(R / 'D22_h1_grid.csv').query('model == "xgb"').set_index('point')
    for c in [c for c, *_ in SPACEB['xgb']]:
        assert np.allclose(GR.loc[gcsv.index, c].values, gcsv[c].values, rtol=1e-12), c
    t5 = pd.read_parquet(R / 'D22_h1_xgb_h1_top5_val.parquet', columns=['step', 'point', 'rank']).drop_duplicates()
    G.update(A=A2, cols=cols, h=1, cfg=D22_CFG, btag='D22_h1')
    G['ref'] = {'xgb': L.make_ref('xgb', A2, cols, 1, D22_CFG, 'D22_h1')}
    jobs = [('xgb', int(r.step), {k: GR.loc[int(r.point), k] for k, *_ in SPACEB['xgb']}, 'D22_h1', dict(point=int(r.point), rank=int(r.rank)))
            for r in t5.sort_values(['step', 'rank']).itertuples()]
    print(f'[{now()}] c{comp}: тикеров {A2.ticker.nunique()}, строк {len(A2)}, признаков {len(cols)}, обучений {len(jobs)}', flush=True)
    res = pd.concat(run_pool(jobs, workers), ignore_index=True)
    res.drop(columns='model').to_parquet(OUT / f'D22_h1_c{comp}_xgb_h1_top5_val.parquet', index=False)
    print(f'[{now()}] c{comp}: D22_h1 готов ({time.time() - t0:.0f} c)', flush=True)


# ---------------------------------------------------------------- h=5: D12_h5
def run_h5(comp, workers):
    import l2_lib as L
    from tune_L2 import SPACEB
    t0 = time.time()
    tk = COMPS[comp]
    L.FEAT, L.FEAT_CFG = F_V3, R / 'features_pit_v3.config.json'
    patch_split(L)
    A, FC = L.load_base(F_D7_28 if comp == '20' else F_D7, (), 5)
    A = A[A.ticker.isin(tk)].reset_index(drop=True)
    A2, cols, _, _ = L.build_features(A, FC, 5, D12_CFG)
    sel = pd.read_csv(R / 'D12_h5_selected_h5.csv')
    G.update(A=A2, cols=cols, h=5, cfg=D12_CFG, btag='Bv2b')
    G['ref'] = {m: L.make_ref(m, A2, cols, 5, D12_CFG, 'Bv2b') for m in ('xgb', 'lgbm')}
    jobs = []
    for r in sel[sel.step <= 6].itertuples():
        jobs.append((r.model, int(r.step), {k: getattr(r, k) for k, *_ in SPACEB[r.model]}, 'D12_h5', dict(trial=int(r.trial))))
    print(f'[{now()}] c{comp}: тикеров {A2.ticker.nunique()}, строк {len(A2)}, признаков {len(cols)}, обучений {len(jobs)}', flush=True)
    res = pd.concat(run_pool(jobs, workers), ignore_index=True)
    for m in ('xgb', 'lgbm'):
        res[res.model == m].drop(columns='model').to_parquet(OUT / f'D12_h5_c{comp}_{m}_h5_val.parquet', index=False)
    print(f'[{now()}] c{comp}: D12_h5 готов ({time.time() - t0:.0f} c)', flush=True)


# ---------------------------------------------------------------- оценка
def harj_val(h, A, harj):
    """HAR-J по шагам как gibrid_har.har_forecasts, только val-годы шагов 1–6 (тестовые годы не прогнозируются)."""
    import gibrid_har as GH
    from sklearn.linear_model import LinearRegression
    target = f'rv_target_h{h}'
    out = []
    ys = [y for y in range(2017, 2026) if y in A.year.values]
    for step, ty in enumerate(ys, 1):
        if step > 6:
            break
        tr = A[A.year < ty]
        vy = sorted(tr.year.unique())[-1]
        tp = tr[tr.year != vy].dropna(subset=[target]); vi = tr[tr.year == vy].dropna(subset=[target])
        Xf = lambda d: GH.clean(np.log(d[harj].values.clip(1e-10)))
        m = LinearRegression().fit(Xf(tp), np.log(tp[target].values + 1e-10))
        out.append(pd.DataFrame({'date': vi['date'].values, 'ticker': vi['ticker'].values, 'year': vy, 'step': step,
                                 'rv_actual': vi[target].values, 'rv_pred': np.clip(np.exp(m.predict(Xf(vi))), 1e-10, None)}))
    return pd.concat(out, ignore_index=True)


def avg_pts(df):
    return df.groupby(['date', 'ticker', 'step'], as_index=False).agg(year=('year', 'first'), rv_actual=('rv_actual', 'first'),
                                                                       rv_pred=('rv_pred', 'mean'))


def corrected(hv, bx, h):
    """Строки и поправка как gibrid_compute.compute для V0 (merged: HAR-J ∩ бустинг, rv > 1e-12; Б0.1 B-sm, val шага t — s² шага t−1)."""
    import gibrid_io as GI
    import gibrid_bias as BI
    d = GI.merged(hv, bx, bx)
    d['hcj'] = d['har']  # HAR-CJ не нужен (V0_XGB / V0_HAR), колонка — для BI.resid
    empty = d.iloc[:0].copy()
    BI.correct(d, empty, BI.resid(d), 'sm', h, [])
    return d


def qrow(d, col, tickers=None):
    import gibrid_schemes as S
    out = {}
    for st in DEV:
        g = d[d.step == st]
        if tickers is not None:
            g = g[g.ticker.isin(tickers)]
        y = g['rv_actual'].values
        out[str(int(g['year'].iloc[0]))] = S.qlike(y, np.clip(g[col].values, 1e-10, None))
    out['2017–2021'] = float(np.mean([out[str(y)] for y in range(2017, 2022)]))
    out['2017–2020'] = float(np.mean([out[str(y)] for y in range(2017, 2021)]))
    return out


def design_frame(h, comp, saved=False):
    """Прогнозы финала h (val шагов 1–6) на составе; saved=True — сохранённые файлы финала (17 тикеров)."""
    if h == 1:
        f = R / 'D22_h1_xgb_h1_top5_val.parquet' if saved else OUT / f'D22_h1_c{comp}_xgb_h1_top5_val.parquet'
        t = pd.read_parquet(f)
        return avg_pts(t[t.step <= 6])
    if h == 5:
        if saved:
            w = pd.read_parquet(R / 'D12_h5_xgb_h5_wrap2_val.parquet')
            return w[w.step <= 6][['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']]
        x, l = (pd.read_parquet(OUT / f'D12_h5_c{comp}_{m}_h5_val.parquet') for m in ('xgb', 'lgbm'))
        m = x.merge(l[['date', 'ticker', 'step', 'rv_pred']], on=['date', 'ticker', 'step'], suffixes=('_x', '_l'), validate='one_to_one')
        return m.assign(rv_pred=(m.rv_pred_l + m.rv_pred_x) / 2)[['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']]
    return None


def evaluate():
    import gibrid_har as GH
    rows, sub, ctl = [], [], []
    name = {1: 'D22_h1 top5/V0_XGB/B-sm/L0', 5: 'D12_h5 wrap2/V0_XGB/B-sm/L0', 22: 'HAR-J с поправкой single/V0_HAR/B-sm/L0'}
    for comp in ('17', '16', '20'):
        A, harj = GH.load_pit(F_V3_28 if comp == '20' else F_V3)
        A = A[A.ticker.isin(COMPS[comp])].reset_index(drop=True)
        for h in (1, 5, 22):
            hv = harj_val(h, A, harj)
            variants = [('пересчёт', design_frame(h, comp))] + ([('сохранённые прогнозы финала', design_frame(h, comp, saved=True))] if comp == '17' and h != 22 else [])
            if h == 22:
                variants = [('пересчёт', hv)]
            for src, bx in variants:
                d = corrected(hv, bx, h)
                q_d, q_h = qrow(d, 'xgb'), qrow(d, 'har')
                for who, q in (('финал', q_d), ('HAR-J с поправкой', q_h)) if h != 22 else (('финал = HAR-J с поправкой', q_h),):
                    rows.append(dict(sostav=comp, h=h, istochnik=src, model=who, dizayn=name[h] if who.startswith('финал') else 'single/V0_HAR/B-sm/L0',
                                     n_tikerov=int(d.ticker.nunique()), n_strok_dev=int(d.step.isin(DEV).sum()), **q))
                if h != 22:
                    rows.append(dict(sostav=comp, h=h, istochnik=src, model='отношение финал / HAR-J', dizayn='', n_tikerov=int(d.ticker.nunique()),
                                     n_strok_dev=int(d.step.isin(DEV).sum()), **{k: q_d[k] / q_h[k] for k in q_d}))
                if src == 'пересчёт':  # справка: оценка на общих 16 тикерах (влияние состава обучения при том же наборе оценки)
                    q16d, q16h = qrow(d, 'xgb', COMMON16), qrow(d, 'har', COMMON16)
                    sub.append(dict(sostav=comp, h=h, model='финал' if h != 22 else 'финал = HAR-J с поправкой', **q16d))
                    if h != 22:
                        sub.append(dict(sostav=comp, h=h, model='HAR-J с поправкой', **q16h))
                        sub.append(dict(sostav=comp, h=h, model='отношение финал / HAR-J', **{k: q16d[k] / q16h[k] for k in q16d}))
            print(f'[{now()}] eval c{comp} h={h} готово', flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'qlike_po_godam.csv', index=False, float_format='%.6f')
    pd.DataFrame(sub).to_csv(OUT / 'qlike_obshchie16.csv', index=False, float_format='%.6f')
    # контроль: прогнозы пересчёта 17 против сохранённых
    for h in (1, 5):
        a, b = design_frame(h, '17'), design_frame(h, '17', saved=True)
        m = a.merge(b, on=['date', 'ticker', 'step'], suffixes=('', '_s'), how='outer', indicator=True)
        both = m[m._merge == 'both']
        r = np.abs(both.rv_pred / both.rv_pred_s - 1)
        ctl.append(dict(h=h, strok_obshchih=len(both), tolko_peresch=int((m._merge == 'left_only').sum()), tolko_sohr=int((m._merge == 'right_only').sum()),
                        max_otn_rashozhdenie=float(r.max()), sovpalo_pobitovo=float((both.rv_pred.values == both.rv_pred_s.values).mean())))
        if h == 1:
            for comp_ in ('17',):
                hx = pd.read_parquet(OUT / f'HXP_c{comp_}_lnall.parquet')
                hs = pd.read_parquet(R / 'D22' / 'HXP_lnall.parquet', columns=['date', 'ticker'] + [f'hxp_1_{k}' for k in STEPS])
                mm = hx.merge(hs, on=['date', 'ticker'], suffixes=('', '_s'))
                ctl.append(dict(h='слой 1 HAR-X⁺ (hxp_1_1..6)', strok_obshchih=len(mm), tolko_peresch=len(hx) - len(mm), tolko_sohr=len(hs) - len(mm),
                                max_otn_rashozhdenie=float(max(np.abs(mm[f'hxp_1_{k}'] - mm[f'hxp_1_{k}_s']).max() for k in STEPS)),
                                sovpalo_pobitovo=float(np.mean([(mm[f'hxp_1_{k}'].values == mm[f'hxp_1_{k}_s'].values).mean() for k in STEPS]))))
    pd.DataFrame(ctl).to_csv(OUT / 'kontrol_17.csv', index=False)
    print(T.to_string()); print(pd.DataFrame(ctl).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['data', 'h1', 'h5', 'otsenka'])
    ap.add_argument('--comp', choices=list(COMPS), default='17')
    ap.add_argument('--workers', type=int, default=5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'[{now()}] старт {a.mode} {a.comp if a.mode in ("h1", "h5") else ""}', flush=True)
    {'data': build_data, 'otsenka': evaluate}.get(a.mode, lambda: None)()
    if a.mode == 'h1':
        run_h1(a.comp, a.workers)
    if a.mode == 'h5':
        run_h5(a.comp, a.workers)


if __name__ == '__main__':
    main()
