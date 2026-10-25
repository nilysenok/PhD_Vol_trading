#!/usr/bin/env python3
"""etap5, вариант B: ТОЛЬКО валидационная сводка. QLIKE и сравнения B/A/П1 — по шагам 1–6 (val 2016–2021, разработочный набор);
шаги 7–9 (val 2022–2024) скрыты: для них только параметры/best_iteration. Масштаб прогнозов — по всем годам. Тестовые QLIKE не считаются.
A (json): xgb = xgb_h*_val (параметры json, ES), lgbm = P1_h*_val (параметры json, gbdt+ES). Запуск из корня: python3 scripts/analysis/etap5_B_val.py
Выход: data/out/etap5_B_qlike.csv, etap5_B_masshtab.csv, etap5_B_stability.csv, etap5_B_ranges.csv"""
from pathlib import Path
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]; REG = ROOT / 'data/regen'; OUT = ROOT / 'data/out'; OUT.mkdir(parents=True, exist_ok=True)
DEV = range(1, 7)
qlike = lambda y, p: float(np.mean((r := y / np.clip(p, 1e-10, None)) - np.log(r) - 1))
SRC = {('xgb', 'B'): 'B_xgb', ('xgb', 'A'): 'xgb', ('lgbm', 'B'): 'B_lgbm', ('lgbm', 'A'): 'P1'}  # A lgbm = П1 (json + ES)

rows, sc = [], []
for h in (1, 5, 22):
    for (m, role), pre in SRC.items():
        for kind in ('val', 'test'):
            f = REG / f'{pre}_h{h}_{kind}.parquet'
            if not f.exists(): continue
            d = pd.read_parquet(f); name = f'{m}_{role}'
            r = d.groupby(['ticker', 'year']).apply(lambda g: g.rv_pred.median() / g.rv_actual.median())
            sc.append(dict(variant=name, h=h, kind=kind, n=len(r), out=int(((r < .5) | (r > 2)).sum()), min=r.min(), max=r.max()))
            if kind == 'val':
                for s, g in d[d.step.isin(DEV)].groupby('step'):
                    rows.append(dict(variant=name, h=h, step=int(s), val_year=int(g.year.iloc[0]), qlike_val=qlike(g.rv_actual.values, g.rv_pred.values)))
q = pd.DataFrame(rows); q.to_csv(OUT / 'etap5_B_qlike.csv', index=False)
pd.DataFrame(sc).to_csv(OUT / 'etap5_B_masshtab.csv', index=False)
pv = q.pivot_table(index=['h', 'variant'], columns='step', values='qlike_val'); pv['mean'] = pv.mean(axis=1)
print('Валидационный QLIKE, шаги 1–6 (val 2016–2021)'); print(pv.round(4).to_string())
print('\nB относительно A, шаги 1–6 (% изменения среднего QLIKE; <0 — B лучше)')
for h in (1, 5, 22):
    for m in ('xgb', 'lgbm'):
        a, b = pv.loc[(h, f'{m}_A'), 'mean'], pv.loc[(h, f'{m}_B'), 'mean']
        w = (pv.loc[(h, f'{m}_B'), list(DEV)] < pv.loc[(h, f'{m}_A'), list(DEV)]).sum()
        print(f'h{h} {m}: A {a:.4f} B {b:.4f} ({100*(b/a-1):+.1f}%), B лучше на {w}/6 шагах')
print('\nМасштаб прогнозов, тикер-годов вне [0,5; 2] (все годы)'); print(pd.DataFrame(sc).round(2).to_string(index=False))

sel = pd.read_csv(REG / 'B_selected.csv'); tr = pd.read_csv(REG / 'B_trials.csv').drop_duplicates(['model', 'h', 'step', 'point'])
st = []
for (m, h), g in sel.groupby(['model', 'h']):
    t = tr[(tr.model == m) & (tr.h == h) & tr.step.isin(DEV)]
    gap = t.groupby('step').apply(lambda x: (lambda s: s.iloc[4] / s.iloc[0] - 1)(x.qlike_val.sort_values())).mean()  # top-5 vs top-1, только шаги 1–6
    st.append(dict(model=m, h=h, distinct_points=g.point.nunique(), steps=len(g), median_best_iter=g.best_iteration.median(), top5_vs_top1_qlike_pct_dev=100 * gap))
pd.DataFrame(st).to_csv(OUT / 'etap5_B_stability.csv', index=False); print('\nСтабильность выбора'); print(pd.DataFrame(st).round(2).to_string(index=False))
par = [c for c in sel.columns if c not in ('model', 'h', 'step', 'point', 'best_iteration', 'qlike_val')]
rg = sel.groupby('model')[par + ['best_iteration']].agg(['min', 'median', 'max']).T
rg.to_csv(OUT / 'etap5_B_ranges.csv'); print('\nФактические диапазоны выбранных параметров (шаги 1–9)'); print(rg.round(4).to_string())

# --- ворота бустинга (справочно, отладочные данные): лучший B (XGB/LGBM) против HAR-J, оба с поправкой Б0.1 (ord, вне выборки), шаги 1–6 ---
import json, warnings
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')
MV = ROOT / 'moex_volatility/data/prepared'
parts = [pd.read_parquet(MV / f'{n}.parquet') for n in ('train', 'val', 'test')] + [d for y in range(2020, 2027) if (MV / f'walkforward_{y}.parquet').exists()
         and len(d := pd.read_parquet(MV / f'walkforward_{y}.parquet')) > 100]
AD = pd.concat(parts).sort_values('date').reset_index(drop=True); AD['year'] = pd.to_datetime(AD['date']).dt.year
cfg = json.load(open(MV / 'config.json')); jump = [c for c in AD.columns if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower()]
HARJ = [f for f in cfg['har_features'] + [c for c in jump if c not in cfg['har_features']] if f in AD.columns]  # как walk_forward_full.py:61-100


def har_val(h):  # val-прогнозы HAR-J шагов 1–6 (обучение: окно без val-года, как в walk_forward_full.py)
    t = f'rv_target_h{h}'; out = []
    X = lambda d: np.nan_to_num(np.log(d[HARJ].values.clip(1e-10)), nan=0, posinf=0, neginf=0)
    for step, ty in enumerate([y for y in range(2017, 2026) if y in AD.year.values], 1):
        if step not in DEV: continue
        tr = AD[AD.year < ty]; vy = sorted(tr.year.unique())[-1]; tp = tr[tr.year != vy].dropna(subset=[t]); vi = tr[tr.year == vy].dropna(subset=[t])
        m = LinearRegression().fit(X(tp), np.log(tp[t].values + 1e-10))
        out.append(pd.DataFrame({'date': vi['date'].values, 'ticker': vi['ticker'].values, 'step': step, 'rv_actual': vi[t].values, 'rv_pred': np.clip(np.exp(m.predict(X(vi))), 1e-10, None)}))
    return pd.concat(out)


def corrected_q(d):  # QLIKE по шагам: с F=exp(s2/2), s2 = дисперсия ln-остатков val-года ПРЕДЫДУЩЕГО шага (шаг 1: F=1), и без поправки
    d = d[d.rv_actual > 1e-12]; e = {s: np.var(np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values)) for s, g in d.groupby('step')}
    return {s: (qlike(g.rv_actual.values, g.rv_pred.values * np.exp(e.get(s - 1, 0) / 2)), qlike(g.rv_actual.values, g.rv_pred.values)) for s, g in d.groupby('step')}


gate = []
for h in (1, 5, 22):
    hv = har_val(h); r = {'HAR': corrected_q(hv)}
    for m in ('xgb', 'lgbm'):
        d = pd.read_parquet(REG / f'B_{m}_h{h}_val.parquet'); d = d[d.step.isin(DEV)].merge(hv[['date', 'ticker']], on=['date', 'ticker']); r[m] = corrected_q(d)
    mean = lambda k, i: np.mean([v[i] for v in r[k].values()])
    best = min(('xgb', 'lgbm'), key=lambda k: mean(k, 0)); hq, bq = mean('HAR', 0), mean(best, 0)
    rule = {1: 'без порога (справочно)', 5: 'не хуже HAR-J', 22: 'не хуже HAR-J более чем на 5%'}[h]
    ok = {1: None, 5: bool(bq <= hq), 22: bool(bq <= hq * 1.05)}[h]
    gate.append(dict(h=h, har_corr=hq, xgbB_corr=mean('xgb', 0), lgbmB_corr=mean('lgbm', 0), best=best, best_vs_har_pct=100 * (bq / hq - 1), rule=rule, passed=ok,
                     har_raw=mean('HAR', 1), xgbB_raw=mean('xgb', 1), lgbmB_raw=mean('lgbm', 1)))
G = pd.DataFrame(gate); G.to_csv(OUT / 'etap5_B_gate.csv', index=False); print('\nВорота бустинга (справочно; шаги 1–6; QLIKE с поправкой Б0.1 ord вне выборки)'); print(G.round(4).to_string(index=False))
sel['dev'] = sel.step.isin(DEV); sel['degenerate'] = sel.best_iteration < 10
dg = pd.concat([sel.groupby(['model', 'h']).degenerate.agg(['sum', 'count']).assign(set='шаги 1–9'), sel[sel.dev].groupby(['model', 'h']).degenerate.agg(['sum', 'count']).assign(set='шаги 1–6')])
dg['share'] = dg['sum'] / dg['count']; dg.to_csv(OUT / 'etap5_B_degenerate.csv'); print('\nДоля шагов с best_iteration < 10'); print(dg.round(3).to_string())
sel.drop(columns=['dev', 'degenerate']).assign(qlike_val=lambda t: t.qlike_val.where(t.step.isin(DEV))).to_csv(OUT / 'etap5_B_selected_dev.csv', index=False)  # val QLIKE шагов 7–9 скрыт
