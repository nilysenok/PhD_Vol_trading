#!/usr/bin/env python3
"""Д16.2 — мета-модель согласования горизонтов (третий круг, владелец 26.10 00:55).
ln ŷ_h = a + b1·ln σ̂_1 + b5·ln σ̂_5 + b22·ln σ̂_22 + b0·ln HAR_h; σ̂_k — OOS-прогнозы ноги финалиста горизонта k (до поправки):
  h=1 D12_h1_A_d812 top5/XGB, h=5 D12_h5 wrap2/XGB, h=22 D12_h22 seeds3/XGB; HAR_h — HAR-J (features_pit_v3) с поправкой Б0.1 кода финалиста h
  (h=22 smrob, h=5 sm; вне выборки: val шага t — оценка шага t-1, тест шага t — оценка шага t), либо HAR-X (--harx, файлы агента B).
Коэффициенты: гребень на стандартизованных ln-регрессорах (λ=0 — МНК), пул по 17 тикерам; окно — val-год шага t-1 ('t1') или
объединённые val-годы 1..t-1 ('pool'); эмбарго — последние h торговых дней года оценки (цель уходит в год применения).
val шага t (t>=2) — оценка по шагу t-1 (вне выборки); val шага 1 — заполнитель в выборке (нужен гибриду для s² шага 2, в метрики не входит);
тест шага t (1..10) — оценка по val шага t (как поправка на тесте). Поправка мета-прогноза — тот же код Б0.1 по его val-остаткам (вне выборки).
Отбор (окно, λ) — средний QLIKE val 2017-2020 (шаги 2-5); 2021 (шаг 6) — проверка; шаги 7-9 — только оценка коэффициентов для теста, метрик нет.
Выход: <R>/D16_2_h{h}_{xgb,lgbm}_h{h}_val.parquet (мета в слоте XGB, нога LGB финалиста — эталон гибрида), <R>/D16_2_h{h}_vsx_* (в слоте LGB — σ̂_h),
<R>/D16_2_h{h}_designs.csv, _coefs.csv; тест без метрик — <R>/test_locked/D16_2/D16_2_h{h}_test.parquet и test_locked/D16_2_h{h}{,_vsx}/ (шаги 1-9).
python3 scripts/analysis/d16_2.py --h 22 [--harx <val.parquet> --harx-test <test.parquet> --tag D16_2x]"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gibrid_har as GH
import gibrid_bias as BI
from gibrid_io import load_boost, MAIN_REGEN as R
from final2_build import leg10, har_step10

FIN = {1: ('D12_h1_A_d812', 'top5'), 5: ('D12_h5', 'wrap2'), 22: ('D12_h22', 'seeds3')}  # нога XGB финалистов (V0_XGB)
CODE = {5: 'sm', 22: 'smrob'}  # поправка финалиста h (Б0.1)
K = ['date', 'ticker', 'step']
LAMS = (0.0, 10.0, 100.0, 1000.0, 10000.0)
WINS = ('t1', 'pool')
CONS = ('free', 'sum1')  # без ограничений / Σb = 1
SEL, CHK = (2, 3, 4, 5), 6


def qlike(y, p):
    r = y / p
    return float(np.mean(r - np.log(r) - 1))


def legs(k):
    """σ̂_k: (val шаги 1-10, тест шаги 1-10) — нога XGB финалиста k до поправки; + нога LGB (для эталона гибрида)."""
    b, leg = FIN[k]
    bx, bl = load_boost(b, k, R, leg)
    tx, tl = load_boost(b, k, R, leg, test=True)
    L10 = leg10(b, k, leg)
    out = {}
    for nm, v, t, s10 in (('xgb', bx, tx, L10['xgb']), ('lgb', bl, tl, L10['lgb'])):
        out[nm] = (pd.concat([v['val'], s10[0]], ignore_index=True)[K + ['year', 'rv_actual', 'rv_pred']],
                   pd.concat([t['test'], s10[1]], ignore_index=True)[K + ['year', 'rv_actual', 'rv_pred']])
    return out


def har(h, A, harj, code):
    """HAR-J h: raw и с поправкой code (val шага t — оценка t-1, шаг 1 — F=1; тест шага t — оценка t); шаги 1-10."""
    hv, ht, _ = GH.har_forecasts(h, A, harj)
    v10, t10 = har_step10(A, harj, h)
    hv, ht = pd.concat([hv, v10], ignore_index=True), pd.concat([ht, t10], ignore_index=True)
    hv = hv[hv.rv_actual > 1e-12]
    E = {'har': {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in hv.groupby('step')}}
    fv = {st: (np.exp(BI.s2_est(code, E, 'har', st - 1) / 2) if st > 1 else 1.0) for st in E['har']}
    ft = {st: np.exp(BI.s2_est(code, E, 'har', st) / 2) for st in E['har']}
    hv = hv.assign(har=hv.rv_pred * hv.step.map(fv))
    ht = ht.assign(har=ht.rv_pred * ht.step.map(ft))
    return hv, ht, ft


def wide(part, L, H, h, harx=None):
    i = 0 if part == 'val' else 1
    base = L[h]['xgb'][i][K + ['year', 'rv_actual']].rename(columns={'rv_actual': 'y'})
    for k in (1, 5, 22):
        base = base.merge(L[k]['xgb'][i][K + ['rv_pred']].rename(columns={'rv_pred': f's{k}'}), on=K, how='left')
    base = base.merge(L[h]['lgb'][i][K + ['rv_pred']].rename(columns={'rv_pred': 'lgb'}), on=K, how='left')
    hh = (H[0] if part == 'val' else H[1])[K + ['har', 'rv_pred']].rename(columns={'rv_pred': 'har_raw'})
    base = base.merge(hh, on=K, how='left')
    if harx is not None:
        base = base.drop(columns=['har']).merge(harx[K + ['har']], on=K, how='left')
    if part == 'val':
        base = base[base.y > 1e-12]
    return base.sort_values(K).reset_index(drop=True)


def embargo(d, h):
    last = np.sort(d.date.unique())[-h:]
    return d[~d.date.isin(last)]


def fit(d, lam, cols, sum1=None):
    """Гребень (λ=0 — МНК) на стандартизованных ln-регрессорах, свободный член без штрафа. sum1 = индекс опорного столбца:
    ограничение Σb = 1 (однородность первой степени: рост всех прогнозов в c раз -> рост ŷ в c раз), оценка на разностях с опорным."""
    X, y = np.log(d[cols].values), np.log(d.y.values + 1e-10)
    if sum1 is not None:
        y = y - X[:, sum1]
        X = np.delete(X, sum1, axis=1) - X[:, [sum1]]
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    b = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    bb = b / sd
    a0 = y.mean() - mu @ bb
    if sum1 is not None:
        bb = np.insert(bb, sum1, 1 - bb.sum())
    return a0, bb


def predict(d, ab, cols):
    return np.exp(ab[0] + np.log(d[cols].values) @ ab[1])


def run_design(V, T, h, win, lam, cols, code, sum1=None):
    """-> val (meta raw, F_oos, pred), test (raw, F, pred), коэффициенты."""
    steps = sorted(V.step.unique())
    vo, to, co = [], [], []
    fitrows = lambda st: embargo(V[V.step == st], h) if win == 't1' else pd.concat([embargo(V[V.step == s], h) for s in steps if s <= st])
    for st in steps:
        src = st - 1 if st > 1 else 1  # шаг 1 — заполнитель в выборке
        ab = fit(fitrows(src), lam, cols, sum1)
        vs = V[V.step == st]
        vo.append(vs[K + ['year', 'y']].assign(raw=predict(vs, ab, cols), insample=st == 1))
        co.append(dict(part='val', step=st, fit_on=src, a=ab[0], **{f'b_{c}': x for c, x in zip(cols, ab[1])}, b_sum=ab[1].sum()))
    for st in sorted(T.step.unique()):
        ab = fit(fitrows(st), lam, cols, sum1)
        ts = T[T.step == st]
        to.append(ts[K + ['year', 'y']].assign(raw=predict(ts, ab, cols)))
        co.append(dict(part='test', step=st, fit_on=st, a=ab[0], **{f'b_{c}': x for c, x in zip(cols, ab[1])}, b_sum=ab[1].sum()))
    vo, to = pd.concat(vo), pd.concat(to)
    E = {'m': {st: np.log(g.y.values + 1e-10) - np.log(g.raw.values) for st, g in vo.groupby('step')}}
    fv = {st: (np.exp(BI.s2_est(code, E, 'm', st - 1) / 2) if st > 1 else 1.0) for st in E['m']}
    ft = {st: np.exp(BI.s2_est(code, E, 'm', st) / 2) for st in E['m']}
    vo['pred'] = vo.raw * vo.step.map(fv)
    to['pred'] = to.raw * to.step.map(ft)
    return vo, to, pd.DataFrame(co)


def yearly(vo, col='pred', steps=SEL + (CHK,)):
    return {int(g.year.iloc[0]): qlike(g.y.values, g[col].values) for st, g in vo.groupby('step') if st in steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, default=22)
    ap.add_argument('--tag', default='D16_2')
    ap.add_argument('--harx', default=None, help='val-прогнозы HAR-X h (date, ticker, step, rv_pred) — вместо HAR-J; поправка тем же кодом')
    ap.add_argument('--harx-test', default=None)
    ap.add_argument('--no-har', action='store_true', help='справка: без ln HAR_h (b0 = 0)')
    a = ap.parse_args()
    h, code = a.h, CODE[a.h]
    tag = f'{a.tag}_h{h}' + ('_harx' if a.harx else '') + ('_nohar' if a.no_har else '')
    A, harj = GH.load_pit(R / 'features_pit_v3.parquet')
    L = {k: legs(k) for k in (1, 5, 22)}
    hv, ht, _ = har(h, A, harj, code)
    hx = None
    if a.harx:  # HAR-X агента B: та же процедура поправки (val — оценка t-1, тест — оценка t)
        xv, xt = pd.read_parquet(a.harx), pd.read_parquet(a.harx_test) if a.harx_test else None
        xv = xv[K + ['rv_pred']].merge(hv[K + ['rv_actual']], on=K)
        Ex = {'har': {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in xv.groupby('step')}}
        fv = {st: (np.exp(BI.s2_est(code, Ex, 'har', st - 1) / 2) if st > 1 else 1.0) for st in Ex['har']}
        ft = {st: np.exp(BI.s2_est(code, Ex, 'har', st) / 2) for st in Ex['har']}
        hx = (xv.assign(har=xv.rv_pred * xv.step.map(fv)), None if xt is None else xt.assign(har=xt.rv_pred * xt.step.map(ft)))
    V = wide('val', L, (hv, ht), h, hx[0] if hx else None)
    T = wide('test', L, (hv, ht), h, hx[1] if hx and hx[1] is not None else None)
    cols = ['s1', 's5', 's22'] + ([] if a.no_har else ['har'])
    nmiss = V[cols].isna().sum().to_dict()
    V = V.dropna(subset=cols)
    T = T.dropna(subset=cols)
    out = R
    # эталоны (те же строки, шаги 2-6): σ̂_h с поправкой кода (как V0_XGB финалиста), HAR с поправкой
    Ex = {'x': {st: np.log(g.y.values + 1e-10) - np.log(g[f's{h}'].values) for st, g in V.groupby('step')}}
    fx = {st: (np.exp(BI.s2_est(code, Ex, 'x', st - 1) / 2) if st > 1 else 1.0) for st in Ex['x']}
    ref = V[K + ['year', 'y']].assign(fin=V[f's{h}'] * V.step.map(fx), har=V.har)
    rows = [dict(design=f'ref_sigma{h}_{code}', **{f'q{y}': q for y, q in yearly(ref, 'fin').items()}),
            dict(design=f'ref_{"HARX" if a.harx else "HAR"}_{code}', **{f'q{y}': q for y, q in yearly(ref, 'har').items()})]
    res = {}
    for win, lam, c1 in ((w, l, c) for c in CONS for w in WINS for l in LAMS):
        nm = f'{win}|lam{lam:g}|{c1}'
        vo, to, co = run_design(V, T, h, win, lam, cols, code, cols.index(f's{h}') if c1 == 'sum1' else None)
        res[nm] = (vo, to, co)
        rows.append(dict(design=nm, **{f'q{y}': q for y, q in yearly(vo).items()}))
    D = pd.DataFrame(rows)
    ys = [f'q{2015 + s}' for s in SEL]
    D['mean_2017_2020'] = D[ys].mean(axis=1)
    D['n_years_better_HAR'] = (D[ys].values < D.loc[D.design.str.match('ref_HAR'), ys].values).sum(axis=1)
    cand = D[~D.design.str.startswith('ref_')]
    best = cand.sort_values(['mean_2017_2020']).design.iloc[0]
    D['chosen'] = D.design == best
    D.to_csv(out / f'{tag}_designs.csv', index=False)
    vo, to, co = res[best]
    co.insert(0, 'design', best)
    co.to_csv(out / f'{tag}_coefs.csv', index=False)
    # для гибрида (критерии 1-4, одиночная модель-кандидат V0_XGB): мета до поправки в слоте XGB; слот LGB — нога LGB финалиста h
    # (основной прогон) или σ̂_h финалиста (_vsx: мета против собственного входа). Тест (шаги 1-9) — test_locked/<boost>/ без метрик.
    mv = vo[K + ['year']].assign(rv_actual=vo.y, rv_pred=vo.raw)
    mt = to[K + ['year']].assign(rv_actual=to.y, rv_pred=to.raw)
    for sfx, col in (('', 'lgb'), ('_vsx', f's{h}')):
        b = f'{tag}{sfx}'
        lv = V[K + ['year']].assign(rv_actual=V.y, rv_pred=V[col])
        lt = T[K + ['year']].assign(rv_actual=T.y, rv_pred=T[col])
        mv[mv.step <= 9].to_parquet(out / f'{b}_xgb_h{h}_val.parquet', index=False)
        lv[lv.step <= 9].to_parquet(out / f'{b}_lgbm_h{h}_val.parquet', index=False)
        td = out / 'test_locked' / b
        td.mkdir(parents=True, exist_ok=True)
        mt[mt.step <= 9].to_parquet(td / f'{b}_xgb_h{h}_test.parquet', index=False)
        lt[lt.step <= 9].to_parquet(td / f'{b}_lgbm_h{h}_test.parquet', index=False)
    # тестовые прогнозы мета-модели (шаги 1-10) — БЕЗ МЕТРИК
    tl = out / 'test_locked' / 'D16_2'
    tl.mkdir(parents=True, exist_ok=True)
    tt = to.merge(T[K + ['s1', 's5', 's22', 'har', 'har_raw']], on=K, how='left')
    tt = tt.rename(columns={'y': 'rv_actual', 'pred': 'rv_pred', 'raw': 'rv_pred_raw', 'har': 'har_corr'}).assign(design=f'{tag}|{best}|B-{code}', h=h)
    tt.to_parquet(tl / f'{tag}_test.parquet', index=False)
    print(D.round(4).to_string())
    print('выбран:', best, '| пропуски входов val:', nmiss, '| строк val', len(V), 'тест', len(tt), 'шаги теста', sorted(tt.step.unique()))


if __name__ == '__main__':
    main()
