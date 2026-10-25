#!/usr/bin/env python3
"""Финальная сборка второго круга, шаг 5: тестовые прогнозы финального дизайна h — шаги 1-9 (gibrid_v17.py --build-test) + шаг 10
(обучение <= 2024, val 2025, прогноз 01-02.2026; final2_step10.py). БЕЗ МЕТРИК. Только дизайны V0 (одна компонента, L0).
Шаг 10 V0: прогноз ноги × F, F = exp(s2/2), s2 — оценка кода дизайна по val-остаткам шагов <= 10 (как на тесте шага k: оценка шага k;
sm/smrob — val-годы 2023-2025). Столбцы как в первом круге (final_test.py): rv_pred, har_raw, har_corr, har_ord, xgb/lgb_corr, xgb/lgb_raw,
design, h, scheme_test. Контроль кода шага 10: тот же расчёт на шаге 9 из файлов ноги совпадает с --build-test (печатается расхождение).
python3 scripts/analysis/final2_build.py --boost F2_h5 --h 5 --design "wrap2|V0_XGB|B-sm|L0" [--out data/regen/test_locked/final2]"""
import argparse, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gibrid_har as GH
import gibrid_bias as BI
from gibrid_io import load_boost, MAIN_REGEN

R = MAIN_REGEN
K = ['date', 'ticker', 'step']


def har_step10(A, harj, h):
    from sklearn.linear_model import LinearRegression
    t = f'rv_target_h{h}'
    tr = A[A.year < 2026]
    tp, vi, td = tr[tr.year != 2025].dropna(subset=[t]), tr[tr.year == 2025].dropna(subset=[t]), A[A.year == 2026]
    X = lambda d: GH.clean(np.log(d[harj].values.clip(1e-10)))
    m = LinearRegression().fit(X(tp), np.log(tp[t].values + 1e-10))
    f = lambda d, yr: pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'year': yr, 'step': 10, 'rv_actual': d[t].values,
                                    'rv_pred': np.clip(np.exp(m.predict(X(d))), 1e-10, None)})
    return f(vi, 2025), f(td, 2026)


def leg10(boost, h, leg):
    """-> {'xgb': (val, test), 'lgb': (val, test)} ноги на шаге 10."""
    sd, lk = R / 'final2' / 'step10', R / 'test_locked' / 'final2' / 'step10'
    raw = {m: (pd.read_parquet(sd / f'{boost}_{m}_h{h}_s10_val.parquet'), pd.read_parquet(lk / f'{boost}_{m}_h{h}_s10_test.parquet')) for m in ('xgb', 'lgbm')}
    comps, prim = (), None
    if leg == 'wrap2':
        w = pd.read_csv(R / 'final2' / f'wrap2_{boost}_h{h}.csv')
        comps = tuple(x for x in str(w.final.iloc[0]).split('+') if x != 'none')
        prim = w.model.iloc[0]
    seeds = leg == 'seeds3' or '1b' in comps

    def one(df):
        d = df if seeds else df[df.seed == 42]
        return d.groupby(['date', 'ticker', 'year', 'step'], as_index=False, dropna=False).agg(rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))
    out = {m: tuple(one(raw[m][i]) for i in (0, 1)) for m in raw}
    if '2' in comps:
        avg = []
        for i in (0, 1):
            a, b = out['lgbm'][i], out['xgb'][i][['date', 'ticker', 'rv_pred']]
            mm = a.merge(b, on=['date', 'ticker'], suffixes=('_l', '_x'), validate='one_to_one')
            avg.append(mm.assign(rv_pred=(mm.rv_pred_l + mm.rv_pred_x) / 2).drop(columns=['rv_pred_l', 'rv_pred_x']))
        out[prim] = tuple(avg)
    return {'xgb': out['xgb'], 'lgb': out['lgbm']}


def wide(har, bx, bl):
    m = har.rename(columns={'rv_pred': 'har'})
    for n, d in (('xgb', bx), ('lgb', bl)):
        m = m.merge(d[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': n}), on=['date', 'ticker'])
    return m


def factors(Ev, code, st):
    return {c: float(np.exp(BI.s2_est(code, Ev, c, st) / 2)) for c in ('har', 'xgb', 'lgb')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boost', required=True); ap.add_argument('--h', type=int, required=True); ap.add_argument('--design', required=True)
    ap.add_argument('--out', default=str(R / 'test_locked' / 'final2'))
    a = ap.parse_args()
    h, out = a.h, Path(a.out)
    leg, sel, lab, lam = a.design.split('|')[:4]
    assert sel.startswith('V0_') and lam == 'L0', 'шаг 10 реализован только для V0/L0'
    comp, code = sel[3:].lower(), lab[2:]
    if comp == 'hcj':
        raise SystemExit('V0_HCJ: шаг 10 не реализован')
    feat = R / 'features_pit_v3.parquet'
    tmp = out / f'_tmp_h{h}'
    subprocess.run([sys.executable, '-W', 'ignore', str(Path(__file__).resolve().parent / 'gibrid_v17.py'), '--boost', a.boost, '--h', str(h),
                    '--features', str(feat), '--build-test', a.design, '--test-out', str(tmp)], check=True)
    T9 = pd.read_parquet(tmp / f'final_h{h}_test.parquet')
    A, harj = GH.load_pit(feat)
    hv, ht, _ = GH.har_forecasts(h, A, harj)
    bx, bl = load_boost(a.boost, h, R, leg)  # val шагов 1-9 (остатки для s2; метрики не считаются)
    bxt, blt = load_boost(a.boost, h, R, leg, test=True)
    V = wide(hv, bx['val'], bl['val'])
    V = V[V.rv_actual > 1e-12]
    E = {c: {st: np.log(g.rv_actual.values + 1e-10) - np.log(g[c].values) for st, g in V.groupby('step')} for c in ('har', 'xgb', 'lgb')}
    # контроль: шаг 9 тем же кодом
    F9 = factors(E, code, 9)
    T9w = wide(ht, bxt['test'], blt['test']); T9w = T9w[T9w.step == 9]
    chk = T9w[['date', 'ticker']].assign(p=T9w[comp].values * F9[comp]).merge(T9[T9.step == 9][['date', 'ticker', 'rv_pred']], on=['date', 'ticker'])
    print(f'контроль шага 9: макс. отн. расхождение с --build-test {float(np.max(np.abs(chk.p / chk.rv_pred - 1))):.2e} ({len(chk)} строк)', flush=True)
    # шаг 10
    hv10, ht10 = har_step10(A, harj, h)
    L10 = leg10(a.boost, h, leg)
    V10 = wide(hv10, L10['xgb'][0], L10['lgb'][0])
    V10 = V10[V10.rv_actual > 1e-12]
    for c in ('har', 'xgb', 'lgb'):
        E[c][10] = np.log(V10.rv_actual.values + 1e-10) - np.log(V10[c].values)
    F, Fo = factors(E, code, 10), factors(E, 'ord', 10)
    W = wide(ht10, L10['xgb'][1], L10['lgb'][1])
    o = W[['date', 'ticker', 'year', 'step', 'rv_actual']].assign(
        rv_pred=W[comp].values * F[comp], har_raw=W.har.values, har_corr=W.har.values * F['har'], xgb_corr=W.xgb.values * F['xgb'],
        lgb_corr=W.lgb.values * F['lgb'], xgb_raw=W.xgb.values, lgb_raw=W.lgb.values, har_ord=W.har.values * Fo['har'],
        design=a.design, h=h, scheme_test=sel)
    o = o[T9.columns]
    assert o[['rv_pred', 'har_raw', 'har_corr']].notna().all().all()
    full = pd.concat([T9, o], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out / f'final_h{h}_test.parquet', index=False)
    pd.DataFrame([dict(h=h, step=10, code=code, comp=comp, **{f'F_{c}': F[c] for c in F}, F_har_ord=Fo['har'], n_test=len(o),
                       n_val2025=len(V10))]).to_csv(out / f'final_h{h}_step10_factors.csv', index=False)
    for f in tmp.glob('*'):
        f.unlink()
    tmp.rmdir()
    print(f'h={h}: {a.design} ({a.boost}) -> {out / f"final_h{h}_test.parquet"}: шаги {sorted(full.step.unique())}, строк {len(full)} '
          f'(шаг 10: {len(o)}, {o.date.min().date()}–{o.date.max().date()}); F шага 10 ({code}) {F[comp]:.4f}; метрики не считались', flush=True)


if __name__ == '__main__':
    main()
