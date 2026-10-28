#!/usr/bin/env python3
"""Д26.1 — D22-слой-2 на прямую цель h=5: Д8.12 на HAR-X⁺₅ (ln HAR-X⁺₅ модели шага — признак; на обучении — внутри выборки, как в D22_h1),
признаки v3 + календарь окна h=5 + RRV PK/GK d/w/m + ночной гэп (как слой 1 Д22); винзоризация цели 1/99 по тикеру; протокол B v2
(tune_L2: ЛГК LGBM-100 / XGB-40 с отсечением, подбор на шагах 1–6). Прогноз — среднее ln LGBM и ln XGB (как hyb₁ Д22).
Шаги 7–10 (точка шага 6) — только для дизайна, прошедшего условие входа (--fixed); их val-прогнозы не оцениваются, тест — в
test_locked/D26_1/ без метрик.
  --tune [--workers 6]           шаги 1–6 (seed 42 топ-5, seed 43/44 у точки ранга 1)
  --seeds 45 46 ... [--steps]    дополнительные seed у точки ранга 1 (10 seed: 42–51)
  --fixed --steps 7 8 9 10       точка шага 6, seed 42/43/44 (тест — test_locked/D26_1/)
  --leak                         детектор «будущее не влияет» (+ подложенная утечка)
Выход: data/regen/D26/D26_1_{xgb,lgbm}_h5[_top5|_seeds]_val.parquet, D26_1_selected_h5.csv, D26_1_trials.csv."""
import os
os.environ['L2_FEAT'] = '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3_rrv.parquet'
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
import argparse, json, sys, time, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L
import d22_harxp as P
import tune_L2 as T

R = L.REGEN
OUT = R / 'D26'
TAG = 'D26_1'
LOCK = R / 'test_locked' / TAG  # только запись
H = 5
CFG = {'harf': True, 'wins': True}
EXTRA = ['rrv_pk_d', 'rrv_pk_w', 'rrv_pk_m', 'rrv_gk_d', 'rrv_gk_w', 'rrv_gk_m', 'r2_on'] + [f'{f}@5' for f in P.CAL_FIELDS]
HXP_LN = R / 'D22' / 'HXP_lnall.parquet'
_split0, _bf0, _lb0 = L.split, L.build_features, L.load_base
ST = {'lncol': 'hxp_5_{k}', 'leak': False}


def har_ln_x(tp, parts, h, hc, w=None, huber=False, y=None):
    """ln HAR-X⁺₅ модели шага k = (последний год обучения + 1) − 2015 на строках каждой части (обучение — внутри выборки)."""
    k = int(tp['year'].max()) + 1 - 2015
    col = ST['lncol'].format(k=k)
    return [d[col].values for d in parts]


def _split10(A, h, step):
    if step != 10:
        return _split0(A, h, step)
    t = L.tcol(h)
    tr = A[A.year < 2026]
    return tr[tr.year != 2025].dropna(subset=[t]), tr[tr.year == 2025].dropna(subset=[t]), A[A.year == 2026], 2025, 2026


def load_base_x(matrix=None, blocks=(), h=None):
    A, fc = _lb0(matrix, blocks, h)
    A = A.merge(pd.read_parquet(HXP_LN), on=['date', 'ticker'], how='left', validate='one_to_one')
    assert A['hxp_5_1'].notna().all()
    return A, fc


def build_features_x(A, FC, h, cfg):
    A2, cols, cats, mono = _bf0(A, FC, h, cfg)
    X = P.build_terms(A2, EXTRA)
    for c in EXTRA:
        A2[c] = X[c].values
    cols = cols + [c for c in EXTRA if c not in cols]
    if ST['leak']:  # подложенная утечка для самопроверки детектора: RV через 3 дня (из строк будущего)
        A2['_leak_rv3'] = A2.groupby('ticker')['rv'].shift(-3).values
        cols = cols + ['_leak_rv3']
    return A2, cols, cats, mono


def patch():
    L.har_ln, L.split, L.build_features, L.load_base = har_ln_x, _split10, build_features_x, load_base_x


def frame(D, k, step, pred, **kw):
    df, yr = (D['parts']['va'], D['vy']) if k == 'val' else (D['parts']['te'], D['ty'])
    return pd.DataFrame({'date': df['date'].values, 'ticker': df['ticker'].values, 'year': yr, 'step': step, **kw,
                         'rv_actual': df[L.tcol(H)].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)})


def prepare(A=None):
    patch()
    if A is None:
        A, FC = L.load_base()
    else:
        FC = [c for c in json.load(open(L.FEAT_CFG))['feature_cols'] if c in A.columns]
    A2, cols, _, _ = L.build_features(A, FC, H, CFG)
    return A2, cols


# ---------------------------------------------------------------- протокол B v2, шаги 1–6 (как tune_L2.main)
def tune(workers):
    A2, cols = prepare()
    OUT.mkdir(parents=True, exist_ok=True); LOCK.mkdir(parents=True, exist_ok=True)
    EXC = R / f'{TAG}_excluded_features.csv'  # l2_lib.excluded читает из data/regen/{tag}_excluded_features.csv
    if EXC.exists():
        EXC.unlink()
    GR = pd.concat([T.grid(T.SPACEB, 'lgbm', 100), T.grid(T.SPACEB, 'xgb', 40), T.grid(T.SPACEB, 'xgb', 100, 100)]).set_index(['model', 'point'])
    GR.reset_index().to_csv(OUT / f'{TAG}_grid.csv', index=False)
    T.G.update(A=A2, cols=cols, cfg=CFG, h=H, tag=TAG, grid=GR)
    t0 = time.time()
    print(f'{TAG}: cfg={CFG}, признаков {len(cols)}', flush=True)
    surv = {'lgbm': list(range(100)), 'xgb': list(range(40))}; exp = False; sel, trl, fin = {}, [], []
    ctx = mp.get_context('fork')
    for step in range(1, 7):
        with ctx.Pool(workers) as Pl:
            res = [x for r in Pl.map(T.trials, T.batches(surv, step, workers, 300 if step in (2, 3) else L.NMAX), chunksize=1) for x in r]
            if step == 1:
                comb = pd.DataFrame([r for r, *_ in res]).sort_values('qlike_val').reset_index(drop=True)
                rk = int(comb.index[comb.model == 'xgb'][0]); exp = rk < 0.2 * len(comb)
                pd.DataFrame([dict(h=H, best_xgb_rank=rk + 1, n_combined=len(comb), expanded=exp)]).to_csv(OUT / f'{TAG}_xgb_budget.csv', index=False)
                if exp:
                    res += [x for r in Pl.map(T.trials, T.batches({'xgb': list(range(100, 200))}, step, workers, L.NMAX), chunksize=1) for x in r]
        for r, *_ in res:
            trl.append(r)
        tr = pd.DataFrame([r for r, *_ in res])
        if step == 1:
            ex = []
            for m in ('lgbm', 'xgb'):
                gs = [(g, nm) for (r, _, _, g, nm) in res if r['model'] == m]
                tot = pd.Series(0.0, index=gs[0][1])
                for g, nm in gs:
                    tot = tot.add(pd.Series(g, index=nm), fill_value=0)
                ex += [dict(model=m, h=H, feature=f) for f in tot.index[tot <= 0] if f not in ('har_ln', 'h1_ln')]
            pd.DataFrame(ex, columns=['model', 'h', 'feature']).to_csv(EXC, index=False)
            print(f'исключено после шага 1: {len(ex)} ({", ".join(sorted(set(e["feature"] for e in ex)))})', flush=True)
            T._c.clear(); T._r.clear()
        tasks = []
        for m in ('lgbm', 'xgb'):
            top = tr[tr.model == m].sort_values('qlike_val').point.astype(int).tolist(); sel[m] = top[0]
            tasks += [(m, step, pt, 42, r + 1) for r, pt in enumerate(top[:5])] + [(m, step, top[0], sd, 1) for sd in (43, 44)]
            kn = (T.KEEPX if (m == 'xgb' and not exp) else T.KEEP).get(step); surv[m] = top[:kn] if kn else surv[m]
        with ctx.Pool(workers) as Pl:
            fin += Pl.map(T.final, tasks, chunksize=1)
        print(f'шаг {step} готов, {time.time() - t0:.0f} с; выбраны {sel}', flush=True)
    pd.DataFrame(trl).to_csv(OUT / f'{TAG}_trials.csv', index=False)
    sels = pd.DataFrame([r for r, *_ in fin if r['seed'] == 42 and r['rank'] == 1]).join(GR, on=['model', 'point'])
    sels.to_csv(OUT / f'{TAG}_selected_h{H}.csv', index=False)
    for m in ('lgbm', 'xgb'):
        pick = lambda f, ix: pd.concat([x[ix] for x in fin if f(x[0]) and x[0]['model'] == m])
        for k, ix, d in (('val', 1, OUT), ('test', 2, LOCK)):
            pick(lambda r: r['seed'] == 42 and r['rank'] == 1, ix).to_parquet(d / f'{TAG}_{m}_h{H}_{k}.parquet', index=False)
            pick(lambda r: r['seed'] == 42, ix).to_parquet(d / f'{TAG}_{m}_h{H}_top5_{k}.parquet', index=False)
            pick(lambda r: r['seed'] != 42, ix).to_parquet(d / f'{TAG}_{m}_h{H}_seeds4344_{k}.parquet', index=False)
    print(f'{TAG} готов {time.strftime("%F %T")}, {time.time() - t0:.0f} с', flush=True)


# ---------------------------------------------------------------- точка ранга 1 с другими seed / шаги 7–10
G = {}
PCOLS = {'xgb': [k for k, *_ in T.SPACEB['xgb']], 'lgbm': [k for k, *_ in T.SPACEB['lgbm']]}


def _job(a):
    m, step, seed = a
    D = L.step_data(G['A'], G['cols'], H, step, m, CFG, TAG)
    ref = L.make_ref(m, G['A'], G['cols'], H, CFG, TAG) if step > 1 else None
    p = G['sel'][(m, min(step, 6))]
    pv, pt, bi, _, _ = L.fit(m, p, D, CFG, seed=seed, ref=ref)
    return m, step, seed, bi, frame(D, 'val', step, pv, seed=seed, rank=1), frame(D, 'test', step, pt, seed=seed, rank=1)


def refit(steps, seeds, workers, sfx, A=None):
    A2, cols = prepare(A)
    sel = L.selected_params(str(OUT / f'{TAG}_selected_h{H}.csv'), H)
    G.update(A=A2, cols=cols, sel={(m, s): {k: sel[(m, s)][k] for k in PCOLS[m]} for m in ('xgb', 'lgbm') for s in range(1, 7)})
    jobs = [(m, s, sd) for s in steps for m in ('lgbm', 'xgb') for sd in seeds]
    with mp.get_context('fork').Pool(workers) as Pl:
        res = Pl.map(_job, jobs, chunksize=1)
    if sfx is None:
        return res
    LOCK.mkdir(parents=True, exist_ok=True)
    for m in ('xgb', 'lgbm'):
        rr = [r for r in res if r[0] == m]
        pd.concat([r[4] for r in rr]).to_parquet(OUT / f'{TAG}_{m}_h{H}{sfx}_val.parquet', index=False)
        pd.concat([r[5] for r in rr]).to_parquet(LOCK / f'{TAG}_{m}_h{H}{sfx}_test.parquet', index=False)
    pd.DataFrame([dict(model=r[0], step=r[1], seed=r[2], best_iteration=r[3]) for r in res]).to_csv(OUT / f'{TAG}{sfx}_trees.csv', index=False)
    print(f'{TAG}{sfx}: {len(res)} моделей', flush=True)
    return res


# ---------------------------------------------------------------- детектор «будущее не влияет»
def perturb(A, cut, seed=20261026):
    """Строки с датой ≥ cut: числовые столбцы × exp(N(0,1)), цели перемешаны (внутри будущего)."""
    rng = np.random.default_rng(seed)
    B = A.copy()
    fut = pd.to_datetime(B['date']) >= pd.Timestamp(cut)
    for c in B.columns:
        if c in ('date', 'ticker', 'year') or not np.issubdtype(B[c].dtype, np.number):
            continue
        v = B.loc[fut, c].values.astype('float64')
        B.loc[fut, c] = rng.permutation(v) if c.startswith('rv_target') else v * np.exp(rng.normal(0, 1, len(v)))
    return B


def leak(workers):
    """Шаг 3 (обучение < 2017, val 2018) и шаг 5 (val 2020): входы и цели с 1 января года после val заменены шумом -> val-прогнозы
    шага побайтно равны эталону. HAR-X⁺₅ (признак) — файл слоя 1, его детектор — у Д22 (leak_test_hxp.csv); здесь столбцы hxp_* тоже
    зашумлены в будущем. Самопроверка: подложенная утечка (RV через 3 дня как признак) должна дать расхождение."""
    rows = []
    A0, FC = (patch(), L.load_base())[1]
    for leak_on in (False, True):
        ST['leak'] = leak_on
        for step, cut in ((3, '2019-01-01'), (5, '2021-01-01')):
            outs = []
            for B in (A0, perturb(A0, cut)):
                res = refit([step], [42], workers, None, A=B)
                outs.append({r[0]: r[4].rv_pred.values for r in res})
            for m in ('xgb', 'lgbm'):
                d = float(np.max(np.abs(np.log(outs[0][m]) - np.log(outs[1][m]))))
                rows.append(dict(contour='Д26.1', leak_planted=leak_on, step=step, cut=cut, model=m, max_abs_dln=d, ok=(d == 0.0) != leak_on))
                print(rows[-1], flush=True)
    ST['leak'] = False
    pd.DataFrame(rows).to_csv(OUT / 'leak_d26_1.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tune', action='store_true'); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--seeds', type=int, nargs='*'); ap.add_argument('--steps', type=int, nargs='*', default=list(range(1, 7)))
    ap.add_argument('--fixed', action='store_true'); ap.add_argument('--leak', action='store_true')
    a = ap.parse_args()
    if a.tune:
        tune(a.workers)
    if a.seeds:
        refit(a.steps, a.seeds, a.workers, '_seeds' + '_'.join(map(str, a.seeds)) if len(a.seeds) < 4 else '_seeds10x')
    if a.fixed:
        refit(a.steps, [42, 43, 44], a.workers, '_s' + ''.join(map(str, a.steps)))
    if a.leak:
        leak(a.workers)


if __name__ == '__main__':
    main()
