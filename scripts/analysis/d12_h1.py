#!/usr/bin/env python3
"""Второй круг Д12 на h=1 (решение владельца 25.10, 20:45): кандидаты гибрида против V0 с одной и той же обёрткой.
Обёртка — как у V0 первого круга (B v2b, h=1): для каждого бустинга (XGB, LGBM) ноги single (лучшая точка шага, seed 42),
top5 (среднее RV пяти лучших точек ЛГК-поиска B v2b шага, шаги 1-6; на 7-9 — точка шага 6), seeds3 (лучшая точка, seed 42/43/44).
Точки — ровно те, что в Bv2b_{model}_h1_top5_val.parquet / Bv2b_selected_h1.csv; обучение — l2_lib.fit (ES на val-годе шага, patience 30,
бины окна шага 1 для шагов >= 2). Отличие кандидатов — только cfg (l2_lib) и beta:
  V0        {}                                   — XGB/LGBM на ln RV (база, перегон на v3)
  A_d812    {"harf": true}                       — Д8.12: ln HAR-J как признак (+ компоненты rv_d, rv_w, rv_m, jump* уже в матрице)
  B_b{β}    {"v9": true, "beta": β}              — ŷ = β·ln HAR-J + f(x), f на ln y − β·ln HAR-J (β=1 — V9)
  C_*       V9 + Д12.3 har_oof / Д12.4 rqw / Д12.5 huber (кумулятивно)
Выход (формат gibrid_io.load_boost, boost = D12_h1_<канд>): val шагов 1-9 — data/regen/D12_h1_<канд>_{xgb,lgbm}_h1{,_top5,_seeds4344}_val.parquet;
тест шагов 1-9 — data/regen/test_locked/D12_h1_<канд>/ (не оценивается). Метрики здесь — только шаги 2-6 (разработочный набор, Б0.1 вне выборки).
python3 scripts/analysis/d12_h1.py --cands V0 B_b1 [--legs single top5 seeds3] [--workers 4] [--matrix data/regen/features_pit_v3.parquet]"""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L

R = L.REGEN
H = 1
CANDS = {
    'V0': {},
    'A_d812': {'harf': True},
    'B_b0.25': {'v9': True, 'beta': 0.25}, 'B_b0.5': {'v9': True, 'beta': 0.5}, 'B_b0.75': {'v9': True, 'beta': 0.75},
    'B_b1': {'v9': True},
    'C_oof': {'v9': True, 'har_oof': True},
    'C_rqw': {'v9': True, 'rqw': True},
    'C_huber': {'v9': True, 'huber': True},
}
G = {}
_split0, _rqw0 = L.split, L.rq_weights


def _split17(A, h, step):
    """Д12.1: обучение на всех тикерах матрицы (28), val (ранняя остановка, оценка) и тест — только 17 основных (is_main17)."""
    tp, vi, td, vy, ty = _split0(A, h, step)
    return tp, vi[vi.is_main17.astype(bool)], td[td.is_main17.astype(bool)], vy, ty


def _rqw28(tp):
    """Д12.4 на матрице 28: RQ из столбца rq матрицы (тот же день t), обрезка 1/99, нормировка к среднему 1."""
    if 'rq' not in tp:
        return _rqw0(tp)
    rq = tp['rq'].values.astype(float)
    w = 1.0 / np.where(rq > 0, rq, np.nan)
    w = np.where(np.isnan(w), np.nanmedian(w), w)
    lo, hi = np.percentile(w, [1, 99])
    w = np.clip(w, lo, hi)
    return w / w.mean()


PCOLS = ('num_leaves', 'learning_rate', 'bagging_fraction', 'feature_fraction', 'min_data_in_leaf', 'reg_alpha', 'reg_lambda',
         'max_depth', 'subsample', 'colsample_bytree', 'min_child_weight')


def grid():
    g = pd.concat([pd.read_csv(R / f'Bv2b_grid_{x}.csv') for x in ('xgb40', 'xgb100', 'lgbm100')])
    return {(r['model'], int(r['point'])): {k: r[k] for k in PCOLS} for _, r in g.iterrows()}


def plan(legs):
    """[(model, step, point, seed, rank)]: rank 1..5 — точки top5 (шаги 1-6); rank 1 на шагах 7-9 — выбранная точка (шаг 6); seed 43/44 — seeds3."""
    S = pd.read_csv(R / 'Bv2b_selected_h1.csv')
    out = []
    for m in ('xgb', 'lgbm'):
        T = pd.read_parquet(R / f'Bv2b_{m}_h1_top5_val.parquet').groupby(['step', 'rank']).point.first()
        for s in range(1, 10):
            sp = int(S[(S.model == m) & (S.step == s)].point.iloc[0])
            if s <= 6:
                assert int(T[(s, 1)]) == sp
                for r in range(1, 6 if 'top5' in legs else 2):
                    out.append((m, s, int(T[(s, r)]), 42, r))
            else:
                out.append((m, s, sp, 42, 1))
            if 'seeds3' in legs:
                out += [(m, s, sp, sd, 1) for sd in (43, 44)]
    return out


def job(a):
    cand, m, s, pnt, seed, rank = a
    cfg = CANDS[cand]
    A, cols = G['A'], G['cols'][cand]
    p = G['grid'][(m, pnt)]
    D = L.step_data(A, cols, H, s, m, cfg, 'Bv2b')
    if cfg.get('beta') is not None:
        D['off'] = {k: cfg['beta'] * v for k, v in D['off'].items()}
    key = (cand, m, cfg.get('beta'))
    ref = L.make_ref(m, A, cols, H, cfg, 'Bv2b') if s > 1 else None
    pv, pt, bi, _, _ = L.fit(m, p, D, cfg, seed=seed, ref=ref)
    fr = []
    for part, pred in (('va', pv), ('te', pt)):
        d = D['parts'][part]
        fr.append(pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': d['year'].values, 'step': s, 'point': pnt,
                                'seed': seed, 'rank': rank, 'rv_actual': d[L.tcol(H)].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
    return cand, m, s, pnt, seed, rank, bi, fr[0], fr[1]


def evaluate_dev(V):
    """V: val шагов 1..9 одной модели/ноги -> {код: среднее шагов 2-6}. Шаги 7-9 отбрасываются ДО расчёта (подтверждающий набор закрыт)."""
    e = L.evaluate(V[V.step <= 6])
    return {k: v[0] for k, v in e.items()} | {'sm_steps': e['sm'][1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cands', nargs='+', required=True)
    ap.add_argument('--legs', nargs='+', default=['single', 'top5', 'seeds3'])
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--matrix', default=str(R / 'features_pit_v3.parquet'))
    ap.add_argument('--prefix', default='D12_h1')
    ap.add_argument('--t28', action='store_true', help='Д12.1: матрица features_pit_v3_28, обучение на 28, оценка на 17; имена кандидатов + _28')
    ap.add_argument('--drop-tickers', nargs='*', default=[], help='справка Д12.1: исключить тикеры из обучения (UTAR RGSS -> _26)')
    a = ap.parse_args()
    t0 = time.time()
    if a.t28:
        a.matrix = str(R / 'features_pit_v3_28.parquet')
        L.split, L.rq_weights = _split17, _rqw28
    A, FC = L.load_base(Path(a.matrix))
    if a.drop_tickers:
        A = A[~A.ticker.isin(a.drop_tickers)].reset_index(drop=True)
    sfx_t = ('_%d' % A.ticker.nunique()) if a.t28 else ''
    G.update(A=A, grid=grid(), cols={})
    for c in a.cands:
        _, cols, _, _ = L.build_features(A, FC, H, CANDS[c])
        G['cols'][c] = cols
    pl = plan(a.legs)
    tasks = [(c,) + t for c in a.cands for t in pl]
    print(f'{time.strftime("%T")} матрица {Path(a.matrix).name}, кандидаты {a.cands}, ноги {a.legs}, моделей {len(tasks)}', flush=True)
    with mp.get_context('fork').Pool(a.workers) as P:
        res = P.map(job, tasks, chunksize=1)
    print(f'{time.strftime("%T")} обучено за {time.time() - t0:.0f} с', flush=True)
    import pickle
    pickle.dump(res, open(R / f'{a.prefix}_logs' / f'raw_{"_".join(a.cands)}{sfx_t}.pkl', 'wb'))
    lock = R / 'test_locked' / 'D12_h1'  # тест: test_locked/D12_h1/<boost>/ + ссылка test_locked/<boost> (для gibrid_io.load_boost)
    rows = []
    for c in a.cands:
        boost = (f'{a.prefix}_{c}' if a.prefix else c) + sfx_t
        (lock / boost).mkdir(parents=True, exist_ok=True)
        if not (lock.parent / boost).exists():
            (lock.parent / boost).symlink_to(Path('D12_h1') / boost)
        for m in ('xgb', 'lgbm'):
            rr = [r for r in res if r[0] == c and r[1] == m]
            sel = lambda f: (pd.concat([x[7] for x in rr if f(x)], ignore_index=True), pd.concat([x[8] for x in rr if f(x)], ignore_index=True))
            single = sel(lambda x: x[5] == 1 and x[4] == 42)
            files = {'': single}
            if 'top5' in a.legs:
                files['_top5'] = sel(lambda x: x[2] <= 6 and x[4] == 42)
            if 'seeds3' in a.legs:
                files['_seeds4344'] = sel(lambda x: x[4] in (43, 44))
            for sfx, (v, t) in files.items():
                v.to_parquet(R / f'{boost}_{m}_h1{sfx}_val.parquet', index=False)
                t.to_parquet(lock / boost / f'{boost}_{m}_h1{sfx}_test.parquet', index=False)
            avg = lambda d: d.groupby(['date', 'ticker', 'step'], as_index=False).agg(rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))
            legs = {'single': single[0]}
            if 'top5' in a.legs:
                t5 = avg(files['_top5'][0]); legs['top5'] = pd.concat([t5, single[0][single[0].step > 6][t5.columns]])
            if 'seeds3' in a.legs:
                legs['seeds3'] = avg(pd.concat([single[0], files['_seeds4344'][0]]))
            for lg, V in legs.items():
                e = evaluate_dev(V)
                rows.append(dict(time=time.strftime('%F %T'), matrix=Path(a.matrix).name, cand=c + sfx_t, cfg=json.dumps(CANDS[c]), model=m, leg=lg,
                                 **{k: e[k] for k in ('raw',) + L.CODES}, sm_steps=' / '.join(f'{x:.4f}' for x in e['sm_steps']),
                                 bi=' '.join(str(x[6]) for x in sorted(rr, key=lambda x: x[2]) if x[5] == 1 and x[4] == 42)))
    df = pd.DataFrame(rows)
    out = R / f'{a.prefix}_metrics_dev.csv'
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False, float_format='%.6g')
    pd.set_option('display.width', 250)
    print(pd.DataFrame(rows)[['cand', 'model', 'leg', 'ord', 'sm', 'smrob', 'sm_steps']].to_string(), flush=True)
    print(f'{time.strftime("%T")} готово за {time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
