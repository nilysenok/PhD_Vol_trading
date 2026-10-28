#!/usr/bin/env python3
"""Уровень 2, Д9.3 (+ Д9.4): переподбор в отдельном пространстве для h >= 5 — optuna TPE 30 испытаний (seed 42), свой study на (модель, шаг),
шаги 1-6 (разработочный набор), критерий — val QLIKE шага (как выбор точки B по шагу); снятие на итерации 100 при отставании > 20% от
лучшего завершённого испытания study (как models/tune.py). Обучение испытания — на всех строках окна, ES — нативная метрика, patience 30.
Пространство (владелец 25.10, 19:10): XGB max_depth 2-3, min_child_weight 200-1000 (лог), reg_lambda 10-100 (лог), colsample_bytree 0,3-0,5,
learning_rate 0,01-0,05 (лог); LGBM num_leaves 4-8, min_data_in_leaf 200-1000 (лог), reg_lambda 10-100 (лог), feature_fraction 0,3-0,5,
learning_rate 0,01-0,05 (лог); деревьев до 800; не названные параметры — диапазоны B (subsample/bagging 0,5-1, reg_alpha 1e-3-10 лог).
Д9.4: --extra-trees — extra_trees как бинарный параметр поиска (только LightGBM; у XGBoost такого параметра нет).
python3 scripts/analysis/l2_tune.py --h 22 --name d93 [--extra-trees] [--cfg '{}'] [--workers 10]
Выход: data/regen/L2/h{h}/{name}_trials.csv, {name}_selected.csv, {name}_val.parquet; строка в levers_h{h}.csv."""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L

SPACE93 = {'xgb': [('max_depth', 2, 3, 'int'), ('learning_rate', .01, .05, 'log'), ('subsample', .5, 1, 'lin'), ('colsample_bytree', .3, .5, 'lin'),
                   ('min_child_weight', 200, 1000, 'log'), ('reg_alpha', 1e-3, 10, 'log'), ('reg_lambda', 10, 100, 'log')],
           'lgbm': [('num_leaves', 4, 8, 'int'), ('learning_rate', .01, .05, 'log'), ('bagging_fraction', .5, 1, 'lin'), ('feature_fraction', .3, .5, 'lin'),
                    ('min_data_in_leaf', 200, 1000, 'logint'), ('reg_alpha', 1e-3, 10, 'log'), ('reg_lambda', 10, 100, 'log')]}
G = {}


def suggest(tr, model, et):
    p = {}
    for k, lo, hi, sc in SPACE93[model]:
        p[k] = tr.suggest_int(k, lo, hi, log=sc == 'logint') if sc in ('int', 'logint') else tr.suggest_float(k, lo, hi, log=sc == 'log')
    if et and model == 'lgbm':
        p['extra_trees'] = tr.suggest_categorical('extra_trees', [False, True])
    return p


def study(a):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    model, step = a
    A, cols, cfg = G['A'], G['cols'], G['cfg']
    h, tag = G['h'], G['tag']
    ref = L.make_ref(model, A, cols, h, cfg, tag) if step > 1 else None
    D = L.step_data(A, cols, h, step, model, cfg, tag)
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    rows, best, bestpv = [], None, None
    for _ in range(G['n']):
        tr = st.ask()
        p = suggest(tr, model, G['et'])
        t0 = time.time()
        pv, _, bi, _, pruned = L.fit(model, p, D, cfg, ref=ref, prune=(100, 1.2 * best) if best is not None else None)
        q = L.qlike(D['y']['va'], np.exp(pv))
        if pruned:
            tr.report(q, 100); st.tell(tr, state=optuna.trial.TrialState.PRUNED)
        else:
            st.tell(tr, q)
            if best is None or q < best:
                best, bestpv = q, pv
        rows.append(dict(model=model, h=h, step=step, trial=tr.number, **p, best_iteration=bi, qlike_val=q, pruned=pruned, sec=time.time() - t0))
    vi = D['parts']['va']
    fr = pd.DataFrame({'date': vi.date.values, 'ticker': vi.ticker.values, 'year': D['vy'], 'step': step, 'model': model,
                       'rv_actual': D['y']['va'], 'rv_pred': np.clip(np.exp(bestpv), 1e-10, None)})
    return rows, fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--name', required=True); ap.add_argument('--tag', default='Bv2b')
    ap.add_argument('--cfg', default='{}'); ap.add_argument('--extra-trees', action='store_true'); ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--workers', type=int, default=10); ap.add_argument('--lever', default='Д9.3')
    a = ap.parse_args()
    cfg = {'nmax': 800, **json.loads(a.cfg)}
    t0 = time.time()
    A, FC = L.load_base()
    A2, cols, _, _ = L.build_features(A, FC, a.h, cfg)
    G.update(A=A2, cols=cols, cfg=cfg, h=a.h, tag=a.tag, n=a.trials, et=a.extra_trees)
    jobs = [(m, s) for m in ('lgbm', 'xgb') for s in range(6, 0, -1)]
    with mp.get_context('fork').Pool(a.workers) as P:
        res = P.map(study, jobs, chunksize=1)
    od = L.L2DIR / f'h{a.h}'; od.mkdir(parents=True, exist_ok=True)
    T = pd.DataFrame([r for rr, _ in res for r in rr])
    T.to_csv(od / f'{a.name}_trials.csv', index=False, float_format='%.17g')
    ok = T[~T.pruned]
    sel = ok.loc[ok.groupby(['model', 'step']).qlike_val.idxmin()].drop(columns=['trial', 'pruned', 'sec'])
    sel['qlike_val'] = np.where(sel.step <= 6, sel.qlike_val, np.nan)
    sel.to_csv(od / f'{a.name}_selected.csv', index=False, float_format='%.17g')
    V = pd.concat([fr for _, fr in res], ignore_index=True)
    V.to_parquet(od / f'{a.name}_val.parquet', index=False)
    S = L.summarize({m: V[V.model == m] for m in ('xgb', 'lgbm')}, gam=cfg.get('obj') == 'gamma')
    row = dict(time=time.strftime('%F %T'), h=a.h, tag=a.tag, lever=a.lever, name=a.name, cfg=json.dumps({**cfg, 'extra_trees_search': a.extra_trees}),
               n_features=len(cols), **{k: v for k, v in S.items() if k != 'per_step'},
               per_step=json.dumps({m: [round(x, 5) for x in q] for m, q in S['per_step'].items()}),
               n_trials=len(T), n_pruned=int(T.pruned.sum()))
    out = L.L2DIR / f'levers_h{a.h}.csv'
    df = pd.DataFrame([row])
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"{a.name}: метрика {S['metric']:.4f} ({S['best_model']}), лучший из 4: {S['best4']:.4f} ({S['best4_mc']}); xgb ord {S['xgb_ord']:.4f}, "
          f"lgbm ord {S['lgbm_ord']:.4f}; испытаний {len(T)}, снято {int(T.pruned.sum())}; {time.time() - t0:.0f} с", flush=True)


if __name__ == '__main__':
    main()
