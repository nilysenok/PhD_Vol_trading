#!/usr/bin/env python3
"""Уровень 2: полный переподбор итоговой комбинации — optuna TPE 30 испытаний, seed 42 (решение владельца 25.10, 19:25; процедура
поиска уровня 2 по плану; семантика как models/tune.py: свой study на (модель, шаг), критерий — val QLIKE шага, снятие на итерации 100 при
отставании > 20% от лучшего завершённого испытания study). Пространство — диапазоны B v2 (learning_rate 0,03-0,2, деревьев <= 1000), max_bin 255.
Подбор — шаги 1-6 (испытание обучается на всех строках окна, ES — нативная метрика на val-годе, patience 30); шаги 7-9 — точка шага 6
дообучается без перебора (seed 42/43/44). Топ-5 — пять лучших испытаний шага (1-6); seed 43/44 — выбранная точка.
Тестовые прогнозы — только в data/regen/test_locked/{TAG}/ (не открывать); val шагов 7-9 (подтверждающий набор) не печатается и не входит в qlike_val.
python3 scripts/analysis/tune_L2_optuna.py --h 22 --cfg '{"v9": true, "linear_lambda": 1}' [--workers 6] [--tag L2_h22]"""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L
from tune_L2 import SPACEB

G = {}


def suggest(tr, model):
    p = {}
    for k, lo, hi, sc in SPACEB[model]:
        p[k] = tr.suggest_int(k, lo, hi, log=sc == 'logint') if sc in ('int', 'logint') else tr.suggest_float(k, lo, hi, log=sc == 'log')
    return p


def frame(D, k, step, pred, **kw):
    df, yr = (D['parts']['va'], D['vy']) if k == 'val' else (D['parts']['te'], D['ty'])
    return pd.DataFrame({'date': df['date'].values, 'ticker': df['ticker'].values, 'year': yr, 'step': step, **kw,
                         'rv_actual': df[L.tcol(G['h'])].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)})


def _ref(model, step):
    return L.make_ref(model, G['A'], G['cols'], G['h'], G['cfg'], G['btag']) if step > 1 else None


def study(a):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    model, step = a
    D = L.step_data(G['A'], G['cols'], G['h'], step, model, G['cfg'], G['btag'])
    ref = _ref(model, step)
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    rows, preds, best = [], {}, None
    for _ in range(G['n']):
        tr = st.ask()
        p = suggest(tr, model)
        t0 = time.time()
        pv, pt, bi, _, pruned = L.fit(model, p, D, G['cfg'], ref=ref, prune=(100, 1.2 * best) if best is not None else None)
        q = L.qlike(D['y']['va'], np.exp(pv))
        if pruned:
            tr.report(q, 100); st.tell(tr, state=optuna.trial.TrialState.PRUNED)
        else:
            st.tell(tr, q); best = q if best is None else min(best, q)
            preds[tr.number] = (pv, pt)
        rows.append(dict(model=model, h=G['h'], step=step, trial=tr.number, **p, best_iteration=bi, qlike_val=q, pruned=pruned, sec=time.time() - t0))
    T = pd.DataFrame(rows)
    top = T[~T.pruned].sort_values('qlike_val').trial.tolist()[:5]
    fv = [frame(D, 'val', step, preds[t][0], trial=t, seed=42, rank=r + 1) for r, t in enumerate(top)]
    ft = [frame(D, 'test', step, preds[t][1], trial=t, seed=42, rank=r + 1) for r, t in enumerate(top)]
    av = pd.concat([frame(D, 'val', step, preds[t][0], trial=t) for t in preds])
    at = pd.concat([frame(D, 'test', step, preds[t][1], trial=t) for t in preds])
    return rows, fv, ft, av, at


def refit(a):
    model, step, p, seed, trial = a
    D = L.step_data(G['A'], G['cols'], G['h'], step, model, G['cfg'], G['btag'])
    pv, pt, bi, _, _ = L.fit(model, p, D, G['cfg'], seed=seed, ref=_ref(model, step))
    kw = dict(trial=trial, seed=seed, rank=1)
    return dict(model=model, step=step, seed=seed, best_iteration=bi), frame(D, 'val', step, pv, **kw), frame(D, 'test', step, pt, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--cfg', default='{}'); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--tag', default=None); ap.add_argument('--btag', default='Bv2b', help='исключения признаков шагов >= 2 (как в точке B)')
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--blocks', default='', help='d7,d10 — блоки признаков (как l2_run.py)')
    a = ap.parse_args()
    h, cfg = a.h, json.loads(a.cfg)
    TAG = a.tag or f'L2_h{h}'
    OUT, LOCK = L.REGEN, L.REGEN / 'test_locked' / TAG
    LOCK.mkdir(parents=True, exist_ok=True)
    A, FC = L.load_base(None, tuple(x for x in a.blocks.split(',') if x), h)
    A2, cols, _, _ = L.build_features(A, FC, h, cfg)
    G.update(A=A2, cols=cols, cfg=cfg, h=h, btag=a.btag, n=a.trials)
    t0 = time.time()
    print(f'{TAG}: optuna TPE {a.trials} (seed 42), cfg={cfg}, блоки={a.blocks or "-"}, признаков {len(cols)}, воркеров {a.workers}', flush=True)
    ctx = mp.get_context('fork')
    with ctx.Pool(a.workers) as P:
        res = P.map(study, [(m, s) for m in ('lgbm', 'xgb') for s in range(6, 0, -1)], chunksize=1)
    T = pd.DataFrame([r for x in res for r in x[0]])
    T.to_csv(OUT / f'{TAG}_trials.csv', index=False, float_format='%.17g')
    sel = T[~T.pruned].loc[lambda d: d.groupby(['model', 'step']).qlike_val.idxmin()]
    print(f'подбор шагов 1-6: {time.time() - t0:.0f} с, испытаний {len(T)}, снято {int(T.pruned.sum())}', flush=True)
    prm = lambda r, m: {k: getattr(r, k) for k, *_ in SPACEB[m]}
    tasks = []
    for r in sel.itertuples():
        tasks += [(r.model, r.step, prm(r, r.model), sd, r.trial) for sd in (43, 44)]
        if r.step == 6:
            tasks += [(r.model, s, prm(r, r.model), sd, r.trial) for s in (7, 8, 9) for sd in (42, 43, 44)]
    with ctx.Pool(a.workers) as P:
        rf = P.map(refit, tasks, chunksize=1)
    S6 = sel.copy()
    S6['seed'] = 42
    add = pd.DataFrame([dict(model=r['model'], h=h, step=r['step'], best_iteration=r['best_iteration'], seed=42,
                             **sel[(sel.model == r['model']) & (sel.step == 6)].iloc[0][[k for k, *_ in SPACEB[r['model']]] + ['trial']].to_dict())
                        for r, _, _ in rf if r['step'] >= 7 and r['seed'] == 42])
    selall = pd.concat([S6, add], ignore_index=True).sort_values(['model', 'step'])
    selall['qlike_val'] = np.where(selall.step <= 6, selall.qlike_val, np.nan)
    selall.drop(columns=['pruned', 'sec'], errors='ignore').to_csv(OUT / f'{TAG}_selected_h{h}.csv', index=False, float_format='%.17g')
    for m in ('lgbm', 'xgb'):
        R = [x for x, (mm, _) in zip(res, [(mm, s) for mm in ('lgbm', 'xgb') for s in range(6, 0, -1)]) if mm == m]
        for k, ix, d in (('val', 1, OUT), ('test', 2, LOCK)):
            top = pd.concat([f for x in R for f in x[ix]])
            main = pd.concat([top[top['rank'] == 1]] + [fr[ix] for fr in rf if fr[0]['model'] == m and fr[0]['seed'] == 42])
            seeds = pd.concat([fr[ix] for fr in rf if fr[0]['model'] == m and fr[0]['seed'] != 42])
            main.sort_values(['step', 'date', 'ticker']).to_parquet(d / f'{TAG}_{m}_h{h}_{k}.parquet', index=False)
            top.to_parquet(d / f'{TAG}_{m}_h{h}_top5_{k}.parquet', index=False)
            seeds.to_parquet(d / f'{TAG}_{m}_h{h}_seeds4344_{k}.parquet', index=False)
            pd.concat([x[3 if k == 'val' else 4] for x in R]).to_parquet(d / f'{TAG}_allpts_{m}_h{h}_{k}.parquet', index=False)
    V = {m: pd.read_parquet(OUT / f'{TAG}_{m}_h{h}_val.parquet') for m in ('xgb', 'lgbm')}
    S = L.summarize({m: v[v.step <= 6] for m, v in V.items()}, gam=cfg.get('obj') == 'gamma')
    print(f"{TAG} готов {time.strftime('%F %T')}, {time.time() - t0:.0f} с: метрика (шаги 2-6) {S['metric']:.4f} ({S['best_model']}), "
          f"лучший из 4 {S['best4']:.4f} ({S['best4_mc']}); xgb ord {S['xgb_ord']:.4f}, lgbm ord {S['lgbm_ord']:.4f}", flush=True)


if __name__ == '__main__':
    main()
