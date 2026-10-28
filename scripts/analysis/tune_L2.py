#!/usr/bin/env python3
"""Уровень 2: полный поиск на итоговой комбинации рычагов по протоколу B v2 (tune_Bv2.py, владелец 25.10 18:15-18:35; max_bin 255 как Bv2b).
ЛГК (seed 42): LightGBM 100 -> 50 (шаг 2) -> 25 (шаг 3) -> 15 (шаги 4-6); XGBoost 40 -> 20 -> 10 -> 6, расширение до ЛГК-100 (id 100-199), если
лучшая точка XGB-40 на шаге 1 в лучших 20% объединённого рейтинга. Подбор — шаги 1-6 на 50% строк окна (страт. тикер x год, seed 42), ранжирование
по val QLIKE шага; шаги 2-3 — потолок 300 деревьев; снятие на итерации 100 при отставании > 20% от лучшего в пачке; после шага 1 — исключение
признаков с нулевым суммарным gain (пара модель x h). Финал шага — все строки окна (топ-5 и seed 43/44 выбранной). Шаги 7-9 — точка шага 6 без
перебора. Тестовые прогнозы — только в data/regen/test_locked/{TAG}/ (не открывать); val шагов 7-9 (подтверждающий набор) не печатается.
Пространство: --space B (диапазоны B v2) | d93 (Д9.3, с --extra-trees — Д9.4). Рычаги — --cfg (как l2_run.py).
python3 scripts/analysis/tune_L2.py --h 22 --cfg '{"v9": true}' --space B [--workers 10]   -> data/regen/L2_h{h}_*"""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L
from l2_tune import SPACE93

SPACEB = {'xgb': [('max_depth', 3, 12, 'int'), ('learning_rate', .03, .2, 'log'), ('subsample', .5, 1, 'lin'), ('colsample_bytree', .4, 1, 'lin'),
                  ('min_child_weight', 1, 100, 'logint'), ('reg_alpha', 1e-3, 10, 'log'), ('reg_lambda', 1e-3, 10, 'log')],
          'lgbm': [('num_leaves', 15, 255, 'int'), ('learning_rate', .03, .2, 'log'), ('bagging_fraction', .5, 1, 'lin'), ('feature_fraction', .4, 1, 'lin'),
                   ('min_data_in_leaf', 5, 200, 'logint'), ('reg_alpha', 1e-3, 10, 'log'), ('reg_lambda', 1e-3, 10, 'log')]}
KEEP = {1: 50, 2: 25, 3: 15}
KEEPX = {1: 20, 2: 10, 3: 6}
G = {}


def lhs(n, d):
    from scipy.stats import qmc
    return qmc.LatinHypercube(d=d, seed=42).random(n)


def grid(space, m, n, off=0, et=False):
    sp = space[m] + ([('extra_trees', 0, 1, 'bin')] if (et and m == 'lgbm') else [])
    rows = []
    for i, u in enumerate(lhs(n, len(sp))):
        r = {'model': m, 'point': off + i}
        for j, (k, lo, hi, sc) in enumerate(sp):
            if sc == 'bin':
                r[k] = bool(u[j] >= .5); continue
            v = lo + u[j] * (hi - lo) if sc in ('lin', 'int') else float(np.exp(np.log(lo) + u[j] * (np.log(hi) - np.log(lo))))
            r[k] = int(round(v)) if sc in ('int', 'logint') else v
        rows.append(r)
    return pd.DataFrame(rows)


_c = {}


def data(model, step, half):
    k = (model, step, half)
    if k not in _c:
        if len(_c) > 4:
            _c.clear()
        A, cols, cfg = G['A'], G['cols'], G['cfg']
        _c[k] = L.step_data(A, cols, G['h'], step, model, cfg, G['tag'], half=half)
    return _c[k]


_r = {}


def ref(model, step):
    if step == 1:
        return None
    if model not in _r:
        _r[model] = L.make_ref(model, G['A'], G['cols'], G['h'], G['cfg'], G['tag'])
    return _r[model]


def trials(a):
    m, step, pts, nmax = a
    D = data(m, step, True); out = []; best = None
    cfg = {**G['cfg'], 'nmax': nmax}
    for pt in pts:
        t0 = time.time()
        p = G['grid'].loc[(m, pt)].to_dict()
        pv, pte, bi, gain, pruned = L.fit(m, p, D, cfg, ref=ref(m, step), prune=(100, 1.2 * best) if best is not None else None)
        q = L.qlike(D['y']['va'], np.exp(pv)); best = q if best is None else min(best, q)
        out.append((dict(model=m, h=G['h'], step=step, point=pt, qlike_val=q, best_iteration=bi, pruned=pruned, sec=time.time() - t0), pv, pte, gain, D['names']))
    return out


def final(a):
    m, step, pt, seed, rank = a
    D = data(m, step, False)
    p = G['grid'].loc[(m, pt)].to_dict()
    pv, pte, bi, _, _ = L.fit(m, p, D, G['cfg'], seed=seed, ref=ref(m, step))
    return (dict(model=m, h=G['h'], step=step, point=pt, seed=seed, rank=rank, best_iteration=bi, n_features=len(D['names']),
                 qlike_val=L.qlike(D['y']['va'], np.exp(pv)) if step <= 6 else np.nan),
            frame(D, 'val', step, pv, point=pt, seed=seed, rank=rank), frame(D, 'test', step, pte, point=pt, seed=seed, rank=rank))


def frame(D, k, step, pred, **kw):
    df, yr = (D['parts']['va'], D['vy']) if k == 'val' else (D['parts']['te'], D['ty'])
    return pd.DataFrame({'date': df['date'].values, 'ticker': df['ticker'].values, 'year': yr, 'step': step, **kw,
                         'rv_actual': df[L.tcol(G['h'])].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)})


def batches(surv, step, W, nmax):
    return [(m, step, pts.tolist(), nmax) for m in surv for pts in np.array_split(np.array(surv[m]), min(W, len(surv[m])))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--cfg', default='{}'); ap.add_argument('--space', default='B', choices=['B', 'd93'])
    ap.add_argument('--extra-trees', action='store_true'); ap.add_argument('--workers', type=int, default=10); ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    h, cfg = a.h, json.loads(a.cfg)
    TAG = a.tag or f'L2_h{h}'
    OUT, LOCK = L.REGEN, L.REGEN / 'test_locked' / TAG
    LOCK.mkdir(parents=True, exist_ok=True)
    EXC = OUT / f'{TAG}_excluded_features.csv'
    if EXC.exists():
        EXC.unlink()
    space = SPACE93 if a.space == 'd93' else SPACEB
    if a.space == 'd93':
        cfg.setdefault('nmax', 800)
    NM = int(cfg.get('nmax', L.NMAX))
    GR = pd.concat([grid(space, 'lgbm', 100, et=a.extra_trees), grid(space, 'xgb', 40), grid(space, 'xgb', 100, 100)]).set_index(['model', 'point'])
    GR.reset_index().to_csv(OUT / f'{TAG}_grid.csv', index=False)
    A, FC = L.load_base()
    A2, cols, _, _ = L.build_features(A, FC, h, cfg)
    G.update(A=A2, cols=cols, cfg=cfg, h=h, tag=TAG, grid=GR)
    t0 = time.time()
    print(f'{TAG}: cfg={cfg}, пространство {a.space}{" + extra_trees" if a.extra_trees else ""}, признаков {len(cols)}', flush=True)
    surv = {'lgbm': list(range(100)), 'xgb': list(range(40))}; exp = False; sel, trl, allp, fin = {}, [], {}, []
    ctx = mp.get_context('fork')
    for step in range(1, 7):
        with ctx.Pool(a.workers) as P:  # новый пул на шаг: в воркерах свежие кэши (после шага 1 меняется список исключений)
            res = [x for r in P.map(trials, batches(surv, step, a.workers, 300 if step in (2, 3) else NM), chunksize=1) for x in r]
            if step == 1:
                comb = pd.DataFrame([r for r, *_ in res]).sort_values('qlike_val').reset_index(drop=True)
                rk = int(comb.index[comb.model == 'xgb'][0]); exp = rk < 0.2 * len(comb)
                pd.DataFrame([dict(h=h, best_xgb_rank=rk + 1, n_combined=len(comb), expanded=exp)]).to_csv(OUT / f'{TAG}_xgb_budget.csv', index=False)
                if exp:
                    res += [x for r in P.map(trials, batches({'xgb': list(range(100, 200))}, step, a.workers, NM), chunksize=1) for x in r]
        for r, pv, pte, gain, names in res:
            trl.append(r)
            D = data(r['model'], step, True)
            allp.setdefault((r['model'], 'val'), []).append(frame(D, 'val', step, pv, point=r['point']))
            allp.setdefault((r['model'], 'test'), []).append(frame(D, 'test', step, pte, point=r['point']))
        tr = pd.DataFrame([r for r, *_ in res])
        if step == 1:
            ex = []
            for m in ('lgbm', 'xgb'):
                gs = [(g, nm) for (r, _, _, g, nm) in res if r['model'] == m]
                tot = pd.Series(0.0, index=gs[0][1])
                for g, nm in gs:
                    tot = tot.add(pd.Series(g, index=nm), fill_value=0)
                ex += [dict(model=m, h=h, feature=f) for f in tot.index[tot <= 0] if f not in ('har_ln', 'h1_ln')]
            pd.DataFrame(ex, columns=['model', 'h', 'feature']).to_csv(EXC, index=False)
            print(f'исключено после шага 1: {len(ex)} ({", ".join(sorted(set(e["feature"] for e in ex)))})', flush=True)
            _c.clear(); _r.clear()
        tasks = []
        for m in ('lgbm', 'xgb'):
            top = tr[tr.model == m].sort_values('qlike_val').point.astype(int).tolist(); sel[m] = top[0]
            tasks += [(m, step, pt, 42, r + 1) for r, pt in enumerate(top[:5])] + [(m, step, top[0], sd, 1) for sd in (43, 44)]
            kn = (KEEPX if (m == 'xgb' and not exp) else KEEP).get(step); surv[m] = top[:kn] if kn else surv[m]
        with ctx.Pool(a.workers) as P:
            fin += P.map(final, tasks, chunksize=1)
        print(f'шаг {step} готов, {time.time() - t0:.0f} с; выбраны {sel}', flush=True)
    tasks = [(m, step, sel[m], sd, 1) for step in (7, 8, 9) for m in ('lgbm', 'xgb') for sd in (42, 43, 44)]
    with ctx.Pool(a.workers) as P:
        fin += P.map(final, tasks, chunksize=1)
    pd.DataFrame(trl).to_csv(OUT / f'{TAG}_trials.csv', index=False)
    sels = pd.DataFrame([r for r, *_ in fin if r['seed'] == 42 and r['rank'] == 1]).join(GR, on=['model', 'point'])
    sels.to_csv(OUT / f'{TAG}_selected_h{h}.csv', index=False)
    for m in ('lgbm', 'xgb'):
        pick = lambda f, ix: pd.concat([x[ix] for x in fin if f(x[0]) and x[0]['model'] == m])
        for k, ix, d in (('val', 1, OUT), ('test', 2, LOCK)):
            pd.concat(allp[(m, k)]).to_parquet(d / f'{TAG}_allpts_{m}_h{h}_{k}.parquet', index=False)
            pick(lambda r: r['seed'] == 42 and r['rank'] == 1, ix).to_parquet(d / f'{TAG}_{m}_h{h}_{k}.parquet', index=False)
            pick(lambda r: r['seed'] == 42 and r['step'] <= 6, ix).to_parquet(d / f'{TAG}_{m}_h{h}_top5_{k}.parquet', index=False)
            pick(lambda r: r['seed'] != 42, ix).to_parquet(d / f'{TAG}_{m}_h{h}_seeds4344_{k}.parquet', index=False)
    V = {m: pd.read_parquet(OUT / f'{TAG}_{m}_h{h}_val.parquet') for m in ('xgb', 'lgbm')}
    S = L.summarize({m: v[v.step <= 6] for m, v in V.items()}, gam=cfg.get('obj') == 'gamma')
    print(f"{TAG} готов {time.strftime('%F %T')}, {time.time() - t0:.0f} с: метрика (шаги 2-6) {S['metric']:.4f} ({S['best_model']}), лучший из 4 {S['best4']:.4f} ({S['best4_mc']})", flush=True)


if __name__ == '__main__':
    main()
