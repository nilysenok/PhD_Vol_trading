#!/usr/bin/env python3
"""Этап 5, B на features_pit_v2: последовательное отсечение точек ЛГК-100 (протокол владельца 25.10, 18:15–18:17).
Шаги 1–6: подбор на 50% строк окна (страт. тикер x год, seed 42), 100 -> 50 (шаг 2) -> 25 (шаг 3) -> 15 (шаги 4–6) по val-QLIKE (отдельно модель x h).
Шаги 7–9: точка, выбранная на шаге 6, дообучается на окне без перебора. После шага 1 признаки с нулевым суммарным gain по 100 точкам исключаются (модель x h).
Финал каждого шага: обучение на ВСЕХ строках окна (основные прогнозы), топ-5 точек (rank), seed 43/44 выбранной. ES: QLIKE на val-годе, patience 30, max_bin 63/64.
Запуск из корня: python3 moex_volatility/scripts/tune_Bv2.py [--workers 12] [--hs 22,5,1]   Выход: data/regen/Bv2_*, test-прогнозы — data/regen/test_locked/."""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[k] = '1'
import argparse, ast, json, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
warnings.filterwarnings('ignore')
_pre = argparse.ArgumentParser(add_help=False); _pre.add_argument('--tag', default='Bv2'); _pre.add_argument('--max-bin', type=int, default=63)
PRE, _ = _pre.parse_known_args()  # читается и в дочерних процессах (spawn); --max-bin 255 -> LightGBM и XGBoost 255, иначе 63/64
MBL = PRE.max_bin; MBX = PRE.max_bin + 1 if PRE.max_bin == 63 else PRE.max_bin
ROOT = Path(__file__).resolve().parents[2]; MV = ROOT / 'moex_volatility'; OUT = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen'); TAG = PRE.tag; LOCK = OUT / 'test_locked' / (TAG if TAG != 'Bv2' else '')
FEAT = OUT / 'features_pit_v2.parquet'; EXC = OUT / f'{TAG}_excluded_features.csv'
ns = {'np': np}
for n in ast.parse((MV / 'scripts/walk_forward_full.py').read_text()).body:
    if isinstance(n, ast.FunctionDef) and n.name in ('qlike', 'clean_features'): exec(compile(ast.Module([n], []), 'wff', 'exec'), ns)
qlike, clean_features = ns['qlike'], ns['clean_features']
N_PTS, MODELS, KEEP_N, PAT, NMAX, PRUNE_AT, PRUNE_X = 100, ('lgbm', 'xgb'), {1: 50, 2: 25, 3: 15}, 30, 1000, 100, 1.2  # lgbm первым (длинные первыми)
SPACE = {'xgb': [('max_depth', 3, 12, 'int'), ('learning_rate', .03, .2, 'log'), ('subsample', .5, 1, 'lin'), ('colsample_bytree', .4, 1, 'lin'),
                 ('min_child_weight', 1, 100, 'logint'), ('reg_alpha', 1e-3, 10, 'log'), ('reg_lambda', 1e-3, 10, 'log')],
         'lgbm': [('num_leaves', 15, 255, 'int'), ('learning_rate', .03, .2, 'log'), ('bagging_fraction', .5, 1, 'lin'), ('feature_fraction', .4, 1, 'lin'),
                  ('min_data_in_leaf', 5, 200, 'logint'), ('reg_alpha', 1e-3, 10, 'log'), ('reg_lambda', 1e-3, 10, 'log')]}


def lhs(n):  # ЛГК 7 измерений, seed 42
    try:
        from scipy.stats import qmc
        return qmc.LatinHypercube(d=7, seed=42).random(n)
    except ImportError:
        rng = np.random.default_rng(42); return np.column_stack([(rng.permutation(n) + rng.random(n)) / n for _ in range(7)])


def grid(m, n, off=0):
    rows = []
    for i, u in enumerate(lhs(n)):
        r = {'model': m, 'point': off + i}
        for j, (k, lo, hi, sc) in enumerate(SPACE[m]):
            v = lo + u[j] * (hi - lo) if sc in ('lin', 'int') else float(np.exp(np.log(lo) + u[j] * (np.log(hi) - np.log(lo))))
            r[k] = int(round(v)) if sc in ('int', 'logint') else v
        rows.append(r)
    return pd.DataFrame(rows)


def make_grid():  # lgbm: ЛГК-100 (id 0-99); xgb: ЛГК-40 (id 0-39) и, если бюджет расширен, ЛГК-100 (id 100-199)
    return pd.concat([grid('lgbm', 100), grid('xgb', 40), grid('xgb', 100, 100)]).set_index(['model', 'point'])


os.chdir(MV)
FC = json.load(open(FEAT.with_suffix('.config.json')))['feature_cols']
ALL = pd.read_parquet(FEAT).sort_values('date').reset_index(drop=True); ALL['year'] = pd.to_datetime(ALL['date']).dt.year
assert not [f for f in FC if 'target' in f] and all(f in ALL.columns for f in FC)
TEST_YEARS = [y for y in range(2017, 2026) if y in ALL.year.values]; GRID = make_grid()
tcol = lambda h: f'rv_target_h{h}' if h == 1 else f'rv_target_mean_h{h}'  # ln среднего RV за [t+1, t+h]
_c = {}


def SD(model, h, step):  # данные шага (кэш на воркере); признаки без исключённых для шагов >= 2
    k = (model, h, step)
    if k not in _c:
        _c.clear(); ty = TEST_YEARS[step - 1]; t = tcol(h); tr = ALL[ALL.year < ty]; vy = sorted(tr.year.unique())[-1]
        tp, vi, td = tr[tr.year != vy].dropna(subset=[t]), tr[tr.year == vy].dropna(subset=[t]), ALL[ALL.year == ty].dropna(subset=[t])
        keep = list(range(len(FC)))
        if step > 1 and EXC.exists():
            ex = pd.read_csv(EXC); ex = set(ex[(ex.model == model) & (ex.h == h)].feature); keep = [i for i, f in enumerate(FC) if f not in ex]
        th = tp.groupby(['ticker', 'year'], group_keys=False).sample(frac=.5, random_state=42)
        X = lambda d: clean_features(d[FC].values).astype(np.float32)[:, keep]
        _c[k] = dict(vi=vi, td=td, vy=vy, ty=ty, t=t, keep=keep, Xf=X(tp), Xh=X(th), Xv=X(vi), Xt=X(td), yf=np.log(tp[t].values + 1e-10),
                     yh=np.log(th[t].values + 1e-10), yv=np.log(vi[t].values + 1e-10), ds={})
    return _c[k]


def qmetric(y, p): d = y - p; return float(np.mean(np.exp(d) - d - 1))  # QLIKE = RV/F - ln(RV/F) - 1, F = exp(p), y = ln RV


class XPrune(xgb.callback.TrainingCallback):
    def __init__(self, ref, Xv, yv): self.ref, self.Xv, self.yv, self.pruned = ref, Xv, yv, False

    def after_iteration(self, model, epoch, evals_log):  # QLIKE на итерации 100: predict по 100 деревьям; ES — по нативному rmse (l2)
        if self.ref is not None and epoch == PRUNE_AT - 1 and qmetric(self.yv, model.inplace_predict(self.Xv)) > PRUNE_X * self.ref: self.pruned = True
        return self.pruned


_ref = {}


def refds(model, h, keep):  # бины строятся на шаге 1 (все строки обучающего окна шага 1) и передаются через reference на шаги 2–9 (по паре модель x h)
    if (model, h) not in _ref:
        t = tcol(h); tr = ALL[ALL.year < TEST_YEARS[0]]; vy = sorted(tr.year.unique())[-1]; tp = tr[tr.year != vy].dropna(subset=[t])
        X, y = clean_features(tp[FC].values).astype(np.float32)[:, keep], np.log(tp[t].values + 1e-10)
        _ref[(model, h)] = xgb.QuantileDMatrix(X, y, max_bin=MBX) if model == 'xgb' else lgb.Dataset(X, y, params={'max_bin': MBL, 'feature_pre_filter': False, 'verbosity': -1}, free_raw_data=False)
    return _ref[(model, h)]


def train(model, h, step, pt, seed, full, ref=None, nmax=NMAX):  # -> (pv, pt, best_iteration, gain, pruned); ref — лучший val QLIKE пачки для отсечения
    s = SD(model, h, step); p = GRID.loc[(model, pt)].to_dict(); X, y = (s['Xf'], s['yf']) if full else (s['Xh'], s['yh'])
    if model == 'xgb':
        if full not in s['ds']:
            rd = refds(model, h, s['keep']) if step > 1 else None; d0 = xgb.QuantileDMatrix(X, y, max_bin=MBX, ref=rd); s['ds'][full] = (d0, xgb.QuantileDMatrix(s['Xv'], s['yv'], ref=d0, max_bin=MBX))
        dtr, dva = s['ds'][full]; xp = XPrune(ref, s['Xv'], s['yv'])
        prm = dict(tree_method='hist', max_bin=MBX, max_depth=int(p['max_depth']), eta=p['learning_rate'], subsample=p['subsample'], colsample_bytree=p['colsample_bytree'],
                   min_child_weight=p['min_child_weight'], alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'], nthread=1, seed=seed, verbosity=0, eval_metric='rmse')
        b = xgb.train(prm, dtr, nmax, evals=[(dva, 'val')], early_stopping_rounds=PAT, verbose_eval=False, callbacks=[xp])
        bi = PRUNE_AT - 1 if xp.pruned else int(b.best_iteration); ir = (0, bi + 1); g = b[:bi + 1].get_score(importance_type='total_gain')
        gain = np.array([g.get(f'f{i}', 0.0) for i in range(X.shape[1])])
        return b.inplace_predict(s['Xv'], iteration_range=ir), b.inplace_predict(s['Xt'], iteration_range=ir), bi, gain, xp.pruned
    if full not in s['ds']:
        d0 = lgb.Dataset(X, y, reference=refds(model, h, s['keep']) if step > 1 else None, params={'max_bin': MBL, 'feature_pre_filter': False, 'verbosity': -1}, free_raw_data=False)
        s['ds'][full] = (d0, lgb.Dataset(s['Xv'], s['yv'], reference=d0, free_raw_data=False))
    dtr, dva = s['ds'][full]
    prm = dict(objective='regression', boosting_type='gbdt', max_depth=-1, num_leaves=int(p['num_leaves']), learning_rate=p['learning_rate'], bagging_fraction=p['bagging_fraction'],
               bagging_freq=1, feature_fraction=p['feature_fraction'], min_data_in_leaf=int(p['min_data_in_leaf']), reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
               max_bin=MBL, verbosity=-1, seed=seed, num_threads=1, metric='l2', force_col_wise=True)
    pr = []

    def prune(env):
        if ref is not None and env.iteration == PRUNE_AT - 1 and qmetric(s['yv'], env.model.predict(s['Xv'])) > PRUNE_X * ref:
            pr.append(1); raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)

    b = lgb.train(prm, dtr, nmax, valid_sets=[dva], callbacks=[prune, lgb.early_stopping(PAT, verbose=False)])
    return b.predict(s['Xv']), b.predict(s['Xt']), int(b.best_iteration), b.feature_importance('gain'), bool(pr)


def trials(a):  # пачка точек одного (модель, h, шаг): один Dataset/DMatrix на пачку; отсечение относительно лучшего посчитанного в пачке
    m, h, step, pts, nmax = a; out = []; best = None
    for pt in pts:
        t0 = time.time(); pv, pte, bi, gain, pruned = train(m, h, step, pt, 42, False, best, nmax); s = SD(m, h, step)
        q = qlike(s['vi'][s['t']].values, np.clip(np.exp(pv), 1e-10, None)); best = q if best is None else min(best, q)
        out.append((dict(model=m, h=h, step=step, point=pt, qlike_val=q, best_iteration=bi, pruned=pruned, sec=time.time() - t0), pv, pte, gain, s['keep']))
    return out


def final(a):  # финал: все строки окна
    m, h, step, pt, seed, rank = a; pv, pte, bi, _, _ = train(m, h, step, pt, seed, True); s = SD(m, h, step)
    return dict(model=m, h=h, step=step, point=pt, seed=seed, rank=rank, best_iteration=bi, n_features=len(s['keep']),
                qlike_val=qlike(s['vi'][s['t']].values, np.clip(np.exp(pv), 1e-10, None))), pv, pte


def frame(s, k, step, pred, **kw):
    df, yr = (s['vi'], s['vy']) if k == 'val' else (s['td'], s['ty'])
    return pd.DataFrame({'date': df['date'].values, 'ticker': df['ticker'].values, 'year': yr, 'step': step, **kw, 'rv_actual': df[s['t']].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)})


def batches(surv, h, step, W):  # по W пачек на модель (lgbm первым)
    return [(m, h, step, pts.tolist(), 300 if step in (2, 3) else NMAX) for m in surv for pts in np.array_split(np.array(surv[m]), min(W, len(surv[m])))]


def run_h(h, P, cnt, t0):
    surv = {'lgbm': list(range(100)), 'xgb': list(range(40))}; exp = False; sel, trl, allp, fin = {}, [], {}, []  # fin: (row, frames val, frames test)
    def log100(n):
        if cnt[0] < 100 <= cnt[0] + n: print(f'ЗАМЕР: первые 100 задач за {time.time()-t0:.0f}s -> оценка всего B ~ {(time.time()-t0)/100*(924+2*306)/60:.0f} мин (924 подбора + 306 финалов x2)', flush=True)
        cnt[0] += n
    for step in range(1, 7):
        res = [x for r in P.map(trials, batches(surv, h, step, P._processes), chunksize=1) for x in r]; log100(len(res)); s = SD('xgb', h, step)
        if step == 1:  # бюджет XGBoost: лучшая точка XGB-40 в лучших 20% объединённого рейтинга (LGBM-100 + XGB-40) по val QLIKE шага 1 -> XGB получает ЛГК-100
            comb = pd.DataFrame([r for r, *_ in res]).sort_values('qlike_val').reset_index(drop=True); rk = int(comb.index[comb.model == 'xgb'][0]); exp = rk < 0.2 * len(comb)
            pd.DataFrame([dict(h=h, best_xgb_rank=rk + 1, n_combined=len(comb), expanded=exp)]).to_csv(OUT / f'{TAG}_xgb_budget.csv', mode='a', header=not (OUT / f'{TAG}_xgb_budget.csv').exists(), index=False)
            if exp: res += [x for r in P.map(trials, batches({'xgb': list(range(100, 200))}, h, step, P._processes), chunksize=1) for x in r]; log100(100)
        for r, pv, pte, gain, keep in res:
            trl.append(r); allp.setdefault((r['model'], 'val'), []).append(frame(s, 'val', step, pv, point=r['point'])); allp.setdefault((r['model'], 'test'), []).append(frame(s, 'test', step, pte, point=r['point']))
        tr = pd.DataFrame([r for r, *_ in res])
        if step == 1:  # признаки с нулевым суммарным gain по 100 точкам
            ex = [dict(model=m, h=h, feature=FC[i]) for m in MODELS for i in np.where(sum(g for (r, _, _, g, _) in res if r['model'] == m) <= 0)[0]]  # по всем точкам шага 1 пары
            pd.DataFrame(ex, columns=['model', 'h', 'feature']).to_csv(EXC, mode='a', header=not EXC.exists(), index=False)
        tasks = []
        for m in MODELS:
            top = tr[tr.model == m].sort_values('qlike_val').point.astype(int).tolist(); sel[m] = top[0]
            tasks += [(m, h, step, pt, 42, r + 1) for r, pt in enumerate(top[:5])] + [(m, h, step, top[0], sd, 1) for sd in (43, 44)]
            kn = ({1: 20, 2: 10, 3: 6} if (m == 'xgb' and not exp) else KEEP_N).get(step); surv[m] = top[:kn] if kn else surv[m]
        for r, pv, pte in P.map(final, tasks, chunksize=1): log100(1); fin.append((r, frame(s, 'val', step, pv, point=r['point'], seed=r['seed'], rank=r['rank']), frame(s, 'test', step, pte, point=r['point'], seed=r['seed'], rank=r['rank'])))
    tasks = [(m, h, step, sel[m], sd, 1) for step in (7, 8, 9) for m in MODELS for sd in (42, 43, 44)]
    for r, pv, pte in P.map(final, tasks, chunksize=1):
        s = SD('xgb', h, r['step']); fin.append((r, frame(s, 'val', r['step'], pv, point=r['point'], seed=r['seed'], rank=1), frame(s, 'test', r['step'], pte, point=r['point'], seed=r['seed'], rank=1)))
    LOCK.mkdir(parents=True, exist_ok=True); pd.DataFrame(trl).to_csv(OUT / f'{TAG}_trials.csv', mode='a', header=not (OUT / f'{TAG}_trials.csv').exists(), index=False)
    sels = pd.DataFrame([r for r, *_ in fin if r['seed'] == 42 and r['rank'] == 1]); sels = sels.join(GRID.reset_index().set_index(['model', 'point']), on=['model', 'point'])
    sels.to_csv(OUT / f'{TAG}_selected_h{h}.csv', index=False)
    for m in MODELS:
        pick = lambda f, ix: pd.concat([x[ix] for x in fin if f(x[0]) and x[0]['model'] == m])
        for k, ix, d in (('val', 1, OUT), ('test', 2, LOCK)):
            pd.concat(allp[(m, k)]).to_parquet(d / f'{TAG}_allpts_{m}_h{h}_{k}.parquet', index=False)
            pick(lambda r: r['seed'] == 42 and r['rank'] == 1, ix).to_parquet(d / f'{TAG}_{m}_h{h}_{k}.parquet', index=False)
            pick(lambda r: r['seed'] == 42 and r['step'] <= 6, ix).to_parquet(d / f'{TAG}_{m}_h{h}_top5_{k}.parquet', index=False)
            pick(lambda r: r['seed'] != 42, ix).to_parquet(d / f'{TAG}_{m}_h{h}_seeds4344_{k}.parquet', index=False)
    print(f'h={h} готов (v2) {time.strftime("%F %T")}, {time.time()-t0:.0f}s от старта', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=10); ap.add_argument('--hs', default='22,5,1'); ap.add_argument('--bench', action='store_true'); ap.add_argument('--tag'); ap.add_argument('--max-bin')
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); G = GRID.reset_index(); G[G.model == 'lgbm'].to_csv(OUT / f'{TAG}_grid_lgbm100.csv', index=False); G[(G.model == 'xgb') & (G.point < 100)].to_csv(OUT / f'{TAG}_grid_xgb40.csv', index=False); G[(G.model == 'xgb') & (G.point >= 100)].to_csv(OUT / f'{TAG}_grid_xgb100.csv', index=False); t0 = time.time(); cnt = [0]
    if a.bench:  # 10 задач шага 9 (полные строки) — только проверка работоспособности и времени
        with Pool(a.workers) as P: P.map(trials, [(m, 22, 9, [0], NMAX) for m in MODELS for _ in range(a.workers)], chunksize=1); t0 = time.time(); P.map(trials, batches({m: list(range(48)) for m in MODELS}, 22, 9, a.workers), chunksize=1); print(f'workers={a.workers}: 96 задач шага 9 за {time.time()-t0:.0f}s', flush=True); return
    with Pool(a.workers) as P:
        for h in [int(x) for x in a.hs.split(',')]: run_h(h, P, cnt, t0)


if __name__ == '__main__':
    main()
