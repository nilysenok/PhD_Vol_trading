#!/usr/bin/env python3
"""Тест 2019 (табл. 3.4–3.6) на features_pit_v3: исходное разбиение обучение <= 2017, валидация 2018, тест 2019 (решение владельца 25.10, 20:45).
Отдельное разбиение 2014–2019 (не шаг СВ): тестовые годы СВ 2017–2025 не открываются, 2019 здесь — тест этого разбиения.
Цели: h=1 rv_target_h1, h=5/22 rv_target_mean_h{h}; модели в ln RV, прогноз exp(ŷ).
  har    — HAR-J (8 предикторов, МНК в логах, как gibrid_har.har_forecasts) + Б0.1: F = exp(s²/2), s² — дисперсия остатков val 2018 (ord), ко всем компонентам.
  boost  — XGBoost / LightGBM (gbdt), протокол B v2b (tune_Bv2.py --max-bin 255) с одним шагом: ЛГК seed 42 (LGBM 100; XGB 40 + расширение до 100
           по правилу бюджета), подбор на 50% строк обучения (страт. тикер x год, seed 42), отсечение на дереве 100 (> 1,2 x лучшего в пачке),
           ранняя остановка по l2 на val 2018 (patience 30, потолок 1000); выбор точки — val-QLIKE; финал — все строки обучения, seed 42.
  garch  — GARCH(1,1) и GJR-GARCH(1,1) по тикеру на дневных лог-доходностях close (x100), параметры по <= 2017, фильтр на 2018–2019 с теми же
           параметрами; прогноз на h — среднее ожидаемых σ² за [t+1, t+h]; основная строка — σ² как есть (как в коде); справка — с множителем mean(RV)/mean(σ²) на обучении (<= 2017).
  hyb    — гибрид V1–V7 (gibrid_schemes, смешивание по RV, V7 lag=h) на val 2018: схема — минимум val-QLIKE; как в коде (без поправки) и с Б0.1.
  tables — QLIKE/RMSE/MAE (уровень RV, как final_test.py), DM (номинальный; NW лаг h−1 на пуле; основной — дневной ряд средней по тикерам, NW лаг h−1).
python3 scripts/analysis/test2019_v3.py --stage har boost garch hyb tables [--workers 4] [--features <parquet>] [--hs 1 5 22]"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[k] = '1'
import argparse, json, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent))
MAIN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
FEAT = MAIN / 'regen/features_pit_v3.parquet'
OUT = MAIN / 'regen/test2019_v3'
TR_END, VY, TY = 2017, 2018, 2019
HS = (1, 5, 22)
N_LGB, N_XGB, N_XGB2, PAT, NMAX, PRUNE_AT, PRUNE_X, MB = 100, 40, 100, 30, 1000, 100, 1.2, 255
if os.environ.get('T2019_SMOKE'): N_LGB, N_XGB, N_XGB2 = 8, 8, 8  # только дымовая проверка кода
SPACE = {'xgb': [('max_depth', 3, 12, 'int'), ('learning_rate', .03, .2, 'log'), ('subsample', .5, 1, 'lin'), ('colsample_bytree', .4, 1, 'lin'),
                 ('min_child_weight', 1, 100, 'logint'), ('reg_alpha', 1e-3, 10, 'log'), ('reg_lambda', 1e-3, 10, 'log')],
         'lgbm': [('num_leaves', 15, 255, 'int'), ('learning_rate', .03, .2, 'log'), ('bagging_fraction', .5, 1, 'lin'), ('feature_fraction', .4, 1, 'lin'),
                  ('min_data_in_leaf', 5, 200, 'logint'), ('reg_alpha', 1e-3, 10, 'log'), ('reg_lambda', 1e-3, 10, 'log')]}
tcol = lambda h: 'rv_target_h1' if h == 1 else f'rv_target_mean_h{h}'


def grid(m, n, off=0):  # как tune_Bv2.grid: ЛГК 7 измерений, seed 42
    from scipy.stats import qmc
    rows = []
    for i, u in enumerate(qmc.LatinHypercube(d=7, seed=42).random(n)):
        r = {'model': m, 'point': off + i}
        for j, (k, lo, hi, sc) in enumerate(SPACE[m]):
            v = lo + u[j] * (hi - lo) if sc in ('lin', 'int') else float(np.exp(np.log(lo) + u[j] * (np.log(hi) - np.log(lo))))
            r[k] = int(round(v)) if sc in ('int', 'logint') else v
        rows.append(r)
    return pd.DataFrame(rows)


GRID = pd.concat([grid('lgbm', N_LGB), grid('xgb', N_XGB), grid('xgb', N_XGB2, 100)]).set_index(['model', 'point'])


def qlike(y, p): p = np.clip(p, 1e-10, None); return float(np.mean(y / p - np.log(y / p) - 1))
def qm(y, lp): d = y - lp; return float(np.mean(np.exp(d) - d - 1))  # QLIKE по ln RV
def clean(X): return np.nan_to_num(X, nan=0, posinf=0, neginf=0)


def load(path):
    a = pd.read_parquet(path).sort_values(['date', 'ticker']).reset_index(drop=True)
    a['date'] = pd.to_datetime(a['date']); a['year'] = a.date.dt.year
    a = a[a.year <= TY].reset_index(drop=True)
    cfg = Path(path).with_suffix('.config.json')
    fc = json.load(open(cfg))['feature_cols'] if cfg.exists() else json.load(open(MAIN / 'regen/features_pit_v2.config.json'))['feature_cols']
    fc = [f for f in fc if f in a.columns and 'target' not in f]
    return a, fc


def split(a, h):
    t = tcol(h)
    return (a[a.year <= TR_END].dropna(subset=[t]), a[a.year == VY].dropna(subset=[t]), a[a.year == TY].dropna(subset=[t]))


def frame(df, part, pred, t):
    return pd.DataFrame({'date': df.date.values, 'ticker': df.ticker.values, 'year': df.year.values, 'part': part,
                         'rv_actual': df[t].values, 'rv_pred': np.clip(pred, 1e-10, None)})


# ---------------- HAR-J ----------------
def stage_har(a, hs):
    from sklearn.linear_model import LinearRegression
    jump = [c for c in a.columns if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower()]
    harj = [f for f in ['rv_d', 'rv_w', 'rv_m'] + [c for c in jump if c not in ('rv_d', 'rv_w', 'rv_m')] if f in a.columns]
    info = []
    for h in hs:
        t = tcol(h); tp, vi, td = split(a, h)
        X = lambda d: clean(np.log(d[harj].values.clip(1e-10)))
        m = LinearRegression().fit(X(tp), np.log(tp[t].values + 1e-10))
        f = pd.concat([frame(d, p, np.exp(m.predict(X(d))), t) for d, p in ((vi, 'val'), (td, 'test'))])
        f.to_parquet(OUT / f'har_h{h}.parquet', index=False)
        info.append(dict(h=h, n_pred=len(harj), preds=' '.join(harj), coef=' '.join(f'{c:.4f}' for c in m.coef_), intercept=m.intercept_))
    pd.DataFrame(info).to_csv(OUT / 'har_info.csv', index=False); print('HAR-J готов', harj, flush=True)


# ---------------- бустинг ----------------
_C = {}


def SD(h):
    if h not in _C:
        _C.clear(); a, fc = load(ARGS_FEAT); t = tcol(h); tp, vi, td = split(a, h)
        th = tp.groupby(['ticker', 'year'], group_keys=False).sample(frac=.5, random_state=42)
        X = lambda d: clean(d[fc].values).astype(np.float32)
        _C[h] = dict(t=t, vi=vi, td=td, fc=fc, Xf=X(tp), Xh=X(th), Xv=X(vi), Xt=X(td), yf=np.log(tp[t].values + 1e-10),
                     yh=np.log(th[t].values + 1e-10), yv=np.log(vi[t].values + 1e-10), ds={})
    return _C[h]


def train(m, h, pt, seed, full, ref=None):
    import lightgbm as lgb, xgboost as xgb
    s = SD(h); p = GRID.loc[(m, pt)].to_dict(); X, y = (s['Xf'], s['yf']) if full else (s['Xh'], s['yh'])
    if m == 'xgb':
        if (m, full) not in s['ds']:
            d0 = xgb.QuantileDMatrix(X, y, max_bin=MB); s['ds'][(m, full)] = (d0, xgb.QuantileDMatrix(s['Xv'], s['yv'], ref=d0, max_bin=MB))
        dtr, dva = s['ds'][(m, full)]; pr = [False]

        class XP(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, ev):
                if ref is not None and epoch == PRUNE_AT - 1 and qm(s['yv'], model.inplace_predict(s['Xv'])) > PRUNE_X * ref: pr[0] = True
                return pr[0]
        prm = dict(tree_method='hist', max_bin=MB, max_depth=int(p['max_depth']), eta=p['learning_rate'], subsample=p['subsample'], colsample_bytree=p['colsample_bytree'],
                   min_child_weight=p['min_child_weight'], alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'], nthread=1, seed=seed, verbosity=0, eval_metric='rmse')
        b = xgb.train(prm, dtr, NMAX, evals=[(dva, 'val')], early_stopping_rounds=PAT, verbose_eval=False, callbacks=[XP()])
        bi = PRUNE_AT - 1 if pr[0] else int(b.best_iteration); ir = (0, bi + 1)
        return b.inplace_predict(s['Xv'], iteration_range=ir), b.inplace_predict(s['Xt'], iteration_range=ir), bi, pr[0]
    if (m, full) not in s['ds']:
        d0 = lgb.Dataset(X, y, params={'max_bin': MB, 'feature_pre_filter': False, 'verbosity': -1}, free_raw_data=False)
        s['ds'][(m, full)] = (d0, lgb.Dataset(s['Xv'], s['yv'], reference=d0, free_raw_data=False))
    dtr, dva = s['ds'][(m, full)]
    prm = dict(objective='regression', boosting_type='gbdt', max_depth=-1, num_leaves=int(p['num_leaves']), learning_rate=p['learning_rate'], bagging_fraction=p['bagging_fraction'],
               bagging_freq=1, feature_fraction=p['feature_fraction'], min_data_in_leaf=int(p['min_data_in_leaf']), reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
               max_bin=MB, verbosity=-1, seed=seed, num_threads=1, metric='l2', force_col_wise=True)
    pr = []

    def prune(env):
        if ref is not None and env.iteration == PRUNE_AT - 1 and qm(s['yv'], env.model.predict(s['Xv'])) > PRUNE_X * ref:
            pr.append(1); raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    b = lgb.train(prm, dtr, NMAX, valid_sets=[dva], callbacks=[prune, lgb.early_stopping(PAT, verbose=False)])
    return b.predict(s['Xv']), b.predict(s['Xt']), int(b.best_iteration), bool(pr)


def trials(arg):
    m, h, pts = arg; out, best = [], None
    for pt in pts:
        t0 = time.time(); pv, _, bi, pr = train(m, h, pt, 42, False, best); s = SD(h)
        q = qlike(s['vi'][s['t']].values, np.exp(pv)); best = q if best is None else min(best, q)
        out.append(dict(model=m, h=h, point=pt, qlike_val=q, best_iteration=bi, pruned=pr, sec=time.time() - t0))
    return out


def final(arg):
    m, h, pt = arg; pv, pte, bi, _ = train(m, h, pt, 42, True); s = SD(h)
    return m, h, pt, bi, pv, pte


def stage_boost(hs, W):
    tr_all, sel = [], []
    with Pool(W) as P:
        for h in hs:
            t0 = time.time()
            bt = lambda m, pts: [(m, h, list(x)) for x in np.array_split(np.array(pts), W)]
            res = [r for x in P.map(trials, bt('lgbm', range(N_LGB)) + bt('xgb', range(N_XGB)), chunksize=1) for r in x]
            comb = pd.DataFrame(res).sort_values('qlike_val').reset_index(drop=True); rk = int(comb.index[comb.model == 'xgb'][0]); exp = rk < 0.2 * len(comb)
            if exp: res += [r for x in P.map(trials, bt('xgb', range(100, 100 + N_XGB2)), chunksize=1) for r in x]
            tr = pd.DataFrame(res).assign(xgb_budget_rank=rk + 1, xgb_expanded=exp); tr_all.append(tr)
            top = {m: int(tr[tr.model == m].sort_values('qlike_val').point.iloc[0]) for m in ('lgbm', 'xgb')}
            print(f'h={h}: подбор {len(tr)} точек за {time.time() - t0:.0f} с, XGB-40 место {rk + 1}/140 -> расширение {exp}, точки {top}', flush=True)
            for m, h_, pt, bi, pv, pte in P.map(final, [(m, h, top[m]) for m in ('lgbm', 'xgb')]):
                s_vi, s_td, t = SD_main(h)
                f = pd.concat([frame(s_vi, 'val', np.exp(pv), t), frame(s_td, 'test', np.exp(pte), t)])
                f.to_parquet(OUT / f'{m}_h{h}.parquet', index=False)
                sel.append(dict(model=m, h=h, point=pt, best_iteration=bi, qlike_val_full=qlike(f[f.part == 'val'].rv_actual.values, f[f.part == 'val'].rv_pred.values),
                                **GRID.loc[(m, pt)].dropna().to_dict()))
            print(f'h={h}: финалы готовы, {time.time() - t0:.0f} с', flush=True)
    pd.concat(tr_all).to_csv(OUT / 'boost_trials.csv', index=False); pd.DataFrame(sel).to_csv(OUT / 'boost_selected.csv', index=False)


def SD_main(h):
    a, _ = load(ARGS_FEAT); _, vi, td = split(a, h); return vi, td, tcol(h)


# ---------------- GARCH ----------------
def stage_garch(a, hs):
    from arch import arch_model
    rows, prm = {h: [] for h in hs}, []
    for tk, g in a.groupby('ticker'):
        g = g.sort_values('date').reset_index(drop=True); r = 100 * np.log(g.close).diff()
        ok = r.notna().values; gg, rr = g[ok].reset_index(drop=True), r[ok].values
        ntr = int((gg.year <= TR_END).sum())
        for name, o in (('garch', 0), ('gjr', 1)):
            res = arch_model(rr[:ntr], mean='Constant', vol='GARCH', p=1, o=o, q=1, dist='normal').fit(disp='off', show_warning=False)
            pa = res.params; om, al, be = pa['omega'], pa['alpha[1]'], pa['beta[1]']; ga = pa.get('gamma[1]', 0.0)
            fx = arch_model(rr, mean='Constant', vol='GARCH', p=1, o=o, q=1, dist='normal').fix(pa)
            e = rr - pa['mu']; s2 = fx.conditional_volatility ** 2  # s2[t] — σ²_t по информации до t−1
            nxt = om + (al + ga * (e < 0)) * e ** 2 + be * s2  # σ²_{t+1} по информации до t
            phi = al + ga / 2 + be; lr = om / (1 - phi) if phi < 1 else np.mean(rr[:ntr] ** 2)
            prm.append(dict(ticker=tk, model=name, omega=om, alpha=al, gamma=ga, beta=be, persistence=phi))
            for h in hs:
                ks = np.arange(h); f = (lr + np.outer(nxt - lr, phi ** ks)).mean(axis=1) / 1e4
                t = tcol(h); d = gg.assign(fc=f).dropna(subset=[t])
                trm = d.year <= TR_END; c = d[trm][t].mean() / d[trm].fc.mean()
                for part, yr in (('val', VY), ('test', TY)):
                    dd = d[d.year == yr]
                    rows[h].append(frame(dd, part, dd.fc.values, t).assign(model=name, rv_pred_scaled=np.clip(dd.fc.values * c, 1e-10, None), scale=c))
    for h in hs: pd.concat(rows[h]).to_parquet(OUT / f'garch_h{h}.parquet', index=False)
    pd.DataFrame(prm).to_csv(OUT / 'garch_params.csv', index=False); print('GARCH готов', flush=True)


# ---------------- гибрид ----------------
def stage_hyb(hs):
    import gibrid_schemes as S
    S.MODE = 'rv'; sel = []
    for h in hs:
        rd = lambda n: pd.read_parquet(OUT / f'{n}_h{h}.parquet')
        m = rd('har').rename(columns={'rv_pred': 'har'})
        for n, c in (('xgb', 'xgb'), ('lgbm', 'lgb')): m = m.merge(rd(n)[['date', 'ticker', 'part', 'rv_pred']].rename(columns={'rv_pred': c}), on=['date', 'ticker', 'part'])
        m = m[m.rv_actual > 1e-12].sort_values(['date', 'ticker']).reset_index(drop=True)
        out = m[['date', 'ticker', 'year', 'part', 'rv_actual']].copy()
        for code in ('bez', 'ord'):
            d = m.copy()
            if code == 'ord':  # Б0.1: s² по остаткам val 2018 каждой компоненты, ко всем компонентам (val — в выборке: шаг один)
                for c in ('har', 'xgb', 'lgb'):
                    v = d.part == 'val'; s2 = float(np.var(np.log(d[v].rv_actual + 1e-10) - np.log(d[v][c]))); d[c] *= np.exp(s2 / 2)
                    out[f'{c}_B01'] = d[c].values; sel.append(dict(h=h, code='B01', comp=c, s2=s2, F=np.exp(s2 / 2)))
            vs, ts = d[d.part == 'val'], d[d.part == 'test']; y = vs.rv_actual.values
            w1, meth, ml = S.v1(vs); W, reg = S.weights_v2_v5(vs, None, 1.0); W = {'V1_Adaptive': np.array(w1), **W}
            pv = {k: S.blend(w, vs) for k, w in W.items()}; pt = {k: S.blend(w, ts) for k, w in W.items()}
            pv['V6_TrimmedW'], pt['V6_TrimmedW'] = S.v6(vs)[0], S.v6(ts)[0]
            pv['V7_Rolling_W20'], pt['V7_Rolling_W20'] = S.v7(vs, 20, h)[0].values, S.v7(ts, 20, h)[0].values
            q = {k: qlike(y, p) for k, p in pv.items()}; best = min(q, key=q.get)
            sfx = '' if code == 'bez' else '_B01'
            out[f'hyb{sfx}'] = np.nan
            out.loc[d.part == 'val', f'hyb{sfx}'] = pv[best]; out.loc[d.part == 'test', f'hyb{sfx}'] = pt[best]
            sel.append(dict(h=h, code=code, scheme=best, v1_ml=ml, v1_method=meth, **{f'q_val_{k}': v for k, v in q.items()},
                            **{f'w_{k}': ' '.join(f'{x:.3f}' for x in w) for k, w in W.items()}))
        out.to_parquet(OUT / f'hyb_h{h}.parquet', index=False)
    pd.DataFrame(sel).to_csv(OUT / 'hyb_choice.csv', index=False); print('гибрид готов', flush=True)


def main():
    global ARGS_FEAT
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', nargs='+', required=True); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--features', default=str(FEAT)); ap.add_argument('--hs', type=int, nargs='+', default=list(HS)); ap.add_argument('--out', default=str(OUT))
    a = ap.parse_args(); ARGS_FEAT = a.features; os.environ['T2019_FEAT'] = a.features
    globals()['OUT'] = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); os.environ['T2019_OUT'] = a.out
    print(time.strftime('%F %T'), 'старт', a.stage, a.features, flush=True)
    if 'har' in a.stage or 'garch' in a.stage:
        A, _ = load(a.features)
        if 'har' in a.stage: stage_har(A, a.hs)
        if 'garch' in a.stage: stage_garch(A, a.hs)
    if 'boost' in a.stage: stage_boost(a.hs, a.workers)
    if 'hyb' in a.stage: stage_hyb(a.hs)
    if 'tables' in a.stage:
        import test2019_v3_tables as T
        T.run(OUT)
    print(time.strftime('%F %T'), 'конец', flush=True)


ARGS_FEAT = os.environ.get('T2019_FEAT', str(FEAT))
OUT = Path(os.environ.get('T2019_OUT', str(OUT)))
if __name__ == '__main__':
    main()
