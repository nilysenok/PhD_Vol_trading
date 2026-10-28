"""Gain-отбор признаков блока Д7: LightGBM на точках Bv2b_selected_h{h}.csv (по шагам 1-6, walk-forward как regen_boosting.py:
тест-год = 2016+шаг, валидация = последний год обучающего окна, ранняя остановка patience 50 по l2 на ln RV).
Признаки d7_ с нулевым суммарным gain на шагах 1-6 исключаются из data/regen/features_pit_v2_d7_h{h}.parquet.
Запуск: python3 d7_gain.py [--no-write]. Выход: docs/tables/d7_gain_h{h}.csv (gain по шагам) и список исключённых."""
import argparse, json, os
import numpy as np, pandas as pd
import lightgbm as lgb
import d7_block as b

REG = os.path.join(b.DATA, 'regen')
TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'tables')


def run(h, steps=range(1, 7), nthreads=2):
    m = pd.read_parquet(os.path.join(REG, 'features_pit_v2_d7_h%d.parquet' % h))
    cfg = json.load(open(os.path.join(REG, 'features_pit_v2.config.json')))['feature_cols']
    exc = pd.read_csv(os.path.join(REG, 'Bv2b_excluded_features.csv'))
    exc = set(exc[(exc.h == h) & (exc.model == 'lgbm')].feature)
    d7 = [c for c in m if c.startswith('d7_')]
    feats = [c for c in cfg if c in m and c not in exc] + d7
    sel = pd.read_csv(os.path.join(REG, 'Bv2b_selected_h%d.csv' % h))
    sel = sel[sel.model == 'lgbm'].set_index('step')
    tgt = 'rv_target_h%d' % h
    m['year'] = pd.to_datetime(m['date']).dt.year
    m = m.dropna(subset=[tgt])
    gains = {}
    for st in steps:
        ty, r = 2016 + st, sel.loc[st]
        tr = m[m.year < ty]
        vy = tr.year.max()
        tp, vi = tr[tr.year != vy], tr[tr.year == vy]
        clean = lambda x: np.nan_to_num(x[feats].values.astype('float64'), nan=0.0, posinf=0.0, neginf=0.0)
        p = dict(num_leaves=int(r.num_leaves), learning_rate=r.learning_rate, bagging_fraction=r.bagging_fraction, bagging_freq=1,
                 feature_fraction=r.feature_fraction, min_data_in_leaf=int(r.min_data_in_leaf), reg_alpha=r.reg_alpha,
                 reg_lambda=r.reg_lambda, n_estimators=500, random_state=42, verbosity=-1, n_jobs=nthreads)
        mod = lgb.LGBMRegressor(**p).fit(clean(tp), np.log(tp[tgt].values + 1e-10), eval_set=[(clean(vi), np.log(vi[tgt].values + 1e-10))],
                                         eval_metric='l2', callbacks=[lgb.early_stopping(50, verbose=False)])
        gains[st] = pd.Series(mod.booster_.feature_importance('gain'), index=feats)
        print('h=%d шаг %d (тест %d): деревьев %d (было %d), признаков %d' % (h, st, ty, mod.best_iteration_, r.best_iteration, len(feats)), flush=True)
    g = pd.DataFrame(gains).loc[d7]
    g['sum'] = g.sum(axis=1)
    return m, g


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--no-write', action='store_true'); a = ap.parse_args()
    for h in (5, 22):
        _, g = run(h)
        g.to_csv(os.path.join(TAB, 'd7_gain_h%d.csv' % h), index_label='feature')
        zero = g.index[g['sum'] == 0].tolist()
        print('h=%d: d7 до %d, нулевой gain %d, после %d; исключены: %s' % (h, len(g), len(zero), len(g) - len(zero), zero), flush=True)
        if not a.no_write:
            p = os.path.join(REG, 'features_pit_v2_d7_h%d.parquet' % h)
            pd.read_parquet(p).drop(columns=zero).to_parquet(p, index=False)
