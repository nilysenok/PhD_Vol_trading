"""Финальное обучение по протоколу B v2 (tune_Bv2.py: train(..., full=True), ветка etap5-B) — для ворот на Bv2_* и дизайнов на точке B v2.
Совпадает построчно: X = clean_features(...) -> float32 -> [:, keep] (keep — без признаков Bv2_excluded_features.csv на шагах >= 2,
объединение по (модель, h)); бины шагов >= 2 — от Dataset/QuantileDMatrix шага 1 (все строки train_pure шага 1) через reference;
lgbm: max_bin 63, force_col_wise, ES l2 patience 30, 1000 итераций; xgb: QuantileDMatrix max_bin 64, eval rmse, ES 30; прогноз —
inplace_predict по best_iteration. Датасеты строятся в воркере (QuantileDMatrix не сохраняется в файл)."""
import numpy as np, pandas as pd
from . import cache as K, config as C, data as D

LGB_DS = {'max_bin': 63, 'feature_pre_filter': False, 'verbosity': -1}


def keep_cols(excl_path, fc, model, h, step):
    if step <= 1 or not excl_path:
        return list(range(len(fc)))
    p = C.data_path(excl_path)
    ex = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=['model', 'h', 'feature'])
    ex = set(ex[(ex.model == model) & (ex.h == h)].feature)
    return [i for i, f in enumerate(fc) if f not in ex]


def arrays(t, nm, fk=None):
    X = D.X_rows(t['dk'], fk or t['fk'], nm, True)
    return np.ascontiguousarray(X[:, t['keep']])


_ref = {}


def datasets(t, extra=None, yoff=None):
    """extra: {nm: доп. колонки} (рычаги), yoff: {nm: смещение цели} (V9). Возвращает (dtr, dva, Xv, Xt, Xn)."""
    fd, fd1 = K.stage_dir('features', t['fk']), K.stage_dir('features', t['fk1'])
    X = {nm: arrays(t, nm) for nm in ('tp', 'vi', 'td', 'nx')}
    if extra:
        X = {nm: np.hstack([x, extra[nm].astype(np.float32)]) for nm, x in X.items()}
    y, yv = np.load(fd / 'ytp.npy'), np.load(fd / 'yvi.npy')
    if yoff:
        y, yv = y - yoff['tp'], yv - yoff['vi']
    rk = (t['model'], t['fk1'], tuple(t['keep']), bool(extra))
    if t['step'] > 1 and rk not in _ref:
        X1 = arrays(t, 'tp', t['fk1'])
        if extra:
            X1 = np.hstack([X1, extra['tp1'].astype(np.float32)])
        y1 = np.load(fd1 / 'ytp.npy') - (yoff['tp1'] if yoff else 0)
        if t['model'] == 'xgb':
            import xgboost as xgb
            _ref[rk] = xgb.QuantileDMatrix(X1, y1, max_bin=64)
        else:
            import lightgbm as lgb
            _ref[rk] = lgb.Dataset(X1, y1, params=dict(LGB_DS), free_raw_data=False)
    ref = _ref.get(rk) if t['step'] > 1 else None
    if t['model'] == 'xgb':
        import xgboost as xgb
        d0 = xgb.QuantileDMatrix(X['tp'], y, max_bin=64, ref=ref)
        dva = xgb.QuantileDMatrix(X['vi'], yv, ref=d0, max_bin=64)
    else:
        import lightgbm as lgb
        d0 = lgb.Dataset(X['tp'], y, reference=ref, params=dict(LGB_DS), free_raw_data=False)
        dva = lgb.Dataset(X['vi'], yv, reference=d0, free_raw_data=False)
    return d0, dva, X


def fit(model, p, seed, dtr, dva, train):
    n, pat = int(train.get('n_estimators', 1000)), int(train.get('patience', 30))
    if model == 'xgb':
        import xgboost as xgb
        prm = dict(tree_method='hist', max_bin=64, max_depth=int(p['max_depth']), eta=p['learning_rate'], subsample=p['subsample'],
                   colsample_bytree=p['colsample_bytree'], min_child_weight=p['min_child_weight'], alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
                   nthread=1, seed=seed, verbosity=0, eval_metric='rmse')
        b = xgb.train(prm, dtr, n, evals=[(dva, 'val')], early_stopping_rounds=pat, verbose_eval=False)
        bi = int(b.best_iteration)
        return (lambda X: b.inplace_predict(X, iteration_range=(0, bi + 1))), bi, b
    import lightgbm as lgb
    prm = dict(objective='regression', boosting_type='gbdt', max_depth=-1, num_leaves=int(p['num_leaves']), learning_rate=p['learning_rate'],
               bagging_fraction=p['bagging_fraction'], bagging_freq=1, feature_fraction=p['feature_fraction'], min_data_in_leaf=int(p['min_data_in_leaf']),
               reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'], max_bin=63, verbosity=-1, seed=seed, num_threads=1, metric='l2', force_col_wise=True)
    b = lgb.train(prm, dtr, n, valid_sets=[dva], callbacks=[lgb.early_stopping(pat, verbose=False)])
    return (lambda X: b.predict(X)), int(b.best_iteration), b
