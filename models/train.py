"""Одно обучение (модель, h, шаг, гиперпараметры) в воркере: num_threads=1, ранняя остановка по l2 (протокол B) или по QLIKE
(кастомная метрика: прогноз на ln-шкале -> exp, QLIKE = RV/F - ln(RV/F) - 1, как qlike walk_forward_full.py), прогнозы val/test.
Параметры моделей — как tune_B.fit (XGBRegressor/lgb.train); проверено: xgb.train на DMatrix из save_binary = XGBRegressor (maxdiff 0.0)."""
import json, os, time
import warnings
import numpy as np, pandas as pd
from . import cache as K, config as C, data as D

warnings.filterwarnings('ignore')  # как tune_B.py / walk_forward_full.py; логи — в data/run/

_ds = {}  # кэш загруженных Dataset в процессе воркера


def task_key(fk, model, params, train, seed):
    return K.key('boost', features=fk, model=model, params=params, train=train, seed=seed, code=C.code_hash('train.py', 'data.py'))


def _qlike_np(rv, pred_ln):
    return float(C.qlike(rv, np.clip(np.exp(pred_ln), 1e-10, None)))


def _load(dk, fk, model, max_bin):
    dsk, built = D.build_dataset(dk, fk, model, max_bin)
    if dsk not in _ds:
        _ds.clear()
        d = K.stage_dir('dataset', dsk)
        if model == 'lgbm':
            import lightgbm as lgb
            tr = lgb.Dataset(str(d / 'tr.bin'), params=D.lgb_params(max_bin))
            _ds[dsk] = (tr, lgb.Dataset(str(d / 'va.bin'), reference=tr))
        else:
            import xgboost as xgb
            cwd = os.getcwd()
            os.chdir(d)  # загрузчик URI xgboost не открывает путь с кириллицей ('Старое') — читаем по относительному имени
            try:
                _ds[dsk] = (xgb.DMatrix('tr.dm', silent=True), xgb.DMatrix('va.dm', silent=True))
            finally:
                os.chdir(cwd)
    return _ds[dsk], built


def fit(model, p, train, seed, dtr, dva, rv_vi):
    n, pat, es, mb = int(train['n_estimators']), int(train['patience']), train['es_metric'], train['max_bin']
    if model == 'xgb':
        import xgboost as xgb
        prm = dict(objective='reg:squarederror', tree_method='hist', max_depth=int(p['max_depth']), learning_rate=p['learning_rate'],
                   subsample=p['subsample'], colsample_bytree=p['colsample_bytree'], min_child_weight=p['min_child_weight'],
                   reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'], verbosity=0, seed=seed, nthread=1)
        if mb is not None:
            prm['max_bin'] = int(mb)
        kw = {}
        if es == 'qlike':
            prm['disable_default_eval_metric'] = 1
            kw['custom_metric'] = lambda pr, dm: ('qlike', _qlike_np(rv_vi, pr))
        b = xgb.train(prm, dtr, n, evals=[(dva, 'val')], early_stopping_rounds=pat, verbose_eval=False, **kw)
        bi = int(b.best_iteration)
        return (lambda X: b.predict(xgb.DMatrix(X, nthread=1), iteration_range=(0, bi + 1))), bi
    import lightgbm as lgb
    prm = dict(objective='regression', boosting_type='gbdt', max_depth=-1, num_leaves=int(p['num_leaves']), learning_rate=p['learning_rate'],
               bagging_fraction=p['bagging_fraction'], bagging_freq=1, feature_fraction=p['feature_fraction'],
               min_data_in_leaf=int(p['min_data_in_leaf']), reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'], verbosity=-1,
               seed=seed, num_threads=1, metric='l2')
    if mb is not None:
        prm['max_bin'] = int(mb)
    kw = {}
    if es == 'qlike':
        prm['metric'] = 'None'
        kw['feval'] = lambda pr, ds: ('qlike', _qlike_np(rv_vi, pr), False)
    b = lgb.train(prm, dtr, n, valid_sets=[dva], callbacks=[lgb.early_stopping(pat, verbose=False)], **kw)
    return (lambda X: b.predict(X)), int(b.best_iteration)  # predict использует best_iteration


def run_task(t):
    """t: dict(dk, fk, model, h, step, params, train, seed[, tag]). Результат кэшируется по ключу обучения."""
    k = task_key(t['fk'], t['model'], t['params'], t['train'], t['seed'])
    jp = K.pred_path('val', 'boost', k).with_suffix('.json')
    if jp.exists():
        return {**json.loads(jp.read_text()), 'hit': True}
    t0 = time.time()
    fd = K.stage_dir('features', t['fk'])
    ids_vi, ids_td = pd.read_parquet(fd / 'ids_vi.parquet'), pd.read_parquet(fd / 'ids_td.parquet')
    (dtr, dva), built = _load(t['dk'], t['fk'], t['model'], t['train']['max_bin'])
    t1 = time.time()
    pred, bi = fit(t['model'], t['params'], t['train'], t['seed'], dtr, dva, ids_vi['rv_actual'].values)
    t2 = time.time()
    out = {}
    for scope, ids, nm in (('val', ids_vi, 'vi'), ('test', ids_td, 'td')):
        df = ids.assign(rv_pred=np.clip(np.exp(pred(D.X_rows(t['dk'], t['fk'], nm))), 1e-10, None))
        K.write_atomic(df, K.pred_path(scope, 'boost', k))
        out[scope] = df
    q = float(C.qlike(out['val']['rv_actual'].values, out['val']['rv_pred'].values))
    res = dict(key=k, model=t['model'], h=t['h'], step=t['step'], best_iteration=bi, qlike_val=q, tag=t.get('tag'),
               sec_dataset=t1 - t0, sec_train=t2 - t1, sec_total=time.time() - t0, dataset_built=built, pid=os.getpid())
    tmp = jp.with_suffix(f'.json.{os.getpid()}')
    tmp.write_text(json.dumps(res))
    os.replace(tmp, jp)
    return {**res, 'hit': False}
