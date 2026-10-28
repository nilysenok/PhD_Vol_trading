"""Одно обучение (модель, h, шаг, гиперпараметры) в воркере: num_threads=1, ранняя остановка по l2 (протокол B) или по QLIKE
(кастомная метрика: прогноз на ln-шкале -> exp, QLIKE = RV/F - ln(RV/F) - 1, как qlike walk_forward_full.py), прогнозы val/test.
Параметры моделей — как tune_B.fit (XGBRegressor/lgb.train); проверено: xgb.train на DMatrix из save_binary = XGBRegressor (maxdiff 0.0)."""
import json, os, time
import warnings
import numpy as np, pandas as pd
from . import cache as K, config as C, data as D

warnings.filterwarnings('ignore')  # как tune_B.py / walk_forward_full.py; логи — в data/run/

_ds = {}  # кэш загруженных Dataset в процессе воркера


def task_key(fk, model, params, train, seed, prune=None, **ex):
    extra = {'prune': prune} if prune else {}
    code = C.code_hash('train.py', 'data.py', *(('v2.py', 'levers.py') if train.get('protocol') == 'b_v2' else ()))
    return K.key('boost', features=fk, model=model, params=params, train=train, seed=seed, code=code, **extra, **ex)


def _qlike_np(rv, pred_ln):
    return float(C.qlike(rv, np.clip(np.exp(pred_ln), 1e-10, None)))


def _load(dk, fk, model, max_bin, f32=False):
    dsk, built = D.build_dataset(dk, fk, model, max_bin, f32)
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


def fit(model, p, train, seed, dtr, dva, rv_vi, prune=None):
    """prune=(итерация, порог QLIKE): снять, если val-QLIKE на этой итерации выше порога (правило 25.10: хуже лучшего на 20%)."""
    n, pat, es, mb = int(train['n_estimators']), int(train['patience']), train['es_metric'], train['max_bin']
    assert prune is None or es == 'qlike', 'отсечение — по QLIKE-метрике ранней остановки'
    pr = []
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
        class Prune(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, log):
                if prune and epoch == prune[0] - 1 and log['val']['qlike'][-1] > prune[1]:
                    pr.append(epoch)
                    return True
                return False
        b = xgb.train(prm, dtr, n, evals=[(dva, 'val')], early_stopping_rounds=pat, verbose_eval=False, callbacks=[Prune()], **kw)
        bi = int(b.best_iteration)
        return (lambda X: b.predict(xgb.DMatrix(X, nthread=1), iteration_range=(0, bi + 1))), bi, bool(pr)
    import lightgbm as lgb
    prm = dict(objective='regression', boosting_type='gbdt', max_depth=-1, num_leaves=int(p['num_leaves']), learning_rate=p['learning_rate'],
               bagging_fraction=p['bagging_fraction'], bagging_freq=1, feature_fraction=p['feature_fraction'],
               min_data_in_leaf=int(p['min_data_in_leaf']), reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'], verbosity=-1,
               seed=seed, num_threads=1, metric='l2')
    if mb is not None:
        prm['max_bin'] = int(mb)
    if train.get('force_col_wise'):
        prm['force_col_wise'] = True  # ускорение 25.10 (не в протоколе B)
    kw = {}
    if es == 'qlike':
        prm['metric'] = 'None'
        kw['feval'] = lambda pr, ds: ('qlike', _qlike_np(rv_vi, pr), False)
    def lprune(env):
        if prune and env.iteration == prune[0] - 1 and env.evaluation_result_list[0][2] > prune[1]:
            pr.append(env.iteration)
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
    cb = [lgb.early_stopping(pat, verbose=False)] + ([lprune] if prune else [])
    b = lgb.train(prm, dtr, n, valid_sets=[dva], callbacks=cb, **kw)
    return (lambda X: b.predict(X)), int(b.best_iteration), bool(pr)  # predict использует best_iteration


def run_task(t):
    """t: dict(dk, fk, model, h, step, params, train, seed[, tag, prune, keep, fk1, levers, h1task, save_all]). Кэш по ключу обучения;
    прогнозы сохраняются и у снятых отсечением (модель на момент снятия, флаг pruned). train.protocol == 'b_v2' — обучение models/v2.py."""
    ex = {x: t[x] for x in ('keep', 'levers', 'h1task') if t.get(x)}
    k = task_key(t['fk'], t['model'], t['params'], t['train'], t['seed'], t.get('prune'), **ex)
    jp = K.pred_path('val', 'boost', k).with_suffix('.json')
    ap = K.stage_dir('allpred', k) / 'pred.npy'
    if jp.exists() and (not t.get('save_all') or ap.exists()):
        return {**json.loads(jp.read_text()), 'hit': True}
    t0 = time.time()
    fd = K.stage_dir('features', t['fk'])
    ids = {nm: pd.read_parquet(fd / f'ids_{nm}.parquet') for nm in ('vi', 'td', 'nx')}
    v2 = t['train'].get('protocol') == 'b_v2'
    if v2:
        from . import levers as LV, v2 as V2
        extra, yoff = LV.prepare(t)
        dtr, dva, Xs = V2.datasets(t, extra, yoff)
        built, t1 = True, time.time()
        pred, bi, _ = V2.fit(t['model'], t['params'], t['seed'], dtr, dva, t['train'])
        pruned, off = False, yoff or {}
    else:
        f32 = bool(t['train'].get('float32'))
        (dtr, dva), built = _load(t['dk'], t['fk'], t['model'], t['train']['max_bin'], f32)
        t1 = time.time()
        pred, bi, pruned = fit(t['model'], t['params'], t['train'], t['seed'], dtr, dva, ids['vi']['rv_actual'].values, t.get('prune'))
        Xs, off = {nm: D.X_rows(t['dk'], t['fk'], nm, f32) for nm in ('vi', 'td', 'nx')}, {}
    t2 = time.time()
    out = {}
    for scope, kind, nm in (('val', 'boost', 'vi'), ('test', 'boost', 'td'), ('test', 'boost_next', 'nx')):
        p = pred(Xs[nm]) + off.get(nm, 0) if len(Xs[nm]) else np.zeros(0)
        out[nm] = ids[nm].assign(rv_pred=np.clip(np.exp(p), 1e-10, None))
        K.write_atomic(out[nm], K.pred_path(scope, kind, k))
    if t.get('save_all'):  # ln-прогноз на всех строках матрицы (признак Д8.6)
        X = np.load(K.stage_dir('data', t['dk']) / 'X.npy', mmap_mode='r')
        X = np.ascontiguousarray(X[:, t['keep']]).astype(np.float32) if v2 else np.asarray(X, dtype=np.float32 if t['train'].get('float32') else None)
        ap.parent.mkdir(parents=True, exist_ok=True)
        np.save(ap, pred(X))
    q = float(C.qlike(out['vi']['rv_actual'].values, out['vi']['rv_pred'].values))
    res = dict(key=k, model=t['model'], h=t['h'], step=t['step'], best_iteration=bi, qlike_val=q, pruned=pruned, tag=t.get('tag'),
               sec_dataset=t1 - t0, sec_train=t2 - t1, sec_total=time.time() - t0, dataset_built=built, pid=os.getpid())
    tmp = jp.with_suffix(f'.json.{os.getpid()}')
    tmp.write_text(json.dumps(res))
    os.replace(tmp, jp)
    return {**res, 'hit': False}
