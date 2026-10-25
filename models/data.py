"""Стадии «данные» -> «признаки» -> «Dataset».
Данные: склейка train/val/test + walkforward_{2020..2026} (len>100) или одна матрица (features_pit_v2.parquet), сортировка по date —
как tune_B.load / gibrid_har.load_all; путь к матрице и колонки цели по h — параметры дизайна (data.features, data.target);
X = clean_features(all[feature_cols]) один раз на весь массив (поэлементно, совпадает с применением к подвыборкам).
Признаки (шаг x h): индексы строк train_pure / val-года / тест-года и цель ln(rv_target_h + 1e-10) — как tune_B.step_data.
Dataset (шаг x h x модель x max_bin): lgb.Dataset.save_binary (train + val с reference) и xgb.DMatrix.save_binary."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from . import cache as K, config as C

PARTS = ('train', 'val', 'test')


def source(design, cli_root=None):
    """Источник матрицы из дизайна (data.features: каталог prepared/ или один .parquet с <имя>.config.json; data.target: {h: колонка};
    data.har_features). --data-root (детектор утечек) подменяет data.features."""
    d = dict(design.get('data') or {})
    f = cli_root or d.get('features') or C.DATA_ROOT
    f = Path(f) if os.path.isabs(str(f)) else C.ROOT / f
    cfg = d.get('config') or (f / 'config.json' if f.is_dir() else f.with_name(f.stem + '.config.json'))
    cfg = Path(cfg) if os.path.isabs(str(cfg)) else C.ROOT / cfg
    tg = {int(h): str(c) for h, c in (d.get('target') or {}).items()}
    return dict(features=str(f), config=str(cfg), target={h: tg.get(h, f'rv_target_h{h}') for h in C.HS}, har_features=d.get('har_features'))


def input_files(src):
    f = Path(src['features'])
    if not f.is_dir():
        return [f, Path(src['config'])]
    fs = [f / f'{n}.parquet' for n in PARTS] + [f / 'config.json']
    return fs + [f / f'walkforward_{y}.parquet' for y in range(2020, 2027) if (f / f'walkforward_{y}.parquet').exists()]


def data_key(src):
    return K.key('data', inputs=K.file_digest(input_files(src)), target=src['target'], har=src['har_features'], code=C.code_hash('data.py'))


def load_all(src):
    """Каталог: склейка как tune_B.load; файл: одна матрица. Цель h -> колонка rv_target_h{h} (исходная — rv_target_h{h}_orig),
    чтобы HAR-J/гибрид (gibrid_har с жёстким именем цели) работали без правок."""
    f = Path(src['features'])
    cfg = json.load(open(src['config']))
    if f.is_dir():
        parts = [pd.read_parquet(f / f'{n}.parquet') for n in PARTS]
        for y in range(2020, 2027):
            fp = f / f'walkforward_{y}.parquet'
            if fp.exists():
                d = pd.read_parquet(fp)
                if len(d) > 100:
                    parts.append(d)
        a = pd.concat(parts)
    else:
        a = pd.read_parquet(f)
    a = a.sort_values('date').reset_index(drop=True)
    a['year'] = pd.to_datetime(a['date']).dt.year
    for h, c in src['target'].items():
        if c != f'rv_target_h{h}':
            if f'rv_target_h{h}' in a.columns:
                a[f'rv_target_h{h}_orig'] = a[f'rv_target_h{h}']
            a[f'rv_target_h{h}'] = a[c]
    har = src['har_features'] or cfg.get('har_features') or json.load(open(C.DATA_ROOT / 'config.json'))['har_features']
    jump = [c for c in a.columns if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower()]
    harj = [x for x in har + [c for c in jump if c not in har] if x in a.columns]  # gibrid_har.load_all
    return a, cfg['feature_cols'], harj


def stage_data(src, T):
    k = data_key(src)
    if K.done('data', k):
        T.add('данные', hit=True)
        return k
    with T.stage('данные'):
        a, fc, harj = load_all(src)
        d = K.stage_dir('data', k)
        d.mkdir(parents=True, exist_ok=True)
        a.to_parquet(d / 'all.parquet', index=False)
        np.save(d / 'X.npy', C.clean_features(a[fc].values))
        K.mark('data', k, fc=fc, harj=harj, n=len(a), src=src)
    T.add('данные', hit=False)
    return k


def read_all(dk):
    m = K.meta('data', dk)
    return pd.read_parquet(K.stage_dir('data', dk) / 'all.parquet'), m['harj']


def features_key(dk, h, step):
    return K.key('features', data=dk, h=h, step=step, code=C.code_hash('data.py', C.WFF))


def stage_features(dk, pairs, T):
    """pairs: [(h, step)] -> {(h, step): ключ}. Данные читаются только если чего-то нет в кэше."""
    keys = {p: features_key(dk, *p) for p in pairs}
    todo = [p for p, k in keys.items() if not K.done('features', k)]
    T.add('признаки', hit=True) if not todo else None
    if todo:
        with T.stage('признаки'):
            a, _ = read_all(dk)
            for h, step in todo:
                ty, t = C.YEARS[step - 1], f'rv_target_h{h}'
                tr = a[a.year < ty]
                vy = sorted(tr.year.unique())[-1]
                tp, vi, td = tr[tr.year != vy].dropna(subset=[t]), tr[tr.year == vy].dropna(subset=[t]), a[a.year == ty].dropna(subset=[t])
                d = K.stage_dir('features', keys[(h, step)])
                d.mkdir(parents=True, exist_ok=True)
                for nm, df in (('tp', tp), ('vi', vi), ('td', td)):
                    np.save(d / f'idx_{nm}.npy', df.index.values)
                np.save(d / 'ytp.npy', np.log(tp[t].values + 1e-10))
                np.save(d / 'yvi.npy', np.log(vi[t].values + 1e-10))
                for nm, df, yr in (('vi', vi, vy), ('td', td, ty)):
                    pd.DataFrame({'date': df['date'].values, 'ticker': df['ticker'].values, 'year': yr, 'step': step,
                                  'rv_actual': df[t].values}).to_parquet(d / f'ids_{nm}.parquet', index=False)
                K.mark('features', keys[(h, step)], h=h, step=step, val_year=int(vy), test_year=ty, data=dk)
        T.add('признаки', hit=False)
    return keys


def dataset_key(fk, model, max_bin):
    return K.key('dataset', features=fk, model=model, max_bin=max_bin if model == 'lgbm' else None, code=C.code_hash('data.py'))


def X_rows(dk, fk, nm, _mm={}):
    if dk not in _mm:
        _mm[dk] = np.load(K.stage_dir('data', dk) / 'X.npy', mmap_mode='r')
    return np.ascontiguousarray(_mm[dk][np.load(K.stage_dir('features', fk) / f'idx_{nm}.npy')])


def lgb_params(max_bin):
    p = {'feature_pre_filter': False, 'verbosity': -1}  # как tune_B: датасет подходит любому min_data_in_leaf
    if max_bin is not None:
        p['max_bin'] = int(max_bin)
    return p


def build_dataset(dk, fk, model, max_bin):
    """В воркере; запись атомарная (tmp + rename), повторная сборка тем же ключом безопасна."""
    k = dataset_key(fk, model, max_bin)
    if K.done('dataset', k):
        return k, False
    d = K.stage_dir('dataset', k)
    d.mkdir(parents=True, exist_ok=True)
    fd = K.stage_dir('features', fk)
    Xtp, Xvi, ytp, yvi = X_rows(dk, fk, 'tp'), X_rows(dk, fk, 'vi'), np.load(fd / 'ytp.npy'), np.load(fd / 'yvi.npy')
    pid = os.getpid()
    if model == 'lgbm':
        import lightgbm as lgb
        tr = lgb.Dataset(Xtp, ytp, free_raw_data=False, params=lgb_params(max_bin))
        va = lgb.Dataset(Xvi, yvi, reference=tr, free_raw_data=False)
        for nm, ds in (('tr', tr), ('va', va)):
            ds.save_binary(str(d / f'{nm}.bin.{pid}'))
            os.replace(d / f'{nm}.bin.{pid}', d / f'{nm}.bin')
    else:
        import xgboost as xgb
        for nm, X, y in (('tr', Xtp, ytp), ('va', Xvi, yvi)):
            xgb.DMatrix(X, y, nthread=1).save_binary(str(d / f'{nm}.dm.{pid}'), silent=True)
            os.replace(d / f'{nm}.dm.{pid}', d / f'{nm}.dm')
    K.mark('dataset', k, model=model, max_bin=max_bin, features=fk)
    return k, True
