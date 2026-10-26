#!/usr/bin/env python3
"""R2, агент S: кэш обучающих наборов бустинга (решение владельца 16:25: весь R2 на Маке, ускоряем кодом).

Ключ набора — SHA-256 канонического JSON частей: шаг, набор признаков (имена по порядку), h, цель, плюс всё, что меняет бины
(max_bin, эмбарго, тикеры, версия данных — SHA матрицы, reference-ключ для бинов окна шага 1). Источник истины — data/;
кэш — производный и одноразовый: RAM-диск /Volumes/RAMcache (если смонтирован) или локальный SSD (~/Library/Caches/phd_r2/r2_ds).

  LightGBM: lgb_dataset(key, build) — Dataset строится один раз (construct + save_binary, атомарно, под файловой блокировкой: при
            одновременном запросе одного ключа строит ровно один процесс), дальше любой процесс читает бинарный файл.
  XGBoost:  xgb_qdm(key, build, ref=None) — массивы (X, y, weight, base_margin) пишутся один раз в <key>.xgb/*.npy, QuantileDMatrix
            строится из них и держится в памяти процесса по тому же ключу; warm(...) в родителе до fork — воркеры получают готовые
            QuantileDMatrix копией-при-записи.
  Пул:      run(fn, tasks, workers=10) — r2_core.run_pool (fork, данные в r2_core.SHARED грузятся один раз в родителе, num_threads=1).

API:
  cache_root() -> Path
  data_sha(path) -> str                                  SHA-256 файла данных с памятью по (путь, размер, mtime)
  ds_key(**parts) -> str                                 64 hex; parts — JSON-совместимые значения (списки сохраняют порядок)
  lgb_dataset(key, build, params=None) -> lgb.Dataset    build() -> dict(X, y[, weight, init_score, reference, categorical_feature,
                                                         feature_name]); reference — Dataset (бины окна шага 1, как l2_lib.make_ref)
  xgb_qdm(key, build, ref=None, max_bin=255, **kw) -> xgb.QuantileDMatrix   build() -> dict(X, y[, weight, base_margin])
  warm(specs)                                            [(вид 'lgb'|'xgb', key, build, kw)] — построить в родителе до fork
  stats() -> dict                                        счётчики процесса: hit/build по видам
  run(fn, tasks, workers=10)                             пул r2_core
python3 scripts/analysis/r2_dataset_cache.py smoke [--workers 10]   дым: SBER, LKOH, AFLT × шаги 1–3 × h ∈ {1, 5, 22}
python3 scripts/analysis/r2_dataset_cache.py info                   корень кэша, число и объём наборов
"""
import os

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse
import fcntl
import hashlib
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r2_core as C  # noqa: E402

RAM = Path('/Volumes/RAMcache')
SSD = Path.home() / 'Library' / 'Caches' / 'phd_r2'
VERSION = 'r2ds-v1'
_STATS = {'lgb_hit': 0, 'lgb_build': 0, 'xgb_hit_mem': 0, 'xgb_hit_disk': 0, 'xgb_build': 0}
_XQ = {}
_SHA_MEMO = {}


def cache_root():
    env = os.environ.get('R2_DS_CACHE')
    if env:
        r = Path(env)
    elif RAM.is_dir() and os.access(RAM, os.W_OK):
        r = RAM / 'r2_ds'
    else:
        r = SSD / 'r2_ds'
    r.mkdir(parents=True, exist_ok=True)
    return r


def data_sha(path):
    p = Path(path)
    st = p.stat()
    k = (str(p), st.st_size, st.st_mtime_ns)
    if k not in _SHA_MEMO:
        memo = cache_root() / '_data_sha.json'
        M = json.loads(memo.read_text()) if memo.exists() else {}
        mk = f'{k[0]}|{k[1]}|{k[2]}'
        if mk not in M:
            M[mk] = C.sha_file(p)
            tmp = memo.with_name(f'.{memo.name}.{os.getpid()}.tmp')
            tmp.write_text(json.dumps(M, ensure_ascii=False))
            os.replace(tmp, memo)
        _SHA_MEMO[k] = M[mk]
    return _SHA_MEMO[k]


def _canon(x):
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return C._norm(float(x))
    if isinstance(x, Path):
        return str(x)
    return x


FEATURES_R2 = C.REGEN / 'R2' / 'features' / 'features_r2_v3.parquet'   # признаки R2 (агент F, PR #176)
FEATURES_R2_SHA = '977a87d31f0474c5f4d113aa524f040fe3fe1a5f290da1c27c22ff1878af7a72'


def features_r2_sha():
    """SHA-256 актуальной матрицы признаков R2 (features_r2_v3): сверка с константой и с файлом .sha256 агента F; расхождение — ошибка."""
    got = data_sha(FEATURES_R2)
    side = FEATURES_R2.with_name(FEATURES_R2.name + '.sha256')
    want_side = side.read_text().split()[0] if side.exists() else got
    if not (got == FEATURES_R2_SHA == want_side):
        raise AssertionError(f'{FEATURES_R2.name}: SHA {got[:12]}… ≠ {FEATURES_R2_SHA[:12]}… (или .sha256 {want_side[:12]}…)')
    return got


def ds_key(**parts):
    """Ключ набора: шаг, признаки, h, цель и всё, что меняет бины. Обязательные части: step, features, h, target.
    В ключ всегда входят SHA матрицы признаков R2 (features_r2_v3) и, если передана, подвыборка строк (row_frac, row_seed)."""
    for need in ('step', 'features', 'h', 'target'):
        if need not in parts:
            raise ValueError(f'ds_key: нет части {need}')
    parts = dict(parts, features_r2=dict(file=FEATURES_R2.name, sha256=features_r2_sha()))
    txt = json.dumps(dict(_canon(parts), _v=VERSION), ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(txt.encode('utf-8')).hexdigest()


@contextmanager
def _lock(key):
    f = cache_root() / f'.{key[:32]}.lock'
    with open(f, 'a+') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _note_build(key, kind):
    with open(cache_root() / f'{key[:32]}.{kind}.builds', 'a') as f:
        f.write(f'{os.getpid()} {time.strftime("%F %T")}\n')


# ---------------------------------------------------------------- LightGBM
LGB_PARAMS = {'max_bin': 255, 'feature_pre_filter': False, 'verbosity': -1, 'num_threads': 1}


def lgb_path(key):
    return cache_root() / f'{key[:32]}.lgb.bin'


def lgb_dataset(key, build, params=None):
    """Dataset по ключу: есть бинарный файл — читается; нет — строится один раз (под блокировкой) и сохраняется save_binary."""
    import lightgbm as lgb
    prm = dict(LGB_PARAMS, **(params or {}))
    path = lgb_path(key)
    if not path.exists():
        with _lock(key):
            if not path.exists():
                spec = build()
                ds = lgb.Dataset(spec['X'], spec['y'], weight=spec.get('weight'), init_score=spec.get('init_score'),
                                 reference=spec.get('reference'), params=prm, categorical_feature=spec.get('categorical_feature', 'auto'),
                                 feature_name=spec.get('feature_name', 'auto'), free_raw_data=False)
                ds.construct()
                tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
                if tmp.exists():
                    tmp.unlink()
                ds.save_binary(str(tmp))
                os.replace(tmp, path)
                _note_build(key, 'lgb')
                _STATS['lgb_build'] += 1
                return ds
    _STATS['lgb_hit'] += 1
    return lgb.Dataset(str(path), params=prm, free_raw_data=False).construct()


# ---------------------------------------------------------------- XGBoost
XGB_PARTS = ('X', 'y', 'weight', 'base_margin')


def xgb_dir(key):
    return cache_root() / f'{key[:32]}.xgb'


def _xgb_arrays(key, build):
    d = xgb_dir(key)
    if not (d / 'DONE').exists():
        with _lock(key):
            if not (d / 'DONE').exists():
                spec = build()
                d.mkdir(parents=True, exist_ok=True)
                for k in XGB_PARTS:
                    if spec.get(k) is not None:
                        np.save(d / f'{k}.npy', np.ascontiguousarray(spec[k]))
                (d / 'DONE').write_text(time.strftime('%F %T'))
                _note_build(key, 'xgb')
                _STATS['xgb_build'] += 1
                return {k: spec.get(k) for k in XGB_PARTS}
    _STATS['xgb_hit_disk'] += 1
    return {k: (np.load(d / f'{k}.npy', mmap_mode='r') if (d / f'{k}.npy').exists() else None) for k in XGB_PARTS}


def xgb_qdm(key, build, ref=None, max_bin=255, **kw):
    """QuantileDMatrix по ключу: память процесса -> массивы на диске -> build(). nthread = 1."""
    import xgboost as xgb
    mk = (key, None if ref is None else id(ref), max_bin)
    if mk in _XQ:
        _STATS['xgb_hit_mem'] += 1
        return _XQ[mk]
    a = _xgb_arrays(key, build)
    dm = xgb.QuantileDMatrix(np.asarray(a['X']), np.asarray(a['y']), weight=None if a['weight'] is None else np.asarray(a['weight']),
                             base_margin=None if a['base_margin'] is None else np.asarray(a['base_margin']), max_bin=max_bin, ref=ref,
                             nthread=1, **kw)
    _XQ[mk] = dm
    return dm


def warm(specs):
    """В родителе до fork: [(вид, key, build, kw)] — наборы строятся/читаются один раз, воркеры получают их готовыми."""
    out = {}
    for kind, key, build, kw in specs:
        out[key] = lgb_dataset(key, build, **(kw or {})) if kind == 'lgb' else xgb_qdm(key, build, **(kw or {}))
    return out


def stats():
    return dict(_STATS)


def run(fn, tasks, workers=10, label='dataset-cache'):
    return C.run_pool(fn, tasks, workers=workers, label=label)


# ================================================================ дым
SM_TICKERS = ('SBER', 'LKOH', 'AFLT')
SM_STEPS = (1, 2, 3)
LGB_P = dict(objective='regression', num_leaves=8, learning_rate=0.05, min_data_in_leaf=50, bagging_fraction=0.8, bagging_freq=1,
             feature_fraction=0.8, seed=42, num_threads=1, force_col_wise=True, verbosity=-1, max_bin=255)
XGB_P = dict(tree_method='hist', max_depth=3, eta=0.05, subsample=0.8, colsample_bytree=0.8, nthread=1, seed=42, verbosity=0,
             objective='reg:squarederror', max_bin=255)
NROUND = 60


def _sm_cols(A):
    cols = [c for c in C._num_cols(A) if not c.startswith('rv_target') and c not in ('rv',)]
    return cols[:40]


def _sm_build(h, k):
    def b():
        A, cols = C.SHARED['A'], C.SHARED['cols']
        m = C.train_mask(A, h, C.step_years(k)[0])
        return dict(X=A.loc[m, cols].to_numpy(np.float64), y=np.log(A.loc[m, f'rv_target_h{h}'].to_numpy() + 1e-10), feature_name=list(cols))
    return b


def _sm_key(h, k, extra=None):
    parts = dict(step=k, features=C.SHARED['cols'], h=h, target=f'ln rv_target_h{h}', embargo=True, tickers=list(SM_TICKERS),
                 data=C.SHARED['data_sha'], max_bin=255)
    parts.update(extra or {})
    return ds_key(**parts)


def _sm_val(h, k):
    A, cols = C.SHARED['A'], C.SHARED['cols']
    m = C.val_mask(A, h, C.step_years(k)[0])
    return A.loc[m, cols].to_numpy(np.float64)


def _sm_train(t):
    """Задача пула: модель из кэша (lgb/xgb) -> сумма прогнозов val (детерминирована при num_threads = 1)."""
    import lightgbm as lgb
    import xgboost as xgb
    h, k, kind = t['h'], t['step'], t['kind']
    key = _sm_key(h, k)
    if kind == 'lgb':
        b = lgb.train(LGB_P, lgb_dataset(key, _sm_build(h, k)), NROUND)
        return float(np.sum(b.predict(_sm_val(h, k))))
    dm = xgb_qdm(key, _sm_build(h, k))
    b = xgb.train(XGB_P, dm, NROUND)
    return float(np.sum(b.predict(xgb.DMatrix(_sm_val(h, k)))))


def smoke(a):
    import lightgbm as lgb
    import pandas as pd
    import xgboost as xgb
    t0 = time.time()
    root = cache_root()
    rows = []
    ok = lambda name, cond, **kw: rows.append(dict(proverka=name, ok=bool(cond), **kw))
    A, harj = C.load_d0(SM_TICKERS)
    C.share(A=A, cols=_sm_cols(A), data_sha=data_sha(C.FEAT))
    ok(f'корень кэша: {root} ({"RAM-диск" if str(root).startswith(str(RAM)) else "локальный SSD"})', root.exists())
    ok('ключ различает шаг, h, признаки, цель', len({_sm_key(5, 1), _sm_key(5, 2), _sm_key(22, 1), _sm_key(5, 1, {'target': 'x'}),
                                                    ds_key(step=1, features=C.SHARED['cols'][:-1], h=5, target='ln rv_target_h5')}) == 5)
    ok(f'в ключе — SHA {FEATURES_R2.name} (PR #176) = {FEATURES_R2_SHA[:12]}… и подвыборка строк',
       features_r2_sha() == FEATURES_R2_SHA and _sm_key(5, 1) != _sm_key(5, 1, {'row_frac': 0.5, 'row_seed': 20261026}))
    # 0) смыть наборы дыма (повторный дым должен строить заново)
    for h in C.HS:
        for k in SM_STEPS:
            key = _sm_key(h, k)
            for f in (lgb_path(key), cache_root() / f'{key[:32]}.lgb.builds', cache_root() / f'{key[:32]}.xgb.builds'):
                if f.exists():
                    f.unlink()
            d = xgb_dir(key)
            if d.exists():
                for f in d.iterdir():
                    f.unlink()
                d.rmdir()
    # 1) LightGBM: свежий Dataset == бинарный из кэша (прогнозы побайтно), повтор — без перестройки
    h, k = 5, 2
    key = _sm_key(h, k)
    spec = _sm_build(h, k)()
    fresh = lgb.Dataset(spec['X'], spec['y'], params=dict(LGB_PARAMS), feature_name=spec['feature_name'], free_raw_data=False)
    p0 = lgb.train(LGB_P, fresh, NROUND).predict(_sm_val(h, k))
    t1 = time.time()
    d1 = lgb_dataset(key, _sm_build(h, k))
    t_build = time.time() - t1
    t1 = time.time()
    d2 = lgb_dataset(key, _sm_build(h, k))
    t_read = time.time() - t1
    p1 = lgb.train(LGB_P, d1, NROUND).predict(_sm_val(h, k))
    p2 = lgb.train(LGB_P, d2, NROUND).predict(_sm_val(h, k))
    ok('LightGBM: свежий = построенный = прочитанный из save_binary (max |Δ| = 0)', np.array_equal(p0, p1) and np.array_equal(p0, p2),
       detal=f'строк {d2.num_data()}, признаков {d2.num_feature()}; построение {t_build:.2f} с, чтение {t_read:.2f} с')
    # 2) XGBoost: свежий QuantileDMatrix == из массивов кэша
    q0 = xgb.train(XGB_P, xgb.QuantileDMatrix(spec['X'], spec['y'], max_bin=255, nthread=1), NROUND).predict(xgb.DMatrix(_sm_val(h, k)))
    _XQ.clear()
    q1 = xgb.train(XGB_P, xgb_qdm(key, _sm_build(h, k)), NROUND).predict(xgb.DMatrix(_sm_val(h, k)))
    _XQ.clear()
    q2 = xgb.train(XGB_P, xgb_qdm(key, _sm_build(h, k)), NROUND).predict(xgb.DMatrix(_sm_val(h, k)))
    ok('XGBoost: свежий QuantileDMatrix = из кэша (построение и чтение с диска)', np.array_equal(q0, q1) and np.array_equal(q0, q2))
    # 3) один писатель: 10 процессов одновременно просят новый ключ — строит ровно один
    key_c = _sm_key(1, 3)
    res = run(_sm_train, [dict(h=1, step=3, kind='lgb', cost=1) for _ in range(a.workers)], workers=a.workers, label='один ключ × 10')
    nb = len((cache_root() / f'{key_c[:32]}.lgb.builds').read_text().splitlines())
    ok(f'один ключ из {a.workers} воркеров: построен {nb} раз, результаты одинаковы', nb == 1 and len({r['result'] for r in res if r['ok']}) == 1
       and all(r['ok'] for r in res))
    # 4) пул fork × 10: XGB-наборы построены в родителе (warm), все (h, шаг, вид) = последовательный счёт
    warm([('xgb', _sm_key(hh, kk), _sm_build(hh, kk), None) for hh in C.HS for kk in SM_STEPS])
    tasks = [dict(h=hh, step=kk, kind=kind, cost=hh) for hh in C.HS for kk in SM_STEPS for kind in ('lgb', 'xgb')]
    t1 = time.time()
    res = run(_sm_train, tasks, workers=a.workers, label='пул × 10')
    t_pool = time.time() - t1
    par = {(r['task']['h'], r['task']['step'], r['task']['kind']): r['result'] for r in res if r['ok']}
    seq = {(t['h'], t['step'], t['kind']): _sm_train(t) for t in tasks}
    ok(f'пул fork × {a.workers}: {len(tasks)} задач = последовательно, ошибок {sum(not r["ok"] for r in res)}', par == seq,
       detal=f'пул {t_pool:.1f} с, pid {len({r["pid"] for r in res})}')
    st = stats()
    ok('счётчики: XGB в воркерах взят из памяти родителя, LGB — из бинарных файлов', st['xgb_build'] >= 1 and st['lgb_hit'] >= 1,
       detal=json.dumps(st))
    T = pd.DataFrame(rows)
    out = C.S_DIR / 'smoke'
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / 'smoke_dataset_cache.csv', index=False)
    with pd.option_context('display.max_colwidth', 110, 'display.width', 220):
        print(T.to_string(index=False))
    C.log(f'дым r2_dataset_cache: {"ВСЕ OK" if T.ok.all() else "ЕСТЬ ОШИБКИ"} за {time.time() - t0:.0f} с -> {out / "smoke_dataset_cache.csv"}')
    return 0 if T.ok.all() else 1


def info(a):
    r = cache_root()
    fs = list(r.glob('*.lgb.bin')) + [p for d in r.glob('*.xgb') for p in d.glob('*.npy')]
    print(f'корень {r}: lgb {len(list(r.glob("*.lgb.bin")))}, xgb {len(list(r.glob("*.xgb")))}, объём {sum(f.stat().st_size for f in fs) / 2**20:.1f} МБ')
    return 0


def main():
    ap = argparse.ArgumentParser(description='R2 S: кэш наборов LightGBM/XGBoost')
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('smoke')
    p.add_argument('--workers', type=int, default=10)
    sp.add_parser('info')
    a = ap.parse_args()
    return smoke(a) if a.cmd == 'smoke' else info(a)


if __name__ == '__main__':
    sys.exit(main())
