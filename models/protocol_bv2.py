"""`models tune --protocol b_v2`: протокол B v2 (владелец 25.10, 18:15-18:25) — код tune_Bv2.py (ветка etap5-B) исполняется как есть:
последовательное отсечение ЛГК 100 -> 50 -> 25 -> 15 на шагах 1-6, подбор на 50% строк, max_bin 63/64, ES patience 30, отсечение на
итерации 100 (x1,2 к лучшему), шаги 7-9 без перебора, исключение признаков с нулевым gain после шага 1. Отличие только в выходах:
префикс Bv2pipe и каталог data/models_cache/bv2 (val) / test_locked/bv2 (test), чтобы не затирать data/regen/Bv2_* агента B.
Кэш: ключ = (версия tune_Bv2.py, содержимое матрицы признаков, h)."""
import importlib.util, sys, time
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
from . import cache as K, config as C

CANDIDATES = [C.MV / 'scripts/tune_Bv2.py',
              Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/.claude/worktrees/agent-afcfa1385cb335411/moex_volatility/scripts/tune_Bv2.py')]
OUT, LOCK, TAG = C.CACHE / 'bv2', C.CACHE / 'test_locked/bv2', 'Bv2pipe'


def script():
    for p in CANDIDATES:
        if p.exists():
            return p
    raise SystemExit('нет tune_Bv2.py (ветка etap5-B)')


def _module():
    p = script()
    if str(p.parent) not in sys.path:
        sys.path.insert(0, str(p.parent))
    if 'tune_Bv2' not in sys.modules:
        spec = importlib.util.spec_from_file_location('tune_Bv2', p)
        m = importlib.util.module_from_spec(spec)
        sys.modules['tune_Bv2'] = m
        spec.loader.exec_module(m)  # модуль грузит матрицу признаков при импорте (как в tune_Bv2.py)
    m = sys.modules['tune_Bv2']
    m.OUT, m.LOCK, m.TAG, m.EXC = OUT, LOCK, TAG, OUT / f'{TAG}_excluded_features.csv'
    return m


def init_worker():  # дочерние процессы spawn импортируют tune_Bv2 заново — те же подмены путей
    _module()


def run(h, workers, smoke=False):
    if smoke:
        raise SystemExit('--protocol b_v2: smoke не предусмотрен (шаги и размеры пачек зашиты в tune_Bv2.py); проверка — --bench скрипта')
    p = script()
    feat = C.MAIN / 'data/regen/features_pit_v2.parquet'
    k = K.key('bv2', code=C.blob_hash(p), feat=K.file_digest([feat]), h=h)
    sel = OUT / f'{TAG}_selected_h{h}.csv'
    t0 = time.time()
    if not K.done('bv2', k):
        OUT.mkdir(parents=True, exist_ok=True); LOCK.mkdir(parents=True, exist_ok=True)
        m = _module()
        with Pool(workers, initializer=init_worker) as P:
            m.run_h(h, P, [0], time.time())
        K.mark('bv2', k, h=h, selected=str(sel))
    s = pd.read_csv(sel)
    rows = []
    for mdl, g in s.groupby('model'):
        d = g[g.step <= C.DEV]
        rows.append(dict(h=h, модель=mdl, набор='разраб. 2016-2021', шагов=len(d), qlike_bv2=round(float(d.qlike_val.mean()), 4),
                         признаков=int(d.n_features.iloc[-1]) if 'n_features' in d else None))
    out = pd.DataFrame(rows)
    out['сек'] = round(time.time() - t0, 1)
    return out
