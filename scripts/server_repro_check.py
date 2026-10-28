#!/usr/bin/env python3
"""Проверка воспроизводимости первого раунда на сервере (до любого счёта R2).

Пересчитывает D12_h5, seed 42, val шагов 1–6, XGBoost и LightGBM (12 обучений) кодом первого раунда без изменений — тем же путём,
что контроль F_5 агента Б2 (`r2_roli_h5_h22.py f5`: d24_1.load_cfg -> l2_lib.step_data / make_ref / fit), — и сравнивает с рядами,
сохранёнными на Маке (`data/regen/D12_h5_{xgb,lgbm}_h5_val.parquet`, их SHA-256 записаны ниже). На Маке (Apple M4 Pro, arm64)
этот пересчёт дал max |Δ ln rv_pred| = 0 (XGB) и 1,8e-15 (LGBM) — `data/regen/R2/cache/B2/_control/F5.csv`.

Порог: max |Δ ln rv_pred| ≤ --tol (по умолчанию 1e-6) и совпадение строк по (date, ticker, step) — иначе выход 1.
Диагностика: SHA входов, best_iteration по шагам против Мака, квантили |Δ ln|, доля строк выше порога.
Что делать, если порог не держится, — docs/server_r2.md, раздел «Воспроизводимость».

Метрик качества нет (QLIKE не считается). Читаются только шаги ≤ 6 (годы ≤ 2021). test_locked не читается и не пишется.
Выход: data/regen/R2/repro/repro_D12_h5_s42_<host>_<время>.{csv,json} (забирается `scripts/server.sh pull`).

python3 scripts/server_repro_check.py --only-inputs          # только SHA входов и версии, без обучения (секунды)
python3 scripts/server_repro_check.py --workers 4            # полный пересчёт (12 обучений)
python3 scripts/server_repro_check.py --lgb-deterministic    # диагностика: LightGBM с deterministic=True (см. docs/server_r2.md)
"""
import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / 'analysis'
sys.path.insert(0, str(ANALYSIS))

MAC_ROOT = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
REGEN = MAC_ROOT / 'data' / 'regen'          # на сервере — ссылка на копию проекта (env/setup_server.sh)
OUT = REGEN / 'R2' / 'repro'
DEV = 6
K = ['date', 'ticker', 'step']

# SHA-256 входов на Маке (26.10.2026). Матрицы — FINAL_DESIGN §6; ряды — data/regen первого раунда.
SHA_INPUTS = {
    'features_pit_v3.parquet': '8c51bfa1eb6ee8624f7e9e0759c693152402a4c76b9e438d69a8f542aadfe0bc',
    'features_pit_v3.config.json': '557649804b910725a4b0597d526792b482be328a6124c213c6ecba132dcb14e1',
    'features_pit_v3_d7_h5.parquet': 'ce4aef543fb85e2b75497023df51e771dcf3973e03c6f6ee73a6e275b418398c',
    'D12_h5_selected_h5.csv': '7458aa714d39d73c1d6e85494d092f80df549e463bbab9de240c43e5973cd1ef',
    'D12_h5_xgb_h5_val.parquet': '2445ec18c318f56ad6e6ba2d0e4744af1e7f97dff45c2e1e86b7afc257e2c37a',
    'D12_h5_lgbm_h5_val.parquet': '6a041baa830335316b6cd6c6f3567b51a65254b76c11281fa4c9d32e96696373',
    'D12_h5_xgb_h5_wrap2_val.parquet': '9b8e8ff758cc2b228f7376d80b94512bca9eddea37b6692f8a98f48c132a3bb7',
}
# SHA-256 кода первого раунда, которым идёт пересчёт (ветка kanon-v22-pereschet, 26.10.2026)
SHA_CODE = {
    'l2_lib.py': '4de0e9f0b1cd8fe223c6f5429e2e1533c0028f4d294bcd08f3f0a14fe70dabf8',
    'd24_1.py': '8e8682eccae076e1b5f86cc4b05607397e4403990ce9385146ab044090b01789',
    'r2_roli_h5_h22.py': '8b684eccc4bb433b0ba016b8b68f09414aefae757ffd03b09b9ade8255b6e30b',
}
# best_iteration seed 42 на Маке (data/regen/R2/cache/B2/_trees_F5_D12_h5.csv; XGBoost — отсчёт от 0)
BI_MAC = {('xgb', 1): 2, ('xgb', 2): 171, ('xgb', 3): 51, ('xgb', 4): 46, ('xgb', 5): 21, ('xgb', 6): 47,
          ('lgbm', 1): 12, ('lgbm', 2): 144, ('lgbm', 3): 32, ('lgbm', 4): 38, ('lgbm', 5): 30, ('lgbm', 6): 22}
VERSIONS = {'numpy': '2.0.2', 'pandas': '2.3.3', 'scipy': '1.13.1', 'sklearn': '1.6.1', 'lightgbm': '4.6.0', 'xgboost': '2.1.4',
            'statsmodels': '0.14.6', 'pyarrow': '21.0.0'}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def cpu_name():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name') or line.startswith('Model'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def check_inputs():
    """-> (ok, rows): SHA входов и кода, версии пакетов, отсутствие test_locked на сервере."""
    rows, ok = [], True
    for f, want in SHA_INPUTS.items():
        p = REGEN / f
        got = sha(p) if p.exists() else 'нет файла'
        rows.append(dict(kind='data', name=f, want=want, got=got, ok=got == want))
    for f, want in SHA_CODE.items():
        got = sha(ANALYSIS / f)
        # r2_roli_h5_h22.py может обновиться в ветке Б2 — расхождение только печатается
        rows.append(dict(kind='code', name=f, want=want, got=got, ok=got == want, strict=f != 'r2_roli_h5_h22.py'))
    for m, want in VERSIONS.items():
        got = __import__(m).__version__
        rows.append(dict(kind='pkg', name=m, want=want, got=got, ok=got == want))
    pv = platform.python_version()
    rows.append(dict(kind='pkg', name='python', want='3.9.6', got=pv, ok=pv == '3.9.6'))
    locked = REGEN / 'test_locked'
    rows.append(dict(kind='guard', name='data/regen/test_locked отсутствует', want='нет', got='есть' if locked.exists() else 'нет',
                     ok=not locked.exists() or os.environ.get('PHD_R2_SERVER') != '1'))
    for r in rows:
        strict = r.get('strict', True)
        if not r['ok'] and strict:
            ok = False
        flag = 'OK' if r['ok'] else ('ОШИБКА' if strict else 'внимание')
        print(f"  {r['kind']:5s} {r['name']:36s} {flag:8s} {r['got'][:16]}{'…' if len(r['got']) > 16 else ''}")
    return ok, rows


def patch_lgb_deterministic():
    """Диагностика: LightGBM deterministic=True (вместе с уже стоящими num_threads=1, force_col_wise=True в l2_lib.fit)."""
    import lightgbm as lgb
    _train = lgb.train

    def train(params, *a, **kw):
        return _train(dict(params, deterministic=True), *a, **kw)

    lgb.train = train


def recompute(workers):
    import r2_roli_h5_h22 as B2
    spec = B2.Y['f5']
    n = B2.setup(spec)
    P = B2.points(spec)
    log(f'D12_h5: признаков {n}, обучений {len(P)} (seed 42, шаги 1–{DEV}), воркеров {workers}')
    V, tr = B2.fit_all(P, [42], workers)
    return B2, V, tr


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--only-inputs', action='store_true')
    ap.add_argument('--lgb-deterministic', action='store_true')
    a = ap.parse_args()

    host = socket.gethostname().split('.')[0]
    meta = dict(host=host, system=platform.system(), machine=platform.machine(), cpu=cpu_name(), python=platform.python_version(),
                threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                tol=a.tol, lgb_deterministic=a.lgb_deterministic, started=time.strftime('%F %T'))
    log(f'проверка воспроизводимости: {host}, {meta["system"]} {meta["machine"]}, {meta["cpu"]}')
    if not REGEN.exists():
        log(f'ОШИБКА: нет {REGEN} — на сервере нужна ссылка на копию проекта (scripts/server.sh setup)')
        sys.exit(1)
    log('входы, код, версии:')
    ok_in, rows_in = check_inputs()
    if not ok_in:
        log('ОШИБКА: входы/код/версии не совпадают с Маком — пересчёт не имеет смысла. Сначала scripts/server.sh push и setup.')
        sys.exit(1)
    if a.only_inputs:
        log('входы совпадают (--only-inputs: без обучения)')
        return

    import numpy as np
    import pandas as pd
    if a.lgb_deterministic:
        patch_lgb_deterministic()
    t0 = time.time()
    B2, V, tr = recompute(a.workers)
    meta['fit_seconds'] = round(time.time() - t0, 1)
    log(f'пересчёт за {meta["fit_seconds"]} с')

    def saved(f):
        d = pd.read_parquet(REGEN / f, filters=[('step', '<=', DEV)])  # годы ≥ 2022 (шаги 7–9) не читаются
        return B2.pick(d, seed=42)

    rows = []
    mine = {}
    for m in ('xgb', 'lgbm'):
        mine[m] = B2.pick(V, 42, m)
        rows.append((f'D12_h5_{m}', mine[m], saved(f'D12_h5_{m}_h5_val.parquet')))
    rows.append(('D12_h5_wrap2', B2.level_mean(mine['xgb'], mine['lgbm']), saved('D12_h5_xgb_h5_wrap2_val.parquet')))

    out = []
    for name, x, y in rows:
        mm = x[K + ['rv_pred']].merge(y[K + ['rv_pred']], on=K, suffixes=('_srv', '_mac'))
        dl = np.abs(np.log(mm.rv_pred_srv.values) - np.log(mm.rv_pred_mac.values))
        dv = np.abs(mm.rv_pred_srv.values - mm.rv_pred_mac.values)
        per_step = mm.assign(dl=dl).groupby('step').dl.max().to_dict()
        r = dict(ryad=name, rows_srv=len(x), rows_mac=len(y), rows_common=len(mm),
                 max_dln=float(dl.max()) if len(mm) else float('inf'), q99_dln=float(np.quantile(dl, 0.99)) if len(mm) else float('inf'),
                 max_drv=float(dv.max()) if len(mm) else float('inf'), share_gt_tol=float((dl > a.tol).mean()) if len(mm) else 1.0,
                 max_dln_by_step=json.dumps({int(k): float(v) for k, v in per_step.items()}))
        r['ok'] = bool(r['rows_srv'] == r['rows_mac'] == r['rows_common'] and r['max_dln'] <= a.tol)
        out.append(r)
    bi = []
    for _, t in tr.iterrows():
        k = (t.model, int(t.step))
        bi.append(dict(model=t.model, step=int(t.step), bi_srv=int(t.best_iteration), bi_mac=BI_MAC.get(k), same=int(t.best_iteration) == BI_MAC.get(k)))
    BI = pd.DataFrame(bi).sort_values(['model', 'step'])
    T = pd.DataFrame(out)

    ok = bool(T.ok.all())
    meta.update(ok=ok, finished=time.strftime('%F %T'), best_iteration_all_same=bool(BI.same.all()))
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f'repro_D12_h5_s42_{host}_{time.strftime("%Y%m%d_%H%M%S")}'
    T.to_csv(f'{stem}.csv', index=False)
    BI.to_csv(f'{stem}_trees.csv', index=False)
    Path(f'{stem}.json').write_text(json.dumps(dict(meta=meta, inputs=rows_in, result=out, trees=bi), ensure_ascii=False, indent=1))

    print(T.drop(columns=['max_dln_by_step']).to_string(index=False))
    print(BI.to_string(index=False))
    log(f'отчёт: {stem}.csv / .json')
    if ok:
        log(f'OK: сервер воспроизводит первый раунд (max |Δ ln rv_pred| ≤ {a.tol:g}). Можно запускать R2.')
        return
    worst = T.max_dln.max()
    if not BI.same.all():
        why = 'разошлось число деревьев (ранняя остановка) — расхождение по ветке обучения, а не только по округлению'
    elif worst < 1e-3:
        why = 'деревья те же, расхождение на уровне округления (SIMD/FMA/libm другой архитектуры)'
    else:
        why = 'деревья те же, но расхождение крупное — вероятно, другое разбиение (порог бина) на шаге с max_dln_by_step'
    log(f'НЕ ВОСПРОИЗВОДИТСЯ: max |Δ ln rv_pred| = {worst:.3g} > {a.tol:g}; {why}.')
    log('Счёт R2 не запускать. Порядок действий — docs/server_r2.md, раздел «Воспроизводимость»; решение — за владельцем.')
    sys.exit(1)


if __name__ == '__main__':
    main()
