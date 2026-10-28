#!/usr/bin/env python3
"""R2, агент S: критерий фазы A «сервер принят» (протокол v3, раздел 6, строка S; models/designs/r2/r2_S_server.yaml, repro_A).

Повтор дизайнов R1 на сервере и сравнение с рядами Мака по QLIKE:
  (1) |QLIKE_сервер(Y) − QLIKE_Мак(Y)| / QLIKE_Мак(Y) ≤ 1e-4 для каждого ряда и каждого года Y ∈ 2017–2021 (val шагов 2–6);
  (2) тот же порядок рангов на наборе дизайнов R1 — по каждому году и по среднему QLIKE за 2017–2021.
Иначе — выход 1 и, по протоколу, весь R2 — на Маке. Проверка ln-прогнозов до 1e-6 — отдельно: scripts/server_repro_check.py.

Набор дизайнов R1 (h = 5, seed 42, val шагов 1–6; ранги — по годам 2017–2021):
  D12_h5_xgb, D12_h5_lgbm — финалист R1 D12_h5 (V9 + Д8.12 + Д7 + Huber), пересчёт тем же путём, что контроль F_5 агента Б2
      (r2_roli_h5_h22.setup/points/fit_all; код R1 без изменений); Мак — data/regen/D12_h5_{m}_h5_val.parquet;
  D12_h5_wrap2 — (RV_xgb + RV_lgbm)/2 (F_5 R1); Мак — D12_h5_xgb_h5_wrap2_val.parquet;
  HARJ_d0 — эталон (а) HAR-J d0 + эмбарго (ВЫР-1), пересчёт r2_core (load_d0 + train_mask + МНК на ln); Мак —
      data/regen/vyr/etalon_d0_h5_val.parquet, колонка har_raw.
QLIKE года — mean(y/p − ln(y/p) − 1) по (дата, тикер), rv_actual > 1e-12; y — rv_actual рядов Мака для обеих сторон
(отдельно сверяется, что rv_actual пересчёта совпадает с Маком). Годы ≥ 2022 не читаются (filters step ≤ 6).

python3 scripts/analysis/r2_repro_A.py --self            # Мак, дым: сохранённые ряды против самих себя + HAR-J d0 пересчётом (секунды)
python3 scripts/analysis/r2_repro_A.py --workers 4       # сервер: полный повтор (12 обучений бустинга + HAR-J)
Выход: data/regen/R2/S/repro_A.csv (+ repro_A_ranks.csv, repro_A.json; при --self — repro_A_self*.csv)."""
import os

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import r2_core as C  # noqa: E402

H = 5
STEPS = range(1, 7)
YEARS = (2017, 2018, 2019, 2020, 2021)
K = ['date', 'ticker', 'step']
TOL = float(C.CFG.get('repro_A', {}).get('qlike_rel_tol', 1e-4))
SAVED = {  # ряд -> (файл data/regen, колонка прогноза)
    'D12_h5_xgb': ('D12_h5_xgb_h5_val.parquet', 'rv_pred'),
    'D12_h5_lgbm': ('D12_h5_lgbm_h5_val.parquet', 'rv_pred'),
    'D12_h5_wrap2': ('D12_h5_xgb_h5_wrap2_val.parquet', 'rv_pred'),
    'HARJ_d0': ('vyr/etalon_d0_h5_val.parquet', 'har_raw'),
}
SHA_EXTRA = {'vyr/etalon_d0_h5_val.parquet': '9c437706bede8e03f1d91d2a2afcf4020d583f1aff1a1ad7df78da4370d54e5c'}


def saved(name):
    f, col = SAVED[name]
    d = pd.read_parquet(C.REGEN / f, filters=[('step', '<=', max(STEPS))])  # годы ≥ 2022 не читаются
    if 'seed' in d:
        d = d[d.seed == 42]
    if 'rank' in d:
        d = d[d['rank'] == 1]
    d = d[d.step.isin(list(STEPS))].rename(columns={col: 'rv_pred'})
    d['date'] = pd.to_datetime(d['date'])
    return d[K + ['year', 'rv_actual', 'rv_pred']].reset_index(drop=True)


def harj_d0():
    """Эталон (а) HAR-J d0 + эмбарго на всех 17 тикерах, шаги 1–6 (r2_core; детерминирован)."""
    A, harj = C.load_d0()
    V = C._smoke_series(A, harj, H, list(STEPS))
    V['date'] = pd.to_datetime(V['date'])
    return V


def boost_d12(workers):
    import r2_roli_h5_h22 as B2
    spec = B2.Y['f5']
    B2.setup(spec)
    V, tr = B2.fit_all(B2.points(spec), [42], workers)
    V['date'] = pd.to_datetime(V['date'])
    out = {f'D12_h5_{m}': B2.pick(V, 42, m) for m in ('xgb', 'lgbm')}
    out['D12_h5_wrap2'] = B2.level_mean(out['D12_h5_xgb'], out['D12_h5_lgbm'])
    return out


def qlike_years(V, Y):
    """QLIKE по годам: прогноз V на ключах Y, фактическая RV — Y.rv_actual (Мак)."""
    m = Y[K + ['year', 'rv_actual']].merge(V[K + ['rv_pred']], on=K, how='left')
    if m.rv_pred.isna().any():
        raise AssertionError(f'нет прогноза на {int(m.rv_pred.isna().sum())} строках Мака')
    m = m[m.rv_actual > 1e-12]
    r = m.rv_actual / m.rv_pred
    m = m.assign(q=r - np.log(r) - 1)
    return m.groupby('year').q.mean().reindex(YEARS)


def ranks(Q):
    """Q: {ряд: Series по годам} -> DataFrame ранги по годам и по среднему (1 — лучший)."""
    T = pd.DataFrame(Q).T
    R = T.rank(axis=0, method='average')
    R['mean_2017_2021'] = T.mean(axis=1).rank(method='average')
    return T, R


def compare(mine, ref, label):
    rows = []
    Qm, Qr = {}, {}
    for n in ref:
        Qr[n] = qlike_years(ref[n], ref[n])
        Qm[n] = qlike_years(mine[n], ref[n])
        a = mine[n][K + ['rv_actual']].merge(ref[n][K + ['rv_actual']], on=K, suffixes=('_s', '_m')) if 'rv_actual' in mine[n] else None
        same_y = bool(a is None or np.allclose(a.rv_actual_s, a.rv_actual_m, rtol=0, atol=0))
        for y in YEARS:
            rel = abs(Qm[n][y] - Qr[n][y]) / Qr[n][y]
            rows.append(dict(rezhim=label, ryad=n, god=y, qlike_mak=float(Qr[n][y]), qlike_server=float(Qm[n][y]), rel=float(rel),
                             rv_actual_sovpadaet=same_y, ok=bool(rel <= TOL and same_y)))
    Tm, Rm = ranks(Qm)
    Tr, Rr = ranks(Qr)
    rk = []
    for col in Rr.columns:
        om = list(Rm[col].sort_values(kind='mergesort').index)
        orf = list(Rr[col].sort_values(kind='mergesort').index)
        rk.append(dict(rezhim=label, srez=str(col), poryadok_mak=' < '.join(orf), poryadok_server=' < '.join(om), ok=bool(om == orf)))
    return pd.DataFrame(rows), pd.DataFrame(rk)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--self', dest='self_', action='store_true', help='дым на Маке: сохранённые ряды против самих себя (+ HAR-J пересчётом)')
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    meta = dict(host=socket.gethostname().split('.')[0], system=platform.system(), machine=platform.machine(),
                python=platform.python_version(), tol=TOL, rezhim='self' if a.self_ else 'server', started=time.strftime('%F %T'))
    C.log(f'repro_A ({meta["rezhim"]}): {meta["host"]} {meta["system"]} {meta["machine"]}, порог |ΔQLIKE|/QLIKE ≤ {TOL:g}')

    # входы: SHA матриц, рядов Мака, кода R1 и версии пакетов (как server_repro_check) + эталон d0
    import server_repro_check as SR
    ok_in, rows_in = SR.check_inputs()
    for f, want in SHA_EXTRA.items():
        got = C.sha_file(C.REGEN / f)
        rows_in.append(dict(kind='data', name=f, want=want, got=got, ok=got == want))
        ok_in &= got == want
    if not ok_in:
        C.log('ОШИБКА: входы/код/версии не совпадают с Маком — сравнение не имеет смысла')
        return 1

    ref = {n: saved(n) for n in SAVED}
    if a.self_:
        mine = {n: ref[n].copy() for n in SAVED}          # против самих себя: ожидается rel = 0 и тот же порядок
        mine['HARJ_d0'] = harj_d0()                        # детерминированный пересчёт (секунды) — проверка пути r2_core
    else:
        mine = boost_d12(a.workers)
        mine['HARJ_d0'] = harj_d0()
    meta['seconds'] = round(time.time() - t0, 1)
    Q, Rk = compare(mine, ref, meta['rezhim'])
    ok = bool(Q.ok.all() and Rk.ok.all())
    meta.update(ok=ok, max_rel=float(Q.rel.max()), finished=time.strftime('%F %T'))

    C.S_DIR.mkdir(parents=True, exist_ok=True)
    sfx = '_self' if a.self_ else ''
    Q.to_csv(C.S_DIR / f'repro_A{sfx}.csv', index=False)
    Rk.to_csv(C.S_DIR / f'repro_A{sfx}_ranks.csv', index=False)
    (C.S_DIR / f'repro_A{sfx}.json').write_text(json.dumps(dict(meta=meta, inputs=rows_in), ensure_ascii=False, indent=1, default=str))
    with pd.option_context('display.width', 200, 'display.max_colwidth', 80):
        print(Q.pivot(index='ryad', columns='god', values='rel').to_string(float_format=lambda x: f'{x:.1e}'))
        print(Rk[['srez', 'poryadok_mak', 'ok']].to_string(index=False))
    C.log(f'repro_A: max rel = {meta["max_rel"]:.2e}; порядок рангов {"совпал" if Rk.ok.all() else "НЕ совпал"} -> '
          f'{C.S_DIR / f"repro_A{sfx}.csv"}')
    if ok:
        C.log('OK: фаза A — сервер принят' if not a.self_ else 'OK: дым repro_A на Маке')
        return 0
    C.log('НЕ ПРИНЯТ: по протоколу (раздел 6, строка S) весь R2 — на Маке; решение — дирижёру/владельцу. Разбор — docs/server_r2.md, §5.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
