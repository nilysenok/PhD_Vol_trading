#!/usr/bin/env python3
"""Д10.1-Д10.3: тест «будущее не влияет». Всё с датой > t заменяется шумом (строки матрицы, цены 18:50, ряд RVI, события ЦБ с known_from > t сдвинуты
вперёд, свечи дня t после 18:50); признаки на датах <= t (включая вероятность марковской модели, пересчитанную целиком) совпадают побайтно.
Самопроверка: признак со сдвигом на -1 (заглядывание) ловится. python3 tests/test_d10_block.py -> код выхода 0 = всё в порядке"""
import sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
import d10_block as B

DATES = ['2019-06-28', '2020-12-31', '2021-06-30', '2022-03-15']
rng = np.random.default_rng(0)


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def main():
    F = pd.read_parquet(B.DATA / 'regen/features_pit_v2.parquet')
    P0, R0, C0 = B.prices_1850(), B.rvi_1850(), pd.read_csv(B.DATA / 'calendar/cbr.csv')
    base = B.build_d10(F, P0, R0, C0, workers=1)
    log = []
    for t in DATES:
        T = pd.Timestamp(t)
        G = F.copy(); m = G['date'] > T
        num = [c for c in G.columns if c not in ('date', 'ticker') and pd.api.types.is_numeric_dtype(G[c])]
        G.loc[m, num] = G.loc[m, num].values * rng.uniform(0.2, 5.0, size=(m.sum(), len(num)))
        P = P0.copy(); fm = P.index > T; P[fm] = P[fm] * rng.uniform(0.5, 2, size=(fm.sum(), P.shape[1]))
        R = R0.copy(); R[R.index > T] *= rng.uniform(0.2, 5, (R.index > T).sum())
        C = C0.copy(); k = pd.to_datetime(C['known_from']) > T
        C.loc[k, 'date'] = (pd.to_datetime(C.loc[k, 'date']) + pd.to_timedelta(rng.integers(1, 60, k.sum()), 'D')).dt.strftime('%Y-%m-%d')
        alt = B.build_d10(G, P, R, C, workers=1)
        mk = (F['date'] <= T).values
        bad = [c for c in base.columns if not same(base.loc[mk, c].values, alt.loc[mk, c].values)]
        log.append((f'build_d10 (все 23 признака, Markov пересчитан) t={t}', not bad, bad))
        # срез дня t: свечи после 18:50 дня t (RVI, IMOEX, SBER) не влияют на закрытие 18:50
        for nm, f in (('RVI', B.RVI10), ('IMOEX', B.CAND / 'external/IMOEX.parquet'), ('SBER', B.CAND / 'stocks/SBER.parquet')):
            c = pd.read_parquet(f); ts = pd.to_datetime(c['timestamp'] if 'timestamp' in c else c['begin'])
            mm = ts >= T + pd.Timedelta(hours=18, minutes=50); c2 = c.copy(); c2.loc[mm, 'close'] = c2.loc[mm, 'close'] * rng.uniform(0.2, 5, mm.sum())
            with tempfile.TemporaryDirectory() as d:
                c2.to_parquet(Path(d) / 'x.parquet')
                a, b = B._cut_last(f), B._cut_last(Path(d) / 'x.parquet')
            log.append((f'срез 18:50 {nm} t={t}', same(a[a.index <= T].values, b[b.index <= T].values), []))
    # самопроверка: признак с заглядыванием (rv_IMOEX следующего дня) должен ловиться тем же сравнением
    t = pd.Timestamp(DATES[2]); G = F.copy(); m = G['date'] > t
    G.loc[m, 'rv_IMOEX'] = G.loc[m, 'rv_IMOEX'] * 3.0
    leak = lambda X: B._by_date(X, ['rv_IMOEX'])['rv_IMOEX'].shift(-1)
    la, lb = leak(F), leak(G)
    log.append(('самопроверка: сдвиг -1 ловится', not same(la[la.index <= t].values, lb[lb.index <= t].values), []))
    for nm, ok, bad in log:
        print(('OK   ' if ok else 'FAIL ') + nm + (f' {bad}' if bad else ''))
    sys.exit(0 if all(o for _, o, _ in log) else 1)


if __name__ == '__main__':
    main()
