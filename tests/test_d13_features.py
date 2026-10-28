#!/usr/bin/env python3
"""Д13: тест «будущее не влияет» для каждого нового признака до оценки (матрица features_pit_v3).
Входы с датой > t (строки матрицы; 10-минутные бары с началом >= t 18:50) заменяются шумом — признаки на датах <= t обязаны совпасть побайтно.
Д13.3 cs, Д13.4 cat — по матрице; Д13.7 intraday — по барам; Д13.1 каскад h=1 — по матрице (модель шага 3 и фолды по времени).
Самопроверка: признак со сдвигом −1 обязан упасть.
python3 tests/test_d13_features.py [cs|intra|casc ...]   (код выхода 0 — всё в порядке)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
import d13_lib as T
import l2_lib as L

DATES = ['2018-06-29', '2021-06-30', '2021-12-30']
rng = np.random.default_rng(0)


def noisy(A, t):
    B = A.copy()
    m = pd.to_datetime(B['date']) > pd.Timestamp(t)
    num = [c for c in B.columns if c not in ('date', 'ticker', 'year') and pd.api.types.is_numeric_dtype(B[c])]
    B.loc[m, num] = B.loc[m, num].values * rng.uniform(0.2, 5.0, size=(m.sum(), len(num)))
    return B


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


def main(which):
    log = []
    A, FC = T.load(1)
    if 'cs' in which:
        for t in DATES:
            B = noisy(A, t); m = (pd.to_datetime(A['date']) <= pd.Timestamp(t)).values
            for nm, f in (('Д13.3 cs', L.f_cs), ('Д13.4 cat', L.f_cat)):
                fa, fb = f(A), f(B)
                bad = [c for c in fa.columns if not same(fa.loc[m, c], fb.loc[m, c])]
                log.append((f'{nm} t={t}', not bad, bad[:5]))
    if 'intra' in which:
        tks = ['SBER', 'LSRG', 'AFLT']
        bars = {tk: pd.read_parquet(T.BARS / f'{tk}.parquet') for tk in tks}
        Ia = T.f_intraday(bars, tickers=tks)
        for t in DATES:
            cut = pd.Timestamp(t) + pd.Timedelta(hours=18, minutes=50)
            b2 = {}
            for tk, c in bars.items():
                c = c.copy(); fm = pd.to_datetime(c['begin']) >= cut
                for col in ('open', 'high', 'low', 'close'):
                    c.loc[fm, col] = c.loc[fm, col] * rng.uniform(0.5, 2, fm.sum())
                b2[tk] = c
            Ib = T.f_intraday(b2, tickers=tks)
            M = Ia.merge(Ib, on=['date', 'ticker'], suffixes=('', '_b'))
            mm = (pd.to_datetime(M['date']) <= pd.Timestamp(t)).values
            bad = [c for c in T.INTRA_COLS if not same(M.loc[mm, c], M.loc[mm, c + '_b'])]
            log.append((f'Д13.7 intraday t={t} (дат <= t: {mm.sum()})', not bad and len(Ia) == len(Ib), bad))
            # самопроверка внутри: признак дня t+1 (сдвиг −1) обязан измениться
            sa = Ia.sort_values(['ticker', 'date']).groupby('ticker')['rv_half2'].shift(-1)
            sb = Ib.sort_values(['ticker', 'date']).groupby('ticker')['rv_half2'].shift(-1)
            ma = (pd.to_datetime(Ia.sort_values(['ticker', 'date'])['date']) <= pd.Timestamp(t)).values
            log.append((f'самопроверка intraday сдвиг −1 t={t} поймана', not same(sa.values[ma], sb.values[ma]), []))
    if 'casc' in which:
        step, t = 3, '2018-06-29'
        B = noisy(A, t)
        jobs = [j for j in T.h1_cascade_jobs(A, FC, [step])]
        ca = pd.concat([T.h1_cascade_task(A, FC, *j) for j in jobs]); cb = pd.concat([T.h1_cascade_task(B, FC, *j) for j in jobs])
        M = ca.merge(cb, on=['date', 'ticker', 'step'], suffixes=('', '_b'))
        mm = (pd.to_datetime(M['date']) <= pd.Timestamp(t)).values
        log.append((f'Д13.1 каскад h1_ln шаг {step} t={t} (строк <= t: {mm.sum()})', same(M.h1_ln.values[mm], M.h1_ln_b.values[mm]), []))
    leak = lambda X: X.groupby('ticker')['rv'].shift(-1)
    t = DATES[0]; B = noisy(A, t); m = (pd.to_datetime(A['date']) <= pd.Timestamp(t)).values
    ok_self = not same(leak(A)[m], leak(B)[m])
    for n, ok, bad in log:
        print(('OK   ' if ok else 'FAIL ') + n + ('' if ok else f'  {bad}'))
    print(f'самопроверка (заглядывание на день): {"поймано" if ok_self else "НЕ ПОЙМАНО"}')
    fails = sum(not ok for _, ok, _ in log)
    print(f'ИТОГО: {len(log) - fails} из {len(log)} проверок пройдено')
    return 0 if fails == 0 and ok_self else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:] or ['cs', 'intra', 'casc']))
