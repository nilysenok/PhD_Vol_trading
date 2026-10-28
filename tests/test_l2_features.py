#!/usr/bin/env python3
"""Уровень 2: тест «будущее не влияет» для каждого нового признака (Д3, Д4, Д6-добавки, Д7.1-Д7.4, Д8.12/Д7.5 HAR-J, Д8.6 h=1).
Все входы с датой > t (строки матрицы, свечи RVI с началом >= t 18:50, события календаря с known_from > t) заменяются шумом;
признаки на датах <= t обязаны совпасть с эталоном побайтно (NaN = NaN). Плюс самопроверка: признак с заглядыванием (сдвиг на -1) ловится.
python3 tests/test_l2_features.py   (код выхода 0 — всё в порядке)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
import l2_lib as L

DATES = ['2019-06-28', '2021-06-30', '2021-12-30']
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


def check_rows(name, fa, fb, mask, log):
    bad = [c for c in fa.columns if not same(fa.loc[mask, c].values, fb.loc[mask, c].values)]
    log.append((name, not bad, bad[:5]))


def by_rows(A, G):
    return pd.DataFrame({c: A['date'].map(G[c]).values for c in G.columns}, index=A.index) if not G.index.equals(A.index) else G


def main():
    A, FC = L.load_base()
    ev = L.load_calendar()
    s1850 = L.rvi_1850()
    c10 = pd.read_parquet(L.RVI10)
    log = []
    for t in DATES:
        T = pd.Timestamp(t)
        B = noisy(A, t)
        mask = (pd.to_datetime(A['date']) <= T).values
        for nm, f in (('Д3 cs', L.f_cs), ('Д4 cat', L.f_cat), ('Д7.1 agg h22', lambda X: L.f_agg(X, 22)), ('Д7.1 agg h5', lambda X: L.f_agg(X, 5)),
                      ('Д7.2 pc1', L.f_pc1), ('Д6 hadd h22', lambda X: L.f_hadd(X, 22)), ('Д6 hadd h5', lambda X: L.f_hadd(X, 5))):
            check_rows(f'{nm} t={t}', by_rows(A, f(A)), by_rows(B, f(B)), mask, log)
        # Д7.3: события с known_from > t сдвинуты случайно; строки матрицы после t — шум
        ev2 = {}
        for ty, (d, kf) in ev.items():
            d2 = d.copy(); fut = kf > np.datetime64(T)
            d2[fut] = d2[fut] - np.array(rng.integers(1, 60, fut.sum()), dtype='timedelta64[D]')
            ev2[ty] = (d2, kf)
        for h in (5, 22):
            check_rows(f'Д7.3 cal h{h} t={t}', by_rows(A, L.f_cal(A['date'], h, ev)), by_rows(B, L.f_cal(B['date'], h, ev2)), mask, log)
        # Д7.4: свечи RVI с началом >= t 18:50 — шум; строки матрицы после t — шум
        c2 = c10.copy(); fm = pd.to_datetime(c2['begin']) >= T + pd.Timedelta(hours=18, minutes=50)
        c2.loc[fm, 'close'] = c2.loc[fm, 'close'] * rng.uniform(0.2, 5, fm.sum())
        p2 = Path(L.L2DIR / '_rvi_noise.parquet'); c2.to_parquet(p2)
        check_rows(f'Д7.4 rvi t={t}', by_rows(A, L.f_rvi(A, s1850)), by_rows(B, L.f_rvi(B, L.rvi_1850(p2))), mask, log)
        p2.unlink()
    # Д10.1-Д10.4: шум после t в матрице, в ценах 18:50 и в свечах RVI; Markov — на двух тикерах
    c18 = L.close_1850()
    sub = A[A.ticker.isin(['SBER', 'LSRG'])].reset_index(drop=True)
    for t in DATES[1:]:
        T = pd.Timestamp(t); B = noisy(A, t); mask = (pd.to_datetime(A['date']) <= T).values
        cb = c18.copy(); fm = pd.to_datetime(cb['date']) > T; cb.loc[fm, 'close1850'] *= rng.uniform(0.5, 2, fm.sum())
        check_rows(f'Д10.1 state t={t}', by_rows(A, L.f_state(A, c18)), by_rows(B, L.f_state(B, cb)), mask, log)
        check_rows(f'Д10.3 rate t={t}', by_rows(A, L.f_rate(A)), by_rows(B, L.f_rate(B)), mask, log)
        check_rows(f'Д10.4 season t={t}', by_rows(A, L.f_season(A)), by_rows(B, L.f_season(B)), mask, log)
        ms_a = L.markov_filtered(sub); ms_b = L.markov_filtered(noisy(sub, t))
        m2 = sub[['date', 'ticker']].merge(ms_a, how='left', on=['date', 'ticker']).merge(ms_b, how='left', on=['date', 'ticker'], suffixes=('', '_b'))
        mm = (pd.to_datetime(m2['date']) <= T).values
        log.append((f'Д10.2 markov (SBER, LSRG) t={t}', same(m2.ms_p_high.values[mm], m2.ms_p_high_b.values[mm]), []))
        ra = L.f_regime(A, ms=ms_a, s1850=s1850); rb = L.f_regime(B, ms=ms_a, s1850=s1850)
        check_rows(f'Д10.2 regime (без Markov) t={t}', ra.drop(columns='ms_p_high'), rb.drop(columns='ms_p_high'), mask, log)
    # Д7.5/Д8.12: ln HAR-J шага — обучение только на окне шага; шум после конца обучающего окна (вкл. val/test) не меняет прогноз строк <= t
    for step, t in ((3, '2018-06-29'), (6, '2021-06-30')):
        for h in (5, 22):
            tp, vi, td, _, _ = L.split(A, h, step)
            B = noisy(A, t); tpb, vib, _, _, _ = L.split(B, h, step)
            hc = L.harj_cols(A)
            a1 = L.har_ln(tp, [vi], h, hc)[0]; b1 = L.har_ln(tpb, [vib], h, hc)[0]
            m = (pd.to_datetime(vi['date']) <= pd.Timestamp(t)).values
            log.append((f'Д8.12/Д7.5 har_ln h{h} шаг {step} t={t}', same(a1[m], b1[m]), []))
    # Д8.6: ln-прогноз h=1 на val шага — модель на окне шага; шум после t (в val-годе) не меняет строки <= t
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
    import l2_run as R
    for step, t in ((6, '2021-06-30'),):
        tg = 'Bv2b' if (L.REGEN / 'Bv2b_selected_h1.csv').exists() else 'Bv2'
        a = R.h1_oof(A, FC, tg, step); b = R.h1_oof(noisy(A, t), FC, tg, step)
        m = (pd.to_datetime(a['date']) <= pd.Timestamp(t)).values
        log.append((f'Д8.6 h1_ln шаг {step} t={t}', same(a.h1_ln.values[m], b.h1_ln.values[m]), []))
    # самопроверка: признак, заглядывающий на день вперёд, обязан упасть
    leak = lambda X: pd.DataFrame({'leak': X.groupby('ticker')['rv'].shift(-1)}, index=X.index)
    B = noisy(A, DATES[0]); mask = (pd.to_datetime(A['date']) <= pd.Timestamp(DATES[0])).values
    tmp = []; check_rows('самопроверка', leak(A), leak(B), mask, tmp)
    ok_self = not tmp[0][1]
    for n, ok, bad in log:
        print(('OK   ' if ok else 'FAIL ') + n + ('' if ok else f'  {bad}'))
    print(f'самопроверка (заглядывание на день): {"поймано" if ok_self else "НЕ ПОЙМАНО"}')
    fails = sum(not ok for _, ok, _ in log)
    print(f'ИТОГО: {len(log) - fails} из {len(log)} проверок пройдено')
    return 0 if fails == 0 and ok_self else 1


if __name__ == '__main__':
    sys.exit(main())
