#!/usr/bin/env python3
"""Тест «будущее не влияет» для сборки features_pit_v3 (m0_pit_lib.assemble, fill='asof').
Все исходные входы с датой > t заменяются шумом: Yahoo (сырые закрытия), макро (сырые значения), индексы MOEX (rv_moex: RV,
close, n_bars), свечи акций (open/high/low/close/volume, begin > t) и ISS rv_1850. Признаки v3 (feature_cols: 234 + 8 «дней
с обновления») на строках с date <= t обязаны совпасть с эталоном побайтно (tobytes). Три даты t.
Самопроверка: «подтекающая» сборка (внешние признаки следующего дня календаря) обязана упасть; v2 (медиана train <= 2017) тоже
обязана упасть на t до конца 2017 (медиана берёт значения после t).
Запуск: python3 tests/test_pit_v3_leak.py [--dates 2019-06-30,2021-12-31,2023-06-30] (код выхода 0 — всё в порядке)"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'analysis'))
import m0_pit_lib as L  # noqa: E402
from m0_common import calendar, load_prepared  # noqa: E402

DATES = ['2019-06-30', '2021-12-31', '2023-06-30']
ORIG = dict(yahoo_raw=L.yahoo_raw, macro_raw=L.macro_raw, rv_moex_pit=L.rv_moex_pit, load_candles=L.CutCalc.load_candles,
            asof_ext=L.asof_ext)


def _noise(df, mask, cols, rng):
    df = df.copy()
    for c in cols:
        v = df.loc[mask, c].astype(float).values
        df.loc[mask, c] = v * np.exp(rng.normal(0, 0.5, len(v))) + rng.normal(0, 1, len(v))
    return df


def patch(t, seed=20251025):
    """Подменяет загрузчики так, что все наблюдения с датой > t — шум."""
    rng = np.random.default_rng(seed)

    def yahoo_raw(c):
        r = ORIG['yahoo_raw'](c)
        return _noise(r, r.date > t, ['close'], rng)

    def macro_raw(c):
        r = ORIG['macro_raw'](c)
        return _noise(r, r.date > t, ['value'], rng)

    def rv_moex_pit(*a, **k):
        r = ORIG['rv_moex_pit'](*a, **k)
        return _noise(r, r.date > t, ['rv_daily', 'close', 'rv_annualized'], rng)

    def load_candles(self, ticker):
        import polars as pl
        d = ORIG['load_candles'](self, ticker)
        fut = d['begin'].dt.date() > t.date()
        n = len(d)
        k = pl.Series(np.exp(rng.normal(0, 0.3, n)))
        return d.with_columns([pl.when(pl.lit(fut)).then((pl.col(c) * k).cast(d[c].dtype)).otherwise(pl.col(c)).alias(c)
                               for c in ('open', 'high', 'low', 'close', 'volume') if c in d.columns])

    L.yahoo_raw, L.macro_raw, L.rv_moex_pit, L.CutCalc.load_candles = yahoo_raw, macro_raw, rv_moex_pit, load_candles


def unpatch():
    L.yahoo_raw, L.macro_raw, L.rv_moex_pit = ORIG['yahoo_raw'], ORIG['macro_raw'], ORIG['rv_moex_pit']
    L.CutCalc.load_candles = ORIG['load_candles']


def build(cal, prep, r18, fill):
    out, _, NEWF, *_ = L.assemble(cal, prep, r18, fill, log=lambda s: None)
    return out, NEWF


def noised_r18(r18, t, seed=7):
    rng = np.random.default_rng(seed)
    return _noise(r18, r18.date > t, ['rv', 'bv', 'j', 'rsv_pos', 'rsv_neg'], rng)


def compare(ref, new, cols, t):
    m = (pd.to_datetime(ref['date']) <= t).values
    bad = [c for c in cols if ref.loc[m, c].values.tobytes() != new.loc[m, c].values.tobytes()]
    return int(m.sum()), bad


def check(cal, prep, r18, dates, fill, log):
    ref, cols = build(cal, prep, r18, fill)
    ok = True
    for t in dates:
        t = pd.Timestamp(t)
        patch(t)
        try:
            new, _ = build(cal, prep, noised_r18(r18, t), fill)
        finally:
            unpatch()
        n, bad = compare(ref, new, cols, t)
        nf = int((pd.to_datetime(ref['date']) > t).sum())
        chg = sum(ref.loc[pd.to_datetime(ref['date']) > t, c].values.tobytes() != new.loc[pd.to_datetime(new['date']) > t, c].values.tobytes() for c in cols)
        log(f'fill={fill} t={t.date()}: строк<=t {n}, признаков {len(cols)}, расхождений до t {len(bad)} {bad[:5]}; '
            f'после t изменено признаков {chg} из {len(cols)} (строк {nf}) — шум дошёл')
        ok &= not bad and chg > 0
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', default=','.join(DATES))
    ap.add_argument('--no-selfcheck', action='store_true')
    a = ap.parse_args()
    cal, prep = calendar(), load_prepared()
    r18 = pd.read_parquet(L.RV1850)
    r18['date'] = pd.to_datetime(r18['date'])
    ok = check(cal, prep, r18, a.dates.split(','), 'asof', print)
    print('ТЕСТ v3:', 'PASS' if ok else 'FAIL', flush=True)
    if a.no_selfcheck:
        return 0 if ok else 1

    def leaky(ext, ext_cols, day_cols, dates):  # значение следующего дня календаря — заглядывание вперёд
        e = ext.copy()
        e[list(ext_cols)] = e[list(ext_cols)].shift(-1)
        return ORIG['asof_ext'](e, ext_cols, day_cols, dates)
    L.asof_ext = leaky
    caught1 = not check(cal, prep, r18, ['2019-06-30'], 'asof', lambda s: print('  [самопроверка сдвиг -1] ' + s))
    L.asof_ext = ORIG['asof_ext']
    caught2 = not check(cal, prep, r18, ['2016-06-30'], 'median', lambda s: print('  [самопроверка v2-медиана] ' + s))
    print('САМОПРОВЕРКА (утечка поймана): сдвиг -1', 'PASS' if caught1 else 'FAIL', '| медиана v2', 'PASS' if caught2 else 'FAIL')
    return 0 if ok and caught1 and caught2 else 1


if __name__ == '__main__':
    sys.exit(main())
