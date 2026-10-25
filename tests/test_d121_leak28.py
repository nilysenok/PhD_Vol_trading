#!/usr/bin/env python3
"""Д12.1: тест «будущее не влияет» для features_pit_v3_28 (d121_features28.build) — те же подмены, что tests/test_pit_v3_leak.py:
все входы с датой > t заменяются шумом (Yahoo, макро, индексы MOEX, свечи акций begin > t, ISS rv_1850_noquar_28 вкл. rq).
Признаки (feature_cols v3 + rq, rq_d) на строках date <= t обязаны совпасть с эталоном побайтно — отдельно по 11 новым тикерам
и по 17 основным. Самопроверка: «подтекающий» rq_d (RQ следующей строки) обязан упасть.
Запуск: python3 tests/test_d121_leak28.py [--dates 2019-06-30,2023-06-30] (код выхода 0 — всё в порядке)"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts', 'analysis'))
sys.path.insert(0, HERE)
import d121_features28 as F  # noqa: E402
import test_pit_v3_leak as T  # noqa: E402
from m0_common import calendar, load_prepared  # noqa: E402

RV28 = F.MAIN / 'data/iss_check/rv_1850_noquar_28.parquet'


def build(cal, prep, rv_path, leak=False):
    out, NEWF = F.build(rv_path, cal, prep, log=lambda s: None)
    if leak:
        o = out.sort_values(['ticker', 'date'])
        out.loc[o.index, 'rq_d'] = o.groupby('ticker')['rq'].shift(-1).values
    return out, NEWF + ['rq', 'rq_d']


def noised_rv(t, tmp, seed=7):
    r = pd.read_parquet(RV28)
    d = pd.to_datetime(r['date'])
    r = T._noise(r, d > t, ['rv', 'bv', 'j', 'rsv_pos', 'rsv_neg', 'rq'], np.random.default_rng(seed))
    r.to_parquet(tmp, index=False)
    return tmp


def check(cal, prep, dates, tmp, leak=False, log=print):
    ref, cols = build(cal, prep, RV28, leak=leak)  # эталон той же (возможно, подтекающей) сборкой
    ok = True
    for t in dates:
        t = pd.Timestamp(t)
        T.patch(t)
        try:
            new, _ = build(cal, prep, noised_rv(t, tmp), leak=leak)
        finally:
            T.unpatch()
        for lab, g in (('11 новых', ~ref.is_main17), ('17 основных', ref.is_main17)):
            m = (g & (pd.to_datetime(ref['date']) <= t)).values
            bad = [c for c in cols if ref.loc[m, c].values.tobytes() != new.loc[m, c].values.tobytes()]
            f = (g & (pd.to_datetime(ref['date']) > t)).values
            chg = sum(ref.loc[f, c].values.tobytes() != new.loc[f, c].values.tobytes() for c in cols)
            log(f't={t.date()} {lab}: строк<=t {int(m.sum())}, признаков {len(cols)}, расхождений до t {len(bad)} {bad[:5]}; '
                f'после t изменено {chg} из {len(cols)} (шум дошёл)')
            ok &= (not bad) and chg > 0
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', default='2019-06-30,2023-06-30')
    a = ap.parse_args()
    cal, prep = calendar(), load_prepared()
    tmp = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'd121_rv28_noised.parquet')
    ok = check(cal, prep, a.dates.split(','), tmp)
    print('ТЕСТ 28:', 'PASS' if ok else 'FAIL', flush=True)
    caught = not check(cal, prep, ['2019-06-30'], tmp, leak=True, log=lambda s: print('  [самопроверка rq_d = RQ(t+1)] ' + s))
    print('САМОПРОВЕРКА (утечка поймана):', 'PASS' if caught else 'FAIL')
    return 0 if ok and caught else 1


if __name__ == '__main__':
    sys.exit(main())
