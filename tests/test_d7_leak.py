#!/usr/bin/env python3
"""Тест «будущее не влияет» для блока Д7 (scripts/analysis/d7_block.py).
Все числовые входы строк с date > t заменяются шумом (бутстрап значений столбца до t, seed 20251025; маска NaN не сохраняется -
шум строже). Признаки d7_* на строках с date <= t обязаны совпасть с эталоном побайтно (tobytes) для h=5 и h=22.
Самопроверка: с «подтекающей» версией агрегата (центрированное окно) тест обязан упасть.
Запуск: python3 tests/test_d7_leak.py   (код выхода 0 - всё в порядке)"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'analysis'))
import d7_block as b

DATES = ['2019-06-30', '2021-12-31', '2023-06-30']


def noised(df, t, seed=20251025):
    rng = np.random.default_rng(seed)
    out = df.copy()
    past, fut = (df['date'] <= t).values, (df['date'] > t).values
    for c in df.columns:
        if c not in ('date', 'ticker') and pd.api.types.is_numeric_dtype(df[c]):
            pool = df.loc[past, c].values
            out.loc[fut, c] = rng.choice(pool, fut.sum())
    return out


def check(df, hs=(5, 22), dates=DATES, log=print):
    ok = True
    for h in hs:
        ref = b.build_d7(df, h)
        for t in dates:
            t = pd.Timestamp(t)
            new = b.build_d7(noised(df, t), h)
            m = (df['date'] <= t).values
            bad = [c for c in ref.columns if ref.loc[m, c].values.tobytes() != new.loc[m, c].values.tobytes()]
            log('h=%d t=%s: строк<=t %d, колонок %d, расхождений %d %s' % (h, t.date(), m.sum(), ref.shape[1], len(bad), bad[:5]))
            ok &= not bad
    return ok


def main():
    cols = ['date', 'ticker'] + sorted(set(b.KINDS) | {'imoex_low_bars'})
    df = pd.read_parquet(os.path.join(b.DATA, 'regen', 'features_pit_v2.parquet'), columns=cols)
    ok = check(df)
    print('ТЕСТ:', 'PASS' if ok else 'FAIL')
    orig = b._agg  # самопроверка: центрированное окно обязано ловиться
    def leaky(x, kind, w, name):
        r = orig(x, kind, w, name)
        r['d7_%s_leak' % name] = x.rolling(w, center=True, min_periods=1).mean()
        return r
    b._agg = leaky
    caught = not check(df, hs=(22,), dates=DATES[:1], log=lambda s: print('  [самопроверка] ' + s))
    b._agg = orig
    print('САМОПРОВЕРКА (утечка поймана):', 'PASS' if caught else 'FAIL')
    return 0 if ok and caught else 1


if __name__ == '__main__':
    sys.exit(main())
