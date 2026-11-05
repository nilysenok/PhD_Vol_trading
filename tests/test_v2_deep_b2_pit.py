"""G3 B2: признак строки d побитно не зависит от баров после 18:50 d (синтетика и реальные бары 2019); мутация «день d+1» ловится в каждой группе."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import prov
from models_v2.s2 import features as ft
from models_v2.s5 import common
from models_v2.s5.b2 import feat, synth_b2

D = 560


def same(x: pd.DataFrame, y: pd.DataFrame) -> bool:
    return np.ascontiguousarray(x.to_numpy(float)).tobytes() == np.ascontiguousarray(y.to_numpy(float)).tobytes()


@pytest.fixture(scope="module")
def tables():
    a, b, _ = synth_b2.pair(700, 21)
    d = pd.Timestamp(ft.build(a).dates[D])
    fa, fb = ft.build(a), None
    base = feat.build(fa, a, b)
    a2, b2 = synth_b2.damage(a, d), synth_b2.damage(b, d)
    dmg = feat.build(ft.build(a2), a2, b2)
    return base, dmg


@pytest.mark.parametrize("g", list(feat.GROUPS))
def test_rows_up_to_d_bitwise_invariant(tables, g):
    base, dmg = tables
    cols = list(feat.GROUPS[g])
    assert same(base[cols].iloc[:D + 1], dmg[cols].iloc[:D + 1])
    assert not same(base[cols].iloc[D + 1:], dmg[cols].iloc[D + 1:])   # порча реально дошла до данных после d


@pytest.mark.parametrize("g", list(feat.GROUPS))
def test_mutation_group_with_day_d_plus_1_caught(tables, g):
    base, dmg = tables
    cols = list(feat.GROUPS[g])
    mb, md = base[cols].shift(-1), dmg[cols].shift(-1)    # признак группы берёт строку d+1 (мутация № 1)
    assert not same(mb.iloc[:D + 1], md.iloc[:D + 1])


def test_real_bars_rows_up_to_d_invariant_to_cut():
    bars, imoex = pd.read_parquet(prov.BARS), pd.read_parquet(common.IMOEX)
    d = pd.Timestamp("2019-06-03")
    lim = lambda x, t: x[x["begin"] < t]   # noqa: E731
    cut_d = pd.Timestamp("2019-06-03 18:50")
    full_end = pd.Timestamp("2020-01-01")
    A, B = lim(bars, cut_d), lim(imoex, cut_d)
    A2, B2 = lim(bars, full_end), lim(imoex, full_end)
    f1, f2 = ft.build(A), ft.build(A2)
    t1, t2 = feat.build(f1, A, B), feat.build(f2, A2, B2)
    assert pd.Timestamp(f1.dates[-1]) == d
    assert same(t1.loc[:d], t2.loc[:d])
