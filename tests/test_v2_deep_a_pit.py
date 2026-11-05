"""A7: PIT — признаки, режимы, группы, ε и сдвиг масштаба не используют данные после d (главный тест обрезки + мутации № 1, 22)."""
import numpy as np
import pandas as pd

from models_v2 import prov
from models_v2.s2 import features as ft
from models_v2.s4 import mkt
from models_v2.s5 import groups, pit

D = np.datetime64("2019-06-03")
K = 400


def _bars(damage: bool) -> pd.DataFrame:
    b = pd.read_parquet(prov.BARS)
    b = b[(b["begin"] >= "2016-01-01") & (b["begin"] < "2020-01-01")].copy()
    if damage:
        late = b["begin"].dt.normalize() > pd.Timestamp(D)
        for c in ("open", "high", "low", "close"):
            b.loc[late, c] = b.loc[late, c] * 1.37
    return b


def _rows(builder):
    outs = []
    for dmg in (False, True):
        F = ft.build(_bars(dmg))
        i = int(np.searchsorted(F.dates, D))
        outs.append((builder(F, i), i))
    return outs


def _same(a, b) -> bool:
    return np.array_equal(a, b, equal_nan=True)


def test_priznaki_stroka_d_pobitno_ne_menyaetsya():
    (a, i), (b, _) = _rows(lambda F, i: np.column_stack([F.X, F.rv])[:i + 1])
    assert i > 300 and _same(a, b)


def test_mut_priznaki_plus_odin_den_lovitsya():
    (a, i), (b, _) = _rows(lambda F, i: np.roll(np.column_stack([F.X, F.rv]), -1, axis=0)[:i + 1])
    assert not _same(a, b)


def test_rynochnye_priznaki_pobitno_ne_menyayutsya():
    outs = []
    for dmg in (False, True):
        bars = _bars(dmg)
        F = ft.build(bars)
        M, _ = mkt.block(None, F.X[:, [0, 1, 2]], F.seg, F.dates, [mkt.paper(bars)], None)
        i = int(np.searchsorted(F.dates, D))
        outs.append(M.iloc[:i + 1].to_numpy())
    assert _same(*outs)


def test_kvantili_i_gruppy_ne_zavisyat_ot_budushchego():
    rng = np.random.default_rng(0)
    x = np.exp(rng.normal(size=800))
    y = x.copy()
    y[K + 1:] = 1e6
    a, b = pit.past_bucket(x, groups.INNER), pit.past_bucket(y, groups.INNER)
    assert _same(a[:K + 1], b[:K + 1])


def test_mut_kvantil_s_tekushchim_dnem_lovitsya():
    rng = np.random.default_rng(0)
    x = np.exp(rng.normal(size=800))
    ok = pit.past_bucket(x, groups.INNER)
    bad = pit.past_bucket(x, groups.INNER, include_current=True)
    assert not _same(ok, bad)   # окно с текущим днём меняет группы


def test_mut_gruppy_po_rv_celi_lovyatsya():
    rng = np.random.default_rng(1)
    rv_d = np.exp(rng.normal(size=800))
    rv_d2 = rv_d.copy()
    rv_d2[K + 1:] = 1e6   # порча после d = K
    assert _same(pit.past_bucket(rv_d, groups.INNER)[:K + 1], pit.past_bucket(rv_d2, groups.INNER)[:K + 1])
    rv_t, rv_t2 = np.roll(rv_d, -1), np.roll(rv_d2, -1)   # мутация: группа по RV цели d+1
    assert not _same(pit.past_bucket(rv_t, groups.INNER)[:K + 1], pit.past_bucket(rv_t2, groups.INNER)[:K + 1])


def test_sdvig_masshtaba_ne_zavisit_ot_budushchego_i_mut_ob_pare_d():
    rng = np.random.default_rng(2)
    n = 700
    d = pd.date_range("2018-01-01", periods=n).to_numpy()
    td = np.roll(d, -1)
    td[-1] = d[-1] + np.timedelta64(1, "D")
    rv, s2 = np.exp(rng.normal(size=n)), np.exp(rng.normal(size=n))
    rv2 = rv.copy()
    rv2[K:] = 1e6
    a, b = pit.scale_shift(rv, s2, d, td), pit.scale_shift(rv2, s2, d, td)
    assert _same(a[:K + 1], b[:K + 1])
    assert not _same(pit.scale_shift(rv, s2, d, td, extra_closed=1)[:K + 1], pit.scale_shift(rv2, s2, d, td, extra_closed=1)[:K + 1])
