"""V6-1a: rel_ryad — синхронные бары и td_rel. Синтетика; мутации ловятся тестами-проверками."""
import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850 as d18
from fastbt import rel_ryad as R


def _bars(days=4, seed=1, p0=100.0):
    rng = np.random.default_rng(seed)
    ts = []
    for d in pd.bdate_range("2021-03-01", periods=days):
        ts += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=50), d + pd.Timedelta(hours=23, minutes=40), freq="10min"))
    n = len(ts)
    c = p0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    o = np.r_[p0, c[:-1]]
    return pd.DataFrame({"timestamp": ts, "open": o, "high": np.maximum(o, c) * 1.0005,
                         "low": np.minimum(o, c) * 0.9995, "close": c,
                         "volume": rng.integers(10, 1000, n)})


def _scaled(b, k):
    x = b.copy()
    x[["open", "high", "low", "close"]] *= k
    return x


def _const(b, v):
    x = b.copy()
    x[["open", "high", "low", "close"]] = v
    return x


def test_konstanta_indeksa_daet_td_pobaytno():
    t = _bars()
    got = R.td_rel(t, _const(t, 7.0), "AAA")
    pd.testing.assert_frame_equal(got, d18.aggregate(t, "AAA"), check_exact=True)


def test_tiker_ravnyj_indeksu_ryad_ploskij_net_signala():
    t = _bars()
    got = R.td_rel(t, t, "AAA")
    for c in ("open", "close"):
        assert np.ptp(got[c]) < 1e-9


def test_tiker_proportsionalen_indeksu_ploskij():
    i = _bars(seed=2)
    got = R.td_rel(_scaled(i, 3.0), i, "AAA")
    assert np.allclose(got[["open", "close"]].to_numpy(), got["close"].iloc[0])


def test_vypadayut_bary_bez_pary():
    t, i = _bars(), _bars()
    i = i.drop(i.index[[5, 6, 100]])
    s = R.sinhronnye_bary(t, i)
    assert len(s) == len(t) - 3
    assert not set(t["timestamp"].iloc[[5, 6, 100]]) & set(s["timestamp"])


def test_pustoe_pereseche_nie_pustoj_kadr():
    t = _bars(1)
    i = t.copy()
    i["timestamp"] += pd.Timedelta(days=30)
    assert R.td_rel(t, i, "AAA").empty and R.sinhronnye_bary(t, i).empty


def test_pustaya_lenta():
    t = _bars(1)
    assert R.td_rel(t.iloc[0:0], t, "AAA").empty


def test_vechernie_bary_ne_vhodyat_v_den():
    t = _bars()
    ev = t[t["timestamp"].dt.hour >= 19]
    t2 = t.copy()
    t2.loc[ev.index, "close"] *= 5            # вечерние бары не должны менять day-close
    pd.testing.assert_frame_equal(R.td_rel(t, _const(t, 1.0), "A"), R.td_rel(t2, _const(t2, 1.0), "A"))
    assert R.td_rel(t, t, "A")["volume"].sum() == d18.aggregate(t, "A")["volume"].sum()


def test_obem_tikera_nulevoj_obem_ok():
    t = _bars()
    t["volume"] = 0
    assert (R.td_rel(t, _bars(seed=9), "A")["volume"] == 0).all()


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_plohaya_tsena_otkaz(bad):
    t, i = _bars(1), _bars(1)
    i.loc[3, "close"] = bad
    with pytest.raises(ValueError):
        R.td_rel(t, i, "A")


def test_dubl_i_net_kolonok_otkaz():
    t = _bars(1)
    with pytest.raises(ValueError):
        R.td_rel(pd.concat([t, t.iloc[:1]]), t, "A")
    with pytest.raises(ValueError):
        R.td_rel(t.drop(columns=["volume"]), t, "A")


def _bez_buduschego_ok(fn, t, i):
    """Значение дня d не меняется, если отрезать все бары позже d (в обоих рядах)."""
    full = fn(t, i, "A").set_index("date")
    days = full.index
    for d in days[:-1]:
        end = d + pd.Timedelta(days=1)
        cut = fn(t[t["timestamp"] < end], i[i["timestamp"] < end], "A").set_index("date")
        if not cut.loc[d].equals(full.loc[d]):
            return False
    return True


def test_znachenie_dnya_ne_zavisit_ot_budushchego():
    t, i = _bars(5, 3), _bars(5, 4)
    assert _bez_buduschego_ok(R.td_rel, t, i)


def test_mutatsiya_close_indeksa_d_plus_1_lovitsya():
    t, i = _bars(5, 3), _bars(5, 4)

    def mut(t, i, name):  # индекс берётся завтрашним (84 бара вперёд), где завтра есть во входе
        i2 = i.copy()
        for c in ("open", "close"):
            i2[c] = i[c].shift(-84).fillna(i[c])
        return R.td_rel(t, i2, name)
    assert not _bez_buduschego_ok(mut, t, i)
    assert _bez_buduschego_ok(R.td_rel, t, i)


def test_mutatsiya_sdvig_bara_indeksa_na_plyus_odin_lovitsya():
    t = _bars(3, 5)
    i = _scaled(t, 2.0)
    i["timestamp"] += pd.Timedelta(minutes=10)      # индекс опоздал на бар
    ok = R.td_rel(t, _scaled(t, 2.0), "A")
    assert np.ptp(ok["close"]) < 1e-9                 # без сдвига ряд плоский
    assert np.ptp(R.td_rel(t, i, "A")["close"]) > 1e-3  # со сдвигом тест это видит


def test_mutatsiya_vneshnee_obedinenie_lovitsya():
    t, i = _bars(2), _bars(2)
    i = i.drop(i.index[:20])
    ok = R.sinhronnye_bary(t, i)
    bad = t.merge(i[["timestamp", "close"]], on="timestamp", how="left")   # мутация: left-join
    assert len(ok) == len(t) - 20 and len(bad) == len(t) != len(ok)
