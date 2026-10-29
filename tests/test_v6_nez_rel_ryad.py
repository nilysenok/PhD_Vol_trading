"""Независимые тесты V6-1a (rel_ryad: синхронные бары, td_rel). Тестировщик, не автор.
Известные ответы посчитаны руками / циклом; сборка баров своя."""
import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850 as d18
from fastbt import rel_ryad as R


def mk(days=3, seed=0, p0=100.0, first="2020-05-04"):
    rng = np.random.default_rng(seed)
    ts = []
    for d in pd.bdate_range(first, periods=days):
        ts += list(pd.date_range(d + pd.Timedelta(hours=9, minutes=50), d + pd.Timedelta(hours=23, minutes=40), freq="10min"))
    n = len(ts)
    c = p0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.r_[p0, c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.0005, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.0005, n)))
    return pd.DataFrame({"timestamp": ts, "open": o, "high": hi, "low": lo, "close": c,
                         "volume": rng.integers(10, 1000, n)})


def _rel_loop(t, i):
    """Эталон циклом по общим timestamp."""
    idx = {r.timestamp: r for r in i.itertuples()}
    rows, b0 = [], None
    for r in t.sort_values("timestamp").itertuples():
        q = idx.get(r.timestamp)
        if q is None:
            continue
        if b0 is None:
            b0 = q.open
        o, c = r.open * b0 / q.open, r.close * b0 / q.close
        rows.append((r.timestamp, o, max(o, c, r.high * b0 / q.close), min(o, c, r.low * b0 / q.close), c, r.volume))
    return pd.DataFrame(rows, columns=R.BAR_COLS)


# ---------- известный ответ, руками ----------
def test_ruchnoy_raschet_dvuh_barov():
    ts = [pd.Timestamp("2020-05-04 10:00"), pd.Timestamp("2020-05-04 10:10")]
    idx = pd.DataFrame({"timestamp": ts, "open": [100.0, 110.0], "high": [111.0, 122.0],
                        "low": [99.0, 109.0], "close": [110.0, 121.0], "volume": [1, 1]})
    tk = pd.DataFrame({"timestamp": ts, "open": [50.0, 55.0], "high": [60.0, 66.0],
                       "low": [45.0, 50.0], "close": [55.0, 60.5], "volume": [7, 9]})
    b = R.sinhronnye_bary(tk, idx)
    assert b["open"].tolist() == pytest.approx([50.0, 50.0])
    assert b["close"].tolist() == pytest.approx([50.0, 50.0])
    assert b["high"].tolist() == pytest.approx([60 * 100 / 110, 66 * 100 / 121])
    assert b["low"].tolist() == pytest.approx([45 * 100 / 110, 50 * 100 / 121])
    assert b["volume"].tolist() == [7, 9]
    d = R.td_rel(tk, idx, "T")
    assert len(d) == 1 and d["ticker"][0] == "T"
    assert d["open"][0] == pytest.approx(50.0) and d["close"][0] == pytest.approx(50.0)
    assert d["high"][0] == pytest.approx(60 * 100 / 110) and d["low"][0] == pytest.approx(45 * 100 / 110)
    assert d["volume"][0] == 16


def test_protiv_cikla_sluchaynyj_ryad_s_dyrami():
    t, i = mk(4, 1), mk(4, 2, p0=3000.0)
    rng = np.random.default_rng(3)
    t = t.drop(rng.choice(len(t), 40, replace=False)).reset_index(drop=True)
    i = i.drop(rng.choice(len(i), 40, replace=False)).reset_index(drop=True)
    got = R.sinhronnye_bary(t, i)
    exp = _rel_loop(t, i)
    pd.testing.assert_frame_equal(got, exp, check_dtype=False, rtol=1e-12)
    assert len(got) == len(set(t.timestamp) & set(i.timestamp))


def test_logarifm_otnosheniya_s_konstantoy():
    """close_rel = P·B0/B -> ln close_rel − (ln P − ln B) = ln B0 на всех барах."""
    t, i = mk(3, 4), mk(3, 5, p0=777.0)
    b = R.sinhronnye_bary(t, i)
    diff = np.log(b["close"]) - (np.log(t["close"]) - np.log(i["close"]))
    assert np.allclose(diff, np.log(i["open"].iloc[0]), atol=1e-12)


# ---------- прямые тесты внутреннего объединения (вместо слабой мутации left-join) ----------
def test_vnutrennee_obedinenie_tochno_peresechenie_bez_nan():
    t, i = mk(2, 6), mk(2, 7)
    only_t = t.index[[0, 10, 11, 50]]
    only_i = i.index[[3, 4, 60]]
    t2, i2 = t.drop(only_i).reset_index(drop=True), i.drop(only_t).reset_index(drop=True)
    # пересечение: из t убраны бары only_i, из i убраны бары only_t
    got = R.sinhronnye_bary(t2, i2)
    common = sorted(set(t2.timestamp) & set(i2.timestamp))
    assert got["timestamp"].tolist() == common
    assert len(common) == len(t) - len(only_t) - len(only_i)
    assert not got.isna().any().any()
    assert set(t.timestamp[only_t]).isdisjoint(got.timestamp)
    assert set(t.timestamp[only_i]).isdisjoint(got.timestamp)


def test_bar_tolko_u_tikera_ne_popadaet_v_den_i_zakrytie_po_sinhronnomu():
    """Последний бар дня до 18:50 есть только у тикера: закрытие дня = последний ОБЩИЙ бар."""
    t, i = mk(1, 8), mk(1, 9, p0=500.0)
    last_before_cut = pd.Timestamp("2020-05-04 18:40")
    i2 = i[i.timestamp != last_before_cut].reset_index(drop=True)
    d = R.td_rel(t, i2, "T")
    ref = _rel_loop(t, i2)
    ref = ref[ref.timestamp < pd.Timestamp("2020-05-04 18:50")]
    assert d["close"][0] == pytest.approx(ref["close"].iloc[-1])
    assert ref["timestamp"].iloc[-1] == pd.Timestamp("2020-05-04 18:30")


def test_net_peresecheniya_i_pustye_vhody():
    t = mk(1, 10)
    i = mk(1, 11, first="2020-06-01")
    assert R.sinhronnye_bary(t, i).empty
    d = R.td_rel(t, i, "T")
    assert d.empty and list(d.columns) == d18.COLS
    e = t.iloc[:0]
    assert R.sinhronnye_bary(e, i).empty and R.td_rel(e, t, "T").empty
    assert R.td_rel(t, e, "T").empty


def test_odin_obshchiy_bar():
    t, i = mk(1, 12), mk(1, 13, p0=50.0)
    b = R.sinhronnye_bary(t.iloc[[5]], i.iloc[[5]])
    assert len(b) == 1 and b["open"][0] == pytest.approx(t.open.iloc[5])


# ---------- масштабы, константный индекс, выкл. ----------
def test_konstantnyj_indeks_daet_td_tikera_pobaytno():
    t = mk(3, 14)
    for v in (1.0, 37.0, 12345.0):
        idx = t.copy()
        idx[["open", "high", "low", "close"]] = v
        got = R.td_rel(t, idx, "T")
        exp = d18.aggregate(t, "T")
        pd.testing.assert_frame_equal(got, exp, check_exact=True)


def test_masshtab_indeksa_nevazhen_masshtab_tikera_lineen():
    t, i = mk(2, 15), mk(2, 16, p0=900.0)
    base = R.td_rel(t, i, "T")
    i7 = i.copy()
    i7[["open", "high", "low", "close"]] *= 7.0
    pd.testing.assert_frame_equal(R.td_rel(t, i7, "T"), base, rtol=1e-12)
    t3 = t.copy()
    t3[["open", "high", "low", "close"]] *= 3.0
    r3 = R.td_rel(t3, i, "T")
    for c in ("open", "high", "low", "close"):
        assert np.allclose(r3[c], 3.0 * base[c], rtol=1e-12)
    assert (r3["volume"] == base["volume"]).all()


def test_neuporyadochennyj_vvod_ravnyj_uporyadochennomu():
    t, i = mk(2, 17), mk(2, 18, p0=60.0)
    a = R.sinhronnye_bary(t, i)
    rng = np.random.default_rng(0)
    b = R.sinhronnye_bary(t.sample(frac=1, random_state=1), i.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(a, b)


# ---------- причинность ----------
def test_prichinnost_porcha_posle_K_ne_menyaet_dni_do_K():
    t, i = mk(5, 19), mk(5, 20, p0=400.0)
    base = R.td_rel(t, i, "T")
    K = pd.Timestamp("2020-05-06 12:00")     # середина третьего дня
    t2, i2 = t.copy(), i.copy()
    for x in (t2, i2):
        m = x.timestamp > K
        for c in ("open", "high", "low", "close"):
            x.loc[m, c] = x.loc[m, c] * 9.0 + 1.0
        x.loc[m, "volume"] = 10**6
    bad = R.td_rel(t2, i2, "T")
    # дни 04.05, 05.05 — целиком до K и строки не меняются
    pd.testing.assert_frame_equal(base.iloc[:2], bad.iloc[:2])
    assert not base.iloc[2].equals(bad.iloc[2])


def test_vechernie_bary_ne_vlijajut_na_den():
    t, i = mk(2, 21), mk(2, 22, p0=250.0)
    base = R.td_rel(t, i, "T")
    t2, i2 = t.copy(), i.copy()
    for x in (t2, i2):
        m = (x.timestamp.dt.hour * 60 + x.timestamp.dt.minute) >= 18 * 60 + 50
        for c in ("open", "high", "low", "close"):
            x.loc[m, c] = x.loc[m, c] * 5.0
    # ни вечерние бары тикера, ни индекса не меняют значение дня при неизменном B0
    pd.testing.assert_frame_equal(R.td_rel(t2, i2, "T").drop(columns="volume"), base.drop(columns="volume"))


# ---------- валидация ----------
def test_validatsiya_vhoda():
    t, i = mk(1, 23), mk(1, 24)
    bad = t.copy()
    bad.loc[3, "close"] = np.nan
    with pytest.raises(ValueError):
        R.sinhronnye_bary(bad, i)
    bad.loc[3, "close"] = 0.0
    with pytest.raises(ValueError):
        R.sinhronnye_bary(t, bad)
    with pytest.raises(ValueError):
        R.sinhronnye_bary(pd.concat([t, t.iloc[[0]]]), i)
    with pytest.raises(ValueError):
        R.sinhronnye_bary(t.drop(columns="volume"), i)
