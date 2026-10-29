"""П5 (Ж-125 п. 3): ось tf для S1–S6 v5 — fastbt/tf_os_v5.py. Только синтетика, без сети и данных.

Мутации: сигнал на незакрытом баре (бар t включает будущие минуты; mutations.md № 1 — сдвиг входа),
исполнение на баре сигнала (Д2), бар через ночь (агрегат склеивает вечер d и утро d+1), комиссия закрытия
в конце дня не взята (mutations.md № 5 — комиссия на границе).
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import bars_agg as ba
from fastbt import tf_os_v5 as T
from fastbt.indicators import compute_base
from fastbt.signals import v5_s56
from fastbt.signals.interface import TickerData, get_family

GRIDS = {
    "S1": [dict(ma_tip="sma", ma_window=20, dev_tip="std", z_entry=0.5, z_exit=0.0),
           dict(ma_tip="ema", ma_window=10, dev_tip="atr", z_entry=1.0, z_exit="net")],
    "S2": [dict(ma_tip="sma", bb_window=20, bb_std=1.0, podtv="none"),
           dict(ma_tip="sma", bb_window=10, bb_std=1.5, podtv="rsi30")],
    "S3": [dict(okno=20, podtv="none"), dict(okno=10, podtv="volume")],
    "S4": [dict(period=14, mnozh=2.0), dict(period=5, mnozh=1.0)],
    "S5": [dict(pivot_type=t, pivot_period=p, uroven=1, atr_filtr=0, filter_set="net")
           for t in ("classic", "camarilla") for p in ("den", "nedelya")],
    "S6": [dict(vwap_window=20, dev_mult=0.5, dir_bias="OFF", filter_set="net"),
           dict(vwap_window=5, dev_mult=1.0, dir_bias="ON", filter_set="net")],
}


def make_b10(days=60, seed=0):
    rng = np.random.default_rng(seed)
    n = days * ba.SLOTS
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    o = np.r_[100.0, c[:-1]] * np.exp(rng.normal(0, 0.001, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.002, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.002, n))
    d = pd.bdate_range("2018-01-01", periods=days)
    return pd.DataFrame(dict(date=np.repeat(d, ba.SLOTS), slot=np.tile(np.arange(ba.SLOTS), days),
                             open=o, high=h, low=l, close=c, volume=rng.uniform(1e3, 5e3, n)))


def make_td_den(n=700, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = c * np.exp(rng.normal(0, 0.006, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.012, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.012, n))
    d = pd.bdate_range("2018-01-01", periods=n).values.astype("datetime64[ns]")
    return TickerData("SYN", d, o, h, l, c, rng.uniform(1e5, 5e5, n), np.r_[np.log(c[1:] / c[:-1]), 0.0])


def _same(a, b):
    return all(np.array_equal(getattr(a, k), getattr(b, k), equal_nan=True)
               for k in ("sig", "exit_l", "exit_s", "strength")) and \
        all(np.array_equal(a.extra[k], b.extra[k], equal_nan=True) for k in a.extra)


@pytest.mark.parametrize("sid", list(GRIDS))
def test_den_pobajtno_semya_bez_osi(sid):
    td = make_td_den()
    ind = compute_base(td.close, td.high, td.low, td.open, td.volume)
    assert _same(T.signaly(sid, td, GRIDS[sid], "den"), get_family(sid, "v5")(td, ind, GRIDS[sid]))


@pytest.mark.parametrize("tip", ["classic", "fibonacci", "woodie", "camarilla", "demark"])
def test_den_pobajtno_gruppa_den_kak_nomer_bara(tip):
    """До П5 группа den = np.arange(n): на дневках новая группа (календарный день) обязана дать то же побайтно."""
    td = make_td_den()
    a = v5_s56.pivots(np.arange(td.n), td.open, td.high, td.low, td.close, tip)
    b = v5_s56.pivots(v5_s56._group_ids(td.dates, "den"), td.open, td.high, td.low, td.close, tip)
    assert all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))


def test_den_pobajtno_bary():
    b10 = make_b10(5)
    f = T.bary(b10, "den")
    pd.testing.assert_frame_equal(f.drop(columns="ts"), ba.daily(b10))
    assert (f["ts"] == f["date"]).all()


def test_agregaciya_1h_izvestnyj_otvet():
    """2 дня 10-минуток 09:50…19:00: 09:50 и ≥ 18:50 отброшены; 1 ч = 9 баров в день, последний 18:00–18:50 (5 слотов)."""
    ts = [pd.Timestamp(d) + pd.Timedelta(minutes=m) for d in ("2018-03-01", "2018-03-02")
          for m in range(9 * 60 + 50, 19 * 60 + 10, 10)]
    mod = np.array([t.hour * 60 + t.minute for t in ts], float)
    raw = pd.DataFrame(dict(timestamp=ts, open=mod, high=mod + 1, low=mod - 1, close=mod + 0.5, volume=1.0))
    f = T.bary(ba.prep_10m(raw), "1h")
    assert len(f) == 18 and (f.groupby("date").size() == 9).all()
    for d, g in f.groupby("date"):
        start = 600 + 60 * np.arange(9)
        endm = np.minimum(start + 50, 18 * 60 + 40)
        assert np.array_equal(g["open"], start) and np.array_equal(g["close"], endm + 0.5)
        assert np.array_equal(g["high"], endm + 1) and np.array_equal(g["low"], start - 1)
        assert np.array_equal(g["volume"], [6.0] * 8 + [5.0])
        assert list(g["ts"]) == [d + pd.Timedelta(minutes=int(m)) for m in start]


def _bary_s_budushchim(b10, tf):
    """Мутация: close бара t = close первой 10-минутки бара t+1 (бар включает будущие минуты)."""
    k, f = ba.K[tf], T.bary(b10, tf)
    nxt = b10.set_index(["date", "slot"])["close"]
    key = pd.MultiIndex.from_arrays([f["date"], (f["slot"] + 1) * k])
    return f.assign(close=nxt.reindex(key).fillna(pd.Series(f["close"].values, index=key)).values)


def _zavisit_ot_budushchego(build, tf="1h", sids=("S1", "S6")):
    """Возмущаем 10-минутки после конца бара t; True, если сигналы на барах ≤ t изменились хоть в одном срезе."""
    b10 = make_b10(60)
    k = ba.K[tf]
    for day in (40, 47, 55):
        for j in (1, 4, 7):
            d = b10["date"].unique()[day]
            fut = (b10["date"] > d) | ((b10["date"] == d) & (b10["slot"] >= (j + 1) * k))
            b2 = b10.copy()
            b2.loc[fut, ["open", "high", "low", "close"]] *= 1.07
            f1, f2 = build(b10, tf), build(b2, tf)
            t = int(np.flatnonzero((f1["date"] == d).values)[j])
            for sid in sids:
                s1 = T.signaly(sid, T.ticker_data(f1, "X"), GRIDS[sid], tf)
                s2 = T.signaly(sid, T.ticker_data(f2, "X"), GRIDS[sid], tf)
                for a, b in ((s1.sig, s2.sig), (s1.exit_l, s2.exit_l), (s1.strength, s2.strength)):
                    if not np.array_equal(a[:, : t + 1], b[:, : t + 1], equal_nan=True):
                        return True
    return False


def test_nezakrytyj_bar_mutaciya_lovitsya():
    assert not _zavisit_ot_budushchego(T.bary)
    assert _zavisit_ot_budushchego(_bary_s_budushchim)


def _pozicii_na_bare_signala(sig, ex_l, ex_s, den):
    """Мутация Д2: решение по close t применяется уже на баре t."""
    return np.r_[T.pozicii(sig, ex_l, ex_s, den)[1:], 0].astype(np.int8)


def test_ispolnenie_na_bare_signala_lovitsya():
    o = np.array([10.0, 11, 12, 13, 14, 15]); c = o + 0.5
    den = np.zeros(6, np.int64); sig = np.array([0, 1, 0, 0, 0, 0], np.int8); no = np.zeros(6, bool)
    ozhid = (13 / 12 - 1) + (14 / 13 - 1) + (15 / 14 - 1) + (15.5 / 15 - 1)    # вход по open бара 2
    for fn, verno in ((T.pozicii, True), (_pozicii_na_bare_signala, False)):
        g, _, _ = T.dohod(fn(sig, no, no, den), o, c, den)
        assert bool(np.isclose(g.sum(), ozhid)) is verno
    rng = np.random.default_rng(1)
    den2 = np.repeat(np.arange(30), 9)
    s = rng.choice(np.array([-1, 0, 1], np.int8), len(den2))
    e = rng.random(len(den2)) < 0.3
    a = T.pozicii(s, e, e, den2)
    for t in (50, 100, 200):                       # не последние бары дня: решение t исполняется на t+1
        s2 = s.copy(); s2[t] = -a[t + 1] if a[t + 1] else 1     # решение на t — другое
        b = T.pozicii(s2, e, e, den2)
        assert a[t + 1] != b[t + 1]
        assert np.array_equal(a[: t + 1], b[: t + 1])
        m1, m2 = _pozicii_na_bare_signala(s, e, e, den2), _pozicii_na_bare_signala(s2, e, e, den2)
        assert not np.array_equal(m1[: t + 1], m2[: t + 1])


def test_signal_na_poslednem_bare_dnya_ne_ispolnyaetsya_utrom():
    den = np.repeat([0, 1], 3); sig = np.array([0, 0, 1, 0, 0, 0], np.int8); no = np.zeros(6, bool)
    assert T.pozicii(sig, no, no, den).tolist() == [0] * 6
    allp = np.ones(6, np.int8)
    assert T.pozicii(allp, no, no, den).tolist() == [0, 1, 1, 0, 1, 1]


def _bary_cherez_noch(b10, tf):
    """Мутация: укрупнение по сквозному номеру 10-минутки — бар склеивает вечер d и утро d+1."""
    k = ba.K[tf]
    b = b10.sort_values(["date", "slot"]).reset_index(drop=True)
    g = b.assign(g=np.arange(len(b)) // k).groupby("g").agg({"date": "first", **ba._AGG}).reset_index(drop=True)
    g["slot"] = g.groupby("date").cumcount()
    return g.assign(ts=g["date"] + pd.to_timedelta(ba.OPEN_MIN + g["slot"] * ba.STEP * k, unit="m"))


@pytest.mark.parametrize("tf", ["30m", "1h", "2h"])
def test_bar_cherez_noch_mutaciya_lovitsya(tf):
    b10 = make_b10(6)
    T.proverka_sessii(b10, T.bary(b10, tf))
    with pytest.raises(ValueError):
        T.proverka_sessii(b10, _bary_cherez_noch(b10, tf))


def test_s5_pivot_proshlogo_dnya_i_nedeli():
    b10 = make_b10(30)
    f = T.bary(b10, "1h")
    td = T.ticker_data(f, "X")
    ss = T.signaly("S5", td, GRIDS["S5"], "1h")
    dd, wk = ba.daily(b10), ba.weekly(ba.daily(b10))
    p_den = ((dd["high"] + dd["low"] + dd["close"]) / 3).shift(1).values
    i_den = np.searchsorted(dd["date"].values, f["date"].values)
    assert np.allclose(ss.extra["P"][0], p_den[i_den], equal_nan=True)            # classic, den
    p_wk = ((wk["high"] + wk["low"] + wk["close"]) / 3).shift(1).values
    i_wk = np.searchsorted(wk["date"].values, f["date"].values)
    assert np.allclose(ss.extra["P"][1], p_wk[i_wk], equal_nan=True)              # classic, nedelya


def test_s6_vwap_s_nachala_sessii():
    b10 = make_b10(8)
    f = T.bary(b10, "30m")
    ss = T.signaly("S6", T.ticker_data(f, "X"), GRIDS["S6"], "30m")
    tp = (f["high"] + f["low"] + f["close"]) / 3
    ozhid = (tp * f["volume"]).groupby(f["date"]).cumsum() / f["volume"].groupby(f["date"]).cumsum()
    assert np.allclose(ss.extra["vwap"][0], ozhid.values, rtol=1e-12)
    first = (f["date"] != f["date"].shift()).values
    assert np.allclose(ss.extra["vwap"][0][first], tp.values[first], rtol=1e-12)


def _dohod_bez_zakrytiya(pos, o, c, den, komissiya=T.KOMISSIYA):
    """Мутация (комиссия на границе): закрытие в конце дня бесплатно."""
    g, _, turn = T.dohod(pos, o, c, den, komissiya)
    last = np.r_[den[1:] != den[:-1], True]
    turn = turn - np.where(last, np.abs(pos), 0)
    return g, turn * komissiya, turn


def test_izderzhki_kazhdaya_storona_i_sdelki_v_den():
    den = np.repeat([0, 1], [5, 3]); o = np.full(8, 100.0); c = o.copy()
    pos = np.array([0, 1, 1, -1, -1, 0, 1, 1], np.int8)
    _, cst, turn = T.dohod(pos, o, c, den, 0.0004)
    assert turn.tolist() == [0, 1, 0, 2, 1, 0, 1, 1] and np.isclose(cst.sum(), 6 * 0.0004)
    assert not np.isclose(_dohod_bez_zakrytiya(pos, o, c, den, 0.0004)[1].sum(), 6 * 0.0004)
    assert T.sdelki(pos, den) == 3
    r = T.stroka_izderzhek("S1", "1h", {}, [(0.0, float(cst.sum()), 3, 2)], 1.0, 0.0004)
    assert r["sdelok_v_den"] == 1.5 and np.isclose(r["izderzhki_pct_god"], 0.24)
    assert np.isclose(r["netto_pct_god"], -0.24) and r["komissiya_storona_pct"] == 0.04


def test_izderzhki_progon_stroki_i_den_otkaz():
    tds = {"X": T.ticker_data(T.bary(make_b10(40), "1h"), "X")}
    rows = T.progon("S1", "1h", tds, GRIDS["S1"], "2018-01-15", "2018-03-01")
    assert len(rows) == 2 and all(r["sdelok"] > 0 and r["izderzhki_pct_god"] > 0 for r in rows)
    assert all(np.isclose(r["netto_pct_god"], r["brutto_pct_god"] - r["izderzhki_pct_god"]) for r in rows)
    with pytest.raises(ValueError):
        T.progon("S1", "den", tds, GRIDS["S1"], "2018-01-15", "2018-03-01")
