"""Л6.1 (Ж-146, п. 6): состояние дня на T = 17:00 / 18:00 / 18:40 и цена исполнения в T. Синтетика с известным ответом.

G1 (выкл.): T = 18:50 == daily_1850.aggregate побайтно; G2/G3 — мутации «бар >= T в индикаторе», «цена раньше T»;
G4 — день без бара T даёт NaN с флагом, без ffill (Д4). Мутация «d+1» — состояние соседнего дня не равно ожидаемому."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850 as D
from fastbt import t_reshenie_v7 as R

DAYS = [pd.Timestamp("2019-03-04"), pd.Timestamp("2019-03-05"), pd.Timestamp("2019-03-06")]
TS = {"17:00": 17 * 60, "18:00": 18 * 60, "18:40": 18 * 60 + 40}


def _px(k: int, mod: int) -> float:
    return 1000.0 * (k + 1) + (mod - 590) / 10.0          # open бара: день k, минута mod


def bars(drop: dict | None = None) -> pd.DataFrame:
    """Бары 09:50..19:00 каждые 10 мин; open=_px, close=open+0.5, high=close+0.25, low=open-0.25, volume=10."""
    rows = []
    for k, d in enumerate(DAYS):
        for mod in range(590, 19 * 60 + 10, 10):
            if drop and mod in drop.get(k, ()):
                continue
            o = _px(k, mod)
            rows.append((d + pd.Timedelta(minutes=mod), o, o + 0.75, o - 0.25, o + 0.5, 10))
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def expected_state(k: int, tmin: int) -> dict:
    last = max(m for m in range(590, tmin, 10))
    return {"open": _px(k, 590), "high": _px(k, last) + 0.75, "low": _px(k, 590) - 0.25,
            "close": _px(k, last) + 0.5, "volume": 10 * len(range(590, tmin, 10))}


@pytest.mark.parametrize("T", list(TS))
def test_sostoyanie_dnya_izvestnyj_otvet(T):
    out = R.td_T(bars(), "AAAA", T)
    assert list(out["date"]) == DAYS and set(out["ticker"]) == {"AAAA"}
    for k in range(3):
        e = expected_state(k, TS[T])
        r = out.iloc[k]
        for c in ("open", "high", "low", "close"):
            assert r[c] == e[c], (T, k, c)
        assert r["volume"] == e["volume"]
    assert out["close"].max() < 1000.0 * 3 + 100         # бары >= T не видны


@pytest.mark.parametrize("T", list(TS))
def test_m_T_bar_s_nachalom_T_ne_v_sostoyanii(T):
    """G2: бар с началом T (и позже) с огромной ценой не попадает в состояние дня."""
    b = bars()
    b.loc[b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute >= TS[T], ["high", "close"]] = 1e9
    out = R.td_T(b, "AAAA", T)
    assert (out["high"] < 1e8).all() and (out["close"] < 1e8).all()


@pytest.mark.parametrize("T", list(TS))
def test_m_T_mutatsiya_cut_plyus_bar_lovitsya(T):
    """Мутант: срез T+10 (бар T попадает в индикатор). Тест на известный ответ его ловит."""
    mut = D.aggregate(bars(), "AAAA", cut_min=TS[T] + 10)
    good = R.td_T(bars(), "AAAA", T)
    assert not mut[["high", "close", "volume"]].equals(good[["high", "close", "volume"]])
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(mut["close"].to_numpy(), [expected_state(k, TS[T])["close"] for k in range(3)])


@pytest.mark.parametrize("T", list(TS))
def test_mutatsiya_d_plyus_odin_lovitsya(T):
    """Мутант «d+1»: состояние дня d подменено состоянием d+1 — не равно ожидаемому."""
    good = R.td_T(bars(), "AAAA", T)["close"].to_numpy()
    exp = np.array([expected_state(k, TS[T])["close"] for k in range(3)])
    assert np.array_equal(good, exp)
    assert not np.array_equal(np.roll(good, -1), exp)


def test_cena_T_otkrytie_bara_T():
    for T in ("17:00", "18:00"):
        c = R.cena_T(bars(), T)
        assert list(c["date"]) == DAYS
        for k in range(3):
            assert c.iloc[k]["cena"] == _px(k, TS[T])      # open бара, начинающегося в T
        assert not c["net_bara_T"].any()


def test_cena_T_1840_close_bara_auktsiona():
    c = R.cena_T(bars(), "18:40")
    for k in range(3):
        assert c.iloc[k]["cena"] == _px(k, TS["18:40"]) + 0.5      # close бара 18:40, не open


@pytest.mark.parametrize("T", list(TS))
def test_m_T_tsena_ranshe_T_lovitsya(T):
    """G3: цена исполнения раньше T (бар T-10 / open предыдущего бара) не равна ожидаемой."""
    good = R.cena_T(bars(), T)["cena"].to_numpy()
    early = D.aggregate(bars(), "AAAA", cut_min=TS[T])["close"].to_numpy()      # close последнего бара до T = «цена раньше T»
    assert not np.array_equal(good, early)


def test_G4_net_bara_T_nan_s_flagom_bez_ffill():
    b = bars(drop={1: (TS["17:00"],)})
    c = R.cena_T(b, "17:00")
    assert list(c["date"]) == DAYS
    assert c["net_bara_T"].tolist() == [False, True, False]
    assert np.isnan(c.iloc[1]["cena"])
    assert c.iloc[0]["cena"] == _px(0, TS["17:00"]) and c.iloc[2]["cena"] == _px(2, TS["17:00"])
    assert c.iloc[1]["cena"] != c.iloc[0]["cena"]                      # не протянуто с d-1


def test_G4_den_bez_barov_do_T_net_v_sostoyanii():
    b = bars(drop={1: tuple(range(590, TS["17:00"], 10))})            # день 1: до 17:00 баров нет
    out = R.td_T(b, "AAAA", "17:00")
    assert list(out["date"]) == [DAYS[0], DAYS[2]]


def test_G1_T_1850_pobajtno_ravno_daily_1850():
    b, T = bars(), "18:50"
    pd.testing.assert_frame_equal(R.td_T(b, "AAAA", T), D.aggregate(b, "AAAA"), check_exact=True)
    assert R.td_T(b, "AAAA", T).to_numpy().tobytes() == D.aggregate(b, "AAAA").to_numpy().tobytes()


def test_G1_cut_1850_ne_pokhozh_na_17():
    assert not R.td_T(bars(), "AAAA", "18:40")["volume"].equals(R.td_T(bars(), "AAAA", "18:50")["volume"])


def test_pustaya_lenta():
    e = bars().iloc[0:0]
    assert len(R.td_T(e, "AAAA", "17:00")) == 0
    c = R.cena_T(e, "17:00")
    assert len(c) == 0 and list(c.columns) == ["date", "cena", "net_bara_T"]


@pytest.mark.parametrize("T", ["25:00", "1700", "", "17:60", None])
def test_plokhoe_T_otkaz(T):
    with pytest.raises(ValueError):
        R.td_T(bars(), "AAAA", T)
    with pytest.raises(ValueError):
        R.cena_T(bars(), T)


def test_T_do_otkrytiya_otkaz():
    with pytest.raises(ValueError):
        R.td_T(bars(), "AAAA", "09:50")        # состояние было бы пустым по построению
