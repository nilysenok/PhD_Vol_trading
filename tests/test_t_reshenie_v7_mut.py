"""Тестировщик #839: попытки сломать td_T / cena_T. Мутации подменяются monkeypatch; реальный тикер 2019 — только чтение."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850 as D
from fastbt import t_reshenie_v7 as R
from tests.test_t_reshenie_v7 import DAYS, TS, _px, bars

MAIN = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading")
SBER = MAIN / "moex_discovery" / "data" / "raw" / "candles_10m" / "SBER.parquet"


def test_granitsa_T_bar_17_00_ne_v_sostoyanii_17_00():
    out = R.td_T(bars(), "AAAA", "17:00")
    assert out["close"].tolist() == [_px(k, 16 * 60 + 50) + 0.5 for k in range(3)]   # последний закрытый к 17:00 — бар 16:50
    assert out["high"].tolist() == [_px(k, 16 * 60 + 50) + 0.75 for k in range(3)]


def test_propusk_16_50_sostoyanie_do_T_ne_tyanetsya():
    b = bars(drop={1: (16 * 60 + 50,)})
    out = R.td_T(b, "AAAA", "17:00")
    assert out.iloc[1]["close"] == _px(1, 16 * 60 + 40) + 0.5        # последний существующий бар, не ffill
    assert out.iloc[1]["volume"] == 10 * (len(range(590, 17 * 60, 10)) - 1)
    c = R.cena_T(b, "17:00")                                          # цена T не зависит от дыры до T
    assert c["net_bara_T"].tolist() == [False] * 3 and c.iloc[1]["cena"] == _px(1, 17 * 60)


def test_den_bez_barov_posle_T_ostayotsya_v_sostoyanii_cena_nan():
    b = bars(drop={1: tuple(range(17 * 60, 19 * 60 + 10, 10))})
    assert list(R.td_T(b, "AAAA", "17:00")["date"]) == DAYS
    c = R.cena_T(b, "17:00")
    assert c["net_bara_T"].tolist() == [False, True, False] and np.isnan(c.iloc[1]["cena"])


def test_korotkii_den_net_bara_1840():
    b = bars(drop={0: (18 * 60 + 40, 18 * 60 + 50, 19 * 60)})
    c = R.cena_T(b, "18:40")
    assert c["net_bara_T"].tolist() == [True, False, False] and np.isnan(c.iloc[0]["cena"])
    assert c.iloc[1]["cena"] == _px(1, 18 * 60 + 40) + 0.5            # close, не протянут


def test_rabochaya_subbota_v_dnyakh():
    sat = pd.Timestamp("2019-03-09")
    assert sat.dayofweek == 5
    rows = [(sat + pd.Timedelta(minutes=m), 5.0 + m / 1e4, 6.0, 4.0, 5.5, 3) for m in range(590, 19 * 60, 10)]
    b = pd.concat([bars(), pd.DataFrame(rows, columns=bars().columns)], ignore_index=True)
    assert sat in set(R.td_T(b, "A", "18:00")["date"]) and sat in set(R.cena_T(b, "18:00")["date"])


def test_vhodnoj_poryadok_ne_vazhen():
    b = bars().sample(frac=1, random_state=1).reset_index(drop=True)
    pd.testing.assert_frame_equal(R.td_T(b, "A", "18:00"), R.td_T(bars(), "A", "18:00"))
    pd.testing.assert_frame_equal(R.cena_T(b, "18:00"), R.cena_T(bars(), "18:00"))


@pytest.mark.parametrize("T", list(TS))
def test_cena_ne_pozzhe_T_plyus_10(T):
    """Цена — бар, начинающийся ровно в T: сорванные цены на соседних барах не влияют."""
    b = bars()
    mod = b["timestamp"].dt.hour * 60 + b["timestamp"].dt.minute
    b.loc[mod != TS[T], ["open", "close"]] = 7.0
    c = R.cena_T(b, T)
    assert (c["cena"] != 7.0).all() and not c["net_bara_T"].any()


def test_M_srez_men_she_ravno_lovitsya(monkeypatch):
    """Мутация `<` → `<=`: бар T попал бы в состояние."""
    orig = D.aggregate
    monkeypatch.setattr(D, "aggregate", lambda b, t, cut_min=D.CUT_MIN, sessions=None: orig(b, t, cut_min + 10, sessions))
    mut = R.td_T(bars(), "A", "17:00")
    assert mut["close"].tolist() != [_px(k, 16 * 60 + 50) + 0.5 for k in range(3)]   # тест границы выше падал бы


def test_M_ffill_pri_otsutstvii_lovitsya():
    c = R.cena_T(bars(drop={1: (17 * 60,)}), "17:00")
    c2 = c.assign(cena=c["cena"].ffill())                               # мутант
    assert np.isnan(c.iloc[1]["cena"]) and not np.isnan(c2.iloc[1]["cena"])
    assert c["net_bara_T"].iloc[1]


def test_M_1840_open_vmesto_close_lovitsya():
    c = R.cena_T(bars(), "18:40")
    assert all(c.iloc[k]["cena"] != _px(k, 18 * 60 + 40) for k in range(3))


def test_realnyj_SBER_2019_T1850_pobajtno_i_srezy_monotonny():
    if not SBER.exists():
        pytest.skip("нет общих данных")
    b = D.read_bars(SBER, end=pd.Timestamp("2020-01-01"))
    b = b[b["timestamp"] >= "2019-01-01"]
    base = D.aggregate(b, "SBER")
    new = R.td_T(b, "SBER", "18:50")
    num = ["open", "high", "low", "close", "volume"]
    assert new[num].to_numpy().tobytes() == base[num].to_numpy().tobytes() and len(base) > 200
    v = {T: R.td_T(b, "SBER", T).set_index("date")["volume"] for T in ("17:00", "18:00", "18:40", "18:50")}
    for a, c in (("17:00", "18:00"), ("18:00", "18:40"), ("18:40", "18:50")):
        j = v[c].reindex(v[a].index)
        assert (j >= v[a]).all() and (v[c] > v[a].reindex(v[c].index).fillna(0)).any()
    cn = R.cena_T(b, "18:40")
    assert cn["cena"].notna().mean() > 0.9 and (cn["cena"] > 0).where(cn["cena"].notna(), True).all()
