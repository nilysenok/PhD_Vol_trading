"""К-16…К-20: детекторы событий T16–T20 на синтетике с известным ответом; сеть и данные ≥ 2022 не нужны.

Мутации (подмена константы модуля в памяти, тест обязан падать): 16 — IMOEX вне ±m0 не отсекается (`FILTR_IMOEX`);
17 — окна Rg и H_c захватывают d+1 (`SDVIG`; по контракту §2.3 «текущий день» = d+1 для решения на close d);
18 — вход на d+1 после отсечки вместо d+2 (`ZAD`); 19 — «первое снижение» по RV5 d+1 (`VPERED`);
20 — SMA20 объёма с текущим днём (`SMA_SDVIG`).
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import t16_shok, t17_szhatie, t18_div_gap, t19_rv_povorot, t20_obyom
from fastbt.t_obshchee import postroit_sig

DATES = pd.bdate_range("2019-01-01", periods=40).to_numpy("datetime64[D]")


def _inp(n=40, **tick):
    return {"dates": DATES[:n], "tickers": {"A": {k: np.asarray(v, float) for k, v in tick.items()}}}


# ---------- T16 ----------
def _t16(imoex_d20):
    c = np.full(40, 100.0)
    c[20] = 96.0
    m = np.full(40, 100.0)
    m[20] = imoex_d20
    s = np.full(40, 0.01)
    inp = _inp(close=c, sigma=s)
    inp["imoex"] = {"close": m, "sigma": s}
    return inp


def test_t16_known_event():
    ev = t16_shok.sobytiya(_t16(100.0), {"z0": 3.0, "m0": 1.0})
    assert ev == [("A", 20, 1, pytest.approx(-np.log(0.96) / 0.01))]   # сила = |z| в σ, знаменатель σ̂_{d−1}
    assert t16_shok.sobytiya(_t16(100.0), {"z0": 5.0, "m0": 1.0}) == []
    _, w, acc = postroit_sig(40, [(d, s, x) for _, d, s, x in ev], 3, np.full(40, 0.01), 3.0)
    assert acc[0][:3] == (20, 21, 1)                                    # вход d+1


def test_t16_imoex_outside_band_filtered_and_mutation(monkeypatch):
    inp = _t16(96.0)                                                    # рынок тоже −4σ
    assert t16_shok.sobytiya(inp, {"z0": 3.0, "m0": 1.0}) == []
    monkeypatch.setattr(t16_shok, "FILTR_IMOEX", False)
    assert len(t16_shok.sobytiya(inp, {"z0": 3.0, "m0": 1.0})) == 1      # мутант событие пропускает


# ---------- T17 ----------
def _t17(zerk=False):
    n = 30
    c, h, lo = np.full(n, 100.0), np.full(n, 102.0), np.full(n, 98.0)
    h[10:13], lo[10:13] = 100.5, 99.5
    h[13], lo[13], h[14], lo[14] = 100.9, 99.5, 100.5, 99.5
    c[15], h[15], lo[15] = 101.0, 101.5, 100.0
    h[16:], lo[16:] = 100.5, 99.5
    if zerk:
        c, h, lo = 200 - c, 200 - lo, 200 - h
    return _inp(n=n, close=c, high=h, low=lo, sigma=np.full(n, 0.01))


def test_t17_breakout_long_and_short():
    osi = {"W": 3, "L": 5}
    assert t17_szhatie.sobytiya(_t17(), osi) == [("A", 15, 1, pytest.approx(np.log(101 / 100.5) / 0.01))]
    assert t17_szhatie.sobytiya(_t17(True), osi) == [("A", 15, -1, pytest.approx(abs(np.log(99 / 99.5)) / 0.01))]


def test_t17_mutation_window_with_next_day(monkeypatch):
    monkeypatch.setattr(t17_szhatie, "SDVIG", 1)
    assert t17_szhatie.sobytiya(_t17(), {"W": 3, "L": 5}) != [("A", 15, 1, pytest.approx(np.log(101 / 100.5) / 0.01))]


# ---------- T18 ----------
def _t18(announce):
    div = pd.DataFrame({"ticker": ["A"], "last_cum_date": [DATES[10]], "announce_date": [announce], "dps_rub": [3.0]})
    inp = _inp(close=np.full(40, 100.0), sigma=np.full(40, 0.01))
    inp["div"] = div
    return inp


def test_t18_known_event_pit_and_threshold():
    assert t18_div_gap.sobytiya(_t18(DATES[5]), {"g0": 2.0}) == [("A", 11, 1, pytest.approx(3.0))]   # решение g=11, вход 12 = d0+2
    assert t18_div_gap.sobytiya(_t18(DATES[5]), {"g0": 4.0}) == []                                  # gap_σ = 3 < 4
    assert t18_div_gap.sobytiya(_t18(DATES[11]), {"g0": 0.0}) != []                                 # announce = g — можно
    assert t18_div_gap.sobytiya(_t18(DATES[12]), {"g0": 0.0}) == []                                 # announce > g — PIT
    assert t18_div_gap.sobytiya(_t18(pd.NaT), {"g0": 0.0}) == []                                    # без даты — нет


def test_t18_mutation_entry_d_plus_1(monkeypatch):
    ev = t18_div_gap.sobytiya(_t18(DATES[5]), {"g0": 2.0})
    _, _, acc = postroit_sig(40, [(d, s, x) for _, d, s, x in ev], 5, np.full(40, 0.01), 2.0)
    assert acc[0][1] == 12                                              # d0 = 10, вход на d0+2
    monkeypatch.setattr(t18_div_gap, "ZAD", 1)
    ev2 = t18_div_gap.sobytiya(_t18(DATES[5]), {"g0": 2.0})
    assert ev2 != ev and ev2[0][1] == 10                                # мутант входил бы на d0+1


# ---------- T19 ----------
def _t19():
    s5 = np.full(30, 0.01)
    s5[10], s5[11], s5[12] = 0.03, 0.04, 0.04
    return _inp(n=30, sigma5=s5, sigma66=np.full(30, 0.01))


def test_t19_known_event():
    assert t19_rv_povorot.sobytiya(_t19(), {"q": 3.0}) == [("A", 13, 1, pytest.approx(9.0))]   # первое снижение на 13
    assert t19_rv_povorot.sobytiya(_t19(), {"q": 20.0}) == []


def test_t19_mutation_decline_from_future(monkeypatch):
    base = t19_rv_povorot.sobytiya(_t19(), {"q": 3.0})
    monkeypatch.setattr(t19_rv_povorot, "VPERED", 1)
    assert t19_rv_povorot.sobytiya(_t19(), {"q": 3.0}) != base


# ---------- T20 ----------
def _t20(c25=100.2):
    c, v = np.full(40, 100.0), np.full(40, 100.0)
    c[25], v[25] = c25, 320.0
    return _inp(close=c, vol=v, sigma=np.full(40, 0.01))


def test_t20_known_event_both_sides():
    assert t20_obyom.sobytiya(_t20(), {"v0": 3.0, "r0": 0.5}) == [("A", 25, 1, pytest.approx(3.2))]
    assert t20_obyom.sobytiya(_t20(99.8), {"v0": 3.0, "r0": 0.5}) == [("A", 25, -1, pytest.approx(3.2))]
    assert t20_obyom.sobytiya(_t20(), {"v0": 3.0, "r0": 0.1}) == []      # |z| = 0,2 ≥ r0
    assert t20_obyom.sobytiya(_t20(100.0), {"v0": 3.0, "r0": 0.5}) == []  # r_d = 0 — события нет


def test_t20_mutation_sma_with_current_day(monkeypatch):
    monkeypatch.setattr(t20_obyom, "SMA_SDVIG", 0)                       # среднее (19·100 + 320)/20 → U = 2,88 < 3
    assert t20_obyom.sobytiya(_t20(), {"v0": 3.0, "r0": 0.5}) == []


# ---------- общее: даты ≥ 2022 ----------
@pytest.mark.parametrize("mod,osi", [(t16_shok, {"z0": 3, "m0": 1}), (t17_szhatie, {"W": 3, "L": 5}),
                                     (t18_div_gap, {"g0": 0}), (t19_rv_povorot, {"q": 3}), (t20_obyom, {"v0": 3, "r0": 1})])
def test_dates_2022_forbidden(mod, osi):
    inp = _t16(100.0)
    inp["dates"] = np.array(["2021-12-30", "2022-01-03"], "datetime64[D]")
    inp["div"] = _t18(DATES[5])["div"]
    with pytest.raises(PermissionError):
        mod.sobytiya(inp, osi)
