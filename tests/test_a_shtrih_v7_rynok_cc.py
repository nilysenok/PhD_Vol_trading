"""Рынок A′-расширенной — close→close того же дня, что ряды центров (#891/#892). Синтетика: портфель = β·рынок -> после хеджа остаток
< 1 % дисперсии; мутация: рынок open→open оставляет десятки процентов."""
import numpy as np
import pandas as pd
import pytest

from fastbt import a_shtrih_v7_vhody as V
from fastbt import a_shtrih_v7_kniga as K

T, BETA = 700, 0.8


def ceny(seed=3):
    rng = np.random.default_rng(seed)
    o = 100 * np.cumprod(1 + rng.normal(0, 0.006, T))                 # open
    c = o * (1 + rng.normal(0, 0.008, T))                            # close дня
    o[1:] = c[:-1] * (1 + rng.normal(0, 0.004, T - 1))               # ночной гэп
    return pd.Series(o, pd.bdate_range("2016-01-04", periods=T)), pd.Series(c, pd.bdate_range("2016-01-04", periods=T))


def ostatok(rynok_knigi, rynok_hedzha_ryad):
    o, c = ceny()
    rcc = V.cc_dohodnost(c, c.index)
    X = (BETA * np.nan_to_num(rcc))[None, None, :]                     # портфель = β·рынок (close→close)
    nul = np.zeros(T)
    r = K.kniga_ryada(X, None, np.array(["a"]), rynok_hedzha_ryad(o, c), rynok_hedzha_ryad(o, c), nul, vesa="vykl",
                      sektor_netto=False, tsel_vol=False, vykl=False)
    w = slice(150, None)
    return float(np.var(r.ryad[w]) / np.var(X[0, 0, w]))


def test_cc_ostatok_menshe_odnogo_procenta():
    assert ostatok(None, lambda o, c: V.cc_dohodnost(c, c.index)) < 0.01


def test_mut_open_open_lovitsya():
    oo = lambda o, c: (o.shift(-1) / o - 1.0).to_numpy(float)          # noqa: E731
    assert ostatok(None, oo) > 0.10                                  # прежний вход: остаётся заметная доля


def test_mix_cc_vnutri_kontrakta_bez_skachka_rolla(tmp_path):
    d = pd.bdate_range("2016-01-04", periods=8)
    def csv(sec, closes, start):
        rows = [dict(BOARDID="RFUD", TRADEDATE=x, SECID=sec, OPEN=v, CLOSE=v) for x, v in zip(d[start:], closes)]
        pd.DataFrame(rows).to_csv(tmp_path / f"{sec}.csv", index=False)
    csv("MXH6_2016", [100, 101, 102, 103, 104], 0)                      # торгуется дни 0–4, экспирация день 4, ролл за 2 дня = день 2
    csv("MXM6_2016", [200, 202, 204, 206, 208, 210, 212, 214][2:], 2)   # новый контракт с дня 2: цены ×2 (скачок цены между контрактами)
    r = V.mix_cc(d, tmp_path)
    assert np.isnan(r[0]) and r[1] == pytest.approx(101 / 100 - 1) and r[2] == pytest.approx(102 / 101 - 1)
    assert r[3] == pytest.approx(206 / 204 - 1)                         # день 3 — внутри нового контракта, скачка 100→200 нет
    assert np.nanmax(np.abs(r)) < 0.05
