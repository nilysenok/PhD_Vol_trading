# -*- coding: utf-8 -*-
"""Предел активности (толкование 27.10 19:35, Ж-044 п. 6) и плато «нет» (Ж-048 п. 1), режим --tolko-itog. Синтетика."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from fastbt import etap1_poisk as P
from fastbt import etap1_poisk_vorota as V
from test_etap1_poisk import AXES, SETKA, YAKOR, World

NET = "net"
SETKA_A = {**SETKA, "yakor": {**YAKOR, "a": 6, "b": 0}}      # якорь вдали от центра плато: 40 сделок/год


class RareCenter(World):
    """Центр плато (a=3) даёт лучшую целевую, но 3 сделки в год; остальные точки — 40."""

    def evaluate(self, p, period):
        r = super().evaluate(p, period)
        return dict(r, sdelok_y={y: (3 if p["a"] == 3 else 40) for y in r["sdelok_y"]})


class AllRare(World):
    def evaluate(self, p, period):
        r = super().evaluate(p, period)
        return dict(r, sdelok_y={y: 3 for y in r["sdelok_y"]})


def go(tmp_path, world, run_id="1"):
    return P.run_strategy("S9", world.evaluate, lambda p: True, out=tmp_path, run_id=run_id, setka=SETKA_A)


def test_tochka_s_3_sdelkami_ne_vybiraetsya_hotya_tselevaya_vyshe(tmp_path):
    assert go(tmp_path / "ref", World())["p_a"] == 3                    # без редкой точки центр — a=3 (контроль теста)
    row = go(tmp_path / "rare", RareCenter())
    assert row["p_a"] != 3 and row["obj_min_msr"] < World.score({"a": 3, "b": 2, "k_s": NET, "R": NET, "tau": NET}) + 1e-9
    f = pd.read_csv(tmp_path / "rare" / "S9_faktor.csv", comment="#")
    assert f[f.a == 3].sdelok_2018.eq(3).all() and f[f.a == 3].storon_2019.eq(6).all()
    assert {"sharpe_2018", "msr_2019", "storon_2018", "sdelok_2019"} <= set(f.columns)
    assert {"sharpe_2018", "sharpe_2019"} <= set(pd.read_csv(tmp_path / "rare" / "S9_itog.csv", comment="#").columns)


def test_mutaciya_bez_filtra_aktivnosti_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "dopustima", lambda led, pt, yakor: True)
    assert go(tmp_path, RareCenter())["p_a"] == 3                      # без фильтра выбрана вырожденная точка
    with pytest.raises(AssertionError):
        test_tochka_s_3_sdelkami_ne_vybiraetsya_hotya_tselevaya_vyshe(tmp_path / "t")


def test_dopustima_poroga_polovina_bazy_i_30_sdelok():
    mk = lambda st, sd: V.Ledger(lambda p, per: dict(storony_y={2018: st, 2019: st}, sdelok_y={2018: sd, 2019: sd}))   # noqa: E731
    yak = {"x": 0}
    led = V.Ledger(lambda p, per: dict(storony_y={2018: 100.0, 2019: 100.0}, sdelok_y={2018: 50, 2019: 50}) if p["x"] == 0 else
                   dict(storony_y={2018: 50.0 if p["x"] == 1 else 49.0, 2019: 100.0}, sdelok_y={2018: 50, 2019: 29 if p["x"] == 3 else 50}))
    assert V.dopustima(led, {"x": 1}, yak) and not V.dopustima(led, {"x": 2}, yak) and not V.dopustima(led, {"x": 3}, yak)
    assert not V.dopustima(mk(float("nan"), 50), {"x": 1}, {"x": 0})    # у якоря стороны не определены -> недопустимо


def test_itog_bez_dopustimoy_tochki_te_zhe_kolonki(tmp_path):
    ok = go(tmp_path, World())
    row = go(tmp_path, AllRare(), "2")                                  # дозапись падает, если колонки разошлись
    d = pd.read_csv(tmp_path / "S9_itog.csv", comment="#")
    assert list(row) == list(ok) and len(d) == 2 and d.verdikt.iloc[1] == "не дошла: нет допустимой точки"
    assert row["N_dopustimykh"] == 0 and math.isnan(row["obj_min_msr"]) and math.isnan(row["sharpe_2018"])


AX = {**AXES, "k_s": [1.0, 2.0, 3.0, 4.0, NET], "R": [1.0, 2.0, 3.0, 4.0, NET], "tau": [1, 3, 10, 20, NET]}
PT = {"a": 3, "b": 2, "k_s": NET, "R": NET, "tau": NET, "sigma": "rv22", "proverka": "zakrytie"}


def net_led(dip: float):
    """Целевая 1,0 везде; при k_s = 4 — на dip хуже."""
    s = lambda p: 1.0 - (dip if p["k_s"] == 4.0 else 0.0)                                            # noqa: E731
    return V.Ledger(lambda p, per: dict(msr_by_year={2018: s(p), 2019: s(p)}, sdelok_y={2018: 40, 2019: 40}))


def test_plato_net_rovnye_sosedi_prohodyat_provalnaya_k_s_4_net():
    assert V.gate_osi(net_led(0.0), PT, AX)["ok"] and V.gate_osi(net_led(0.1), PT, AX)["ok"]         # ровно 0,1 — допустимо
    g = V.gate_osi(net_led(0.11), PT, AX)
    assert not g["ok"] and g["per"]["k_s_net"] is False and g["per"]["tau_net"] is True


def test_mutaciya_porog_plato_net_beskonechnost_lovitsya(monkeypatch):
    monkeypatch.setattr(V, "PLATO_NET", math.inf)
    assert V.gate_osi(net_led(0.5), PT, AX)["ok"]                      # с порогом ∞ провал k_s = 4 пропущен
    with pytest.raises(AssertionError):
        test_plato_net_rovnye_sosedi_prohodyat_provalnaya_k_s_4_net()


def test_tolko_itog_dopisyvaet_i_ne_trogaet_faktor_morris(tmp_path):
    first = go(tmp_path, World())
    before = {n: (tmp_path / f"S9_{n}.csv").read_bytes() for n in ("faktor", "morris")}
    w = World()
    row = P.tolko_itog("S9", w.evaluate, lambda p: True, out=tmp_path, run_id="2", setka=SETKA_A)
    d = pd.read_csv(tmp_path / "S9_itog.csv", comment="#")
    assert len(d) == 2 and list(row) == list(first) and {"plato_staroe", "plato_novoe"} <= set(d.columns)
    assert str(d.plato_staroe.iloc[1]) == str(d.G4_osi.iloc[0]) and row["pbo"] == pytest.approx(first["pbo"])
    assert row["p_a"] == first["p_a"] and row["N_unikalnykh"] >= first["N_unikalnykh"]
    assert before == {n: (tmp_path / f"S9_{n}.csv").read_bytes() for n in ("faktor", "morris")}
    assert set(w.calls) == {"train"}


def test_tolko_itog_novaya_aktivnost_menyaet_vybor(tmp_path):
    go(tmp_path, World())                                              # прежний прогон выбрал a=3
    row = P.tolko_itog("S9", RareCenter().evaluate, lambda p: True, out=tmp_path, run_id="2", setka=SETKA_A)
    assert row["p_a"] != 3 and row["verdikt"] != "не дошла: нет допустимой точки"
