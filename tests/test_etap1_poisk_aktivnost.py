# -*- coding: utf-8 -*-
"""Предел активности (толкование 27.10 19:35, Ж-044 п. 6) и плато «нет» (Ж-048 п. 1), режим --tolko-itog. Синтетика.
Правка теста (Ж-081, 28.10): в test_tolko_itog_... pbo сравнивается с nan_ok=True — после Ж-090 (51867f11) целевая PBO даёт nan
на синтетике (год половины ≤ 0), это законно; проверяется, что повторный прогон даёт то же значение, включая nan."""
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


def test_konechnye_znacheniya_ostayutsya_3_soseda_v_summe():
    pt = {**PT, "k_s": 2.0}                                            # 2 соседа по k_s, остальные оси «нет» -> сумма 2 < 3
    g = V.gate_osi(net_led(0.0), pt, AX)
    assert g["per"]["k_s"] == 2 and not g["ok"]                       # правило min(3, n) пропустило бы
    pt2 = {**PT, "k_s": 2.0, "tau": 3}                                 # k_s: 2, tau: 2 соседа (1 и 10) -> сумма 4, R «нет» односторонне
    assert V.gate_osi(net_led(0.0), pt2, AX)["ok"]


def test_kray_plana_s_net_ne_otsekaetsya():
    from fastbt import sweep_drivers as sd
    ax = {**{k: v[:1] for k, v in AX.items()}, "k_s": [1.0, 2.0, NET]}
    ev = lambda p, per: dict(msr_by_year={2018: 1.0 + (p["k_s"] == NET), 2019: 1.0 + (p["k_s"] == NET)},    # noqa: E731
                             excess_d={2018: [0.0, 1.0], 2019: [0.0, 1.0]}, sdelok_y={2018: 40, 2019: 40})
    led = V.Ledger(ev)
    fact = sd.factorial_sweep(led, {k: v[0] for k, v in ax.items()}, {"k_s": ax["k_s"]}, "train", objective=V.obj)
    c, pt, _ = P.pick_center(led, fact, ax, 0.0, lambda p: True, {k: v[0] for k, v in ax.items()})
    assert c is not None and pt(c)["k_s"] == NET                       # у «нет» соседей < 3, но это не край по Ж-048


AXK = {**AX, "k_s": [1.0, 2.0, 3.0, 4.0, NET], "R": [1.0, 2.0, 3.0, 4.0, NET]}
KRAY = {**PT, "k_s": 1.0, "R": 2.0, "tau": 3}                          # k_s = 1 — край сетки; R, τ внутри


KRAY1 = {**PT, "k_s": 1.0}                                              # единственная конечная ось — k_s (край), R и τ «нет»


def kray_led(m075: float = 1.0, m05: float = 1.0):
    """Целевая 1,0 везде; k_s = 0,75 -> m075, k_s = 0,5 -> m05."""
    s = lambda p: {0.75: m075, 0.5: m05}.get(p["k_s"], 1.0)                                         # noqa: E731
    return V.Ledger(lambda p, per: dict(msr_by_year={2018: s(p), 2019: s(p)}, sdelok_y={2018: 40, 2019: 40}))


def test_kray_setki_bez_rasshireniya_ne_prohodit_s_rasshireniem_prohodit():
    led = kray_led()
    g = V.gate_osi(led, KRAY1, AXK)                                      # соседи k_s: 2,0 + 0,75 + 0,5 = 3
    assert g["ok"] and g["kray"] == "k_s" and g["per"]["k_s"] == 3 and g["per"]["k_s_kray"] is True
    assert {0.75, 0.5} <= {p["k_s"] for p, _ in led.cache.values()}      # наружные точки посчитаны и вошли в N
    assert not V.gate_osi(kray_led(-1.0, -1.0), KRAY1, AXK)["ok"]        # наружные не положительны -> соседей 1 < 3
    assert V.gate_osi(led, {**KRAY1, "k_s": 2.0}, AXK)["kray"] == "-"


def test_kray_odin_naruzhnyy_nepolozhitelnyy_obshchee_pravilo_bez_usloviya_oba():
    g = V.gate_osi(kray_led(1.0, -1.0), KRAY, AXK)                       # 0,5 не положительна; сумма по осям 1+1+... >= 3
    assert g["ok"] and g["per"]["k_s"] == 2


AXT = {**AXK, "tau": [1, 2, 3, 5, 10, 20, NET]}
TAU1 = {**PT, "k_s": 2.0, "R": 2.0, "tau": 1}                            # k_s, R внутри (по 2 положительных соседа), τ = 1 — физический край


def tau_led(dip: float, tau: int = 2):
    """Целевая 1,0 везде; при τ = tau — на dip хуже."""
    s = lambda p: 1.0 - (dip if p["tau"] == tau else 0.0)                                           # noqa: E731
    return V.Ledger(lambda p, per: dict(msr_by_year={2018: s(p), 2019: s(p)}, sdelok_y={2018: 40, 2019: 40}))


def test_tau_1_ravnye_2_i_3_prohodit_proval_na_2_net_ostalnoe_ne_trogaet():
    led = tau_led(0.0)
    g = V.gate_osi(led, TAU1, AXT)
    assert g["ok"] and g["kray"] == "tau" and g["per"]["tau_fiz"] is True and "tau" not in g["per"]    # без отказа «край τ = 1»
    assert {2, 3} <= {p["tau"] for p, _ in led.cache.values()}                                       # τ = 2, 3 досчитаны в N
    assert V.gate_osi(tau_led(0.1), TAU1, AXT)["ok"]                                                 # ровно 0,1 — допустимо
    bad = V.gate_osi(tau_led(0.11), TAU1, AXT)
    assert not bad["ok"] and bad["per"]["tau_fiz"] is False
    assert not V.gate_osi(tau_led(0.5, 3), TAU1, AXT)["ok"]                                          # провал на τ = 3 тоже
    assert V.gates(tau_led(0.0), TAU1, AXT, ["a"], 0.0, lambda p: True)["kray_setki"] == "tau"


def test_mutaciya_tau1_porog_beskonechnost_ili_vsegda_otkaz_lovitsya(monkeypatch):
    monkeypatch.setattr(V, "PLATO_NET", math.inf)
    assert V.gate_osi(tau_led(0.5), TAU1, AXT)["ok"]                    # порог ∞ пропускает провал на τ = 2
    with pytest.raises(AssertionError):
        test_tau_1_ravnye_2_i_3_prohodit_proval_na_2_net_ostalnoe_ne_trogaet()
    monkeypatch.undo()
    monkeypatch.setattr(V, "one_sided", lambda *a, **k: False)          # «τ = 1 всегда не проходит»
    with pytest.raises(AssertionError):
        test_tau_1_ravnye_2_i_3_prohodit_proval_na_2_net_ostalnoe_ne_trogaet()


def test_mutaciya_kray_ne_pomechaetsya_i_rasshirenie_ne_schitaetsya(monkeypatch):
    monkeypatch.setitem(V.OUTER, "k_s", ())                              # расширение не считается: за край не шагаем
    with pytest.raises(AssertionError):
        test_kray_setki_bez_rasshireniya_ne_prohodit_s_rasshireniem_prohodit()
    monkeypatch.undo()
    monkeypatch.setattr(V, "kray_plato", lambda led, pt, a: (None if a not in V.OUTER else
                        all(V.obj(led({**pt, a: v}, "train")) > 0 for v in V.OUTER[a])))   # «оба наружных > 0 обязательны»
    with pytest.raises(AssertionError):
        test_kray_odin_naruzhnyy_nepolozhitelnyy_obshchee_pravilo_bez_usloviya_oba()


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
    assert str(d.plato_staroe.iloc[1]) == str(d.G4_osi.iloc[0]) and row["pbo"] == pytest.approx(first["pbo"], nan_ok=True)
    assert row["p_a"] == first["p_a"] and row["N_unikalnykh"] >= first["N_unikalnykh"]
    assert before == {n: (tmp_path / f"S9_{n}.csv").read_bytes() for n in ("faktor", "morris")}
    assert set(w.calls) == {"train"}


def test_tolko_itog_novaya_aktivnost_menyaet_vybor(tmp_path):
    go(tmp_path, World())                                              # прежний прогон выбрал a=3
    row = P.tolko_itog("S9", RareCenter().evaluate, lambda p: True, out=tmp_path, run_id="2", setka=SETKA_A)
    assert row["p_a"] != 3 and row["verdikt"] != "не дошла: нет допустимой точки"
