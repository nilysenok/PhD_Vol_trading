"""Ворота плато итераций 3/4/10 — как в итерации 1, без отступлений (Ж-051 п. 3): ≥ 3 соседей с min mSR > 0, край и «нет соседей»
не проходят, псевдонимы P1 не плодят N и не считаются соседями дважды."""
import math

import numpy as np
import pandas as pd

from fastbt import etap1_poisk as P, etap1_poisk_vorota as V, zh051_plato as Z
from test_zh051_iter_drv import fake_res


def _setup(drift_of):
    S = P.load_setka("S1")
    row = pd.read_csv("docs/tables/zh047/S1_itog_v2.csv", comment="#").iloc[-1]
    p1 = {a: P._coerce(str(row[f"p_{a}"]), S["axes"][a]) for a in S["axes"]}
    calls = []

    def ev(p, per):
        calls.append(dict(p))
        return fake_res(zlib_seed(p), drift_of(p))
    return S, p1, Z.PlusLedger(ev), calls


def zlib_seed(p):
    import zlib
    return zlib.crc32(repr(sorted(p.items(), key=str)).encode()) % 10 ** 6


def test_canon_plus_psevdonimy_p1_ne_plodyat_N():
    S, p1, led, calls = _setup(lambda p: 0.01)
    r = led(p1, "train")
    for extra in ({"top_k": "net"}, {"sektor_trend": "net", "shirina": "net", "ma_n": 50}, {"ma_n": 100, "porog_shiriny": 0.4}):
        led({**p1, **extra}, "train")
    assert led.n_unique == 1 and len(calls) == 1 and r is led(p1, "train")
    led({**p1, "shirina": "net", "sektor_trend": "po_trendu", "porog_shiriny": 0.4}, "train")
    led({**p1, "shirina": "net", "sektor_trend": "po_trendu", "porog_shiriny": 0.6}, "train")
    assert led.n_unique == 2                                         # при shirina = net порог не различает точки


def test_odnomernaya_setka_net_sosedey_plato_ne_projdeno():
    S, p1, led, _ = _setup(lambda p: 0.01)                           # все точки положительны, но соседей < 3 (край)
    grid = {"top_k": [3, 5, 8, 12, "net"]}
    best, cands = Z.vybor(led, p1, grid, S["axes"], S["yakor"], True)
    assert best is not None and best["net_sosedey"] is True
    assert best["gates"]["G3_plato"] is False and best["gates"]["ok"] is False
    assert best["gates"]["n_sosedey"] < 3 and best["gates"]["net_sosedey"]       # у top_k = 5/8/12 два соседа, у 3 — один


def test_vnutrennyaya_tochka_tri_soseda_i_dedup_psevdonimov():
    S, p1, led, _ = _setup(lambda p: 0.01)
    grid = {"shirina": ["protiv", "net", "po_trendu"], "ma_n": [50, 100, 200]}
    e = {"shirina": "po_trendu", "ma_n": 100}
    nb = Z.uniq_nbrs(p1, e, grid)
    # соседи: (po,50), (po,200) и «net» по shirina при ma_n 50/100/200 — все они P1 (один раз)
    assert len(nb) == 3 and sum(1 for q in nb if not any(k in q for k in ("shirina", "ma_n"))) == 1
    g = Z.gates_plus(led, p1, e, grid, S["axes"], 0.1, True)
    assert g["G3_plato"] and g["n_sosedey"] == 3 and g["G1_pbo"] and g["G2_sdelki"] and g["G5_s9"]


def test_g3_trebuet_polozhitelnyh_sosedey():
    S, p1, led, _ = _setup(lambda p: -0.01 if p.get("ma_n") in (50, 200) else 0.01)
    grid = {"shirina": ["protiv", "net", "po_trendu"], "ma_n": [50, 100, 200]}
    g = Z.gates_plus(led, p1, {"shirina": "po_trendu", "ma_n": 100}, grid, S["axes"], 0.1, True)
    assert g["G3_plato"] is False                                    # положителен один сосед (P1) из трёх
    assert Z.gates_plus(led, p1, {"shirina": "po_trendu", "ma_n": 100}, grid, S["axes"], 0.25, True)["G1_pbo"] is False


def test_vybor_chestnyj_kray_ne_vybiraetsya_esli_est_vnutrennyaya():
    S, p1, led, _ = _setup(lambda p: 0.01)
    grid = {"shirina": ["protiv", "net", "po_trendu"], "ma_n": [50, 100, 200]}
    best, cands = Z.vybor(led, p1, grid, S["axes"], S["yakor"], True)
    assert best is not None and not best["edge"] and not best["net_sosedey"]
    assert math.isfinite(best["plateau"]) and np.isfinite(best["sigma126"])
