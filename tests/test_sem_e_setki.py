"""Сетки C/E (fastbt/designs/sweep/{C,E}_setka_e2.yaml): N пересчитывается из осей, каждая точка принимается sem_e."""
import itertools
from pathlib import Path

import numpy as np
import pytest

from fastbt.sem_e import evaluate, point_params
from test_sem_e import build

yaml = pytest.importorskip("yaml")
DIR = Path(__file__).resolve().parent.parent / "fastbt" / "designs" / "sweep"


def load(name):
    return yaml.safe_load((DIR / name).read_text("utf-8"))


def points(osi):
    ks = list(osi)
    return [dict(zip(ks, v)) for v in itertools.product(*[osi[k] for k in ks])]


def n_of(s):
    return len(points(s["osi"]))


def test_n_c_is_324_recomputed_from_axes():
    d = load("C_setka_e2.yaml")
    n = {k: n_of(s) for k, s in d["semyi"].items()}
    assert n == {"C1": 15, "C2W": 21, "Cstar_filtr": 15, "Cstar_chastichnaya": 21}
    tot = 0
    for r, spec in d["sigmy"].items():
        per = sum(v for k, v in n.items() if not (r == "naive" and d["semyi"][k].get("bez_naive")))
        assert per * len(spec["h"]) == spec["tochek"]
        tot += spec["tochek"]
    assert tot == d["N"]["itogo"] == 324
    assert d["baza"]["A_prim"]["baza"] is True and d["baza"]["A_prim"]["v_N"] is False
    assert "A_prim" not in d["semyi"] and "A_prim" not in d["N"]        # база в N не входит


def test_n_e_recomputed_and_dsr_n_is_with_u():
    d = load("E_setka_e2.yaml")
    n = {k: n_of(s) for k, s in d["semyi"].items()}
    assert n == {"E0": 4, "E1": 192, "E2": 96, "E3": 36}
    assert sum(n.values()) == d["N"]["bez_u"] == 328
    assert sum(n.values()) * len(d["podvyborka_u"]) == d["N"]["s_u"] == d["N"]["v_dsr"] == 656
    assert d["stupeni_sigma"] == d["semyi"]["E1"]["osi"]["sigma"] == d["semyi"]["E2"]["osi"]["sigma"]


def test_every_e_point_value_is_accepted_by_sem_e():
    d = load("E_setka_e2.yaml")
    inp = build()
    for fam in ("E0", "E1", "E2"):
        for pt in points(d["semyi"][fam]["osi"]):
            if pt.get("sigma", "const") != "const":
                continue                                  # все значения осей встречаются при sigma = const
            r = evaluate({"kind": "E", "sigma": "const", "u": "all"}, point_params(pt), "tune2018", 0.04, inp)
            assert np.isfinite(r["sharpe"]) and r["T"] > 200, pt


def test_every_c_point_is_accepted_by_sem_e():
    d = load("C_setka_e2.yaml")
    inp = build()
    seen = 0
    for name, s in d["semyi"].items():
        for pt in points(s["osi"]):
            if pt["W"] != 126:
                continue
            par = {**point_params(pt), "star": name.startswith("Cstar")}
            r = evaluate({"kind": "C", "sigma": "naive"}, par, "train", 0.04, inp)
            assert np.isfinite(r["sharpe"]), (name, pt)
            seen += 1
    assert seen == 5 + 7 + 5 + 7


def test_point_params_maps_percent_pairs_and_ab():
    assert point_params({"W": 126, "p": 80}) == {"W": 126, "p": 0.8}
    assert point_params({"W": 252, "pary": [50, 70]}) == {"W": 252, "p1": 0.5, "p2": 0.7}
    assert point_params({"q": 0.67, "ab": 1.5, "tau": 5, "s": "ezhedn", "sigma": "hyb_h1"}) == \
        {"q": 0.67, "a": 1.5, "b": 1.5, "tau": 5, "s": "ezhedn"}
