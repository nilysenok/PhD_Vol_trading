"""Независимые тесты тестировщика к V6-14 лист 2 (fastbt/morris_v6_zamorozka.py): правило заморозки, csv, yaml.

Правило (объявлено до счёта): ось новая и в КАЖДОМ из трёх окон n_akt >= 20 и U95 < 0,05 -> заморозить; n_akt < 20 в любом
окне -> «не измерено»; NaN не заморозка; оси v5 никогда; замороженный родитель замораживает дочерние новые оси цепочкой.
Статистика окон подаётся готовыми числами (границы 0,05 / 19 / 20 задаются точно), плюс сквозной прогон через morris_v6.
"""
import csv
import itertools

import numpy as np
import pytest
import yaml

from fastbt import morris_v6 as M
from fastbt import morris_v6_zamorozka as Z

W = Z.OKNA


def stat_of(n, u):
    """n, u: списки по осям-для-окна: n[w][i], u[w][i] -> stat[okno] с массивами по осям."""
    return {W[w]: {"n_akt": np.array(n[w]), "mu_star": np.zeros(len(n[w])), "sigma": np.zeros(len(n[w])),
                   "U95": np.array(u[w], float)} for w in range(3)}


def one(u3, n3=(20, 20, 20), novaya=True):
    osi = [{"os": "a", "gruppa": "g", "novaya": novaya}]
    st = stat_of([[n3[0]], [n3[1]], [n3[2]]], [[u3[0]], [u3[1]], [u3[2]]])
    return Z.reshenie(osi, st)["a"]


# ---------------- константы правила ----------------
def test_declared_constants():
    assert Z.OKNA == ("<=2016", "<=2017", "<=2018")
    assert Z.PORIG_U == 0.05 and Z.MIN_AKT == 20


# ---------------- границы ----------------
def test_u95_exactly_005_is_not_frozen():
    z, _ = one((0.05, 0.0, 0.0))
    assert z is False
    assert one((0.0, 0.05, 0.0))[0] is False and one((0.0, 0.0, 0.05))[0] is False
    assert one((0.05, 0.05, 0.05))[0] is False


def test_u95_just_below_005_is_frozen():
    below = np.nextafter(0.05, 0.0)
    assert below < 0.05
    assert one((below, below, below))[0] is True
    assert one((0.0, 0.0, 0.0))[0] is True
    assert one((0.0499, 0.0499, 0.0499))[0] is True


def test_u95_just_above_not_frozen():
    assert one((np.nextafter(0.05, 1.0),) * 3)[0] is False


@pytest.mark.parametrize("w", [0, 1, 2])
def test_n_akt_19_in_any_window_means_not_measured(w):
    n = [20, 20, 20]
    n[w] = 19
    z, why = one((0.0, 0.0, 0.0), n3=n)
    assert z is False and "не измерено" in why


def test_n_akt_20_in_all_windows_is_enough():
    assert one((0.0, 0.0, 0.0), n3=(20, 20, 20))[0] is True
    assert one((0.0, 0.0, 0.0), n3=(500, 20, 77))[0] is True


@pytest.mark.parametrize("w", [0, 1, 2])
def test_one_bad_window_blocks_freeze(w):
    u = [0.0, 0.0, 0.0]
    u[w] = 0.2
    z, why = one(u)
    assert z is False and "не нулев" in why


def test_all_combinations_of_three_windows_all_rule():
    for flags in itertools.product([0.01, 0.2], repeat=3):
        assert one(flags)[0] == all(x < 0.05 for x in flags)


def test_nan_u95_not_frozen_and_nan_with_low_n_is_unmeasured():
    z, why = one((np.nan, 0.0, 0.0))
    assert z is False
    z2, why2 = one((0.0, 0.0, np.nan))
    assert z2 is False
    z3, why3 = one((np.nan, np.nan, np.nan), n3=(0, 0, 0))
    assert z3 is False and "не измерено" in why3


def test_negative_u95_or_zero_n_edge():
    assert one((-0.001, 0.0, 0.0))[0] is True        # U95 < 0 невозможен, но правило «< порога» формально выполнено
    assert one((0.0, 0.0, 0.0), n3=(0, 20, 20))[0] is False


def test_v5_axis_never_frozen_even_with_zero_effect():
    z, why = one((0.0, 0.0, 0.0), novaya=False)
    assert z is False and "v5" in why
    z2, _ = one((0.0, 0.0, 0.0), n3=(0, 0, 0), novaya=False)
    assert z2 is False


# ---------------- окна: вход ----------------
def test_windows_must_be_exactly_three_in_order():
    osi = [{"os": "a", "gruppa": "g", "novaya": True}]
    st = stat_of([[20]] * 3, [[0.0]] * 3)
    ok = Z.reshenie(osi, st)
    assert ok["a"][0] is True
    two = {k: st[k] for k in W[:2]}
    four = dict(st, **{"<=2019": st[W[0]]})
    rev = {k: st[k] for k in reversed(W)}
    with_2019 = {W[0]: st[W[0]], W[1]: st[W[1]], "<=2019": st[W[2]]}
    for bad in (two, four, rev, with_2019, {}):
        with pytest.raises(ValueError):
            Z.reshenie(osi, bad)


# ---------------- родитель -> дочерние ----------------
def _chain(res_parent_frozen=True):
    u_par = 0.0 if res_parent_frozen else 0.5
    osi = [
        {"os": "par", "gruppa": "g", "novaya": True},
        {"os": "kid", "gruppa": "g", "novaya": True, "roditel": "par"},
        {"os": "grandkid", "gruppa": "g", "novaya": True, "roditel": "kid"},
        {"os": "v5kid", "gruppa": "g", "novaya": False, "roditel": "par"},
        {"os": "orphan", "gruppa": "g", "novaya": True, "roditel": "net_takoy"},
        {"os": "strong_kid", "gruppa": "g", "novaya": True, "roditel": "par"},
    ]
    n = [[20, 3, 5, 0, 20, 20]] * 3                  # дети не измерены (n_akt < 20): активны только при родителе «вкл.»
    u = [[u_par, np.nan, np.nan, 0.0, 0.9, 0.7]] * 3
    return Z.reshenie(osi, stat_of(n, u))


def test_frozen_parent_freezes_children_chain_but_not_v5_or_orphan():
    res = _chain(True)
    assert res["par"][0] and res["kid"][0] and res["grandkid"][0]
    assert "родитель par" in res["kid"][1] and "родитель kid" in res["grandkid"][1]
    assert res["strong_kid"][0] is True               # дочерняя с ненулевым эффектом тоже замораживается вместе с родителем
    assert res["v5kid"][0] is False                   # ось v5 не замораживается никогда
    assert res["orphan"][0] is False                  # родитель вне списка осей - ничего не происходит


def test_not_frozen_parent_leaves_children_alone():
    res = _chain(False)
    assert not any(z for z, _ in res.values())


def test_chain_listed_in_reverse_order_still_propagates():
    osi = [
        {"os": "g", "gruppa": "x", "novaya": True, "roditel": "k"},
        {"os": "k", "gruppa": "x", "novaya": True, "roditel": "p"},
        {"os": "p", "gruppa": "x", "novaya": True},
    ]
    st = stat_of([[2, 2, 40]] * 3, [[np.nan, np.nan, 0.0]] * 3)
    res = Z.reshenie(osi, st)
    assert res["p"][0] and res["k"][0] and res["g"][0]


# ---------------- сквозной прогон с morris_v6 ----------------
def test_end_to_end_with_real_statistika():
    L = [2, 2, 2, 2]
    c = {W[0]: [1.0, 0.0, 0.04, 0.06], W[1]: [1.0, 0.0, 0.04, 0.06], W[2]: [1.0, 0.0, 0.04, 0.06]}
    osi = [{"os": n, "gruppa": "g", "novaya": True} for n in ("sil", "nul", "mal", "bol")]
    st = {}
    for okno in W:
        I, por = M.traektorii(L, 30, seed=2)
        Y = (I * np.asarray(c[okno])).sum(2).astype(float)
        keys = np.array([[hash(tuple(p)) for p in tr] for tr in I])
        st[okno] = M.statistika(M.effekty(Y, keys, por), B=300)
    res = Z.reshenie(osi, st)
    assert [res[n][0] for n in ("sil", "nul", "mal", "bol")] == [False, True, True, False]


# ---------------- выход: csv и yaml ----------------
def _fixture():
    osi = [{"os": "a", "gruppa": "g1", "novaya": True}, {"os": "b", "gruppa": "g2", "novaya": False},
           {"os": "c", "gruppa": "g1", "novaya": True, "roditel": "a"}]
    n = [[25, 30, 3]] * 3
    u = [[0.0, 0.0, 0.5]] * 3
    st = stat_of(n, u)
    for i, w in enumerate(W):
        st[w]["mu_star"] = np.array([0.1 * (i + 1), 0.2, 0.123456789])
        st[w]["sigma"] = np.array([0.5, 0.25, 1e-7])
    return osi, st, Z.reshenie(osi, st)


def test_csv_layout_and_values(tmp_path):
    osi, st, res = _fixture()
    p = tmp_path / "morris_Sx.csv"
    Z.zapisat_csv(p, {"код": "abc123", "вход": "def456"}, osi, st, res)
    lines = p.read_text(encoding="utf-8").splitlines()
    head = [l for l in lines if l.startswith("#")]
    assert any("разработка train4, до аудита" in l for l in head)
    assert "# код: abc123" in head and "# вход: def456" in head
    body = [l for l in lines if not l.startswith("#")]
    rows = list(csv.reader(body))
    assert tuple(rows[0]) == Z.KOLONKI and len(rows) == 1 + 3 * 3
    data = [dict(zip(rows[0], r)) for r in rows[1:]]
    assert [d["okno"] for d in data[:3]] == list(W) and [d["os"] for d in data] == ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    assert data[0]["n_akt"] == "25" and data[0]["zamorozit"] == "True" and data[1]["mu_star"] == "0.2"
    assert float(data[2]["mu_star"]) == pytest.approx(0.3) and float(data[8]["mu_star"]) == pytest.approx(0.123457)
    assert data[3]["zamorozit"] == "False" and data[3]["novaya"] == "False"
    assert data[6]["zamorozit"] == "True" and "родитель a" in data[6]["prichina"]     # c заморожена родителем a
    assert data[6]["U95"] == "0.5"                                                    # фактическое U95 сохраняется


def test_yaml_content_and_roundtrip(tmp_path):
    osi, st, res = _fixture()
    p = tmp_path / "zamorozka.yaml"
    zam = Z.zapisat_yaml(p, "2026-10-29", {"kub_sha": "aa", "kod": "bb", "csv": "cc"}, res)
    assert zam == ["a", "c"]
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert d["v"] == 1 and d["data"] == "2026-10-29" and d["vhody"] == {"kub_sha": "aa", "kod": "bb", "csv": "cc"}
    assert d["pravilo"] == {"okna": list(W), "U95_lt": 0.05, "n_akt_ge": 20}
    assert d["osi"] == ["a", "c"] and set(d["prichiny"]) == {"a", "c"}
    assert "b" not in d["osi"]


def test_yaml_empty_freeze_list(tmp_path):
    osi = [{"os": "a", "gruppa": "g", "novaya": True}]
    res = Z.reshenie(osi, stat_of([[20]] * 3, [[0.9]] * 3))
    p = tmp_path / "z.yaml"
    assert Z.zapisat_yaml(p, "d", {}, res) == []
    assert yaml.safe_load(p.read_text(encoding="utf-8"))["osi"] == []
