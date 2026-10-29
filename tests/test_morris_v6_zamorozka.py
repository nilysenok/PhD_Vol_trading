"""V6-14, часть 2: заморозка по Моррису и выход. Известный ответ: линейная функция + нулевая ось → нулевая ось замораживается.

Мутации: «хотя бы в одном окне» вместо «во всех трёх»; без порога n_akt ≥ 20; оси v5 тоже замораживаются; `<=` вместо `<` на границе 0,05.
"""
from __future__ import annotations

import numpy as np
import pytest

from fastbt import morris_v6 as M
from fastbt import morris_v6_zamorozka as Z

OSI = [  # (имя, новая, коэффициенты по окнам ≤2016, ≤2017, ≤2018)
    ("sil", True, (1.0, 1.0, 1.0)),
    ("nul", True, (0.0, 0.0, 0.0)),
    ("v5nul", False, (0.0, 0.0, 0.0)),
    ("mal", True, (0.04, 0.04, 0.04)),
    ("bol", True, (0.06, 0.06, 0.06)),
    ("odno", True, (0.0, 0.0, 0.5)),
    ("rezh", True, (0.0, 0.0, 0.0)),  # дочерняя от sil: активна только при sil = вкл. → n_akt < 20 при r = 24
]


def _stat(r=24, seed=3):
    L = [2] * len(OSI)
    I, por = M.traektorii(L, r, seed)
    kan = I.copy()
    kan[:, :, 6] = np.where(I[:, :, 0] == 0, 0, I[:, :, 6])
    keys = np.array([[hash(tuple(p)) for p in tr] for tr in kan])
    out = {}
    for w, okno in enumerate(Z.OKNA):
        c = [o[2][w] for o in OSI]
        out[okno] = M.statistika(M.effekty((kan * np.asarray(c)).sum(2).astype(float), keys, por), B=300)
    return out


def _osi():
    return [{"os": n, "gruppa": "kontekst", "novaya": nv, **({"roditel": "sil"} if n == "rezh" else {})} for n, nv, _ in OSI]


def test_izvestnyj_otvet_nulevaya_os_zamorazhivaetsya():
    st = _stat()
    res = Z.reshenie(_osi(), st)
    assert res["nul"][0] is True and res["mal"][0] is True  # μ* = 0 и U95 = 0,04 < 0,05 во всех окнах
    assert res["sil"][0] is False and res["bol"][0] is False
    assert st["<=2016"]["mu_star"][0] == pytest.approx(1.0) and st["<=2016"]["mu_star"][1] == 0.0


def test_ne_vo_vseh_oknah_ne_zamorazhivaetsya():
    res = Z.reshenie(_osi(), _stat())
    assert res["odno"][0] is False  # нулевая в двух окнах, 0,5 в третьем (мутация «хотя бы в одном» заморозила бы)


def test_os_v5_ne_zamorazhivaetsya():
    res = Z.reshenie(_osi(), _stat())
    assert res["v5nul"][0] is False and "v5" in res["v5nul"][1]


def test_malo_aktivnyh_shagov_ne_izmereno():
    st = _stat()
    assert 0 < st["<=2016"]["n_akt"][6] < Z.MIN_AKT
    res = Z.reshenie(_osi(), st)
    assert res["rezh"][0] is False and "не измерено" in res["rezh"][1]


def _fake(n, u, k=1):
    return {w: {"n_akt": np.array([n] * k), "U95": np.array([u] * k)} for w in Z.OKNA}


def test_granitsa_poroga_strogo_menshe_i_n_akt_20_vklyuchitelno():
    o = [{"os": "a", "gruppa": "g", "novaya": True}]
    assert Z.reshenie(o, _fake(20, 0.0499))["a"][0] is True
    assert Z.reshenie(o, _fake(20, 0.05))["a"][0] is False  # U95 = 0,05 — не «меньше»
    assert Z.reshenie(o, _fake(19, 0.0))["a"][0] is False
    assert Z.reshenie(o, _fake(50, float("nan")))["a"][0] is False  # NaN — не заморозка


def test_roditel_zamorozhen_zamorazhivaet_dochernie_tsepochkoi():
    osi = [{"os": "p", "gruppa": "g", "novaya": True}, {"os": "d", "gruppa": "g", "novaya": True, "roditel": "p"},
           {"os": "vn", "gruppa": "g", "novaya": True, "roditel": "d"}, {"os": "v5", "gruppa": "g", "novaya": False, "roditel": "p"}]
    st = _fake(30, 0.9, 4)
    for w in Z.OKNA:
        st[w]["U95"][0] = 0.0
    res = Z.reshenie(osi, st)
    assert [res[x][0] for x in ("p", "d", "vn", "v5")] == [True, True, True, False]


def test_okna_tolko_tri_bez_2019():
    st = _fake(30, 0.0)
    st["<=2019"] = st.pop("<=2018")
    with pytest.raises(ValueError):
        Z.reshenie([{"os": "a", "gruppa": "g", "novaya": True}], st)


def test_zapis_csv_i_yaml(tmp_path):
    st, osi = _stat(), _osi()
    res = Z.reshenie(osi, st)
    Z.zapisat_csv(tmp_path / "morris_S1.csv", {"seed": M.SEED, "kod": "abc"}, osi, st, res)
    zam = Z.zapisat_yaml(tmp_path / "zamorozka.yaml", "2026-11-01", {"kub_S1": "sha"}, res)
    rows = (tmp_path / "morris_S1.csv").read_text("utf-8").splitlines()
    assert rows[0] == '# "v": "1"' and "# seed: 20261101" in rows
    i = rows.index(",".join(Z.KOLONKI))
    assert len(rows) - i - 1 == len(OSI) * 3 and rows[i + 4].startswith("nul,kontekst,True,<=2016,")
    import yaml
    d = yaml.safe_load((tmp_path / "zamorozka.yaml").read_text("utf-8"))
    assert d["v"] == 1 and d["osi"] == zam == ["mal", "nul"] and d["pravilo"]["U95_lt"] == 0.05
