"""V4-H (G40): ценность источников сверх K8 — известный ответ на малой таблице: доля строк вне [min(H, M), max(H, M)], снижение потерь на них, доля общего выигрыша; перепутанный знак выигрыша ловится."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_hd_value as V


def frame():
    # H = 1, M = 2 (k2 = k3 = 2): диапазон [1; 2]. RV: 0.5 (ниже), 1.5 (внутри), 4.0 (выше); K8 = √2 · 1 ≈ 1.4142.
    n = 3
    c = pd.DataFrame({"ticker": ["A"] * n, "group": ["M16"] * n, "t_plus": pd.to_datetime(["2017-03-01", "2017-03-02", "2017-03-03"]), "rv_tplus": [0.5, 1.5, 4.0],
                      "k1": [1.0] * n, "k2": [2.0] * n, "k3": [2.0] * n, "K8": [np.sqrt(2.0)] * n, "Hx": [1.0, 1.6, 3.0]})
    return c


def test_out_of_range_known_answer():
    assert list(V.out_of_range(frame())) == [True, False, True]


def test_build_gain_and_shares_known_answer():
    c = frame()
    comp = pd.DataFrame([{"series": s, "window": w, "qlike": 0.0} for s in ("K8", "Hx") for w in ("2017_2021", "2022_2025", "2026", "all")])
    tests = pd.DataFrame(columns=["config", "comparator", "window", "dm_stat", "dm_p", "p_holm"])
    v = V.SOURCE.copy()
    try:
        V.SOURCE["Hx"] = "тест"
        r = V.build(c, comp, tests, ["Hx"])
    finally:
        V.SOURCE.clear()
        V.SOURCE.update(v)
    a = r[r["window"] == "2017_2021"].iloc[0]
    q = lambda f: [x / y - np.log(x / y) - 1 for x, y in zip(c["rv_tplus"], f)]
    gain = np.array(q(c["K8"])) - np.array(q(c["Hx"]))
    assert a["share_rows_out_of_range"] == pytest.approx(2 / 3) and a["n_rows"] == 3
    assert a["gain_per_row_out"] == pytest.approx(gain[[0, 2]].mean()) and a["gain_per_row_in"] == pytest.approx(gain[1])
    assert a["share_of_gain_out"] == pytest.approx(gain[[0, 2]].sum() / gain.sum())
    assert a["gain_per_row_out"] > 0 > a["gain_per_row_in"]                    # знак выигрыша: плюс = H лучше K8 (мутация «перепутан знак» дала бы обратное)
