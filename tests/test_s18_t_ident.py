"""S-T (Ж-256): тождества — Σ вкладов лет = Δ; QLIKE окна = взвешенное по n среднее годовых; срезы и обрезка на синтетике с известным ответом; совпадение исключённых дат."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import r_eval_tab as RT, t_slices as TS, t_table as TT


@pytest.fixture(scope="module")
def b():
    return TT.build()


def test_sum_of_contribs_equals_delta(b):
    y = b["t"]["years"]
    p = y[y["base"] != "-"]
    assert set(p["base"]) == {"W5-G", "X1"} and set(p["series"]) == set(TT.SA + TT.SR + TT.SC)
    for _, g in p.groupby(["series", "base", "kind"]):
        assert abs(g["contrib_pct"].sum() - g["delta_total_pct"].iloc[0]) < 1e-12


def test_period_set_and_n(b):
    y = b["t"]["years"]
    g = y[(y["series"] == "K1") & (y["kind"] == "period")]
    assert list(g["period"]) == [str(k) for k in range(2018, 2026)] + ["2026_до_27.02", "2026_F"] and g["n"].sum() == 2198
    h = y[(y["series"] == "K1") & (y["kind"] == "half")]
    assert list(h["n"]) == [1010, 1188]


def _synth():
    tp = pd.bdate_range("2021-06-01", periods=400)
    n = len(tp)
    d = np.linspace(-0.2, 0.5, n)
    d = d[np.random.default_rng(1).permutation(n)]
    ls = {k: np.ones(n) for k in ("W5-G", "X1", "K2", "K3", "S-CK", "S-A2")}
    ls.update({"S-RCK": 1.0 + d, "S-RA2": 1.0 + d})
    return tp, ls, d


def test_slices_known_answer_on_synthetic():
    tp, ls, d = _synth()
    n = len(tp)
    s = TS.slices(tp, ls)
    g = s[(s["candidate"] == "S-RCK") & (s["base"] == "W5-G")].reset_index(drop=True)
    i22 = int(np.where(tp == pd.Timestamp("2022-02-24"))[0][0])
    y = tp.year.to_numpy()
    srt = np.sort(d)
    want = [100.0 * d.mean(), 100.0 * (d.sum() - d[i22]) / (n - 1), 100.0 * d[y != 2022].sum() / int((y != 2022).sum()), 100.0 * srt[22:n - 22].sum() / (n - 44)]
    assert np.allclose(g["delta_pct"], want, atol=1e-12)
    assert list(g["n"]) == [n, n - 1, int((y != 2022).sum()), n - 44]
    assert np.allclose(g["q_base"], 1.0) and abs(RT.trim_delta(ls["S-RCK"], ls["W5-G"]) - want[3]) < 1e-12


def test_denominator_is_base_on_remaining_dates():
    tp, ls, d = _synth()
    ls["X1"] = 1.0 + np.linspace(0, 1, len(tp))                      # база с трендом: знаменатель меняется со срезом
    s = TS.slices(tp, ls)
    g = s[(s["candidate"] == "S-RCK") & (s["base"] == "X1")].reset_index(drop=True)
    m = TS.masks(tp, ls["S-RCK"] - ls["X1"])
    for i in range(4):
        assert abs(g.at[i, "q_base"] - ls["X1"][m[i]].mean()) < 1e-15
    assert g.at[0, "q_base"] != g.at[2, "q_base"]


def test_trimsets_identity_on_synthetic():
    tp, ls, d = _synth()
    ts = TS.trimsets(tp, ls)
    pr = ts[ts["kind"] == "пара сравнений"]
    assert len(pr) == 28 and (pr["n_each"] == 44).all()
    same = pr[(pr["a"] == "S-RCK | W5-G") & (pr["b"] == "S-RCK | X1")].iloc[0]       # база обеих = 1 → те же даты
    assert same["n_common"] == 44 and same["jaccard"] == 1.0
    assert (abs(pr["jaccard"] - pr["n_common"] / (88 - pr["n_common"])) < 1e-15).all()


def test_real_trimsets_not_identical_and_lists(b):
    ts = b["t"]["trimsets"]
    pr = ts[ts["kind"] == "пара сравнений"]
    assert pr["n_common"].min() < 44 and pr["n_common"].max() < 44
    lst = ts[ts["kind"] == "перечень дат"]
    assert len(lst) == 4 and all(len(x.split("; ")) == 22 for x in lst["dates"])
    assert "не одинаковы" in b["t"]["report"]


def test_report_order_and_marks(b):
    r = b["t"]["report"]
    assert r.index("## 1. Полная таблица") < r.index("## 2. По годам") < r.index("## 3. Срезы") < r.index("## 4. Совпадение")
    assert r.count("не независимый OOS") >= 2 and "диагностика" in r.lower() and "не рассчитывался" in r
