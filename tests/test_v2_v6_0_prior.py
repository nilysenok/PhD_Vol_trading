"""V6-0, G2/G3: прежние гибриды SBER по датам — QLIKE совпадает с отчётами (≤ 1e-4), класс сопоставимости задан с доказательством, ориентир 0,2081 — потолок; ряды без прогнозов — без оценки;
BEST — лучший ряд по окну на общих датах. Мутации: дубль контроля узнаётся только при побитном равенстве; ряд без правила класса — отказ; потолок не попадает в PIT-набор."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_0_best as BE, v6_0_class as CL, v6_0_load as LD, v6_0_prior as PR


@pytest.fixture(scope="module")
def real():
    w, meta = LD.load()
    prior = PR.prior_table(w, meta)
    return w, meta, prior


def test_reported_qlike_matches_programs_within_1e4(real):
    p = real[2]
    x = p[p["reported_qlike"].notna()]
    assert len(x) >= 70 and (x["diff_reported"] <= PR.TOL_REP).all()


def test_reference_02081_is_ceiling_with_evidence(real):
    p = real[2].set_index("series")
    r = p.loc["J210a:A1_roll_126"]
    assert round(r["qlike_own"], 4) == 0.2081 and r["class"] == CL.CEIL and "crit_wf" in r["evidence"] and "GATES_HF.md:55" in r["evidence"]


def test_ceiling_not_in_pit_set_and_every_row_has_class_and_evidence(real):
    p = real[2]
    pit, ceil = BE.split_sets(p)
    assert "J210a:A1_roll_126" in ceil and "J210a:A1_roll_126" not in pit and set(BE.CTRL) <= set(pit)
    prog = p[p["class"].isin([CL.PIT, CL.CEIL])]
    assert prog["evidence"].str.len().gt(20).all() and prog["reason"].str.len().gt(10).all()


def test_no_forecast_rows_without_estimate(real):
    p = real[2]
    x = p[p["class"] == "нет прогнозов"]
    assert "V3 (Ж-221)" in set(x["series"]) and x["qlike_own"].isna().all() and x["n_dates"].isna().all()


def test_a1_roll_126_is_family_winner_by_crit_wf(real):
    """Доказательство потолка в данных: A1_roll_126 — минимум crit_SBER среди 11 родственных правил весов (выбор по метрике на тех же целях)."""
    from models_v2.s13 import v6_0_crit as CR
    w, _, p = real
    fam = [s for s in w.columns if s.startswith(("J210a:A1_", "J211w:H1_"))]
    cls = {s: "потолок" for s in fam}
    t = CR.rows_for(w, fam, cls).set_index("series")["crit"]
    assert t.idxmin() == "J210a:A1_roll_126" and len(fam) >= 11


def test_duplicates_of_controls_detected_bitwise(real):
    w = real[0]
    assert PR.dup_of(w, "DEEP:s2_harj") == "K1" and PR.dup_of(w, "J210comp:XJ") == "K2" and PR.dup_of(w, "J210comp:LF") == "K3"
    w2 = w.copy()
    w2.loc[w2.index[5], "DEEP:s2_harj"] *= 1.0000001                            # мутация: микроскопическое расхождение — уже не дубль
    assert PR.dup_of(w2, "DEEP:s2_harj") == ""


def test_class_without_rule_refuses():
    with pytest.raises(ValueError):
        CL.classify("X:y", "UNKNOWN", "f")


def test_best_table_known_answer_and_common_dates(real):
    w, _, p = real
    best, res = BE.best_table(w, p)
    b = res["best"].set_index(["set", "window"])
    assert res["partial"] == [] and res["common"]["n_dates"].tolist() == [2271, 2271]
    assert round(b.loc[(BE.SETS[1], "all"), "qlike"], 4) == 0.2081 and b.loc[(BE.SETS[1], "all"), "class_"] == CL.CEIL
    assert b.loc[(BE.SETS[0], "all"), "class_"] != CL.CEIL and b.loc[(BE.SETS[0], "all"), "qlike"] > b.loc[(BE.SETS[1], "all"), "qlike"]


def test_best_rows_picks_minimum_synthetic():
    met = pd.DataFrame([dict(series="a", class_="контроль", window="all", n=5, qlike=0.3), dict(series="b", class_="потолок", window="all", n=5, qlike=0.1), dict(series="c", class_="PIT-сопоставим", window="all", n=5, qlike=0.2)])
    r = BE.best_rows(met, ["a", "c"], ["b"]).set_index("set")
    assert r.loc[BE.SETS[0], "series"] == "c" and r.loc[BE.SETS[1], "series"] == "b"
    bad = BE.best_rows(met, ["a", "b", "c"], []).set_index("set")             # мутация: потолок в PIT-наборе — лучший PIT становится потолком
    assert bad.loc[BE.SETS[0], "series"] == "b"
    assert np.isclose(r.loc[BE.SETS[1], "second_qlike"], 0.2)
