"""B1 (Ж-206, волна B, поток XGB): перенос Ж-204 без пересчёта — trial_id и числа равны исходным CSV, чужих алгоритмов в таблицах нет.
Мутации: подмена trial_id; число из другой строки; transferred=false; LightGBM-строка в переносе."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s5 import common
from models_v2.s6.xgb import tablex, transferx


@pytest.fixture(scope="module")
def tabs():
    return transferx.objectives(), transferx.features()


def src(name):
    return pd.read_csv(common.REP / name, comment="#")


def test_objective_rows_match_sources_and_exclude_lightgbm(tabs):
    o, _ = tabs
    assert list(o["id"]) == list(transferx.OBJ_IDS) and "obj_l3l" not in set(o["id"])
    assert transferx.verify(o) == []
    s = src("OBJECTIVE_EXPERIMENTS.csv").set_index("row_id")
    for rid in ("obj_l3x", "obj_l4", "obj_l5b"):
        r = o[o["id"] == rid].iloc[0]
        assert r["trial_id"] == s.loc[rid, "trial_id"] and r["qlike_all"] == s.loc[rid, "qlike_all"] and r["crit_mean"] == s.loc[rid, "q_loyo"]
    assert (o["crit_pen"].dropna() >= o["crit_mean"][o["crit_pen"].notna()]).all()      # штраф за худший год не уменьшает критерий


def test_baseline_b2_row_has_trial_of_base_and_zero_diff(tabs):
    o, _ = tabs
    b = o[o["id"] == "ctrl_b2"].iloc[0]
    assert b["trial_id"] == "5ad9a4572a87" and b["d_b2"] == pytest.approx(0.0, abs=1e-15)


def test_feature_rows_match_sources_no_windows_no_hybrids(tabs):
    _, f = tabs
    assert transferx.verify(f) == []
    ids = set(f["id"])
    assert {"b2_f1", "b2_f4", "b2_top2", "b2_no_imoex", "c1_f4_f1", "c1_f4_range", "c3_f3calm_rv"} <= ids
    assert not any(i.startswith(("w_", "c2_", "c3_r4", "c1_opt")) for i in ids)          # окна, смеси с HAR, режимные правила с HAR, Optuna — не здесь
    r = f[f["id"] == "b2_f4"].iloc[0]
    assert r["trial_id"] == "344764e88789" and r["features"] == "Б-2+f4" and r["transferred"]
    assert f["trial_id"].is_unique and f["id"].is_unique


def test_status_in_every_row(tabs):
    for t in tabs:
        assert t["status"].str.contains("перенос Ж-204 без пересчёта").all() and t["status"].str.contains("HISTORICAL WALK-FORWARD").all()


def test_mutation_swapped_trial_id_is_caught(tabs):
    o, _ = tabs
    m = o.copy()
    m.loc[m["id"] == "obj_l4", "trial_id"] = m.loc[m["id"] == "obj_l3x", "trial_id"].iloc[0]
    assert transferx.verify(m) != []


def test_mutation_number_from_other_row_is_caught(tabs):
    o, _ = tabs
    m = o.copy()
    m.loc[m["id"] == "obj_l4", "qlike_all"] = m.loc[m["id"] == "obj_l3x", "qlike_all"].iloc[0]
    assert transferx.verify(m) != []


def test_mutation_not_transferred_or_unknown_trial_is_caught(tabs):
    o, _ = tabs
    m = o.copy()
    m["transferred"] = m["transferred"].astype(object)
    m.loc[m["id"] == "obj_l4", "transferred"] = False
    assert any("transferred=false" in b for b in transferx.verify(m))
    m2 = o.copy()
    m2.loc[0, "trial_id"] = "ffffffffffff"
    assert any("нет в Ж-204" in b for b in transferx.verify(m2))


def test_tables_do_not_cross_results_between_rows(tabs):
    o, f = tabs
    new = [{"id": "x3_gamma_f4", "trial_id": "t1", "transferred": False, "direction": "objective", "qlike_all": 0.2}]
    assert tablex.no_crossing(tablex.frame(o, new))
    dup = new + [{"id": "x4_qlike_f4", "trial_id": "t1", "transferred": False, "direction": "objective", "qlike_all": 0.3}]
    assert not tablex.no_crossing(tablex.frame(o, dup))                                  # один trial_id у двух строк — чужой результат


def test_split_new_puts_feature_rows_and_control_in_right_tables():
    rows = {"ctrl_f4": {"direction": "control"}, "x3_gamma_f4": {"direction": "objective"}, "f4_f2": {"direction": "feature"}}
    obj, feat = tablex.split_new(rows)
    assert [r["direction"] for r in obj] == ["control", "objective"] and [r["direction"] for r in feat] == ["control", "feature"]
    assert np.isfinite(0.0)
