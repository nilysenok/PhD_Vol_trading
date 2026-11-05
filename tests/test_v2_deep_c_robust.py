"""G7 C: соседи ±1 шаг (одна ось), ранжирование «лучших 3» по внутреннему критерию, Холм с N = все trials (MC6, MC7, MC13, MC14)."""
import numpy as np
import pandas as pd

from models_v2.s5.c import fitx, robust, space, tab_c

HP = fitx.HPX(2, 40.0, 6.0, 0.05, 0.7, 0.6, 300, 0.01, 1.0)


def diff_axes(a, b):
    return [x for x in space.AXES if getattr(a, x) != getattr(b, x)]


def test_neighbor_changes_exactly_one_axis_by_declared_step():
    for ax in space.AXES:
        for sg in (-1, 1):
            n = space.shift(HP, ax, sg)
            if n is not None:
                assert diff_axes(HP, n) == [ax]
    assert space.shift(HP, "depth", 1).depth == 3 and space.shift(HP, "depth", -1).depth == 1
    assert space.shift(fitx.HPX(3, 40.0, 6.0, 0.05, 0.7, 0.6, 300), "depth", 1) is None
    assert space.shift(fitx.HPX(3, 40.0, 6.0, 0.05, 0.7, 0.6, 300), "gamma", -1) is None
    assert abs(space.shift(HP, "lr", 1).lr - 0.075) < 1e-12 and space.shift(HP, "min_child", -1).min_child == 20.0


def test_neighbors_by_year_all_years_or_none():
    hp = {2017: HP, 2018: fitx.HPX(3, 40.0, 6.0, 0.05, 0.7, 0.6, 300, 0.0, 0.0)}
    nb = space.neighbors_by_year(hp)
    assert ("depth", 1) not in nb and ("depth", -1) in nb and ("gamma", -1) not in nb
    for k, v in nb.items():
        assert all(diff_axes(hp[Y], v[Y]) == [k[0]] for Y in hp)


def test_top3_by_internal_criterion_not_description():
    df = pd.DataFrame({"row_id": list("abcde"), "perspective": [True, True, True, True, False], "rel_gain": [0.05, 0.04, 0.03, 0.01, 0.2],
                       "d_harj_no_240222": [-0.001] * 5, "qlike_all": [0.30, 0.21, 0.22, 0.10, 0.1]})
    assert list(tab_c.top3(df)["row_id"]) == ["a", "b", "c"]
    mut = df[df.perspective].sort_values("qlike_all").head(3)          # мутант MC14: отбор по описанию
    assert list(mut["row_id"]) != ["a", "b", "c"]


def test_holm_uses_all_program_trials():
    df = pd.DataFrame({"p_harj": [0.001, 0.02, 0.5], "p_b2": [0.01, 0.2, 0.9]})
    a = tab_c.holm_cols(df)
    ref = tab_c.holm_cols(df, tab_c.FAMILY_C, "_ref40")
    assert tab_c.FAMILY == 90 and tab_c.FAMILY_C == 40
    assert (a["p_holm_harj"] >= ref["p_holm_harj_ref40"] - 1e-15).all() and a["p_holm_harj"].iloc[0] > ref["p_holm_harj_ref40"].iloc[0]
    assert np.isclose(a["p_holm_harj"].iloc[0], 0.001 * 90)       # мутант MC13 (Холм по 3 строкам) дал бы 0,003


def test_seed_table_rule_needs():
    assert robust.needs("c1_opt") == ("opt",) and robust.needs("c3_f3calm_g4") == ("f4", "top2") and robust.needs("c2_h3_roll_opt") == ("opt",)
    assert robust.SEEDS == (42, 43, 44, 45, 46)
