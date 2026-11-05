"""D: мутации проверяющего M3c (N Холма = длина группы) и M4 (лучший ансамбль по описанию qlike_all вместо q_int). Синтетика, без сети."""
import numpy as np
import pandas as pd

from models_v2.s5.d import d2_d, stages_d, stats_d, tab_d


def test_holm_rows_uses_family_n_not_group_length():
    p = np.array([0.004, 0.01, 0.2])
    rids = ["a", "b", "c"]
    rec = [{"test": t, "row_id": r, "base": b, "period": per, "p": pv}
           for t, per in (("DM", "весь период"), ("DM", "без 24.02.2022"), ("boot", "весь период")) for b in ("HAR-J(б)", "Б-2") for r, pv in zip(rids, p)]
    out = d2_d.holm_rows(pd.DataFrame(rec), rids)
    for _, g in out.groupby(["test", "period", "base"]):
        assert np.allclose(g["p_holm_N106"], stats_d.holm_n(g["p"].to_numpy(), tab_d.FAMILY))
        assert np.allclose(g["p_holm_N116"], stats_d.holm_n(g["p"].to_numpy(), tab_d.FAMILY_REF))
        assert (g["p_holm_N106"] > stats_d.holm_n(g["p"].to_numpy(), len(g)) + 1e-9).any()   # N = длина группы (мутант) дал бы меньше
    assert np.isclose(out["p_holm_N106"].iloc[0], min(1.0, 0.004 * tab_d.FAMILY))


def test_holm_rows_other_groups_untouched():
    df = pd.DataFrame([{"test": "DM", "row_id": "a", "base": "Б-2", "period": "2022", "p": 0.001},
                       {"test": "DM", "row_id": "z", "base": "Б-2", "period": "весь период", "p": 0.001}])
    out = d2_d.holm_rows(df, ["a"])
    assert out["p_holm_N106"].isna().all()    # подпериод и не-финалист Холмом не правятся


def test_plateau_empty_set_is_not_checked_not_ok():
    from models_v2.s5.d import robust_d
    pl = pd.DataFrame({"unit": ["f4", "opt"], "kind": ["booster_C", "booster_C"], "plateau_ok": [True, True]})
    assert robust_d.plateau_status(pl, "e3_roll", "e3_roll", 1.0) == robust_d.NOT_CHECKED   # HAR + LGBM: XGB-членов нет
    assert not robust_d.plateau_flags(pl, "e3_roll", "e3_roll", 1.0)
    assert robust_d.plateau_status(pl, "c1_opt", "e3_roll", 1.0) == "ok"
    assert robust_d.plateau_status(pl[pl.unit != "opt"], "c1_opt", "e3_roll", 1.0) == "нет"   # бустер без строки соседей — не ok
    assert robust_d.plateau_status(pl, "e1_roll", "e1_roll", 0.5) == "нет"                      # есть XGB-член, но соседей правила мало


def test_pick_best_uses_q_int_not_description_and_not_other_column():
    df = pd.DataFrame({"row_id": ["e1_roll", "e3_roll", "e6_rv"], "q_int": [0.19, 0.17, 0.18], "qlike_all": [0.18, 0.21, 0.17], "q_wf_dev": [0.16, 0.19, 0.15]})
    best, dev = stages_d.pick_best(df)
    assert best == "e3_roll"           # мутант (qlike_all) выбрал бы e6_rv
    assert dev == "e6_rv" and best != dev
