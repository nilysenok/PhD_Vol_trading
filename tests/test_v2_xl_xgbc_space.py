"""C4 (Ж-206, волна C, поток XGB): пространство осей, зажим, дубли опоры, соседи, trial_id, реестр и бюджет HP <= 55. Синтетика.
Мутации: зажим снят (тест свойства краснеет), ось/u/семья не в trial_id, дубль учтён как trial, повторная запись удваивает."""
import dataclasses
import json

import pandas as pd
import pytest

from models_v2.s4 import fit
from models_v2.s6.xgb import c_rows, c_space, cfgx

BASE = {Y: c_space.to_hpc(fit.HP(3, 10.0, 2.0, 0.05, 0.8)) for Y in range(2017, 2027)}


def test_x1_has_18_points_two_per_axis():
    pts = c_space.x1_points()
    assert len(pts) == 18 and len(set(pts)) == 18 and {a for a, _ in pts} == set(c_space.AXES) and len(c_space.AXES) == 9


def test_one_step_values():
    h = BASE[2020]
    assert c_space.shift_hp(h, "max_depth", 1).depth == 4 and c_space.shift_hp(h, "min_child_weight", -1).min_child == 5.0
    assert c_space.shift_hp(h, "learning_rate", 1).lr == pytest.approx(0.075) and c_space.shift_hp(h, "subsample", -1).subsample == pytest.approx(0.7)
    assert c_space.shift_hp(h, "colsample_bytree", -1).colsample == pytest.approx(0.8) and c_space.shift_hp(h, "reg_alpha", 3).alpha == 3.0
    assert c_space.shift_hp(h, "reg_lambda", 1).l2 == pytest.approx(6.0) and c_space.shift_hp(h, "gamma", 4).gamma == pytest.approx(0.2)
    assert c_space.shift_hp(h, "max_delta_step", 1).mds == pytest.approx(0.1)


def clamp_property(shift) -> bool:
    """Любой сдвиг остаётся в границах оси."""
    return all(c_space.AXES[a][3][0] <= getattr(shift(BASE[2020], a, u), c_space.AXES[a][0]) <= c_space.AXES[a][3][1]
               for a in c_space.AXES for u in (-9, -2, -1, 1, 2, 9))


def test_clamp_holds_and_mutation_without_clamp_is_caught():
    assert clamp_property(c_space.shift_hp)

    def no_clamp(hp, axis, u):
        fld, kind, st = c_space.AXES[axis][:3]
        v = getattr(hp, fld)
        return dataclasses.replace(hp, **{fld: v * st ** u if kind == "mul" else v + u * st})
    assert not clamp_property(no_clamp)


def test_point_clamped_in_every_year_is_duplicate_of_base():
    low = {Y: dataclasses.replace(h, depth=1) for Y, h in BASE.items()}
    assert c_space.years_changed(low, c_space.apply(low, {"max_depth": -1.0})) == 0
    assert c_space.years_changed(BASE, c_space.apply(BASE, {"max_depth": -1.0})) == 10
    assert c_space.years_changed(BASE, c_space.apply(BASE, {})) == 0                      # u=0 / пустой сдвиг — опора


def test_clamped_years_counts_only_clamped():
    part = {Y: dataclasses.replace(h, colsample=0.3 if Y < 2020 else 1.0) for Y, h in BASE.items()}
    assert c_space.clamped_years(part, "colsample_bytree", -1.0) == 3                      # 2017–2019: 0,3 − 0,2 < 0,3
    assert c_space.clamped_years(BASE, "learning_rate", 1.0) == 0


def test_apply_two_axes_order_independent_and_per_year():
    a = c_space.apply(BASE, {"gamma": 1.0, "max_depth": 1.0})
    assert all(h.gamma == pytest.approx(0.05) and h.depth == 4 for h in a.values())
    base2 = {**BASE, 2022: dataclasses.replace(BASE[2022], depth=2)}
    assert c_space.apply(base2, {"max_depth": 1.0})[2022].depth == 3 and c_space.apply(base2, {"max_depth": 1.0})[2021].depth == 4


def test_neighbors_one_step_in_range_and_u0_is_base():
    nb = c_space.neighbors({"reg_alpha": 1.0})
    assert {} in nb and {"reg_alpha": 2.0} in nb and len(nb) == 2
    assert c_space.neighbors({"reg_alpha": 5.0}) == [{"reg_alpha": 4.0}]                  # верхняя граница диапазона u
    two = c_space.neighbors({"gamma": 2.0, "max_depth": -1.0})
    assert len(two) == 4 and {"gamma": 2.0, "max_depth": 0.0} not in two and {"gamma": 2.0} in two


def test_trial_id_deterministic_unique_and_depends_on_axis_u_family():
    ids = {c_space.trial_id("l2", {a: u}) for a, u in c_space.x1_points()}
    assert len(ids) == 18 and c_space.trial_id("l2", {"gamma": 1.0}) == c_space.trial_id("l2", {"gamma": 1.0})
    assert c_space.trial_id("l2", {"gamma": 1.0}) != c_space.trial_id("gamma", {"gamma": 1.0})
    assert c_space.trial_id("l2", {"gamma": 1.0}) != c_space.trial_id("l2", {"gamma": 4.0}) != c_space.trial_id("l2", {"max_delta_step": 4.0})
    assert c_space.trial_id("l2", {"a": 1.0, "b": 2.0}) == c_space.trial_id("l2", {"b": 2.0, "a": 1.0})


def rec(tid):
    return {"trial_id": tid, "wave": "C-xgb", "direction": "hyperparameters", "config": "{}", "metric_inner": "m", "seed": 42, "time": "t"}


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    ck = tmp_path / "CHECKPOINT.json"
    ck.write_text(json.dumps({"algorithm": "xgboost", "used": {"obj_feat": 9, "hp": 0, "temporal_regime": 0, "ens_hybrid": 0}}))
    monkeypatch.setattr(c_rows, "TRIALS", tmp_path / "OPTUNA_TRIALS.csv")
    monkeypatch.setattr(c_rows, "CKPT", ck)
    pd.DataFrame([{**rec("b2old"), "wave": "B-xgb", "direction": "objective"}]).to_csv(tmp_path / "OPTUNA_TRIALS.csv", index=False)
    return ck


def test_registry_append_only_and_counts_only_wave_c(reg):
    assert c_rows.register([rec("a1"), rec("a2")]) == 2 and c_rows.register([rec("a1"), rec("a2")]) == 0     # повтор не удваивает
    used = json.loads(reg.read_text())["used"]
    assert used["hp"] == 2 and used["obj_feat"] == 9 and c_rows.n_hp_trials() == 2                              # B-строки не считаются, obj_feat не тронут
    assert len(pd.read_csv(c_rows.TRIALS)) == 3


def test_budget_over_55_raises(reg):
    c_rows.register([rec("t%d" % i) for i in range(55)])
    with pytest.raises(ValueError):
        c_rows.register([rec("extra")])


def test_head_matches_cfgx():
    assert list(rec("x")) == cfgx.TRIAL_HEAD
