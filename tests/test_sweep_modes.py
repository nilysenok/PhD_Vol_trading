"""Э1-а (Ж-031): режимы — OAT, факторный, Моррис (аналитика), Optuna (воспроизводимость, n_jobs ≤ 8)."""
import numpy as np
import pytest

from fastbt import sweep_modes as sm
from fastbt.sweep import Spec, Sweep
from fastbt.sweep_synth import SynthEvaluator

LV = {"x1": [0, .25, .5, .75, 1.0], "x2": [0, .25, .5, .75, 1.0], "x3": [0, .25, .5, .75, 1.0]}


def test_morris_linear_known_mu_star(monkeypatch):
    f = lambda p: 3 * p["x1"] - 1 * p["x2"] + 0 * p["x3"]       # noqa: E731
    m = sm.morris_effects(f, LV, r=10, seed=0)
    assert [m[k]["mu_star"] for k in LV] == pytest.approx([3.0, 1.0, 0.0])
    assert m["x2"]["mu"] == pytest.approx(-1.0) and m["x1"]["sigma"] == pytest.approx(0.0, abs=1e-12)
    assert all(m[k]["n"] == 10 for k in LV)
    real = sm.morris_trajectory
    monkeypatch.setattr(sm, "morris_trajectory", lambda lv, rng: [(i, n, 1.0 if n else 0.0) for i, n, _ in real(lv, rng)])
    bad = sm.morris_effects(f, LV, r=10, seed=0)
    assert bad["x1"]["mu_star"] != pytest.approx(3.0)           # мутация: шаг не нормирован -> тест видит


def test_morris_interaction_has_sigma():
    m = sm.morris_effects(lambda p: p["x1"] * p["x2"], LV, r=10, seed=1)
    assert m["x1"]["sigma"] > 0.05 and m["x3"]["mu_star"] == 0.0
    assert 0.0 < m["x1"]["mu_star"] < 1.0


def test_morris_trajectory_one_step_per_param():
    tr = sm.morris_trajectory(LV, np.random.default_rng(0))
    assert len(tr) == 4 and sorted(n for _, n, _ in tr[1:]) == ["x1", "x2", "x3"]
    for (a, _, _), (b, n, du) in zip(tr, tr[1:]):
        assert abs(b[n] - a[n]) == 1 and abs(du) == pytest.approx(0.25)
        assert sum(a[k] != b[k] for k in a) == 1


def test_oat_and_factorial_points():
    pts = sm.oat_points({"a": 5, "w": 1}, {"a": [3, 5, 10], "w": [1, 2]})
    assert pts[0] == {"a": 5, "w": 1} and len(pts) == 4
    assert len(sm.factorial_points({"c": 0}, {"a": [1, 2, 3], "b": [1, 2, 3]}, names=["a", "b"])) == 9
    assert sm.factorial_points({"a": 1}, {}, points=[{"a": 2}]) == [{"a": 2}]


def _opt(seed, n_jobs=1):
    spec = Spec(obj={"kind": "Synth"}, base={"a": 5, "w": 1.0}, grids={"a": [3, 5, 10, 20, 40], "w": [0.5, 1.0, 2.0]},
                mode="optuna", optuna=dict(n_trials=25, seed=seed, n_jobs=n_jobs))
    return Sweep(spec, SynthEvaluator()).run()


def test_optuna_seed_reproducible():
    a, b = _opt(7), _opt(7)
    assert a["optuna"] == b["optuna"] and a["table"].equals(b["table"])
    assert [t["params"] for t in a["optuna"]] != [t["params"] for t in _opt(8)["optuna"]]


def test_optuna_prunes_and_counts_all_probes():
    r = _opt(7)
    states = {t["state"] for t in r["optuna"]}
    assert "COMPLETE" in states and r["N"] >= len({tuple(t["params"].items()) for t in r["optuna"]}) - 0


def test_optuna_n_jobs_capped():
    with pytest.raises(ValueError):
        _opt(1, n_jobs=9)
