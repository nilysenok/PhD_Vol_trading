"""LGBM волна B: сетка LB0, подвыборка (seed 42), критерий crit_pen, выбор по году, реестр trials и бюджет."""
import json
import types

import numpy as np
import pandas as pd
import pytest

from models_v2.s6.lgbm import crit, grid, job, tab

DECLARED = (17, 18, 19, 42, 89, 111, 133, 145, 154, 157, 163, 177, 205, 207, 213)   # объявлено в GATES_B_lgbm.md до счёта


def test_grid_size_and_axes():
    g = grid.full_grid()
    assert len(g) == 216 and len(set(g)) == 216
    assert {h.max_depth for h in g} == {-1} and max(h.max_trees for h in g) == 300


def test_lb0_subsample_is_declared_list():
    g = grid.full_grid()
    ix = tuple(g.index(p) for p in grid.lb0_points())
    assert ix == DECLARED and len(set(ix)) == 15


def test_mutation_other_seed_changes_subsample(monkeypatch):
    monkeypatch.setattr(grid, "SEED_SUB", 43)
    g = grid.full_grid()
    assert tuple(g.index(p) for p in grid.lb0_points()) != DECLARED


def test_pen_year_known_answer():
    q = np.array([0.2, 0.3, 0.7])
    assert crit.pen_year(q) == pytest.approx(0.4 + 0.25 * (0.7 - 0.4))
    assert crit.pen_year(q, 0.0) == pytest.approx(0.4)
    assert np.isnan(crit.pen_year(np.array([np.nan])))


def test_select_takes_min_pen_and_tie_rule():
    pts = (grid.Hp(31, 30, 2.0, 0.05, 0.5, 1.0), grid.Hp(7, 200, 30.0, 0.02, 0.5, 1.0))
    mk = lambda a: pd.DataFrame({"Y": [2017, 2017], "y": [2015, 2016], "q": a})   # noqa: E731
    ch, _ = crit.select({0: mk([0.2, 0.2]), 1: mk([0.2, 0.2])}, pts)
    assert ch[2017] == pts[1]                    # ничья -> проще (меньше листьев)
    ch, _ = crit.select({0: mk([0.1, 0.1]), 1: mk([0.2, 0.2])}, pts)
    assert ch[2017] == pts[0]


def test_mutation_penalty_changes_choice():
    pts = (grid.Hp(7, 30, 2.0, 0.05, 0.5, 1.0), grid.Hp(15, 30, 2.0, 0.05, 0.5, 1.0))
    a = pd.DataFrame({"Y": [2017] * 2, "y": [2015, 2016], "q": [0.10, 0.50]})   # mean 0.30, worst 0.50
    b = pd.DataFrame({"Y": [2017] * 2, "y": [2015, 2016], "q": [0.31, 0.31]})   # mean 0.31
    assert crit.select({0: a, 1: b}, pts)[0][2017] == pts[1]       # со штрафом выигрывает ровная
    assert crit.pen_year(a["q"].to_numpy(), 0.0) < crit.pen_year(b["q"].to_numpy(), 0.0)   # без штрафа — нет


def test_folds_use_only_past_years_and_count():
    assert len(job.FOLDS) == 65 and all(y < Y for Y, y in job.FOLDS) and {Y for Y, _ in job.FOLDS} == set(range(2017, 2027))


def test_register_dedupes_and_checkpoint_limits(tmp_path, monkeypatch):
    t = tmp_path / "T.csv"
    r = [tab.trial_row("a", "hp", "c", "m", 1.0), tab.trial_row("b", "obj_feat", "c", "m", 1.0)]
    assert tab.register(r, t) == 2 and tab.register(r, t) == 0
    c = tmp_path / "C.json"
    c.write_text(json.dumps({"used": {"hp": 0, "obj_feat": 0}}))
    assert tab.checkpoint(c, t)["used"] == {"hp": 1, "obj_feat": 1}
    pd.DataFrame([tab.trial_row("x%d" % i, "hp", "c", "m", 1.0) for i in range(16)]).to_csv(t, index=False)
    with pytest.raises(ValueError):
        tab.checkpoint(c, t)


def test_weights_rvw_ignore_future_rows():
    n = 400
    F = types.SimpleNamespace(rv=np.ones(n), X=np.random.default_rng(0).normal(size=(n, 3)))
    sp = job.Spec("s", "ln", F.X, np.zeros(n), np.zeros(n), ("rvw", 0.5))
    idx = np.arange(300)
    w1 = job.weights(F, sp, idx, 300)[idx]
    F.X[300:, 1] = 99.0
    assert np.array_equal(w1, job.weights(F, sp, idx, 300)[idx])
    sp2 = job.Spec("s", "ln", F.X, np.zeros(n), np.zeros(n), ("decay", 100))
    w = job.weights(F, sp2, idx, 299)
    assert w[299] == 1.0 and w[199] == pytest.approx(0.5)
