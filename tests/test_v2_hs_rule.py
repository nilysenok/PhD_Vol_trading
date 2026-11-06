"""Ж-212 (HS5, HS11): правило «продолжать направление» (порог −0,0005 и >= 6 лет из 10), блок «полезен», выбор HP LGBM по LOYO (ничья — меньше листьев, затем lr), top2 механически."""
from __future__ import annotations

from models_v2.s8 import hs_bfit, hs_bstage, hs_eval


def r(name, d_crit, yb, d_q=-0.001, d_q2=-0.001, **kw):
    return {"model": name, "d_crit": d_crit, "years_better": yb, "d_qlike": d_q, "d_qlike_no_240222": d_q2, **kw}


def test_hs11_thresholds_boundary_and_years():
    assert hs_eval.hs11([r("a", -0.0005, 6)])["continue"]                 # граница включительно
    assert not hs_eval.hs11([r("a", -0.00049, 6)])["continue"]
    assert not hs_eval.hs11([r("a", -0.01, 5)])["continue"]               # мутация «порог лет 6 -> 5»
    best = hs_eval.hs11([r("a", 0.001, 9), r("b", -0.002, 7)])
    assert best["best"] == "b" and best["continue"]


def test_robust_gain_needs_all_three_conditions():
    assert hs_eval.robust_gain(r("a", -0.001, 7))
    assert not hs_eval.robust_gain(r("a", -0.001, 7, d_q2=0.0001))        # выигрыш держится на 24.02.2022
    assert not hs_eval.robust_gain(r("a", -0.0001, 7))


def test_pick_points_ties_prefer_fewer_leaves_then_lr():
    pts = hs_bfit.POINTS
    years = range(2017, 2027)
    folds = {pt: {(Y, 2016): 0.2 for Y in years} for pt in pts}
    out = hs_bfit.pick_points(folds)
    assert out[2018] == (4, 0.02) and out[2019] == (4, 0.02)
    folds[(8, 0.05)] = {(Y, 2016): (0.1 if Y == 2018 else 0.2) for Y in years}
    out = hs_bfit.pick_points(folds)
    assert out[2018] == (8, 0.05) and out[2019] == (4, 0.02)


def test_top2_picks_two_best_single_blocks_mechanically():
    rows = [{"family": "XGB", "blocks": b, "crit_wf": c} for b, c in (("Bf", 0.3), ("Bc", 0.1), ("Br", 0.2), ("Bg", 0.4), ("Be", 0.5))]
    assert hs_bstage.top2(rows, "XGB") == ("Bc", "Br") and hs_bstage.top2(rows, "LGBM") == ()
