"""V6-1a, G14: модуль оценки на синтетике с известным ответом: метрики по окнам, Δ в % уровня K1 с блоковым интервалом, DM (НУ лаг 5), Холм с N = 722 / 3122 (мутация № 103), условия «go» 1–5 по порогам CHECKPOINT ключ `0`,
правило «новая информация» (п. 2.2), охват с интервалом, стресс-окна. Мутация (подмена N Холма на число строк, № 103) ловится в `test_holm_uses_program_n_and_mutation_n_equals_rows_caught`; остальные проверки — известные ответы (пороги go строже/мягче, роли охвата w = 1 / w = 0, отброшенные крайние даты)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s3 import dm as dmod
from models_v2.s5.d import stats_d as SD
from models_v2.s13 import v6_0_cp as CP0, v6_0_q as Q, v6_1_cov as CV, v6_1_eval as EV

CP = {"N": 722, "N_wide": 3122}


def dates(per_year: int = 120, y0: int = 2017, y1: int = 2021):
    return pd.DatetimeIndex(np.concatenate([pd.date_range("%d-01-04" % y, periods=per_year, freq="B").to_numpy() for y in range(y0, y1 + 1)]))


def test_metrics_known_answer():
    tp = dates(10, 2017, 2018)
    rv = np.ones(len(tp))
    t = EV.metrics_table(rv, {"x": 2 * rv}, tp).set_index("window")
    assert t.loc["2017", "qlike"] == pytest.approx(0.5 + np.log(2) - 1, abs=1e-12) and t.loc["2017_2021", "n"] == 20
    assert t.loc["all", "bias_ln"] == pytest.approx(np.log(2)) and t.loc["2017", "rmse"] == pytest.approx(1.0) and t.loc["2017", "mae"] == pytest.approx(1.0)
    assert "2026" not in t.index                                      # окно без дат не выдумывается


def test_delta_pct_and_interval_known_constant_difference():
    qy = np.full(400, 0.2)
    qx = qy - 0.01
    r = EV.delta_ci(qx, qy, 0.25)
    assert r["delta_pct"] == pytest.approx(-4.0) and r["lo90"] == pytest.approx(-4.0) and r["hi90"] == pytest.approx(-4.0)
    rng = np.random.default_rng(0)
    d = rng.normal(-0.01, 0.05, 600)
    r2 = EV.delta_ci(qy[:1].repeat(600) + d, qy[:1].repeat(600), 0.25)
    assert r2["lo90"] < r2["delta_pct"] < r2["hi90"] and (r2["hi90"] - r2["lo90"]) > 0
    again = EV.delta_ci(qy[:1].repeat(600) + d, qy[:1].repeat(600), 0.25)
    assert again == r2                                                # seed 0 детерминирован


def test_dm_one_sided_direction():
    rng = np.random.default_rng(1)
    rv = np.exp(rng.normal(0, 0.5, 500))
    good, bad = rv * np.exp(rng.normal(0, 0.1, 500)), rv * np.exp(rng.normal(0, 0.6, 500))
    a, b = EV.dm_one(rv, bad, good), EV.dm_one(rv, good, bad)
    assert a["p_one"] < 0.001 and b["p_one"] > 0.999 and a["p_one"] + b["p_one"] == pytest.approx(1.0, abs=1e-12)
    assert a["p_two"] == pytest.approx(dmod.dm(rv, bad, good)["p"])


def test_holm_uses_program_n_and_mutation_n_equals_rows_caught(monkeypatch):
    p = np.array([0.001, 0.01])
    a, aw = EV.holm_both(p, CP)
    assert a == pytest.approx([0.722, 1.0]) and aw == pytest.approx([1.0, 1.0])               # 722·0,001; 721·0,01 -> 1
    tp, rng = dates(), np.random.default_rng(2)
    rv = np.exp(rng.normal(0, 0.5, len(tp)))
    refs = {"K1": rv * np.exp(rng.normal(0, 0.5, len(tp))), "K2": rv * np.exp(rng.normal(0, 0.6, len(tp)))}
    f = rv * np.exp(rng.normal(0, 0.1, len(tp)))
    ok = EV.tests_table(rv, f, refs, tp.to_numpy(), CP, windows=("all",))
    exp = SD.holm_n(ok["p_one"].to_numpy(), 722)
    assert np.allclose(ok["p_holm_N"], exp, rtol=1e-9, atol=0) and (ok["p_holm_N"] >= ok["p_one"]).all()
    monkeypatch.setattr(EV, "holm_both", lambda p_, cp: (SD.holm_n(p_, len(p_)), SD.holm_n(p_, len(p_))))              # N = число строк (№ 103)
    bad = EV.tests_table(rv, f, refs, tp.to_numpy(), CP, windows=("all",))
    assert not np.allclose(bad["p_holm_N"], exp, rtol=1e-6, atol=0)


def go_setup():
    tp = dates(120)
    rng = np.random.default_rng(3)
    rv = np.exp(rng.normal(0, 0.5, len(tp)))
    k2, k3 = rv * np.exp(rng.normal(0, 0.9, len(tp))), rv * np.exp(rng.normal(0, 0.9, len(tp)))
    f = rv * np.exp(rng.normal(0, 0.2, len(tp)))
    return rv, f, tp.to_numpy(), Q.ql(rv, k2), Q.ql(rv, k3)


def cp_with(r: dict, shift: float) -> dict:
    th = {k: {"threshold": r[c] + shift} for k, c in (("cond1_crit", "crit"), ("cond2_crit_no2017", "crit_no2017"), ("cond3_crit_no2017_18", "crit_no2017_18"), ("cond4_crit_trim1", "crit_trim1"))}
    return {"thresholds": {**th, "cond5_years_better_min": 3}}


def test_go_conditions_known_thresholds_and_mutation_sign():
    rv, f, tp, q2, q3 = go_setup()
    r = Q.crit_row(rv, f, tp, q2, q3)
    assert r["years_better_of5"] == 5
    ok = EV.go_conditions(rv, f, tp, q2, q3, cp_with(r, +0.001))
    assert ok["passed"].all() and list(ok["cond"]) == [1, 2, 3, 4, 5]
    bad = EV.go_conditions(rv, f, tp, q2, q3, cp_with(r, -0.001))
    assert list(bad["passed"]) == [False, False, False, False, True]                  # пороги строже -> 1–4 нет; 5 не зависит от порогов 1–4
    worse = EV.go_conditions(rv, rv * 6.0, tp, q2, q3, cp_with(r, +0.001))
    assert not worse["passed"].iloc[0] and not worse["passed"].iloc[4] and worse["value"].iloc[4] == 0       # прогноз 6·RV хуже K2/K3 во всех годах


def test_new_information_rule_three_parts():
    tp = dates().to_numpy()
    rng = np.random.default_rng(4)
    rv = np.exp(rng.normal(0, 0.5, len(tp)))
    a = rng.normal(0, 0.5, len(tp))
    fy = rv * np.exp(a)
    better = EV.new_information(rv, rv * np.exp(0.6 * a), fy, tp)
    assert better["new_information"] and better["years_neg"] == 5 and better["delta"] < 0 and better["p_dm_one"] < 0.05
    worse = EV.new_information(rv, fy, rv * np.exp(0.6 * a), tp)
    assert not worse["new_information"] and not worse["cond_i"] and not worse["cond_iii"]
    y = pd.DatetimeIndex(tp).year.to_numpy()
    mixed = rv * np.exp(np.where(y >= 2020, 1.5 * a, 0.3 * a))                                  # лучше в 3 годах из 5 (2017–2019), хуже в двух, в среднем — смотрим флаги
    m = EV.new_information(rv, mixed, fy, tp)
    assert m["years_neg"] == 3 and m["cond_iii"]
    two = rv * np.exp(np.where(y >= 2019, 0.3 * a, 1.5 * a))                                    # лучше только в 2019–2021 = 3 года; в 2017–2018 хуже
    assert EV.new_information(rv, two, fy, tp)["years_neg"] == 3
    one = rv * np.exp(np.where(y >= 2021, 0.3 * a, 1.5 * a))
    assert EV.new_information(rv, one, fy, tp)["years_neg"] == 1 and not EV.new_information(rv, one, fy, tp)["cond_iii"]


def test_coverage_known_answers_and_roles():
    rng = np.random.default_rng(5)
    rv = np.exp(rng.normal(0, 0.5, 800))
    noisy = rv * np.exp(rng.normal(0, 0.5, 800))
    c1 = CV.coverage(rv, noisy, rv)                       # F = RV: B ничего не добавляет -> w = 1
    c0 = CV.coverage(rv, rv, noisy)                       # B = RV: F ничего не добавляет -> w = 0
    assert c1["w"] == pytest.approx(1.0, abs=1e-6) and c1["lo90"] > 0.9
    assert c0["w"] == pytest.approx(0.0, abs=1e-6) and c0["hi90"] < 0.1
    mid = CV.coverage(rv, noisy, rv * np.exp(rng.normal(0, 0.5, 800)))
    assert 0.0 < mid["lo90"] <= mid["w"] <= mid["hi90"] < 1.0


def test_stress_masks_known_answers():
    tp = pd.DatetimeIndex(np.concatenate([pd.date_range("%d-01-04" % y, periods=250, freq="B").to_numpy() for y in range(2017, 2023)]))
    rng = np.random.default_rng(6)
    h = np.exp(rng.normal(0, 0.5, len(tp)))
    qf = rng.random(len(tp)) * 0.1
    qb = qf.copy()
    w = Q.wmask(tp, "2017_2021")
    big = np.flatnonzero(w)[[3, 50, 400, 700, 900, 1100, 1200, 100, 200, 300, 600, 650]]
    qf[big] += 5.0                                         # 12 крайних дат из 1250 -> ровно round(0,01·1250) = 12 отбрасывается
    m = CV.stress_masks(tp, h, qf, qb)
    assert m["без 2017 (2018–2021)"].sum() == 4 * 250 and m["без 2017–2018 (2019–2021)"].sum() == 3 * 250
    assert (m["без 1 % крайних дат (2017–2021)"] == (w & ~np.isin(np.arange(len(tp)), big))).all()
    assert m["без 1 % крайних дат (2017–2021)"].sum() == 1250 - 12
    y = tp.year.to_numpy()
    assert all(abs(m["H в верхних 10 % года"][y == k].mean() - 0.1) < 0.02 and abs(m["спокойные: H в нижних 50 % года"][y == k].mean() - 0.5) < 0.02 for k in range(2017, 2023))
    cr = m["кризис 2022-02-21 … 2022-06-30"]
    assert (tp[cr].min() >= pd.Timestamp("2022-02-21")) and (tp[cr].max() <= pd.Timestamp("2022-06-30")) and cr.sum() > 50


def test_stress_trim_drops_the_outlier_date():
    tp = pd.DatetimeIndex(np.concatenate([pd.date_range("%d-01-04" % y, periods=250, freq="B").to_numpy() for y in range(2017, 2022)]))
    qf = np.random.default_rng(7).random(len(tp)) * 0.1
    qf[10] += 9.0
    m = CV.stress_masks(tp, np.ones(len(tp)), qf, np.zeros(len(tp)))["без 1 % крайних дат (2017–2021)"]
    assert not m[10] and m.sum() == len(tp) - 12
