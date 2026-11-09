"""V7 лист 1a-M: статистика сравнения (импорт из V6). Одинаковые даты, Δ в % уровня K1, бутстрап 20·2000·seed 0, DM лаг 5, Холм N = 733 / 3166, MCS 90 %. Известные ответы на синтетике; мутации ловятся."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2 import metrics
from models_v2.s13 import v6_1_eval as EV
from models_v2.s14 import v7_1m_stat as STt

N = 1100


def data(seed=0, gain=0.0):
    rng = np.random.default_rng(seed)
    tp = pd.bdate_range("2018-01-02", periods=N)
    f_true = np.exp(rng.normal(0, 0.3, N))
    rv = f_true * rng.gamma(20.0, 1 / 20.0, N)
    k1 = f_true * np.exp(rng.normal(0, 0.25, N))
    k2, k3 = f_true * np.exp(rng.normal(0, 0.12, N)), f_true * np.exp(rng.normal(0, 0.12, N))
    rf = STt.refs(k1, k2, k3, f_true * np.exp(rng.normal(0, 0.15, N)), f_true * np.exp(rng.normal(0, 0.2, N)))
    f = f_true * np.exp(gain * rng.normal(0, 0.05, N) + rng.normal(0, 0.03, N))
    return rv, f, rf, tp


def test_refs_m_and_k8_definitions():
    k1, k2, k3 = np.array([1.0, 2.0]), np.array([4.0, 2.0]), np.array([1.0, 8.0])
    r = STt.refs(k1, k2, k3, [1, 1], [1, 1])
    assert np.allclose(r["M"], [2.0, 4.0]) and np.allclose(r["K8"], np.sqrt(k1 * r["M"])) and list(r) == list(STt.BASES)


def test_params_declared_bootstrap_block_holm_n():
    assert (STt.B, STt.BLOCK, STt.SEED, STt.N_HOLM, STt.N_WIDE) == (2000, 20, 0, 733, 3166)
    assert STt.N_HOLM == 722 + 11 and STt.N_WIDE == 3122 + 11 * 4


def test_delta_in_percent_of_k1_level_known_number():
    rv, f, rf, tp = data()
    t = STt.compare(rv, f, rf, tp, windows=("2018_2021",)).set_index("base")
    m = STt.wmask(tp, "2018_2021")
    q = lambda x: metrics.qlike(rv[m], x[m])
    for b in STt.BASES:
        assert t.loc[b, "delta_pct"] == pytest.approx(100 * (q(f).mean() - q(rf[b]).mean()) / q(rf["K1"]).mean(), rel=1e-12)
    assert t.loc["K1", "n"] == m.sum()


def test_bootstrap_uses_block20_2000_seed0_and_interval_contains_delta():
    rv, f, rf, tp = data()
    t = STt.compare(rv, f, rf, tp, windows=("2018_2021",)).set_index("base")
    m = STt.wmask(tp, "2018_2021")
    qf, qb = metrics.qlike(rv[m], f[m]), metrics.qlike(rv[m], rf["K2"][m])
    ref = EV.delta_ci(qf, qb, float(metrics.qlike(rv[m], rf["K1"][m]).mean()), b=2000, block=20, seed=0)
    assert t.loc["K2", "lo90"] == ref["lo90"] and t.loc["K2", "hi90"] == ref["hi90"]
    other = EV.delta_ci(qf, qb, float(metrics.qlike(rv[m], rf["K1"][m]).mean()), b=2000, block=20, seed=1)
    assert other["lo90"] != ref["lo90"]                                                    # seed 0 в самом деле задан, а не случайный
    assert t.loc["K2", "lo90"] < t.loc["K2", "delta_pct"] < t.loc["K2", "hi90"]


def test_holm_uses_program_N_not_row_count():
    rv, f, rf, tp = data(gain=0.0)
    t = STt.compare(rv, f, rf, tp, windows=("2018_2021",))
    p = t["p_one"].to_numpy()
    ex = lambda n: pytest.approx(min(1.0, p.min() * n), rel=1e-9, abs=0)                    # без абсолютного допуска: p могут быть крошечными
    assert t["p_holm_N"].min() == ex(STt.N_HOLM) and t["p_holm_Nwide"].min() == ex(STt.N_WIDE)
    assert (t["p_holm_N"] >= np.minimum(1.0, p * len(p))).all()                              # мутант N = числу строк дал бы p·7 и упал бы здесь


def test_mutation_holm_n_equals_rows_is_caught(monkeypatch):
    rv, f, rf, tp = data()
    monkeypatch.setattr(STt, "N_HOLM", 7)
    t = STt.compare(rv, f, rf, tp, windows=("2018_2021",))
    assert not t["p_holm_N"].min() == pytest.approx(min(1.0, t["p_one"].min() * 733), rel=1e-9, abs=0)


def test_same_dates_for_all_bases_nan_in_one_series_drops_the_date_everywhere():
    rv, f, rf, tp = data()
    rf2 = {k: v.copy() for k, v in rf.items()}
    rf2["V6:A2"][500] = np.nan
    a = STt.compare(rv, f, rf, tp, windows=("2018_2021",)).set_index("base")
    b = STt.compare(rv, f, rf2, tp, windows=("2018_2021",)).set_index("base")
    assert (b["n"] == a["n"] - 1).all()
    keep = np.arange(N) != 500
    m = STt.wmask(tp, "2018_2021") & keep
    assert b.loc["K1", "delta_pct"] == pytest.approx(100 * (metrics.qlike(rv[m], f[m]).mean() - metrics.qlike(rv[m], rf["K1"][m]).mean()) / metrics.qlike(rv[m], rf["K1"][m]).mean(), rel=1e-12)


def test_windows_by_target_year_and_2017_excluded():
    tp = pd.DatetimeIndex(["2017-12-29", "2018-01-02", "2021-12-30", "2022-01-03", "2026-01-12"])
    assert list(STt.wmask(tp, "2018_2021")) == [False, True, True, False, False] and list(STt.wmask(tp, "all")) == [False, True, True, True, True]
    assert list(STt.wmask(tp, "2019_2021")) == [False, False, True, False, False] and list(STt.wmask(tp, "2026")) == [False, False, False, False, True]


def test_better_model_gets_negative_delta_and_small_dm_p():
    rv, f, rf, tp = data()
    t = STt.compare(rv, rf["K3"], rf, tp, windows=("2018_2021",)).set_index("base")        # F = K3: против шумного K1 лучше, против себя — ничья
    assert t.loc["K1", "delta_pct"] < 0 and t.loc["K1", "p_one"] < 0.05 and t.loc["K3", "delta_pct"] == pytest.approx(0.0, abs=1e-12)


def test_mutation_sign_swapped_delta_is_caught():
    rv, f, rf, tp = data()
    t = STt.compare(rv, rf["K3"], rf, tp, windows=("2018_2021",)).set_index("base")
    swapped = STt.compare(rv, rf["K1"], rf, tp, windows=("2018_2021",)).set_index("base")          # F = K1, база K3: знак противоположный
    assert t.loc["K1", "delta_pct"] < 0 < swapped.loc["K3", "delta_pct"]


def test_mcs_90_keeps_best_drops_dominated():
    rv, f, rf, tp = data(seed=3)
    bad = f * np.exp(0.8)                                                                   # заведомо плохой ряд
    r = STt.mcs90(rv, {"F": f, "K2": rf["K2"], "bad": bad}, tp).set_index("series")
    assert bool(r.loc["F", "in_mcs"]) and not bool(r.loc["bad", "in_mcs"]) and r.loc["F", "mean_qlike"] < r.loc["bad", "mean_qlike"]


def test_new_information_rule_6_2_known_answers():
    rv, f, rf, tp = data(seed=4)
    assert STt.new_information(rv, f, rf["K1"], tp)["new_information"] is True            # f близок к истине, K1 шумный: Δ < 0, DM p мал, лет ≥ 3
    assert STt.new_information(rv, rf["K1"], f, tp)["new_information"] is False
    assert STt.new_information(rv, f, f, tp)["new_information"] is False


def test_years_table_shape_and_sign():
    rv, f, rf, tp = data()
    y = STt.years_table(rv, f, rf, tp)
    assert set(y["base"]) == set(STt.BASES) and y["year"].min() == 2018 and (y[y["base"] == "K1"]["delta"] < 0).all()
