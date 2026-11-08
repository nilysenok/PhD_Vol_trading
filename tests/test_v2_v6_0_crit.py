"""V6-0, G7: меры выбора. crit_SBER = crit_wf (известный ответ вручную), формула весов воспроизводит crit_wf ≤ 1e-12, мутация «равные веса лет» ловится, пороги = min − δ, бутстрап детерминирован (seed 0), trim отбрасывает крайние даты."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_0_crit as CR, v6_0_load as LD, v6_0_q as Q


def _toy(ann):
    d = pd.bdate_range("2017-01-02", "2021-12-31")
    y = d.year.to_numpy()
    q = np.array([ann[k - 2017] for k in y])
    return d, q


ANN = [0.1, 0.3, 0.2, 0.2, 0.2]
KNOWN = np.mean([0.1, 0.2 + 0.25 * 0.1, 0.2 + 0.25 * 0.1, 0.2 + 0.25 * 0.1])         # pen_Y для Y = 2018…2021 вручную


def test_crit_sber_known_answer_by_hand():
    d, q = _toy(ANN)
    assert np.isclose(Q.crit_sber(q, d), KNOWN, atol=1e-12) and np.isclose(KNOWN, 0.19375)


def test_mutation_equal_year_weights_is_caught():
    d, q = _toy(ANN)
    eqy = float(np.mean(ANN))
    assert abs(Q.crit_sber(q, d) - eqy) > 1e-3                                  # равные веса лет дали бы EqY — тест различает
    assert abs(Q.annual(q, d)[2017] - 0.1) < 1e-15 and abs(Q.crit_row(np.ones(len(q)), np.ones(len(q)), d, q, q)["EqY"]) < 1e-12        # rv = f ⇒ QLIKE = 0 ⇒ EqY = 0


def test_formula_check_real_and_mutation_equal_weights_caught(monkeypatch):
    w, _ = LD.load()
    ser = ["K1", "K2", "K3", "K8", "D2"]
    assert CR.formula_check(w, ser) <= 1e-12
    monkeypatch.setattr(CR.CR, "pen_parts", lambda ann: (float(np.mean(ann)), 0.0, float(np.mean(ann))))        # мутация: равные веса, без штрафа
    assert CR.formula_check(w, ser) > 1e-3


def test_years_better_known_answer():
    d, _ = _toy(ANN)
    base = np.full(len(d), 0.2)
    good = np.where(d.year.isin([2017, 2019, 2021]), 0.1, 0.3)
    assert Q.years_better(good, d, base, base + 0.01) == 3
    assert Q.years_better(base, d, base, base) == 0                              # ничья не «лучше»


def test_trim_drops_exactly_one_percent_largest_deviations():
    d = pd.bdate_range("2017-01-02", "2021-12-31")
    rng = np.random.default_rng(1)
    q = 0.15 + 0.01 * rng.random(len(d))
    best = q.copy()
    big = rng.choice(len(d), 13, replace=False)
    q[big] += 5.0                                                                # 13 = round(0,01·1305) крайних дат
    assert len(d) == 1305 and int(round(0.01 * len(d))) == 13
    assert Q.crit_trim(q, d, best) < 0.2 < Q.crit_sber(q, d)


def test_thresholds_are_min_minus_delta_and_only_over_go():
    ser = ["K1", "K2", "K3", "X", "Y"]
    t = pd.DataFrame({"series": ser, "crit": [0.20, 0.19, 0.18, 0.10, 0.17], "crit_no2017": [0.3] * 5, "crit_no2017_18": [0.2] * 5, "crit_trim1": [0.15] * 5,
                      **{"q%d" % y: [0.1 + 0.01 * i for i in range(5)] for y in Q.YEARS}})
    th = CR.thresholds(t, ["K1", "K2", "K3", "Y"])                               # X (потолок) в планку не входит
    assert np.isclose(th["cond1_crit"]["threshold"], 0.17 - 0.0005) and th["cond1_crit"]["attained_by"] == "Y"
    assert np.isclose(th["cond1_crit"]["base5_threshold"], 0.18 - 0.0005) and th["cond1_crit"]["base5_attained_by"] == "K3"          # планка только по K1,K2,K3,K8,D2
    assert th["cond5_annual_best_k23"]["2017"] == min(0.11, 0.12)
    mut = CR.thresholds(t, ser)                                                  # мутация: потолок X в планке — порог падает до 0,0995
    assert np.isclose(mut["cond1_crit"]["threshold"], 0.0995)


def test_noise_bootstrap_deterministic_seed0_and_zero_for_equal_series():
    w, _ = LD.load()
    a, b = CR.noise(w, "K3"), CR.noise(w, "K3")
    assert a == b and a["B"] == 2000 and a["block"] == 20 and a["seed"] == 0 and a["sd_boot"] > 0
    w2 = w.copy()
    w2["K8"] = w2["K3"]
    assert CR.noise(w2, "K3")["sd_boot"] == 0.0
    assert np.isclose(a["mde_95_pct_of_k1"], 100 * 1.96 * a["sd_boot"] / a["k1_qlike"])


def test_real_crit_table_matches_stage_and_thresholds_consistent():
    from models_v2.s13 import v6_0_stage as ST
    r = ST.compute()
    t, th = r["cr"]["table"], r["cr"]["thresholds"]
    go = t[t["series"].isin(th["go_series"])]
    assert np.isclose(th["cond1_crit"]["min_over_go"], go["crit"].min()) and np.isclose(th["cond4_crit_trim1"]["min_over_go"], go["crit_trim1"].min())
    assert (r["checkpoint"]["N"], r["checkpoint"]["N_wide"]) == (722, 3122) and r["cr"]["formula_max_diff"] <= 1e-12
    assert not set(th["go_series"]) & set(r["bres"]["ceil"])
