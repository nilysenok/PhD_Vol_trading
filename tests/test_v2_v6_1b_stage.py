"""V6-1b, G18/G19/G21: правило «новая информация» на синтетике с известным ответом (A_INFO), автоматическое решение по A3 (мутация: правило по паре A2 против A1 вместо A0 ловится), наборы признаков конфигураций (S побитно прежний; Σ — 12, Σ³ — 8 колонок),
DX-A: QLIKE-регрессия и тест Вальда с робастной ковариацией на синтетике (эффект найден, нулевой блок не найден; перемешанный остаток эффект убирает)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_1b_dxa as DXA, v6_1b_sig as SG, v6_1b_stage as ST, v6_1b_tab as TB

CP0 = {"N": 722, "N_wide": 3122}


def synth(n: int = 1500, seed: int = 5):
    rng = np.random.default_rng(seed)
    tp = pd.bdate_range("2017-01-02", periods=n)
    mu = rng.normal(0, 0.3, n)
    rv = np.exp(mu + rng.normal(0, 0.5, n))
    w = pd.DataFrame({"rv": rv, "t_plus": tp, "K1": np.exp(mu) * 1.3}, index=tp - pd.offsets.BDay(1))
    return w, np.exp(mu + 0.125)                       # точная условная средняя E[RV] = exp(μ + σ²/2)


def test_info_table_known_answer_and_rule_direction():
    w, best = synth()
    fs = {"X": best, "Y": best * np.exp(0.4)}
    t = TB.info_table(fs, [("X", "Y"), ("Y", "X")], w, CP0)
    a, b = t[(t["window"] == "2017_2021") & (t["pair"] == "X vs Y")].iloc[0], t[(t["window"] == "2017_2021") & (t["pair"] == "Y vs X")].iloc[0]
    assert a["delta_qlike"] < 0 and a["years_neg"] == 5 and a["dm_p_one"] < 0.05 and bool(a["new_information"]) is True
    assert b["delta_qlike"] > 0 and bool(b["new_information"]) is False and b["dm_p_one"] > 0.95
    assert set(t["window"]) == {"2017_2021", "2022_2025", "2026", "all"} and (t["p_holm_N"].dropna() >= t["dm_p_one"].dropna() - 1e-15).all()


def test_decide_a3_follows_rule_against_a0_not_a1():
    w, best = synth()
    bad, worse = best * np.exp(0.4), best * np.exp(0.2)
    for fs, want in (({"A0": best, "A1": bad, "A2": worse}, False),           # A2 лучше A1, но хуже A0: A3 не считается
                     ({"A0": bad, "A1": bad, "A2": best}, True),                # A2 лучше A0
                     ({"A0": bad, "A1": best, "A2": bad}, True)):               # A1 лучше A0
        info = TB.info_table(fs, [("A1", "A0"), ("A2", "A0"), ("A2", "A1")], w, CP0)
        assert ST.decide_a3(info)["compute"] is want
        mut = info[(info["window"] == "2017_2021") & (info["pair"] == "A2 vs A1")]
        assert (bool(mut["new_information"].iloc[0]) is True) != want or want is True                    # мутация «пара A2 против A1» на первом случае дала бы True
    first = TB.info_table({"A0": best, "A1": bad, "A2": worse}, [("A1", "A0"), ("A2", "A0"), ("A2", "A1")], w, CP0)
    assert bool(first[(first["window"] == "2017_2021") & (first["pair"] == "A2 vs A1")]["new_information"].iloc[0]) is True


def test_feature_sets_s_bitwise_and_column_counts():
    idx = pd.bdate_range("2016-01-04", periods=30)
    S = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 9)), index=idx, columns=["s%d" % i for i in range(1, 10)])
    parts = {"sigma": pd.DataFrame(np.ones((30, 12)), index=idx, columns=SG.SIGMA_NAMES), "sig3": pd.DataFrame(np.ones((30, 8)), index=idx, columns=SG.sigma3_names())}
    x1, x3 = ST.design("A1", S, parts), ST.design("A3", S, parts)
    assert x1.shape[1] == 21 and x3.shape[1] == 29 and (x1[S.columns].to_numpy() == S.to_numpy()).all() and ST.design("A2", S, parts).shape[1] == 21
    assert ST.LEARNER == {"A1": "LIN", "A2": "GBM", "A3": "LIN"} and len(set(SG.SIGMA_NAMES)) == 12 and len(set(SG.sigma3_names())) == 8


def planted(seed: int = 1, n: int = 3000):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    return Z, 0.5 * Z[:, 0] + rng.normal(0, 0.3, n)


def test_dxa_wald_finds_effect_and_not_null_block():
    Z, e = planted()
    X = pd.DataFrame(Z, columns=["z1", "z2", "z3"])
    rows = {r["test"]: r for r in DXA.fit_wald(X, e, {"eff": ["z1"], "null": ["z2", "z3"]})}
    assert rows["Wald:eff"]["p"] < 1e-10 and rows["Wald:null"]["p"] > 0.01 and abs(rows["coef:z1"]["theta"] - 0.5) < 0.05
    perm = np.random.default_rng(7).permutation(len(e))                                                  # мутация: остаток перемешан — эффект исчезает
    assert {r["test"]: r for r in DXA.fit_wald(X, e[perm], {"eff": ["z1"]})}["Wald:eff"]["p"] > 0.01


def test_hac_cov_accounts_for_autocorrelated_scores():
    Z, e = planted(2)
    th = np.array([0.0, 0.5, 0.0, 0.0])
    iid = np.diag(DXA.hac_cov(Z, e, th, lag=0))
    rng = np.random.default_rng(3)
    ar, z1 = np.zeros(len(e)), np.zeros(len(e))
    for t in range(1, len(e)):
        ar[t], z1[t] = 0.7 * ar[t - 1] + rng.normal(0, 0.3), 0.7 * z1[t - 1] + rng.normal()
    Z = Z.copy()
    Z[:, 0] = z1                                                         # скоры (1 − exp(e − g))·z автокоррелированы только если автокоррелированы и регрессор, и ошибка
    e2 = 0.5 * z1 + ar
    assert np.diag(DXA.hac_cov(Z, e2, th, lag=5))[1] > 1.3 * np.diag(DXA.hac_cov(Z, e2, th, lag=0))[1] and np.all(iid > 0)
