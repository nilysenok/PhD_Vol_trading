"""V7 лист 2b, G30: специализация и веса экспертов (6.3, 6.5), условие 9 — синтетика с известным ответом.
Мутации: «без эксперта» с обратным знаком; порог π; доли считаются по всей выборке/2022+; условие 9 по окну 2022+ или по строке другого окна."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s14 import v7_1m_stat as ST, v7_2_evspec as SP

IDX = pd.bdate_range("2018-01-03", "2026-02-26")
N = len(IDX)
TP = IDX.to_numpy("datetime64[ns]")


def frame(pi1, pi2, g1, g2, rv=None, seed=0):
    g = np.random.default_rng(seed)
    H = np.exp(g.normal(0, 0.3, N))
    pi1, pi2, g1, g2 = (np.broadcast_to(np.asarray(a, float), N).copy() for a in (pi1, pi2, g1, g2))
    G = pi1 * g1 + pi2 * g2
    rv = H * np.exp(g.normal(0, 0.3, N)) if rv is None else rv
    return pd.DataFrame({"d": IDX, "t_plus": IDX, "rv": rv, "H": H, "G": G, "F": H * np.exp(G), "fitted": 1.0, "pi0": 1 - pi1 - pi2, "pi1": pi1, "pi2": pi2, "g1": g1, "g2": g2})


def test_gain_known_answer_and_sign():
    df = frame(0.5, 0.2, 0.4, -0.1)
    k = 1
    want = Q.ql(df["rv"], df["H"] * np.exp(df["G"] - 0.5 * 0.4)) - Q.ql(df["rv"], df["F"])
    assert np.allclose(SP.gain(df, 1), want, rtol=0, atol=1e-14) and SP.approx_err(df) < 1e-15
    wrong = Q.ql(df["rv"], df["H"] * np.exp(df["G"] + 0.5 * 0.4)) - Q.ql(df["rv"], df["F"])        # мутация: вклад эксперта добавлен, а не снят
    assert np.abs(wrong - SP.gain(df, k)).max() > 1e-3
    g = df.assign(G=df["G"] + 0.01)
    assert abs(SP.approx_err(g) - 0.01) < 1e-12


def test_expert_that_helps_only_on_days_pi_above_threshold_is_specialized_and_null_expert_is_not():
    on = (np.arange(N) // 20) % 2 == 0                                                      # блоки по 20 дней — как в бутстрапе
    pi1 = np.where(on, 0.6, 0.1)
    g1 = 0.5
    H = np.exp(np.random.default_rng(3).normal(0, 0.3, N))
    rv = np.where(on, H * np.exp(pi1 * g1) * 1.0, H)                                        # на «on» истина = F, на остальных истина = H (эксперт вредит)
    df = frame(pi1, 0.1, g1, 0.0, rv=rv)
    df["H"] = H
    df["F"] = H * np.exp(df["G"])
    t = SP.spec_table({"cfg5": df})
    r = t[(t["k"] == 1) & (t["window"] == "2018_2021")].iloc[0]
    assert r["gain_on"] > 0 > r["gain_rest"] and r["diff_on_rest_lo90"] > 0 and r["specialized"] == "да"
    assert abs(r["share_pi_gt03"] - on[ST.wmask(TP, "2018_2021")].mean()) < 1e-12
    null = frame(pi1, 0.1, 0.0, 0.0)                                                        # g ≡ 0: эксперт ничего не вносит, разность ровно 0
    rn = SP.spec_table({"cfg5": null}).query("k == 1 and window == '2018_2021'").iloc[0]
    assert rn["gain_on"] == 0 and rn["gain_rest"] == 0 and rn["specialized"] == "нет"


def test_switch_frequency_threshold_and_share_threshold():
    flip = np.where(np.arange(N) % 2 == 0, 0.9, 0.05)
    t = SP.spec_table({"cfg6": frame(flip, 0.05, 0.1, 0.1)})
    r = t[(t["k"] == 1) & (t["window"] == "all")].iloc[0]
    assert abs(r["switch_freq"] - 1.0) < 1e-12 and abs(r["share_pi_gt03"] - np.mean(flip > 0.3)) < 1e-12
    calm = SP.spec_table({"cfg6": frame(0.3, 0.3, 0.1, 0.1)}).query("k == 1 and window == 'all'").iloc[0]
    assert calm["switch_freq"] == 0 and calm["share_pi_gt03"] == 0                          # π = 0,3 ровно: условие строгое «> 0,3»
    mid = np.where(np.arange(N) % 2 == 0, 0.6, 0.1)                                         # скачок 0,5 ровно: «> 0,5» не срабатывает
    assert SP.spec_table({"cfg6": frame(mid, 0.1, 0.1, 0.1)}).query("k == 1 and window == 'all'").iloc[0]["switch_freq"] == 0


def test_weights_by_year_and_before_after_2021():
    pi1 = np.where(IDX.year <= 2021, 0.2, 0.5)
    w = SP.weights_table({"cfg5": frame(pi1, 0.1, 0.1, 0.1)})
    g = w[(w["config"] == "cfg5") & (w["k"] == 1)].set_index("period")["mean_pi"]
    assert abs(g["2019"] - 0.2) < 1e-12 and abs(g["2024"] - 0.5) < 1e-12 and abs(g["до 2021 (≤ 2021)"] - 0.2) < 1e-12 and abs(g["после 2021 (≥ 2022)"] - 0.5) < 1e-12
    n0 = w[(w["k"] == 0) & (w["period"] == "2018")].iloc[0]
    assert abs(n0["mean_pi"] - 0.7) < 1e-12 and set(w["period"]) >= {str(y) for y in range(2018, 2027)}


def pairs(rule_main="да", rule_trim="да"):
    rows = [dict(hyp="H2", x="cfg6", y="cfg4", window=w, rule62=r, delta_q=-0.001, p_one=0.01, years_neg=3) for w, r in (("2018_2021", rule_main), ("2018_2021 без 1 % крайних", rule_trim), ("2019_2021", rule_trim), ("2022_2025", "—"))]
    return pd.DataFrame(rows)


def test_cond9_known_answer_both_conditions_needed():
    ok = frame(0.5, 0.5, 0.1, 0.1)
    assert SP.cond9(ok, pairs())["cond9"] is True
    assert SP.cond9(ok, pairs("нет"))["cond9"] is False                                      # нет новой информации против 4
    low = frame(0.5, np.where(IDX.year <= 2021, 0.05, 0.9), 0.1, 0.1)                       # у второго эксперта в 2018–2021 π > 0,3 меньше чем на 10 % дней
    c = SP.cond9(low, pairs())
    assert c["cond9"] is False and c["shares_ok"] is False and c["share_pi2_gt03"] == 0


def test_mutation_cond9_by_window_2022_plus_is_caught():
    """В 2022+ π2 > 0,3 на всех днях, в 2018–2021 — ни разу; новая информация против 4 — «да» только в окне без 2018 (обрезанном): условие 9 обязано остаться ложным."""
    df = frame(0.5, np.where(IDX.year >= 2022, 0.9, 0.0), 0.1, 0.1)
    c = SP.cond9(df, pairs(rule_main="нет", rule_trim="да"))
    assert c["window"] == "2018_2021" and c["share_pi2_gt03"] == 0 and c["cond9"] is False and c["new_information_vs_4"] is False
    mutant_share = float((df["pi2"] > SP.THR).mean())                                        # доля по всей выборке — то, что вернула бы мутация
    assert mutant_share > SP.SHARE_MIN > c["share_pi2_gt03"]
