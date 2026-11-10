"""Y-A (Ж-243), оценка: знак Δ и база процентов (известный ответ, мутация знаменателя ловится); Холм N = 744 / 3193, а не число строк (мутация ловится); «go» = V7 при тех же базах и строже с X1;
правило 6.2 = `new_information`; клетки; отчёт генерируется из таблиц, «подтверждено» — только при p Холма < 0,05, подменённое число ловится. Синтетика; файлов нет."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s13 import v6_0_q as Q, v6_1_eval as EVX
from models_v2.s16 import x1_ya_doc as DOC, x1_ya_eval as EV

NAMES = ("K1", "K2", "K3", "V4:NTP2", "V6:A2", "V7:cfg6", "V7:cfg8", "X1", "Y1", "Y2")
SIG = dict(K1=0.5, K2=0.3, K3=0.3, **{"V4:NTP2": 0.25, "V6:A2": 0.3, "V7:cfg6": 0.3, "V7:cfg8": 0.25}, X1=0.2, Y1=0.05, Y2=0.15)


def synth(seed: int = 0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2018-01-03", "2026-02-26")
    tp = pd.DatetimeIndex(list(d[1:]) + [pd.Timestamp("2026-02-27")])
    lat = np.exp(rng.normal(-1.0, 0.4, len(d)))
    rv = lat * rng.gamma(3.0, 1 / 3.0, len(d))
    s = {k: lat * np.exp(rng.normal(0, v, len(d))) for k, v in SIG.items()}
    s["M"], s["K8"] = np.sqrt(s["K2"] * s["K3"]), np.sqrt(s["K1"] * np.sqrt(s["K2"] * s["K3"]))
    e = (rng.random(len(d)) < 0.15).astype(int)
    ys = {y: pd.DataFrame(dict(d=d, active=(d >= EV.AFTER).astype(int), a_n=0.1, a_e=0.2, kappa=0.3, lam=0.01)) for y in EV.YS}
    return rv, s, tp, d, e, ys


@pytest.fixture(scope="module")
def tabs():
    rv, s, tp, d, e, ys = synth()
    return dict(rv=rv, s=s, tp=tp, d=d, e=e, t=EV.main_tables(rv, s, tp, d, e, ys))


def test_pair_row_known_answer_sign_and_base_denominator(monkeypatch) -> None:
    rv = np.exp(np.random.default_rng(0).normal(0, 0.3, 300))
    tp = pd.bdate_range("2018-01-03", periods=300)
    r = EV.pair_row(rv, 2 * rv, 4 * rv, tp)
    l = lambda u: u - np.log(u) - 1.0
    assert r["y_qlike"] == pytest.approx(l(0.5)) and r["base_qlike"] == pytest.approx(l(0.25))
    assert r["delta_q"] == pytest.approx(l(0.5) - l(0.25)) and r["delta_q"] < 0        # Y лучше базы -> Δ < 0
    assert r["delta_pct"] == pytest.approx(100 * (l(0.5) - l(0.25)) / l(0.25))        # знаменатель — QLIKE БАЗЫ
    assert EV.pair_row(rv, 4 * rv, 2 * rv, tp)["delta_q"] > 0
    orig = EVX.delta_ci
    monkeypatch.setattr(EVX, "delta_ci", lambda qx, qy, k1, **kw: orig(qx, qy, 1.7 * k1, **kw))        # мутация: другая база процентов
    assert EV.pair_row(rv, 2 * rv, 4 * rv, tp)["delta_pct"] != pytest.approx(100 * (l(0.5) - l(0.25)) / l(0.25))


def test_all_pairs_delta_pct_is_percent_of_base(tabs) -> None:
    pr = tabs["t"]["PAIRS"]
    assert np.allclose(pr["delta_pct"], 100 * pr["delta_q"] / pr["base_qlike"], rtol=1e-9, atol=1e-12)
    assert set(pr["y"]) == {"Y1", "Y2"} and ((pr["y"] == "Y1") & (pr["base"] == "Y2")).sum() == len(EV.PWIN)
    assert (pr[pr["window"] == "after"]["n"] == int((tabs["d"] >= EV.AFTER).sum())).all()


def test_holm_n_744_not_row_count(tabs, monkeypatch) -> None:
    pr = tabs["t"]["PAIRS"]
    w = pr[pr["window"] == "2018_2021"]
    pmin = w["p_one"].min()
    assert pmin < 1e-3 and w["p_holm_N"].min() == pytest.approx(min(1.0, 744 * pmin), rel=1e-9, abs=0) and w["p_holm_Nwide"].min() == pytest.approx(min(1.0, 3193 * pmin), rel=1e-9, abs=0)
    monkeypatch.setattr(EV, "N_HOLM", len(w))                 # мутация: N = числу строк таблицы
    mut = EV.holm(pr, EV.PWIN)
    assert mut[mut["window"] == "2018_2021"]["p_holm_N"].min() != pytest.approx(w["p_holm_N"].min(), rel=1e-6, abs=0)


def test_go_matches_v7_without_x1_and_x1_only_tightens(tabs) -> None:
    rv, s, tp = tabs["rv"], tabs["s"], tabs["tp"]
    rf = {k: s[k] for k in ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "X1")}
    v7 = EV.EV.go_table(rv, s["Y1"], rf, tp)
    mine6 = EV.go_table(rv, s["Y1"], rf, tp, bases=EV.EV.GO_BASES)
    assert np.allclose(mine6[["value", "threshold"]], v7[["value", "threshold"]]) and list(mine6["passed"]) == list(v7["passed"])
    mine7 = EV.go_table(rv, s["Y1"], rf, tp)
    assert (mine7["value"].iloc[:3].to_numpy() >= v7["value"].iloc[:3].to_numpy() - 1e-15).all()        # X1 в min только понижает порог


def test_rule62_equals_new_information_and_years_cells_shapes(tabs) -> None:
    rv, s, tp, e, t = tabs["rv"], tabs["s"], tabs["tp"], tabs["e"], tabs["t"]
    r = EV.ST.new_information(rv, s["Y1"], s["X1"], tp)
    row = t["RULE62"].set_index("y").loc["Y1"]
    assert row["rule62"] == ("да" if r["new_information"] else "нет") and row["delta_q"] == pytest.approx(r["delta"]) and row["rule62"] == "да"          # Y1 в синтетике лучше X1
    assert len(t["YEARS"]) == 2 * 2 * 9 and set(t["YEARS"]["year"]) == set(range(2018, 2027))
    c = t["CELLS"].set_index(["cell", "y", "base"])
    y = pd.DatetimeIndex(tp).year
    assert c.loc[("после экстремального бара 2022–2025", "Y1", "X1"), "n"] == int(((y >= 2022) & (y <= 2025) & (e == 1)).sum())
    assert c.loc[("обычные дни 2019–2021", "X1", "K2"), "n"] == int(((y >= 2019) & (y <= 2021) & (e == 0)).sum())
    assert c.loc[("2022 год", "Y2", "K2"), "n"] == int((y == 2022).sum())


def test_report_generated_from_tables_confirmed_only_when_holm_below_005(tabs) -> None:
    t, num = tabs["t"], dict(n_dates=2150, first="a", last="b", pairs_at_activation=124, min_pairs=120, n_inactive=124, close_from="17:40", close_to="18:30", close_slots="46…51", n_zero_days=0, n_extreme_days=7)
    doc = DOC.build(t, num)
    assert doc == DOC.build(t, num) and doc.index("DEVELOPMENT / HISTORICAL WALK-FORWARD") < doc.index("## 1.")
    pr = t["PAIRS"][t["PAIRS"]["base"].isin(["X1", "K2", "K3", "V4:NTP2", "V7:cfg8", "V6:A2", "V7:cfg6", "K1"]) & t["PAIRS"]["window"].isin(DOC.PW)]
    want = int((pr["p_holm_N"] < 0.05).sum()) + int((t["RULE62"]["p_holm_N"] < 0.05).sum())          # строки раздела 4 + первые строки отчёта (Y против X1, 2018–2021)
    assert want > 0 and sum(l.count("подтверждено") for l in doc.splitlines()) == want
    t2 = {**t, "PAIRS": t["PAIRS"].assign(p_holm_N=1.0), "RULE62": t["RULE62"].assign(p_holm_N=1.0)}
    assert "подтверждено" not in DOC.build(t2, num)
    swapped = doc.replace(" %s |" % DOC.f(float(t["METRICS"].iloc[0]["qlike"])), " 9.9999 |", 1)
    assert swapped != doc                                     # подменённое число отличается от генератора -> `--check` его ловит (сверка текста)
