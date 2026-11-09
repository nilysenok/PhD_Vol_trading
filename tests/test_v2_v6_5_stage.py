"""V6-5, G42–G44: известный ответ целей F1/F2 на двух средах, F_YEARS (годовые Δ, СКО ddof = 1, лет с Δ < 0) с подложенными ошибками, регистрация CB1, F1, F2."""
import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q, v6_1_reg as RG, v6_5_fit as R, v6_5_yr as YR


def test_known_answer_f1_f2_two_envs():
    e = np.r_[np.full(10, 0.2), np.full(10, 0.7)]
    lab = np.r_[np.zeros(10, int), np.ones(10, int)]
    A = np.ones((20, 1))
    q = np.array([np.exp(0.2) - 0.2 - 1, np.exp(0.7) - 0.7 - 1])
    th = np.zeros(1)
    assert abs(R.robust_obj(th, A, e, lab, 0.0, "F1")[0] - (q.mean() + 0.5 * abs(q[1] - q[0]) / 2)) <= 1e-12
    assert abs(R.robust_obj(th, A, e, lab, 0.0, "F2")[0] - np.log(np.exp(20 * q).sum()) / 20) <= 1e-12


def test_theta0_not_penalized_and_slopes_are():
    A = np.column_stack([np.ones(30), np.linspace(-1, 1, 30)])
    e, lab = np.zeros(30), np.r_[np.zeros(15, int), np.ones(15, int)]
    th = np.array([0.0, 0.0])
    for kind in ("F1", "F2"):
        d = R.robust_obj(np.array([1.0, 0.0]), A, e, lab, 5.0, kind)[0] - R.robust_obj(np.array([1.0, 0.0]), A, e, lab, 0.0, kind)[0]
        s = R.robust_obj(np.array([0.0, 1.0]), A, e, lab, 5.0, kind)[0] - R.robust_obj(np.array([0.0, 1.0]), A, e, lab, 0.0, kind)[0]
        assert abs(d) <= 1e-12 and abs(s - 5.0) <= 1e-12 and th.sum() == 0


def _w():
    ix = pd.bdate_range("2017-01-02", "2026-03-31")
    n = len(ix)
    rng = np.random.default_rng(5)
    return pd.DataFrame({"t_plus": ix.to_series().shift(-1).ffill().to_numpy(), "rv": np.exp(rng.normal(size=n) * 0.3), "K1": np.exp(rng.normal(size=n) * 0.3)}, index=ix)


def test_years_table_known_answer_and_mutations():
    w = _w()
    rv = w["rv"].to_numpy()
    yr = pd.DatetimeIndex(w["t_plus"]).year.to_numpy()
    fy = {"CB1": rv * 1.1, "F1": rv * 1.1, "F2": rv * (1.02 + 0.005 * (yr - 2017))}               # F2 ближе к rv, чем CB1, в каждом году, но по-разному
    t = YR.years_table(fy, w)
    f1 = t[(t["pair"] == "F1 vs CB1") & (t["section"] == "итог")]
    assert (f1["sd_delta_qlike"] == 0).all() and (f1["years_neg"] == 0).all()                # F1 ≡ CB1: нулевой разброс, лет с Δ<0 нет
    f2 = t[(t["pair"] == "F2 vs CB1") & (t["section"] == "итог")].set_index("window")
    assert f2.loc["2017_2021", "years_n"] == 5 and f2.loc["2017_2025", "years_n"] == 9
    assert f2.loc["2017_2021", "years_neg"] == 5 and f2.loc["2017_2025", "years_neg"] == 9      # F2 ближе к rv: Δ < 0 в каждом году
    yy = t[(t["pair"] == "F2 vs CB1") & (t["section"] == "год")]
    assert set(yy["window"]) == {str(y) for y in range(2017, 2027)}                           # 2026 — частичный, но есть
    d = yy[yy["window"].isin([str(y) for y in range(2017, 2022)])]["delta_qlike"].to_numpy()
    assert abs(f2.loc["2017_2021", "sd_delta_qlike"] - d.std(ddof=1)) <= 1e-15 and abs(f2.loc["2017_2021", "sd_delta_qlike"] - d.std(ddof=0)) > 1e-9      # ddof = 1 (Д7)
    tm = YR.years_table({"CB1": rv * 1.05, "F1": rv * 1.1, "F2": rv * 1.05}, w)                # мутация: знак перепутан (F хуже CB1) -> лет с Δ<0 нет
    assert (tm[(tm["pair"] == "F1 vs CB1") & (tm["section"] == "итог")]["years_neg"] == 0).all()
    assert (tm[(tm["pair"] == "F1 vs CB1") & (tm["section"] == "год")]["delta_qlike"] > 0).all()


def test_years_table_empty_difference_and_levels_use_tplus_year():
    w = _w()
    rv = w["rv"].to_numpy()
    t = YR.years_table({"CB1": rv, "F1": rv, "F2": rv}, w)
    assert (t["delta_qlike"].dropna() == 0).all()
    assert Q.annual(np.ones(len(w)), w["t_plus"].to_numpy(), YR.YEARS)[2026] == 1.0


def test_cb1_f1_f2_registered_and_text_of_prior_rows_unchanged():
    for c in ("CB1", "F1", "F2"):
        assert c in RG.CONFIGS and c in RG.DECLARED and RG.CONFIGS[c]["features"] == ["S", "Sigma", "Q"]
    txt = RG.trial_text(["A0", "CB1", "F1", "F2"])
    assert not RG.trials_bad(txt) and txt.count("Ж-227:") == 4
    assert RG.trials_bad(txt + RG.trial_text(["F2"])[len(RG.trial_text([])):]) != []             # дубль строки F2 отвергается
    assert RG.trials_bad(RG.trial_text(["A0"]) + "Ж-227:ZZ,x\n") != []                         # id вне плана отвергается
