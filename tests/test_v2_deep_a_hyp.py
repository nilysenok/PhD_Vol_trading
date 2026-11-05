"""A12: гипотезы (а)–(е): вывод вычисляется кодом по правилам GATES_A.md."""
import numpy as np
import pandas as pd

from models_v2.s5 import common, hyp


def test_verdict_rho_monotonnaya_svyaz():
    x = np.arange(9.0)
    assert hyp.verdict_rho(hyp.rho(x, -x), -1) == "подтверждена"
    assert hyp.verdict_rho(hyp.rho(x, x), -1) == "отвергнута"
    assert hyp.verdict_rho(0.0, -1) == "отвергнута" and hyp.verdict_rho(-0.4, -1) == "не различима"
    assert hyp.verdict_rho(float("nan"), -1) == "не различима"


def test_mut_znak_pravila_lovitsya():
    r = -0.9
    assert hyp.verdict_rho(r, -1) != hyp.verdict_rho(r, +1)


def test_verdicty_b_c_d():
    assert hyp.verdict_b(1e-5, -0.8) == "подтверждена" and hyp.verdict_b(0.2, -0.8) == "отвергнута" and hyp.verdict_b(0.03, -0.8) == "не различима"
    assert hyp.verdict_c(0.06) == "подтверждена" and hyp.verdict_c(-0.01) == "отвергнута" and hyp.verdict_c(0.02) == "не различима"
    assert hyp.verdict_d(5.0, -1.0, 0.1) == "подтверждена" and hyp.verdict_d(5.0, 4.0, 0.6) == "подтверждена"
    assert hyp.verdict_d(5.0, 4.0, 0.1) == "отвергнута" and hyp.verdict_d(5.0, 4.0, 0.4) == "не различима"


def test_sensitivity_bez_2022_na_sinteticheskoj_tablice():
    n = 40
    td = pd.date_range("2021-12-01", periods=n)
    q = np.where(td.year == 2022, 1.0, 0.1)
    df = pd.DataFrame({"target_date": td, "qlike_b2": q, "qlike_harj": 0.1, "dloss": q - 0.1})
    s = hyp.sensitivity(df).set_index("sample")
    assert abs(s.loc["without_2022", "delta"]) < 1e-12 and s.loc["all", "delta"] > 0.1


def test_real_shest_gipotez_i_dopustimye_vyvody():
    h = common.read_comment_csv(common.REP / "HYPOTHESES_A.csv")
    assert list(h["id"]) == list("абвгде") and set(h.verdict) <= {"подтверждена", "отвергнута", "не различима"}
    top = common.read_comment_csv(common.REP / "TOP20_DAYS.csv")
    assert len(top) == 20 and (pd.to_datetime(top.target_date).dt.year <= 2021).all()
    assert (top["dloss"].abs().diff().dropna() <= 1e-12).all()   # по убыванию |разности|
