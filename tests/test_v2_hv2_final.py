"""Ж-214, лист 5: финалисты по правилу V12, Холм с N по реестру, уровни (2 без 1), N до статистики, 5 крупнейших дней, статус «потолок». Синтетика с известным ответом + мутации:
финалист выбран не по crit_wf; N Холма = числу тестов; уровень 2 без уровня 1; N не объявлено до расчёта."""
import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s5.d import stats_d
from models_v2.s8 import hs_gstage as G, hs_levels as LV
from models_v2.s9 import hv2_fladder as FL, hv2_fpick as FP, hv2_fstats as FS, hv2_ledger as L

N_REG = 585


def synth(n=2400, seed=7):    # 2400 рабочих дней от 03.01.2017 — до середины 2026 (как реальная сетка: есть все периоды)
    rng = np.random.default_rng(seed)
    td = pd.bdate_range("2017-01-03", periods=n)
    lrv = np.zeros(n)
    for i in range(1, n):
        lrv[i] = 0.8 * lrv[i - 1] + rng.normal(0, 0.5)
    f = lambda sd: np.exp(lrv + rng.normal(0, sd, n) + 0.1)
    df = pd.DataFrame({"d": td - pd.offsets.BDay(1), "target_date": td, "rv": np.exp(lrv + 0.1), "HAR": f(0.52), "XJ": f(0.5), "LF": f(0.5)})
    return df, {"M1": f(0.46), "M2": f(0.6)}


@pytest.fixture(scope="module")
def world():
    df, ser = synth()
    fin = pd.DataFrame([{"step": "Л1", "model": "M1", "label": "QLIKE-HAR-J", "crit_wf": 0.0}, {"step": "Л2", "model": "M2", "label": "HARQ-J (без QLIKE)", "crit_wf": 0.0}])
    return df, ser, fin, FS.build_s(df, fin, ser)


def test_choose_by_crit_wf_not_by_qlike():
    crit = {"A": 0.30, "B": 0.25, "C": 0.28}
    qall = {"A": 0.20, "B": 0.26, "C": 0.27}      # минимум QLIKE у A, минимум crit_wf у B
    assert FP.choose(crit, ("A", "B", "C")) == "B"
    assert min(qall, key=qall.get) == "A"          # мутация «по QLIKE всей истории» дала бы A — тест её ловит
    assert FP.choose({"A": 0.1, "B": 0.1}, ("B", "A")) == "B"      # ничья — раньше в списке


def test_choose_missing_candidate_stops():
    with pytest.raises(KeyError):
        FP.choose({"A": 0.3}, ("A", "B"))
    with pytest.raises(KeyError):
        FP.choose({"A": 0.3, "B": float("nan")}, ("A", "B"))


def test_real_finalists_are_argmin_and_h9_not_in_l2():
    fin = FP.finalists()
    assert list(fin["step"]) == ["Л1", "Л2", "Л3", "Л4", "Л5", "Л6", "Л7", "Л8", "Л9", "V8"]
    l2 = fin[fin["step"] == "Л2"].iloc[0]
    assert "H9" not in l2["candidates"].split(",") and "H7" not in l2["candidates"].split(",")
    for _, r in fin.iterrows():
        assert r["crit_wf"] <= r["crit_best_other"] or np.isnan(r["crit_best_other"])
    assert not set(fin["model"]) & {"Z6", "J3"}


def test_holm_N_is_registry_not_number_of_tests(world, monkeypatch):
    df, _, _, s = world
    t, _ = FS.stats_all(df, s, N_REG)
    fam = t[(t["period"] == "all") & t["test"].isin(["A", "B"])]
    p = fam["p_raw"].fillna(1.0).to_numpy()
    assert len(p) == 4 and ((p > 1e-6) & (p < 0.9)).any()
    assert np.allclose(fam["p_holm"].to_numpy(), stats_d.holm_n(p, N_REG)) and not np.allclose(stats_d.holm_n(p, N_REG), stats_d.holm_n(p, len(p))) and (fam["N_holm"] == N_REG).all() and (fam["family_tests"] == 4).all()
    orig = stats_d.holm_n
    monkeypatch.setattr(G.stats_d, "holm_n", lambda pp, N: orig(pp, len(pp)))     # мутация: N = числу тестов
    tm, _ = FS.stats_all(df, s, N_REG)
    fm = tm[(tm["period"] == "all") & tm["test"].isin(["A", "B"])]
    assert not np.allclose(fm["p_holm"].to_numpy(), orig(p, N_REG))


def test_level2_requires_level1():
    rows = []
    for tst, d in (("A", -0.01), ("B", -0.01), ("C", -0.01)):
        for per in ("all", "no_240222", "G_2019_2021", "P2_2022", "P3_2023_2025", "late_2019_2026"):
            rows.append({"model": "M", "test": tst, "period": per, "delta": d, "p_raw": 0.01, "boot_hi95": -0.001, "p_holm": 0.5 if tst == "A" and per == "all" else 0.01})
    t = pd.DataFrame(rows)
    crit, qall = {"M": 0.20, "HAR": 0.25, "S*": 0.23}, {"M": 0.20, "HAR": 0.25}
    o = LV.levels(t, "M", crit, qall, np.full(50, -0.01))
    assert o["L1_c"] and not o["L1_d"] and not o["L1"]              # p Холма теста A 0,5 >= 0,10
    assert o["L2_b"] and o["L2_c"] and o["L2_d"] and not o["L2"] and not o["L3"]       # уровень 2 без 1 — нельзя
    assert o["L4"] == "не проверялся" and o["verdict"] == "L1 только сырой"
    mut = bool(o["L2_b"] and o["L2_c"] and o["L2_d"])             # мутация: L2 без L1
    assert mut != o["L2"]


def test_n_declared_before_stats(tmp_path):
    chk = tmp_path / "CHECKPOINT.json"
    n = FS.declare_n(path=chk)
    st = json.loads(chk.read_text())
    assert n == st["N_declared_before_final"] == st["N_for_corrections"] == st["N_final"] == L.N_PRIOR + st["used_total"] == N_REG
    assert n != 20      # не число тестов семейства (2 теста × 10 финалистов)


def test_top5_days_known_answer():
    n = 40
    df = pd.DataFrame({"target_date": pd.bdate_range("2020-01-01", periods=n), "rv": np.ones(n)})
    s = {"HAR-J": np.full(n, 4.0), "XGBoost": np.full(n, 4.0), "LightGBM": np.full(n, 4.0), "M": np.full(n, 4.0)}
    s["M"][:5] = 1.0           # 5 дней, где M точна
    t = FS.top5_table(df, s)
    a = t[(t["model"] == "M") & (t["ref"] == "A0")].iloc[0]
    assert a["delta_all"] < 0 and a["delta_wo_top5"] == pytest.approx(0.0, abs=1e-12) and a["top5_share_of_gain"] == pytest.approx(1.0)


def test_pipeline_status_in_every_row_and_panel_not_in_package(world):
    df, ser, fin, s = world
    t, lv = FS.stats_all(df, s, N_REG)
    assert lv["ceiling_note"].str.contains("потолок").all() and lv["ceiling_note"].str.contains("HISTORICAL WALK-FORWARD").all()
    assert (lv["L4"] == "не проверялся").all() and not lv["verdict"].str.contains("ACCEPTED").any()
    met = G.metric_rows(df, s)
    tb = FL.p29(met, t, lv, fin, df)
    real = tb[tb["config"].isin(["A0", "XJ", "LF", "M1", "M2"])]
    assert (real["n"] == len(df)).all() and real["date_first"].nunique() == 1 and real["date_last"].nunique() == 1
    assert tb["status"].str.contains("HISTORICAL WALK-FORWARD").sum() == 5
    assert tb.iloc[-1]["model"] == "HAR-J Panel Hybrid" and "не в этом пакете" in tb.iloc[-1]["status"]


def test_ladder_and_rating_rules(world):
    df, ser, fin, s = world
    t, lv = FS.stats_all(df, s, N_REG)
    t5 = FS.top5_table(df, s)
    lad = FL.ladder(df, s, fin, {"T3": ser["M1"], "E1": ser["M1"]}, t5)
    assert list(lad["step"]) == ["Л0", "Л1", "Л2"] and lad.loc[lad["step"] == "Л1", "own_base"].iloc[0] == "A0"
    rk = FL.rating(lad, t, fin)
    assert rk.iloc[0]["model"] == "M1" and rk.iloc[0]["passed"] > rk.iloc[1]["passed"]
    fin3 = pd.concat([fin, pd.DataFrame([{"step": "Л3", "model": "M3", "label": "QLIKE-HARQ-J", "crit_wf": 0.0}])], ignore_index=True)
    s3 = {**s, "Л3:M3": ser["M1"] * 1.01}
    lad3 = FL.ladder(df, s3, fin3, {"T3": ser["M1"], "E1": ser["M1"]}, FS.top5_table(df, s3))
    assert lad3.loc[lad3["step"] == "Л3", "own_base"].iloc[0] in ("M1", "M2") and np.isfinite(lad3.loc[lad3["step"] == "Л3", "d_crit_own"].iloc[0])
    assert FL.own_base("Л3",pd.DataFrame({"step": ["Л1", "Л2"], "model": ["Q", "H"], "crit_wf": [0.3, 0.2]})) == "H"      # лучшая из Л1/Л2 по crit_wf
