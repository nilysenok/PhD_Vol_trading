"""V6-FIN (G50, G51, G54): сводные таблицы. Известный ответ на синтетике (MCS, разложение «обучатель + признаки»), сверка записанных таблиц с таблицами листов 1a–6 (≤ 1e-12), состав рядов и уровней.
Мутации: перевёрнутый знак MCS-потерь, признаки считаются против A0 вместо A0g, ряд выброшен из MCS."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_0_io as IO, v6_0_q as Q, v6_fin_abl as AB, v6_fin_data as D, v6_fin_tabs as TB


def synth(seed: int = 3) -> dict:
    """28 рядов на датах цели 2017…2026: цель rv = exp(шум), прогноз = rv·exp(ε); K1 — заведомо плохой (смещён в 3 раза), остальные — одного качества."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-03", "2026-02-27")
    n = len(idx)
    rv = np.exp(rng.normal(0.0, 0.5, n))
    w = pd.DataFrame({"rv": rv, "t_plus": idx, "K1": rv * 3.0}, index=idx)
    fs = {s: rv * np.exp(rng.normal(0.0, 0.3, n)) for s in D.SER}
    fs["K1"] = rv * 3.0
    return dict(w=w, fs=fs, cp0={"N": 722, "N_wide": 3122})


def test_mcs_excludes_clearly_worse_series_and_keeps_equal_ones():
    d = synth()
    m = TB.mcs(d)
    a = m[m["window"] == "all"].set_index("model")
    assert not bool(a.loc["K1", "in_mcs90"]) and a["in_mcs90"].sum() >= 20 and a.loc["K1", "mcs_p"] < 0.1
    assert len(m) == 3 * len(D.SER)


def test_mcs_is_deterministic_seed_fixed():
    d = synth()
    pd.testing.assert_frame_equal(TB.mcs(d), TB.mcs(d))


def test_mutation_mcs_with_reversed_sign_of_loss_is_caught(monkeypatch):
    """Потери со знаком «минус» превращают худший ряд в лучший: тест известного ответа это ловит."""
    d = synth()
    orig = Q.ql
    monkeypatch.setattr(TB.Q, "ql", lambda rv, f: -orig(rv, f))
    m = TB.mcs(d)
    a = m[m["window"] == "all"].set_index("model")
    assert bool(a.loc["K1", "in_mcs90"]) and a["in_mcs90"].sum() < 20


def test_decomposition_is_additive_learner_plus_features():
    """A2 − A0 = (A0g − A0) + (A2 − A0g) в % одного уровня K1: разложение точное."""
    d = synth()
    for win in AB.WINS:
        tot = AB.row(d, "total_gbm", "A2", "A0", win)["delta_pct"]
        part = AB.row(d, "learner", "A0g", "A0", win)["delta_pct"] + AB.row(d, "features_gbm", "A2", "A0g", win)["delta_pct"]
        assert tot == pytest.approx(part, abs=1e-12)


def test_mutation_features_against_wrong_base_breaks_additivity(monkeypatch):
    d = synth()
    ok = AB.row(d, "features_gbm", "A2", "A0g", "all")["delta_pct"]
    wrong = AB.row(d, "features_gbm", "A2", "A0", "all")["delta_pct"]
    assert abs(ok - wrong) > 1e-6


def test_pairs_cover_four_gbm_and_nine_lin_configs():
    ps = AB.pairs()
    assert sum(p == "features_gbm" for p, _, _ in ps) == 4 and sum(p == "features_lin" for p, _, _ in ps) == 9 and ps[0] == ("learner", "A0g", "A0")


def test_all_series_and_levels_on_disk():
    a = IO.read_csv("FIN_ALL.csv")
    assert set(a["series"]) == set(D.SER) and len(D.SER) == 28 and a.groupby("series")["n"].apply(lambda s: s.max()).eq(2271).all()
    lv = IO.read_csv("FIN_LEVELS.csv")
    assert len(lv) == 90 and set(lv["config"]) == set(D.CFG) and set(lv["level"]) == {1, 2, 3, 4, 5}
    assert (lv[lv["level"] == 5]["passed"] == "нет").all()


def test_fin_tables_equal_sheet_tables_to_1e_12():
    """G50: метрики конфигураций и контролей в FIN_ALL = метрикам таблиц листов (≤ 1e-12)."""
    a = IO.read_csv("FIN_ALL.csv")
    cols = ["n", "qlike", "rmse", "mae", "r2_rv", "r2_ln", "bias_ln"]
    for c in D.CFG + ["K9"]:
        sh = IO.read_csv("%s_METRICS.csv" % c)
        sh, fin = sh[sh["series"] == c].set_index("window"), a[a["series"] == c].set_index("window")
        assert np.abs(sh[cols].to_numpy(float) - fin.loc[sh.index, cols].to_numpy(float)).max() <= 1e-12, c
    for c in ("K1", "K2", "K3", "K8", "V4:NTP2"):
        sh = IO.read_csv("A0_METRICS.csv")
        sh, fin = sh[sh["series"] == c].set_index("window"), a[a["series"] == c].set_index("window")
        assert np.abs(sh[cols].to_numpy(float) - fin.loc[sh.index, cols].to_numpy(float)).max() <= 1e-12, c


def test_mutation_substituted_metric_is_caught_by_sheet_comparison():
    a = IO.read_csv("FIN_ALL.csv")
    sh = IO.read_csv("A2_METRICS.csv")
    sh = sh[sh["series"] == "A2"].set_index("window")
    fin = a[a["series"] == "A2"].set_index("window")
    bad = fin.copy()
    bad.loc["all", "qlike"] += 1e-6
    assert np.abs(sh["qlike"] - fin.loc[sh.index, "qlike"]).max() <= 1e-12 and np.abs(sh["qlike"] - bad.loc[sh.index, "qlike"]).max() > 1e-12
