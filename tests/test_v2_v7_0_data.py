"""V7-0: G1 (контроли), G3 (паспорт, тождество), G5 (число закрытых пар), G6 (карта). Реальные сохранённые файлы, без обучения; мутации — подменой входа."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_0_ctrl as C, v7_0_pass as PA, v7_0_reuse as RE, v7_0_samples as SA, v7_0_seq as SQ


def test_g1_controls_equal_cmp():
    r = C.check_rows()
    assert (r["status"] == "совпало").all() and len(r) == 6 * 7
    assert r[r["check"] == "qlike"]["abs_diff"].max() <= 1e-12


def test_g1_mutation_forecast_shifted_is_caught(monkeypatch):
    orig = C.forecast
    monkeypatch.setattr(C, "forecast", lambda name, path, idx, w: orig(name, path, idx, w) * (1 + 1e-6) if name == "K2" else orig(name, path, idx, w))
    r = C.check_rows()
    bad = r[r["status"] != "совпало"]
    assert set(bad["series"]) == {"K2"} and {"sha256_forecast", "qlike"} <= set(bad["check"])


def test_g1_mutation_wrong_expected_sha_is_caught(monkeypatch):
    orig = C.frozen
    def swapped():
        f = orig().copy()
        f.loc["K1", "sha256_file"] = f.loc["K3", "sha256_file"]
        return f
    monkeypatch.setattr(C, "frozen", swapped)
    r = C.check_rows()
    assert (r[(r["series"] == "K1") & (r["check"] == "sha256_file")]["status"] == "расхождение").all()


def test_g1_alignment_drops_row_without_boosters():
    w = C.aligned()
    assert len(w) == 2271 and w.index.min() == pd.Timestamp("2016-12-30") and w.index.max() == pd.Timestamp("2026-02-26")


def test_g3_identity_all_days_and_dropped_bar_mutation():
    bars, d = PA.load()
    f = SQ.bar_table(bars, d.index)
    rel = PA.identity(f, d)
    assert len(rel) == 2944 and rel.notna().all() and rel.max() <= PA.TOL_ID
    day = d.index[1000]
    b2 = bars[~((bars["begin"].dt.normalize() == day) & (bars["begin"].dt.strftime("%H:%M") == "12:00"))]
    rel2 = PA.identity(SQ.bar_table(b2, d.index), d)
    assert rel2.max() > PA.TOL_ID and rel2.idxmax() == day                   # выброшенный бар ломает тождество на своём дне


def test_g3_passport_years_and_channels():
    tab, info = PA.passport()
    y = tab[tab["section"] == "year"].set_index("year")
    assert y.loc[2016, "bars_mode"] == 53 and y.loc[2016, "bars_min"] == 53 and y.loc[2014, "bars_mode"] == 52
    assert y["zero_volume_bars"].sum() == 0 and y.loc[2025, "excluded_noncanonical_session_bars"] == 3411
    assert y["n_days"].sum() == 2944 and info["max_rel"] <= 1e-9
    ch = tab[tab["section"] == "channel"].set_index("channel")
    assert set(SQ.CH) == set(ch.index) and ch.loc["vwap", "available"].startswith("да")


def test_g5_closed_pairs_boundary_inclusive_and_known_counts():
    t = pd.Series(pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"]))
    assert SA.closed(t, np.ones(3, bool), pd.Timestamp("2020-01-06")) == 2          # t⁺ ≤ D включительно
    assert SA.closed(t, np.array([False, True, True]), pd.Timestamp("2020-01-07")) == 2
    m, info = SA.table()
    m = m.set_index("date")
    assert info["first_a"] == info["first_b"] == "2016-12-30"
    assert m.loc["2017-01-03", "n_a"] == 1 and m.loc["2018-01-03", "n_a"] == 253 and m.loc["2022-01-03", "n_a"] == 1264
    assert (m["n_a"] == m["n_b"]).all() and (np.diff(m["n_a"].to_numpy()) > 0).all() and m["n_a"].iloc[-1] == 2253


def test_g5_mutation_d_le_D_instead_of_tplus_le_D_is_caught():
    w = C.aligned()
    D = pd.Timestamp("2017-01-03")
    honest = SA.closed(w["t_plus"], np.ones(len(w), bool), D)
    mutant = int((w.index <= D).sum())                                               # d ≤ D (ошибка: пара ещё не закрыта)
    assert honest == 1 and mutant == 2 and honest != mutant


def test_g5_ratio_template_and_first_dates():
    t, _ = SA.samples()
    r = t[t["section"] == "ratio"]
    assert set(r["P"].dropna()) == set(SA.P_LIST) and len(r) == 10 * len(SA.P_LIST)
    x = r[(r["date"] == "2018-01-03") & (r["P"] == 1000)].iloc[0]
    assert abs(x["ratio_a"] - 0.253) < 1e-12


def test_g6_reuse_map_items():
    t = RE.table()
    assert set(t["item"]) == set(range(1, 12))
    assert set(t[t["status"] != "есть"]["item"]) == {6, 7, 10, 11}
    k1 = t[t["series"] == "K1"].iloc[0]
    assert abs(k1["q_all"] - 0.23366211569377798) < 1e-12 and abs(t[t["series"] == "V4:NTP2"].iloc[0]["q_2017_2021"] - 0.14849391752663751) < 1e-12


@pytest.mark.parametrize("name", ["V6:A0", "V6:A0g", "V6:A1", "V6:A2", "V6:A3", "V4:B1", "V4:C6", "J210a:A1_roll_126"])
def test_g6_every_mapped_series_is_finite_on_all_dates(name):
    w = C.aligned()
    path = {s: p for _, _, ss, _ in RE.ITEMS for s, _, p in ss}[name]
    f = RE.load_series(name, path, w)
    assert len(f) == 2271 and np.isfinite(f).all() and (f > 0).all()
