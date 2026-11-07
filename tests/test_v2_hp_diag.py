"""P3c, G13 (R15): известный ответ на синтетике для γ, ICC, R² поправки; мутации — E11 попала в диагностику 17; диагностика читает результаты на 28; правила вывода изменены после объявления."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_diag as DG, hp_diag_core as C, hp_diag_data as D, hp_diag_dh as DH, hp_diag_rules as R, hp_engine as E, hp_paths as P


def _lognorm_noise(rng, n, s=0.5):
    return np.exp(s * rng.standard_normal(n) - s * s / 2)          # E = 1


def _synth(gam, n=20000, seed=1):
    rng = np.random.default_rng(seed)
    base = np.exp(rng.normal(0.0, 0.3, n))
    delta = rng.normal(0.0, 0.5, n)
    har = base * np.exp(delta)
    rv = base * np.exp(gam * delta) * _lognorm_noise(rng, n)
    return rv, har, base


def test_gamma_ticker_izvestnyj_otvet():
    rv, har, base = _synth(0.5)
    r = C.gamma_ticker(rv, har, base)
    assert abs(r["gamma"] - 0.5) < 0.1 and r["t"] > 5
    r0 = C.gamma_ticker(*_synth(0.0, seed=2))
    assert abs(r0["gamma"]) < 0.1 and abs(r0["t"]) < 3.5


def test_gamma_panel_izvestnyj_otvet_i_znak():
    k, T = 5, 3000
    rv, har, base = _synth(0.7, n=k * T, seed=3)
    dates = np.tile(np.arange(T), k)
    r = C.gamma_panel(rv, har, base, dates)
    assert abs(r["gamma"] - 0.7) < 0.1 and r["t"] > 5
    r2 = C.gamma_panel(*_synth(0.0, n=k * T, seed=4), dates)
    assert abs(r2["gamma"]) < 0.1 and abs(r2["t"]) < 3.5
    rv_m, har_m, base_m = _synth(0.7, n=k * T, seed=3)
    assert C.gamma_panel(rv_m, har_m, base_m, dates)["gamma"] > 0 > C.gamma_panel(rv_m, base_m * base_m / har_m, base_m, dates)["gamma"]      # перепутанный знак δ — γ меняет знак


def test_icc_izvestnyj_otvet():
    rng = np.random.default_rng(5)
    k, n = 12, 800
    a = np.repeat(rng.normal(0, 1.0, k), n)                          # σ_b² = 1
    r = a + rng.normal(0, np.sqrt(3.0), k * n)                        # σ_w² = 3 → ICC = 0,25
    out = C.icc1(r, np.repeat(np.arange(k), n))
    assert abs(out["icc"] - 0.25) < 0.1 and 0 < out["eta2"] < 0.5
    assert abs(C.icc1(rng.normal(size=k * n), np.repeat(np.arange(k), n))["icc"]) < 0.02
    assert np.isnan(C.icc1(np.arange(5.0), np.zeros(5))["icc"])


def test_r2_izvestnyj_otvet():
    rng = np.random.default_rng(6)
    g = rng.normal(0, 1.0, 50000)
    e = g + rng.normal(0, np.sqrt(3.0), 50000)                        # R² = 1/(1+3) = 0,25
    assert abs(C.r2_correction(e, g)["r2_mz"] - 0.25) < 0.02
    assert abs(C.r2_correction(e, e)["r2_oos"] - 1.0) < 1e-12 and C.r2_correction(e, 0 * e)["r2_oos"] == 0.0
    assert abs(C.r2_correction(e, -g)["r2_mz"] - 0.25) < 0.02 and C.r2_correction(e, -g)["r2_oos"] < 0     # знак поправки перепутан — OOS R² < 0, MZ его не видит


def test_date_mean_t_znak_i_pustota():
    d, dates = np.r_[np.ones(100), np.ones(100)], np.r_[np.arange(100), np.arange(100)]
    r = C.date_mean_t(d + 0.0, dates)
    assert r["mean"] == 1.0 and r["n_dates"] == 100
    assert np.isnan(C.date_mean_t(np.array([1.0, 2.0]), np.array([0, 1]))["mean"])


def _frame(with_e11: bool, e11_junk: bool = False):
    rng = np.random.default_rng(7)
    rows = []
    for tk, grp in [("AAA", "M16"), ("BBB", "M16"), ("CCC", "M16")] + ([("EEE", "E11")] if with_e11 else []):
        d = pd.bdate_range("2017-01-02", periods=400)
        rv = np.exp(rng.normal(0, 0.5, 400))
        k1 = rv * np.exp(rng.normal(0, 0.3, 400))
        f = {"xgb_A1": k1 * 0.9, "xgb_D2": k1 * 1.1, "lgbm_A1": k1 * 0.9, "lgbm_D2": k1 * 1.1}
        if grp == "E11" and e11_junk:
            f = {k: v * 100 for k, v in f.items()}
        rows.append(pd.DataFrame({"ticker": tk, "group": grp, "date": d, "t_plus": d + pd.Timedelta(days=1), "gap": False, "rv": rv, "k1": k1, "ntr_xgb": 1000.0, "ntr_lgbm": 1000.0, **f}))
    return pd.concat(rows, ignore_index=True)


def _m16(rows):
    return sorted((r["window"], r["item"], r["metric"], round(r["value"], 12)) for r in rows if r["population"] == "M16")


def test_e11_ne_vliyaet_na_diagnostiku_17_i_mutaciya_lovitsya(monkeypatch):
    base = _m16(DH.cause_e(_frame(False))[0])
    assert len(base) > 0
    assert _m16(DH.cause_e(_frame(True, e11_junk=True))[0]) == base                 # E11 с мусорными значениями не меняет строки M16
    monkeypatch.setattr(D, "population", lambda df, pop: df)                        # мутация: E11 попала в диагностику 17
    assert _m16(DH.cause_e(_frame(True, e11_junk=True))[0]) != base


def test_mutaciya_chtenie_na_28_lovitsya():
    D.check_names(D.input_names())
    for bad in ("hyb_xgb.parquet", "hyb_lgbm.parquet", "loto.parquet", "engine_panel.parquet", "k4567.parquet", "k23.parquet"):
        with pytest.raises(SystemExit):
            D.check_names(["hyb_xgb_p17.parquet", bad])


def test_load_chitaet_tolko_p17(monkeypatch):
    seen = []
    orig = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: (seen.append(str(p).split("/")[-1]), orig(p, *a, **k))[1])
    df = D.load()
    D.load_loto()
    assert seen and set(seen) <= set(D.input_names())
    assert set(df["group"]) == {"M16", "SBER", "E11"} and df["xgb_A1"].notna().any()


def test_a1_vosproizvoditsya_iz_sohranyonnoj_g():
    h = pd.read_parquet(P.OUT / "hyb_xgb_p17.parquet")
    f = E.apply_corr(h["sigma2_k1"], h["a1_g"], 0.0, h["a1_c"].fillna(1.0).to_numpy(), h["a1_lam"].fillna(1.0).to_numpy(), "qlike")
    assert np.allclose(f[h["a1_g"].notna()], h["sigma2_A1"][h["a1_g"].notna()], rtol=1e-12)          # λ = 1 в (ж) считается из той же g
    f1 = E.apply_corr(h["sigma2_k1"], h["a1_g"], 0.0, h["a1_c"].fillna(1.0).to_numpy(), 1.0, "qlike")
    assert not np.allclose(f1, h["sigma2_A1"], rtol=1e-9) and np.allclose(f1[h["a1_lam"] == 1.0], h["sigma2_A1"][h["a1_lam"] == 1.0], rtol=1e-12)


def test_pravila_vyvoda_tablichnye_otvety():
    ok_a = {"t_k2_all": 0.5, "t_k3_all": -1.0, "t_k2_conf": 1.9, "t_k3_conf": 0.0}
    assert R.verdict("a", ok_a) == "подтверждена"
    assert R.verdict("a", {**ok_a, "t_k3_conf": 2.5}) == "не подтверждена"
    assert R.verdict("a", {**ok_a, "t_k3_conf": float("nan")}) == "не измеримо"
    g = {"lam_med": 0.75, "dq_lam1": 0.004}
    assert R.verdict("g", g) == "подтверждена" and R.verdict("g", {**g, "dq_lam1": -0.004}) == "не подтверждена" and R.verdict("g", {**g, "lam_med": 1.0}) == "не подтверждена"
    assert R.verdict("h", {"gap_share": 0.001, "rel_gap": 9.99}) == "подтверждена" and R.verdict("h", {"gap_share": 0.001, "rel_gap": 0.01}) == "не подтверждена"


def test_mutaciya_pravila_izmeneny_posle_ob_yavleniya_lovitsya(monkeypatch):
    DG.declared()
    s, r = R.CAUSES["f"]
    monkeypatch.setitem(R.CAUSES, "f", (s, r.replace("0,6", "0,3")))
    with pytest.raises(SystemExit):
        DG.declared()
