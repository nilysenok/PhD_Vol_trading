"""P2, G9/G10: архитектура C (hp_hyb_c) — части K7 (k7_parts), цели m/u, forecast_c, select_c на синтетике (сеть и данные проекта не читаются; обучения ML нет).
Правила: λ = 0 ≡ K7 побитно (боевой путь k7_parts против hp_ctrl.k7_series); β только по прошлому (обрезка будущего; мутант «β с днём t⁺» ловится); цели m, u — по известному ответу; выбор (c, λ) по закрытым Y−1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_ctrl as CT, hp_hyb_c as CC


def syn(n: int = 900, seed: int = 0):
    """Бумага с β = 0.77 к индексу (как в test_v2_hp_ctrl): RV = β²·RV_IMOEX + u."""
    rng = np.random.default_rng(seed)
    ix = pd.bdate_range("2015-01-05", periods=n)
    z, own = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        z[t], own[t] = 0.9 * z[t - 1] + 0.2 * rng.normal(), 0.8 * own[t - 1] + 0.2 * rng.normal()
    rm = np.exp(z)
    rv = 0.77 ** 2 * rm + np.exp(-1.5 + own)
    d = pd.DataFrame({"rv": rv, "j": 0.1 * rv * rng.random(n), "seg": 0, "cov": 0.77 * rm * (1 + 0.05 * rng.normal(size=n)), "rv_m_sync": rm}, index=ix)
    imo = pd.DataFrame({"rv": rm}, index=ix)
    hm = pd.DataFrame({"d": ix, "sigma2": imo["rv"].rolling(22).mean().to_numpy()})
    return d, imo, hm


def test_chasti_k7_summiruyutsya_v_k7_pobajtno_i_lambda_nol_dayot_k7():
    d, imo, hm = syn()
    dates = d.index[:-1]
    k7 = CT.k7_series(d, imo, pd.DataFrame({"d": dates}), hm)
    pr = CC.k7_parts(d, imo, dates, hm)
    assert len(pr) == len(k7) and np.array_equal(pr["date"].to_numpy(), k7["d"].to_numpy())
    s = (pr["hm"] + pr["hu"]).to_numpy()
    assert np.isfinite(s).sum() > 300 and np.array_equal(s, k7["sigma2_k7"].to_numpy(), equal_nan=True)
    ok = np.isfinite(s)
    g = np.random.default_rng(1).normal(size=len(pr))
    f0 = CC.forecast_c(pr["hm"].to_numpy(), pr["hu"].to_numpy(), g, g, 1.0, 0.0, 0.0)             # λ_m = λ_u = 0
    assert np.array_equal(f0[ok], s[ok])
    f1 = CC.forecast_c(pr["hm"].to_numpy(), pr["hu"].to_numpy(), g, g, 1.0, 1.0, 1.0)
    assert not np.array_equal(f1[ok], s[ok]) and (f1[ok] > 0).all()


def test_beta_tolko_po_proshlomu_i_mutaciya_beta_s_dnem_t_plus(monkeypatch):
    d, imo, hm = syn()
    d0 = d.index[600]
    rng = np.random.default_rng(5)
    dc = d.copy()
    dc.loc[dc.index > d0, "cov"] = rng.normal(size=int((dc.index > d0).sum())) * 5
    dc.loc[dc.index > d0, "rv_m_sync"] = 1 + rng.random(int((dc.index > d0).sum())) * 9
    dates = d.index[:-1]
    a, b = CC.k7_parts(d, imo, dates, hm), CC.k7_parts(dc, imo, dates, hm)
    past = (a["date"] <= d0).to_numpy() & np.isfinite(a["b2"].to_numpy())
    assert past.sum() > 100 and np.array_equal(a["b2"].to_numpy()[past], b["b2"].to_numpy()[past], equal_nan=True)
    assert np.array_equal(a["hm"].to_numpy()[past], b["hm"].to_numpy()[past], equal_nan=True)
    monkeypatch.setattr(CC, "beta2_at", lambda dd: (CT.beta_past(dd).shift(-1) ** 2).to_numpy(float))      # мутант: β по окну, включающему t⁺
    c = CC.k7_parts(dc, imo, dates, hm)
    assert not np.array_equal(c["b2"].to_numpy()[past], a["b2"].to_numpy()[past], equal_nan=True)


def test_celi_m_i_u_po_izvestnomu_otvetu():
    p = pd.DataFrame({"rv_tplus": [10.0, 4.0, 5.0, 8.0], "mkt_rv_tplus": [10.0, 10.0, np.nan, 5.0]})
    b2 = np.array([0.5, 0.5, 0.5, np.nan])
    m, u = CC.targets(p, b2)
    assert m[:2].tolist() == [5.0, 5.0] and np.isnan(m[2:]).all()
    assert u[0] == 5.0 and u[1] == 0.01 * 4.0 and np.isnan(u[2:]).all()               # u = max(RV − m, ε·RV): пол 0,01·RV, не ноль и не минус


def _panel(n=800, tk=("A", "B")):
    dates = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(3)
    p = pd.concat([pd.DataFrame({"ticker": t, "date": dates[:-1], "t_plus": dates[1:], "rv_tplus": np.exp(rng.normal(size=n - 1))}) for t in tk], ignore_index=True)
    o = np.argsort(p["date"].to_numpy(), kind="stable")
    p = p.iloc[o].reset_index(drop=True)
    return p, np.exp(rng.normal(size=len(p))), np.exp(rng.normal(size=len(p))), {dp: pd.DataFrame({"g": rng.normal(size=len(p))}) for dp in (2, 4)}


def test_vybor_c_po_zakrytym_proshlogo_goda_i_c2_bez_gm():
    p, hm, hu, runs = _panel()
    base = CC.select_c(p, hm, hu, runs, runs, 2017)
    q = p.copy()
    q.loc[q["t_plus"] >= "2017-01-01", "rv_tplus"] = 7.0                              # цели 2017+ выбор на 2017 не меняют
    assert CC.select_c(q, hm, hu, runs, runs, 2017) == base and base["source"] == "closed"
    assert CC.select_c(p, hm, hu, runs, runs, 2015)["source"] == "default"
    s2, ch = CC.finalize_c(p, hm, hu, None, runs, g_m_zero=True, years=[2017])        # C2: runs_m не нужны
    m17 = (pd.DatetimeIndex(p["date"]).year == 2017)
    assert np.isfinite(s2[m17]).all() and (ch["lam_m"] == 0.0).all()
    s0, _ = CC.finalize_c(p, hm, hu, None, runs, g_m_zero=True, years=[2017], lams=(0.0,))
    assert np.array_equal(s0[m17], (hm + hu)[m17])                                    # λ_u = 0 при g_m ≡ 0 -> K7 побитно
