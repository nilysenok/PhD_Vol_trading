"""P1b, G8: контроли K4–K7 (hp_ctrl). K7 — синтетика через боевой путь k7_series (положительность, PIT, пол u); K6 — реальный SBER побитно против E8 Ж-214 (пропуск, если файлов нет),
обрезка будущего (hv2_dec.trunc_diff); K4/K5 — состав признаков без величин из прогноза HAR; таблица QLIKE на общих датах. Мутации — подмена крючков hp_ctrl."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2 import har as H, metrics
from models_v2.s10 import hp_ctrl as CT, hp_paths as P
from models_v2.s9 import hv2_dec as DC, hv2_mkt as MK

E8_FILE = P.ROOT / "data/regen/harj_hybrid_v2/MKT_series.parquet"
real = pytest.mark.skipif(not (P.OUT / "daily_SBER.parquet").exists() or not E8_FILE.exists(), reason="нет данных P0 / Ж-214")


def syn(n: int = 900, seed: int = 0, own_mu: float = -1.5):
    """Бумага: β = 0.77 к индексу, RV = β²·RV_IMOEX + u, u = exp(own) < 1 (чтобы log-шкала давала отрицательные значения)."""
    rng = np.random.default_rng(seed)
    ix = pd.bdate_range("2015-01-05", periods=n)
    z = np.zeros(n)
    own = np.zeros(n)
    for t in range(1, n):
        z[t], own[t] = 0.9 * z[t - 1] + 0.2 * rng.normal(), 0.8 * own[t - 1] + 0.2 * rng.normal()
    rm = np.exp(z)
    rv = 0.77 ** 2 * rm + np.exp(own_mu + own)
    d = pd.DataFrame({"rv": rv, "j": 0.1 * rv * rng.random(n), "seg": 0, "cov": 0.77 * rm, "rv_m_sync": rm}, index=ix)
    imo = pd.DataFrame({"rv": rm}, index=ix)
    return d, imo


def k7_of(d, imo):
    k1 = pd.DataFrame({"d": d.index[:-1]})
    hm = pd.DataFrame({"d": imo.index, "sigma2": imo["rv"].rolling(22).mean().to_numpy()})            # прогноз индекса — только по прошлому
    return CT.k7_series(d, imo, k1, hm), hm


def test_k7_polozhitelen_i_bolshe_rynochnoj_chasti():
    d, imo = syn()
    k7, hm = k7_of(d, imo)
    s = k7.dropna(subset=["sigma2_k7"])
    hmk = hm.set_index("d")["sigma2"].reindex(s["d"]).to_numpy()
    assert len(s) > 200 and (s["sigma2_k7"] > 0).all() and np.isfinite(s["sigma2_k7"]).all()
    assert (s["sigma2_k7"].to_numpy() > s["k7_beta2"].to_numpy() * hmk).all()             # Ĥ(u) = exp(·) > 0 строго


def test_mutaciya_k7_u_bez_eksponenty_lovitsya(monkeypatch):
    def log_scale(p, i, min_pairs=CT.MIN_U_PAIRS):                                          # ошибка: прогноз в лог-шкале (забыли exp)
        cols = list(H.COLS_HARJ)
        Xt, yt = H.pairs(p, i, cols)
        ok = np.isfinite(yt)
        if ok.sum() < min_pairs or not np.isfinite(p.X[i, cols]).all():
            return float("nan")
        b, _, _ = H.ols(Xt[ok], yt[ok])
        return float(b[0] + b[1:] @ p.X[i, cols])
    monkeypatch.setattr(CT, "harj_at", log_scale)
    d, imo = syn()
    k7, hm = k7_of(d, imo)
    s = k7.dropna(subset=["sigma2_k7"])
    hmk = hm.set_index("d")["sigma2"].reindex(s["d"]).to_numpy()
    assert not (s["sigma2_k7"].to_numpy() > s["k7_beta2"].to_numpy() * hmk).all()


def test_k7_pol_u_i_beta_tolko_po_proshlomu(monkeypatch):
    d, imo = syn()
    d.iloc[600:, d.columns.get_loc("rv")] = 0.001                                           # RV почти нулевая: β²·RV_IMOEX > RV, срабатывает пол ε·RV
    u = CT.u_daily(d, imo)
    ok = u["rv"].notna()
    assert (u.loc[ok, "rv"] >= 0.01 * d.loc[ok, "rv"] - 1e-15).all() and (u.loc[ok, "rv"].iloc[-100:] == 0.01 * d["rv"].iloc[-100:]).all()
    assert CT.beta_past(d).iloc[:251].isna().all() and CT.beta_past(d).iloc[251:].notna().all()
    assert np.allclose(CT.beta_past(d).dropna(), 0.77, atol=1e-12)
    monkeypatch.setattr(CT, "EPS_U", 0.0)
    assert (CT.u_daily(d, imo)["rv"].dropna() < 0.01 * d["rv"].reindex(CT.u_daily(d, imo)["rv"].dropna().index)).any()   # пол снят — мутация видна


def corrupt_tail(d, imo, i0, seed=3):
    rng = np.random.default_rng(seed)
    d, imo = d.copy(), imo.copy()
    f = np.exp(rng.normal(0, 1.0, len(d) - i0 - 1))
    for c in ("rv", "j", "cov", "rv_m_sync"):
        d.iloc[i0 + 1:, d.columns.get_loc(c)] = d[c].iloc[i0 + 1:] * f
    imo.iloc[i0 + 1:, 0] = imo["rv"].iloc[i0 + 1:] * f
    return d, imo


def test_k7_obrezka_budushchego_i_mutaciya_beta_po_centrirovannomu_oknu(monkeypatch):
    d, imo = syn()
    i0 = 700
    d2, imo2 = corrupt_tail(d, imo, i0)
    a, b = k7_of(d, imo)[0], k7_of(d2, imo2)[0]
    m = (a["d"] <= d.index[i0]).to_numpy()
    assert np.isfinite(a["sigma2_k7"][m]).sum() > 100 and np.array_equal(a["sigma2_k7"][m], b["sigma2_k7"][m], equal_nan=True)
    monkeypatch.setattr(CT, "beta_past", lambda dd, win=252: (dd["cov"].rolling(win, center=True, min_periods=win).sum() / dd["rv_m_sync"].rolling(win, center=True, min_periods=win).sum()).clip(lower=0.0))
    a2, b2 = k7_of(d, imo)[0], k7_of(d2, imo2)[0]
    assert not np.array_equal(a2["sigma2_k7"][m], b2["sigma2_k7"][m], equal_nan=True)


def test_k45_priznaki_bez_velichin_iz_prognoza_har():
    assert CT.K45_FEATS and not any(c in CT.K45_FEATS for c in ("z_x_hdev", "z_x_err1", "z_x_err5", "z_x_err22"))
    assert not any("hdev" in c or "err" in c for c in CT.K45_FEATS) and "z_c_rank" in CT.K45_FEATS and "z_m_med" in CT.K45_FEATS
    p = pd.DataFrame({"rv_m_t": [2.0, 3.0], "sigma2_k1": [5.0, 6.0]})
    assert np.array_equal(CT.k45_inputs(p)[2], [2.0, 3.0])                                      # offset ln RV_m, не K1


@real
def test_k6_sber_pobajtno_raven_e8_zh214(monkeypatch):
    d, imo, k1 = (pd.read_parquet(P.OUT / "daily_SBER.parquet"), pd.read_parquet(P.OUT / "daily_IMOEX.parquet"), pd.read_parquet(P.OUT / "har_SBER.parquet"))
    e8 = pd.read_parquet(E8_FILE)[["d", "E8"]]

    def diff():
        m = CT.k6_series(d, imo, k1).merge(e8, on="d")
        return len(m), float(np.abs(m["sigma2_k6"] - m["E8"]).max())
    assert diff() == (2271, 0.0)
    monkeypatch.setattr(CT, "E8_BLOCKS", ("semicov",))                                           # мутация: блок idio потерян
    assert diff()[1] > 0.0


@real
def test_k6_obrezka_budushchego_i_mutaciya_lead(monkeypatch):
    d, imo, k1 = (pd.read_parquet(P.OUT / "daily_SBER.parquet"), pd.read_parquet(P.OUT / "daily_IMOEX.parquet"), pd.read_parquet(P.OUT / "har_SBER.parquet"))
    dw = CT.e8_world(d, imo, np.flatnonzero(np.isin(d.index.to_numpy(), k1["d"].to_numpy())))
    i0 = int(dw.el[900])
    assert DC.trunc_diff(lambda x: MK.run(x, False, CT.E8_BLOCKS, CT.E8_WIN)[0], dw, i0) == 0.0
    assert DC.trunc_diff(lambda x: MK.run(x, False, CT.E8_BLOCKS, CT.E8_WIN, lead=1)[0], dw, i0) != 0.0


def test_tablitsa_qlike_na_obshchikh_datah():
    t = pd.Timestamp("2018-01-03")
    k = pd.DataFrame({"ticker": ["SBER"] * 3 + ["AFLT"] * 2, "group": ["SBER"] * 3 + ["M16"] * 2, "date": t, "t_plus": [t, t + pd.Timedelta(days=1), t + pd.Timedelta(days=2)] + [t] * 2,
                      "gap": False, "rv_tplus": [2.0, 3.0, 4.0, 2.0, 2.0], "sigma2_k1": [1.0, 3.0, 4.0, 2.0, 2.0]})
    for s in CT.SIG[1:]:
        k[s] = k["sigma2_k1"] * 2.0
    k.loc[2, "sigma2_k5"] = np.nan                                                              # строка 2 — не общая дата: исключена из всех колонок
    q = CT.qlike_tables(k)
    r = q[(q["name"] == "SBER") & (q["period"] == "all")].iloc[0]
    assert r["n"] == 2 and np.isclose(r["q_k1"], np.mean(metrics.qlike([2.0, 3.0], [1.0, 3.0])))
    assert np.isclose(r["q_k4"], np.mean(metrics.qlike([2.0, 3.0], [2.0, 6.0])))
    assert q[(q["name"] == "all28") & (q["period"] == "all")].iloc[0]["n_tickers"] == 2
