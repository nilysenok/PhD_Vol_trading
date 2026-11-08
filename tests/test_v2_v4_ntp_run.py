"""V4-NTP (G29): боевой путь `v4_ntp_run.forecast` на синтетике с известным устройством — обрезка (PIT), мутации «закрыто + 1», тождества π ≡ 1 → C8, π ≡ 0 → M, LOTO, NTP3 восстанавливает знак гейта."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_b_run as BR, v4_ntp_gate as GT, v4_ntp_run as R, v4_ntp_state as SS, v4_z_feat as ZF
from tests.test_v2_v4_z_feat import synth


@pytest.fixture(scope="module")
def data():
    P, inp = synth(560)
    rng = np.random.default_rng(11)
    G = P[["ticker", "date"]].copy()
    G["F"] = (P["h"] * np.exp(rng.normal(0, 0.2, len(P)))).to_numpy()
    return P, G, inp, ZF.build_panel_z(P, inp)


def day_sep(P):
    d = P["date"]
    return d[d.dt.to_period("M") == pd.Period("2017-09")].min()


@pytest.mark.parametrize("cfg", R.CFGS)
def test_cut_future_data_does_not_change_forecast(data, cfg):
    P, G, inp, _ = data
    assert not R.pit_violation(cfg, P, G, inp, day_sep(P))


@pytest.mark.parametrize("cfg", R.CFGS)
def test_mutation_closed_plus_one_is_caught_via_real_path(data, monkeypatch, cfg):
    P, G, inp, _ = data
    monkeypatch.setattr(SS, "closed_date", lambda t, d: d)
    assert R.pit_violation(cfg, P, G, inp, day_sep(P))


def test_ntp3_mutation_closed_plus_one_in_monthly_fit_is_caught(data, monkeypatch):
    """Оценка гейта по парам, закрытым «плюс три дня» (в т. ч. будущим), ловится обрезкой отдельно от состояния m/s."""
    P, G, inp, _ = data
    monkeypatch.setattr(BR, "closed_mask", lambda t_plus, d0: np.asarray(t_plus) <= np.datetime64(d0) + np.timedelta64(3, "D"))
    assert R.pit_violation("NTP3", P, G, inp, day_sep(P))


@pytest.mark.parametrize("cfg", R.CFGS)
def test_identities_pi_one_is_c8_and_pi_zero_is_m_bitwise(data, cfg):
    P, G, inp, zl = data
    Pm = P.merge(G.rename(columns={"F": "c8"}), on=["ticker", "date"]).sort_values(["date", "ticker"])
    one, zero = (R.forecast(cfg, P, G, zl, force_pi=v)[0]["F"].to_numpy() for v in (1.0, 0.0))
    assert np.array_equal(one, Pm["c8"].to_numpy()) and np.array_equal(zero, np.sqrt(Pm["k2"].to_numpy() * Pm["k3"].to_numpy()))


@pytest.mark.parametrize("cfg", R.CFGS)
def test_forecast_positive_pi_in_unit_interval_and_prior_phase(data, cfg):
    P, G, _, zl = data
    out, w = R.forecast(cfg, P, G, zl)
    assert len(out) == len(P) and (out["F"] > 0).all() and np.isfinite(out["F"]).all() and ((out["pi"] >= 0) & (out["pi"] <= 1)).all()
    assert (out[out["date"] < "2017-02-01"]["pi"] == 0.5).all()                  # до 60 закрытых пар панели/бумаги — приор
    assert {"config", "month", "ticker", "mean_pi"} <= set(w.columns) and "PANEL" in set(w["ticker"])


def test_ntp4_step_bound_on_real_path(data):
    P, G, _, zl = data
    out = R.forecast("NTP4", P, G, zl)[0].sort_values(["date", "ticker"])
    tix = out["ticker"].astype("category").cat.codes.to_numpy()
    assert SS.max_step(out["pi"].to_numpy(), tix, 4) <= SS.STEP + 1e-12
    assert SS.max_step(np.where(np.arange(len(out)) % 2 == 0, 0.9, 0.1), tix, 4) > SS.STEP           # проверка шага не пропускает скачки


@pytest.mark.parametrize("cfg", ["NTP1", "NTP2", "NTP4"])
def test_loto_panel_value_excludes_held_out_real_path(data, cfg):
    P, G, _, zl = data
    Q = P.copy()
    Q.loc[Q["ticker"] == "IRAO", "rv"] *= 5.0
    a, b = R.forecast(cfg, P, G, zl, exclude="IRAO")[0], R.forecast(cfg, Q, G, zl, exclude="IRAO")[0]
    assert np.array_equal(a["pan"].to_numpy(), b["pan"].to_numpy(), equal_nan=True)
    c = R.forecast(cfg, Q, G, zl)[0]
    assert not np.array_equal(a["pan"].to_numpy(), c["pan"].to_numpy(), equal_nan=True)          # без исключения панель видит подмену


def test_ntp3_loto_excluded_ticker_has_zero_delta_and_does_not_enter_training(data):
    P, G, _, zl = data
    Q = P.copy()
    Q.loc[Q["ticker"] == "IRAO", "rv"] *= 7.3
    (a, wa), (b, wb) = R.forecast("NTP3", P, G, zl, exclude="IRAO"), R.forecast("NTP3", Q, G, zl, exclude="IRAO")
    t = wa[wa["ticker"] == "IRAO"]
    assert (t["delta"] == 0).all() and t["excluded"].all()
    p, q = wa[wa["ticker"] == "PANEL"][["alpha", "b_zmq", "b_abs_ell", "b_med_lrv"]], wb[wb["ticker"] == "PANEL"][["alpha", "b_zmq", "b_abs_ell", "b_med_lrv"]]
    assert np.array_equal(p.to_numpy(), q.to_numpy())                         # параметры панели не видят исключённую
    wc = R.forecast("NTP3", Q, G, zl)[1]
    assert not np.array_equal(wc[wc["ticker"] == "PANEL"]["b_zmq"].to_numpy(), p["b_zmq"].to_numpy())


def test_ntp3_known_gate_sign_recovered():
    """Вес на C8 = σ(1,5·u), u — признак |ℓ| в позиции abs_ell: оценка гейта восстанавливает знак (b_abs_ell > 0), прочие коэффициенты малы."""
    rng = np.random.default_rng(5)
    n, nT = 700, 3
    days = pd.bdate_range("2017-01-02", periods=n + 1)
    P = pd.concat([pd.DataFrame({"ticker": t, "date": days[:-1], "t_plus": days[1:]}) for t in ("AFLT", "ALRS", "HYDR")], ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    N = len(P)
    lM, ell, u = rng.normal(0, 0.3, N), rng.normal(0, 0.5, N), rng.normal(0, 1, N)
    P["m"], P["c8"] = np.exp(lM), np.exp(lM + ell)
    P["rv"] = np.exp(lM + 1.0 / (1.0 + np.exp(-1.5 * u)) * ell + rng.normal(0, 0.1, N))
    tix = P["ticker"].map({"AFLT": 0, "ALRS": 1, "HYDR": 2}).to_numpy()
    pi, w = GT.fit_predict(P, tix, ["AFLT", "ALRS", "HYDR"], rng.normal(0, 1, N), np.column_stack([u, rng.normal(0, 1, N)]))
    last = pd.DataFrame(w)
    last = last[last["ticker"] == "PANEL"].iloc[-1]
    assert last["b_abs_ell"] > 0.3 and abs(last["b_zmq"]) < 0.15 and abs(last["b_med_lrv"]) < 0.15 and last["converged"]
    assert np.corrcoef(pi[-300:], 1.0 / (1.0 + np.exp(-1.5 * u[-300:])))[0, 1] > 0.8
