"""V4-CLD (G32): боевой путь `v4_cld_run.forecast` на синтетике с известным устройством — знак веса, приор, обрезка (PIT), мутации «закрыто + 1», «D цели в признаках», LOTO, тождество D̂ ≡ 0 → w = 0,5."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_b_run as BR, v4_cld_run as R, v4_z_feat as ZF
from tests.test_v2_v4_z_feat import synth


def regime(P, ml_better: bool, seed=5):
    """Известное устройство: rv ≈ M (ML лучше, D > 0) или rv ≈ H (HAR-J лучше, D < 0); M расходится с H."""
    rng = np.random.default_rng(seed)
    Q = P.copy()
    ell = rng.normal(0, 0.4, len(Q))
    m = Q["h"].to_numpy() * np.exp(ell)
    Q["k2"], Q["k3"] = m, m
    Q["rv"] = (m if ml_better else Q["h"].to_numpy()) * np.exp(rng.normal(0, 0.05, len(Q)))
    return Q


@pytest.fixture(scope="module")
def data():
    P, inp = synth(560)
    return P, inp, ZF.build_panel_z(P, inp)


def day_sep(P):
    d = P["date"]
    return d[d.dt.to_period("M") == pd.Period("2017-09")].min()


def ql(rv, f):
    r = rv / f
    return r - np.log(r) - 1


@pytest.mark.parametrize("cfg", R.CFGS)
def test_known_answer_weight_follows_sign_of_D(data, cfg):
    P, inp, _ = data
    for better, sign in ((True, 1), (False, -1)):
        Q = regime(P, better)
        out, _, _ = R.forecast(cfg, Q, ZF.build_panel_z(Q, inp))
        late = out[out["date"] >= "2018-01-01"]
        assert sign * (late["w"].mean() - 0.5) > 0.02, (cfg, better)
        q_f, q_h = (ql(late["rv"], late[c]).mean() for c in ("F", "h"))
        assert (q_f < q_h) if better else True


@pytest.mark.parametrize("cfg", R.CFGS)
def test_prior_phase_half_weight_and_positive_forecast(data, cfg):
    P, _, Z = data
    out, w, c = R.forecast(cfg, P, Z)
    assert len(out) == len(P) and (out["F"] > 0).all() and np.isfinite(out["F"]).all() and ((out["w"] >= 0) & (out["w"] <= 1)).all()
    assert (out[out["date"] < "2017-02-01"]["w"] == 0.5).all()
    assert {"config", "month", "ticker", "mean_w"} <= set(w.columns) and "PANEL" in set(w["ticker"]) and set(c["feature"]) == set(R.FEATS)


@pytest.mark.parametrize("cfg", R.CFGS)
def test_identity_dhat_zero_gives_half_weight(data, cfg):
    P, _, Z = data
    out, _, _ = R.forecast(cfg, P, Z, force_dhat=0.0)
    assert (out["w"] == 0.5).all()
    assert np.allclose(out["F"], np.sqrt(out["h"] * np.sqrt(P.sort_values(["date", "ticker"])["k2"].to_numpy() * P.sort_values(["date", "ticker"])["k3"].to_numpy())))


@pytest.mark.parametrize("cfg", R.CFGS)
def test_cut_future_data_does_not_change_forecast(data, cfg):
    P, inp, _ = data
    assert not R.pit_violation(cfg, P, inp, day_sep(P))


@pytest.mark.parametrize("cfg", R.CFGS)
def test_mutation_closed_plus_one_in_D_regression_is_caught_via_real_path(data, monkeypatch, cfg):
    P, inp, _ = data
    monkeypatch.setattr(BR, "closed_mask", lambda t_plus, d0: np.asarray(t_plus) <= np.datetime64(d0) + np.timedelta64(3, "D"))
    assert R.pit_violation(cfg, P, inp, day_sep(P))


@pytest.mark.parametrize("cfg", R.CFGS)
def test_mutation_D_of_future_day_in_features_is_caught(data, cfg):
    P, inp, _ = data
    assert R.pit_violation(cfg, P, inp, day_sep(P), leak=True)


def test_mutation_features_of_day_d_plus_1_is_caught(data):
    P, inp, _ = data
    assert R.pit_violation("CLD1", P, inp, day_sep(P), lead=1)


def test_features_contain_no_label_of_target_day(data):
    P, _, Z = data
    Pp, X, names, *_ = R.prepare(P, Z)
    assert names == R.FEATS and not any(n.startswith("d_") or n == "D" for n in names)
    assert R.prepare(P, Z, leak=True)[2][-1] == "d_future"


def test_loto_excluded_ticker_is_not_in_training_and_not_in_cross_section(data):
    """Подмена RV (и дневных величин) исключённой бумаги не меняет прогнозы остальных при поперечных z по 16; без исключения/пересчёта поперечных — меняет."""
    P, inp, _ = data
    t, others = "IRAO", ["AFLT", "ALRS", "HYDR"]
    Q = P.copy()
    Q.loc[Q["ticker"] == t, "rv"] *= 5.0
    alt = {k: ({"daily": v["daily"].assign(rv=v["daily"]["rv"] * 5, j=v["daily"]["j"] * 5, rv_close=v["daily"]["rv_close"] * 5), "herf": v["herf"]} if k == t else v) for k, v in inp.items()}
    Za, Zb = ZF.build_panel_z(P, inp), ZF.build_panel_z(Q, alt)
    pick = lambda o: o[o["ticker"].isin(others)].sort_values(["date", "ticker"])[["F", "w"]].to_numpy()
    a = pick(R.forecast("CLD1", P, ZF.recross(Za, others), exclude=t)[0])
    b = pick(R.forecast("CLD1", Q, ZF.recross(Zb, others), exclude=t)[0])
    assert np.array_equal(a, b)
    assert not np.array_equal(pick(R.forecast("CLD1", P, Za)[0]), pick(R.forecast("CLD1", Q, Zb)[0]))          # мутация: бумага внутри оценки и медиан
    assert not np.array_equal(a, pick(R.forecast("CLD1", Q, Zb, exclude=t)[0]))                            # мутация: поперечные z по 17 вместо 16

