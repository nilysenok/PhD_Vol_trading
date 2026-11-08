"""V4-I (G42): PIT боевого пути I1–I3 (признаки из дневных рядов + оценка компонент + прогноз): обрезка, мутации «закрыто + 1» (в HAR-C и в моделях скачка раздельно и в целом), «J_{d+1} в признаках p̂»,
тождество θ ≡ 0 -> K8, до 60 пар S = K8, исключённая бумага (LOTO), известный ответ. Данные проекта не читаются."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_b_run as BR, v4_i_comp as IC, v4_i_run as R, v4_i_stage as ST, v4_rqc_calib as CAL

TK = ["AAA", "BBB", "CCC", "DDD"]
DAY = "2017-09-01"


def synth(n=330, seed=0):
    """Панель + дневные ряды. RV следующего дня = цель; J — часть RV (bv случайный)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-02", periods=n + 1)
    dl, rows = {}, []
    for t in TK:
        rv = np.exp(rng.normal(1.0, 0.4, n + 1))
        bv = rv * rng.uniform(0.6, 1.3, n + 1)
        dl[t] = pd.DataFrame({"rv": rv, "j": np.maximum(rv - bv, 0.0), "seg": 0}, index=idx)
        h, k2, k3 = (np.exp(rng.normal(1.0, 0.2, n)) for _ in range(3))
        rows.append(pd.DataFrame({"ticker": t, "date": idx[:-1], "t_plus": idx[1:], "rv": rv[1:], "h": h, "k2": k2, "k3": k3}))
    return pd.concat(rows, ignore_index=True), dl


@pytest.fixture(scope="module")
def data():
    return synth()


def first_day(P, m):
    return P["date"][P["date"].dt.to_period("M") == pd.Period(m)].min()


@pytest.mark.parametrize("cfg", ["I1", "I2", "I3"])
def test_cut_future_data_does_not_change_forecast(data, cfg):
    P, dl = data
    assert not R.pit_violation(cfg, P, dl, DAY)


@pytest.mark.parametrize("cfg", ["I1", "I3"])
def test_mutation_closed_plus_one_is_caught_via_real_path(data, monkeypatch, cfg):
    P, dl = data
    monkeypatch.setattr(BR, "closed_mask", lambda t_plus, d0: np.asarray(t_plus) <= np.datetime64(d0) + np.timedelta64(3, "D"))
    assert R.pit_violation(cfg, P, dl, DAY)


def _future_perturbed(P, Fe, d0, col):
    """Подмена целевой колонки (C или событие) только у пар с t⁺ >= d0 — они не закрыты для прогноза месяца d0."""
    G = Fe.merge(P[["ticker", "date", "t_plus"]], on=["ticker", "date"])
    m = (G["t_plus"] >= d0).to_numpy()
    G.loc[m, col] = G.loc[m, col] * 7.3 if col == "c_t" else 1.0 - G.loc[m, col]
    return G.drop(columns="t_plus")


def _comp_at(P, Fe, day):
    c = R.components(P, Fe)[0]
    return c[c["date"] == pd.Timestamp(day)].sort_values("ticker")


def test_closed_pairs_only_in_har_c_and_in_jump_model_separately(data, monkeypatch):
    """Честно: подмена будущих целей C (события) не меняет Ĉ (p̂) месяца; мутация «закрыто + 1» меняет именно Ĉ (p̂) — каждая из двух моделей ловится отдельно."""
    P, dl = data
    Fe = IC.build(P, dl)
    d0 = first_day(P, "2017-09")
    base = _comp_at(P, Fe, d0)
    hc, he = _comp_at(P, _future_perturbed(P, Fe, d0, "c_t"), d0), _comp_at(P, _future_perturbed(P, Fe, d0, "ev_t"), d0)
    assert np.array_equal(base["chat"].to_numpy(), hc["chat"].to_numpy()) and np.array_equal(base["phat"].to_numpy(), he["phat"].to_numpy())
    monkeypatch.setattr(BR, "closed_mask", lambda t_plus, d: np.asarray(t_plus) <= np.datetime64(d) + np.timedelta64(3, "D"))
    assert not np.array_equal(base["chat"].to_numpy(), _comp_at(P, _future_perturbed(P, Fe, d0, "c_t"), d0)["chat"].to_numpy())
    assert not np.array_equal(base["phat"].to_numpy(), _comp_at(P, _future_perturbed(P, Fe, d0, "ev_t"), d0)["phat"].to_numpy())


def test_mutation_j_of_next_day_in_p_features_is_caught(data):
    P, dl = data
    assert not R.pit_violation("I1", P, dl, DAY, lead=0)
    assert R.pit_violation("I1", P, dl, DAY, lead=1)


def test_identity_theta_zero_gives_k8_bitwise_and_before_60_pairs_k8(data):
    P, dl = data
    Fe = IC.build(P, dl)
    out, cmp, co = R.forecast("I3", P, Fe, zero_theta=True)
    assert np.array_equal(out["F"].to_numpy(), cmp["K8"].to_numpy())
    assert np.array_equal(cmp["K8"].to_numpy(), CAL.k8(cmp["h"], P.sort_values(["date", "ticker"])["k2"], P.sort_values(["date", "ticker"])["k3"]))
    first = cmp["date"] < pd.Timestamp("2017-03-01")                              # < 60 закрытых пар у панели из 4 бумаг — компонент нет
    assert (~np.isfinite(cmp.loc[first, "chat"])).all()
    for cfg in ("I1", "I2"):
        f = R.combine(cfg, cmp, co[["month", "theta"]])
        assert np.array_equal(f.loc[first, "F"].to_numpy(), cmp.loc[first, "K8"].to_numpy())


def test_i2_is_geometric_mean_and_i1_is_sum(data):
    P, dl = data
    Fe = IC.build(P, dl)
    out, cmp, co = R.forecast("I1", P, Fe)
    ok = np.isfinite(cmp["chat"] + cmp["jhat"]).to_numpy()
    assert ok.any() and np.allclose(out["F"].to_numpy()[ok], (cmp["chat"] + cmp["jhat"]).to_numpy()[ok], rtol=0, atol=0)
    f2 = R.combine("I2", cmp, co[["month", "theta"]])["F"].to_numpy()
    assert np.allclose(f2, np.sqrt(cmp["K8"].to_numpy() * cmp["S"].to_numpy()))


def test_loto_excluded_ticker_does_not_enter_estimation_and_mutation_is_caught(data):
    P, dl = data
    Fe = IC.build(P, dl)
    out = R.forecast("I1", P, Fe, exclude="BBB")[0]
    assert ST.zero_ok("I1", P, Fe, "BBB", out)
    full = R.forecast("I1", P, Fe)[0]                                             # мутация: исключённая бумага осталась в панели
    a = full[full["ticker"] != "BBB"].sort_values(["date", "ticker"])["F"].to_numpy()
    b = out[out["ticker"] != "BBB"].sort_values(["date", "ticker"])["F"].to_numpy()
    assert not np.array_equal(a, b)


def test_known_answer_components_recovered_on_synthetic_hajj():
    """Известный ответ: C — AR(1) в логах, события скачка — с частотой 0,5: Ĉ ≈ C по порядку величины, среднее p̂ ≈ 0,5."""
    rng = np.random.default_rng(5)
    n, idx = 700, pd.bdate_range("2017-01-02", periods=701)
    dl, rows = {}, []
    for t in TK:
        lc = np.zeros(n + 1)
        for i in range(1, n + 1):
            lc[i] = 0.8 * lc[i - 1] + rng.normal(0, 0.2)
        c = np.exp(lc + 1.0)
        j = np.where(rng.uniform(size=n + 1) < 0.5, np.exp(rng.normal(-0.5, 0.3, n + 1)), 0.0)
        dl[t] = pd.DataFrame({"rv": c + j, "j": j, "seg": 0}, index=idx)
        rows.append(pd.DataFrame({"ticker": t, "date": idx[:-1], "t_plus": idx[1:], "rv": (c + j)[1:], "h": np.ones(n), "k2": np.ones(n), "k3": np.ones(n)}))
    P = pd.concat(rows, ignore_index=True)
    cmp = R.components(P, IC.build(P, dl))[0]
    ok = np.isfinite(cmp["chat"] + cmp["phat"])
    late = ok & (cmp["date"] > pd.Timestamp("2018-06-01"))
    assert abs(cmp.loc[late, "phat"].mean() - 0.5) < 0.03
    tgt = pd.concat([dl[t]["rv"].sub(dl[t]["j"]).reindex(cmp.loc[late & (cmp["ticker"] == t), "t_plus"]) for t in TK])
    assert abs(np.log(cmp.loc[late, "chat"].mean() / tgt.mean())) < 0.1
    assert abs(cmp.loc[late, "mhat"].mean() - np.exp(-0.5 + 0.045)) < 0.1
