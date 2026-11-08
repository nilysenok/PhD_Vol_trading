"""V6-1a, G13: помесячная переоценка (PIT). Переоценка в первый торговый день месяца; обучение — только пары с t⁺ ≤ d; нормировка — по тем же парам; F > 0; g ≡ 0 → F = H.
Мутации через боевой `walk_forward`: «закрыто + 1» (пара t⁺ = d + 1 в обучении) и нормировка по всей выборке — ловятся инвариантностью: замена исходов и признаков СТРОК ≥ d0 не должна менять g на d0."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_1_fit as FT, v6_1_h as HH, v6_1_wf as WF

N, FIRST = 1100, "2018-09-01"


def synth_tab(seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-07-07", periods=N)
    h = np.exp(rng.normal(0.5, 0.3, N))
    tab = pd.DataFrame({"t_plus": np.r_[idx[1:].to_numpy(), (idx[-1] + pd.offsets.BDay(1)).to_numpy()], "rv_next": h * np.exp(0.3 * rng.normal(size=N) - 0.05), "h": h}, index=idx)
    S = pd.DataFrame(rng.normal(size=(N, 3)) * [1.0, 2.0, 0.5] + [0.0, 1.0, -2.0], index=idx, columns=["s1", "s2", "s3"])
    return tab, S


def run(tab, S, first=FIRST):
    return WF.walk_forward(tab, S, FT.fit_lin, first)


def d0_of(tab, ym="2019-03"):
    return tab.index[WF.refit_positions(tab.index, FIRST)][list(tab.index[WF.refit_positions(tab.index, FIRST)].strftime("%Y-%m")).index(ym)]


def leak_value(tab, S, d0, kind: str) -> float:
    """|g(d0) после замены ... − g(d0) до|: исходы строк ≥ d0 (kind='y') или признаки строк > d0 (kind='x')."""
    base = run(tab, S)["g"].loc[d0]
    t2, S2 = tab.copy(), S.copy()
    pos = tab.index.get_loc(d0)
    rng = np.random.default_rng(5)
    if kind == "y":
        t2.iloc[pos:, t2.columns.get_loc("rv_next")] = t2["rv_next"].to_numpy()[pos:] * np.exp(rng.normal(0, 1.0, len(t2) - pos))
    else:
        S2.iloc[pos + 1:] = rng.normal(size=S2.iloc[pos + 1:].shape) * 5 + 3
    return abs(run(t2, S2)["g"].loc[d0] - base)


def test_refit_first_trading_day_of_month_and_closed_pairs_only():
    tab, S = synth_tab()
    r = run(tab, S)
    pos = WF.refit_positions(tab.index, FIRST)
    assert all(tab.index[p].month != tab.index[p - 1].month for p in pos) and tab.index[pos[0]] >= pd.Timestamp(FIRST)
    for f in r["fits"]:
        assert f["n_train"] == int((tab["t_plus"] <= f["d0"]).sum()) and f["n_train"] == tab.index.get_loc(f["d0"])        # пары d < d0: t⁺ ≤ d0
    g = r["g"]
    assert (g[g.index < pd.Timestamp(FIRST)] == 0).all()
    assert g.loc[tab.index[pos[0]]] != 0
    same = g[(g.index >= tab.index[pos[0]]) & (g.index < tab.index[pos[1]])].to_numpy()
    X = S.to_numpy()[pos[0]:pos[1]]
    assert np.array_equal(same, r["fits"][0]["model"].g(X))                                     # весь месяц — по параметрам первого дня


def test_no_leak_from_outcomes_or_future_features():
    tab, S = synth_tab()
    d0 = d0_of(tab)
    assert leak_value(tab, S, d0, "y") < 1e-12 and leak_value(tab, S, d0, "x") < 1e-12              # допуск на шум округления; мутации дают > 1e-9


def test_mutation_closed_plus_one_caught(monkeypatch):
    tab, S = synth_tab()
    d0 = d0_of(tab)
    monkeypatch.setattr(WF, "closed_rows", lambda tp, d: np.flatnonzero(np.asarray(tp, "datetime64[ns]") <= np.datetime64(d) + np.timedelta64(1, "D") * 3))
    assert leak_value(tab, S, d0, "y") > 1e-9


def test_mutation_normalization_on_whole_sample_caught(monkeypatch):
    tab, S = synth_tab()
    d0 = d0_of(tab)
    cache = {}

    real = FT.norm_stats

    def patched(X):
        full = cache.get("X")
        return real(full) if full is not None else real(X)
    monkeypatch.setattr(FT, "norm_stats", patched)
    orig_wf = WF.walk_forward

    def wf(tab_, S_, fit, first=FIRST):
        cache["X"] = S_.to_numpy(float)                                     # нормировка по ВСЕЙ выборке (включая будущее)
        return orig_wf(tab_, S_, fit, first)
    monkeypatch.setattr(WF, "walk_forward", wf)
    base = WF.walk_forward(tab, S, FT.fit_lin, FIRST)["g"].loc[d0]
    S2 = S.copy()
    S2.iloc[tab.index.get_loc(d0) + 1:] = np.random.default_rng(5).normal(size=S2.iloc[tab.index.get_loc(d0) + 1:].shape) * 5 + 3
    assert abs(WF.walk_forward(tab, S2, FT.fit_lin, FIRST)["g"].loc[d0] - base) > 1e-9


def test_forecast_positive_and_zero_g_gives_h_bitwise():
    tab, S = synth_tab()
    r = run(tab, S)
    F = WF.forecasts(tab, r["g"])
    assert (F > 0).all() and np.isfinite(F).all()
    z = WF.walk_forward(tab, S, lambda X, e: type("Z", (), {"g": staticmethod(lambda X: np.zeros(len(X)))})(), FIRST)
    assert np.array_equal(WF.forecasts(tab, z["g"]).to_numpy(), tab["h"].to_numpy())


def test_real_first_refit_has_378_closed_pairs():
    p, k1 = HH.H.make_panel(HH.daily_sber()), HH.k1_stored()
    tab = HH.h_table(p, k1)
    pos = WF.refit_positions(tab.index)
    assert tab.index[pos[0]] == pd.Timestamp("2017-01-03") and len(WF.closed_rows(tab["t_plus"].to_numpy(), tab.index[pos[0]])) == 378 == pos[0]
