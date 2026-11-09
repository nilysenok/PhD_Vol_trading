"""V6-2, G24/G25: переоценка с индексом u (PIT). Нормировка u и порог верхнего квинтиля — по закрытым парам (t⁺ ≤ d0) на дату переоценки; взаимодействия u·s1…s3 строятся с этим u; без взаимодействий результат = движок листа 1a побитно.
Мутации через боевой `walk_forward_u`: нормировка u по всей выборке (будущие Q меняют g и порог на d0); «закрыто + 1» (исходы строк ≥ d0 меняют g на d0)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_1_fit as FT, v6_1_wf as WF, v6_2_q as QF, v6_2_wf as W2

N, FIRST = 1100, "2018-09-01"


def synth(seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-07-07", periods=N)
    h = np.exp(rng.normal(0.5, 0.3, N))
    tab = pd.DataFrame({"t_plus": np.r_[idx[1:].to_numpy(), (idx[-1] + pd.offsets.BDay(1)).to_numpy()], "rv_next": h * np.exp(0.3 * rng.normal(size=N) - 0.05), "h": h}, index=idx)
    S = pd.DataFrame(rng.normal(size=(N, 3)) * [1.0, 2.0, 0.5] + [0.0, 1.0, -2.0], index=idx, columns=["s1", "s2", "s3"])
    Qd = pd.DataFrame(rng.normal(size=(N, 8)), index=idx, columns=QF.Q_NAMES)
    return tab, S, Qd


def run(tab, S, Qd, inter=True, fit=FT.fit_lin):
    return W2.walk_forward_u(tab, S, Qd, fit, inter, FIRST)


def d0_of(tab, ym="2019-03"):
    ds = tab.index[WF.refit_positions(tab.index, FIRST)]
    return ds[list(ds.strftime("%Y-%m")).index(ym)]


def test_no_interactions_equals_engine_and_columns():
    tab, S, Qd = synth()
    r = run(tab, S, Qd, inter=False)
    base = WF.walk_forward(tab, pd.concat([S, Qd], axis=1), FT.fit_lin, FIRST)
    assert np.abs(r["g"].to_numpy() - base["g"].to_numpy()).max() < 1e-10 and W2.names(S, False) == list(S.columns) + QF.Q_NAMES
    assert W2.names(S, True)[-3:] == ["u_s1", "u_s2", "u_s3"] and r["fits"][0]["model"].mu.shape[0] == 11 and run(tab, S, Qd)["fits"][0]["model"].mu.shape[0] == 14


def test_interaction_columns_are_u_times_s_with_refit_date_norm():
    tab, S, Qd = synth()
    d0 = d0_of(tab)
    tr = WF.closed_rows(tab["t_plus"].to_numpy(), d0)
    X, u, thr = W2.design(S.to_numpy(), Qd.to_numpy(), tr, True)
    assert np.allclose(X[:, -3:], u[:, None] * S.to_numpy()) and abs(u[tr].mean()) < 1e-12 and np.isclose(thr, np.quantile(u[tr], 0.8))
    r = run(tab, S, Qd)
    pos = tab.index.get_loc(d0)
    assert np.isclose(r["thr"].iloc[pos], thr) and np.allclose(r["u"].iloc[pos:pos + 5], u[pos:pos + 5])
    assert (r["thr"].iloc[:WF.refit_positions(tab.index, FIRST)[0]].isna()).all() and not r["high"].iloc[:5].any()


def leak_value(tab, S, Qd, d0, kind: str) -> float:
    """|g(d0) после замены − g(d0) до|: исходы строк ≥ d0 ('y'), признаки S строк > d0 ('x') или Q строк > d0 ('q')."""
    base = run(tab, S, Qd)
    t2, S2, Q2 = tab.copy(), S.copy(), Qd.copy()
    pos = tab.index.get_loc(d0)
    rng = np.random.default_rng(5)
    if kind == "y":
        t2.iloc[pos:, t2.columns.get_loc("rv_next")] = t2["rv_next"].to_numpy()[pos:] * np.exp(rng.normal(0, 1.0, len(t2) - pos))
    elif kind == "x":
        S2.iloc[pos + 1:] = rng.normal(size=S2.iloc[pos + 1:].shape) * 5 + 3
    else:
        Q2.iloc[pos + 1:] = rng.normal(size=Q2.iloc[pos + 1:].shape) * 9 + 4
    new = run(t2, S2, Q2)
    return max(abs(new["g"].loc[d0] - base["g"].loc[d0]), abs(new["thr"].loc[d0] - base["thr"].loc[d0]), abs(new["u"].loc[d0] - base["u"].loc[d0]))


def test_no_leak_from_outcomes_or_future_features_or_future_q():
    tab, S, Qd = synth()
    d0 = d0_of(tab)
    for kind in ("y", "x", "q"):
        assert leak_value(tab, S, Qd, d0, kind) < 1e-12, kind


def test_mutations_caught_through_battle_path(monkeypatch):
    tab, S, Qd = synth()
    d0 = d0_of(tab)
    real_norm, real_closed = QF.u_norm, WF.closed_rows
    monkeypatch.setattr(QF, "u_norm", lambda C, tr: FT.norm_stats(C))                                       # МУТАЦИЯ: нормировка u по всей выборке
    assert leak_value(tab, S, Qd, d0, "q") > 1e-6
    monkeypatch.setattr(QF, "u_norm", real_norm)
    assert leak_value(tab, S, Qd, d0, "q") < 1e-12
    monkeypatch.setattr(WF, "closed_rows", lambda tp, d: np.flatnonzero(np.asarray(tp, "datetime64[ns]") <= np.datetime64(pd.Timestamp(d) + pd.offsets.BDay(1))))   # МУТАЦИЯ: «закрыто + 1»
    assert leak_value(tab, S, Qd, d0, "y") > 1e-9
    monkeypatch.setattr(WF, "closed_rows", real_closed)
    assert leak_value(tab, S, Qd, d0, "y") < 1e-12


def test_threshold_from_closed_pairs_mutation_caught(monkeypatch):
    tab, S, Qd = synth()
    d0 = d0_of(tab)
    monkeypatch.setattr(QF, "u_threshold", lambda u, tr, level=0.8: float(np.quantile(u, level)))            # МУТАЦИЯ: порог по всем строкам
    assert leak_value(tab, S, Qd, d0, "q") > 1e-6


def test_fit_none_gives_groups_only_and_g_zero_gives_f_equal_h():
    tab, S, Qd = synth()
    a, b = W2.walk_forward_u(tab, S, Qd, None, True, FIRST), run(tab, S, Qd)
    assert np.array_equal(a["u"].to_numpy(), b["u"].to_numpy(), equal_nan=True) and np.array_equal(a["high"].to_numpy(), b["high"].to_numpy()) and (a["g"] == 0).all() and a["fits"] == []
    assert (WF.forecasts(tab, a["g"]).to_numpy() == tab["h"].to_numpy()).all()
    share = b["high"][b["thr"].notna()].mean()
    assert 0.1 < share < 0.3                                                                                 # верхний квинтиль закрытых пар даёт ~20 % дней вне обучения
