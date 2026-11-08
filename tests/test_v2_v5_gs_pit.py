"""V5 лист 1a, G11 (3), (4), (5), (6): PIT через боевой путь `v5_gs_fit.run_config`. Свойство: прогнозы строк с днём прогноза d < τ* не меняются при порче данных с даты цели τ* и позже. Мутации: «закрыто + 1» в оценке θ, «s дня t⁺» в обновлении u,
«W/λ по всей выборке», LOTO без исключения бумаги, E11 во входе."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s12 import v5_gs_fit as FI, v5_gs_graph as GR, v5_gs_panel as PA

TICKS = ["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG"]


def synth(n: int = 640, seed: int = 2, extra=()) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ax = pd.bdate_range("2017-01-02", periods=n)
    f = np.cumsum(rng.normal(0, 0.05, n)) * 0.3
    rows = []
    for t in list(TICKS) + list(extra):
        lr = f + rng.normal(0, 0.4, n) + 0.3 * np.cumsum(rng.normal(0, 0.03, n))
        k = [np.exp(pd.Series(lr).shift(1).bfill().to_numpy() * 0.5 + rng.normal(0, 0.25, n)) for _ in range(3)]
        dd = pd.DatetimeIndex([ax[0] - pd.offsets.BDay(1)]).append(ax[:-1])
        rows.append(pd.DataFrame({"ticker": t, "date": dd, "t_plus": ax, "group": "M16", "rv_tplus": np.exp(lr), "k1": k[0], "k2": k[1], "k3": k[2]}))
    return pd.concat(rows, ignore_index=True)


def spoil(k: pd.DataFrame, tau, tickers=None) -> pd.DataFrame:
    """Портит RV и прогнозы со строк, цель которых ≥ tau (для tickers или всех)."""
    k = k.copy()
    who = k["ticker"].isin(tickers) if tickers else True
    for c, m in (("rv_tplus", (k["t_plus"] >= tau) & who), ("k1", (k["date"] >= tau) & who), ("k2", (k["date"] >= tau) & who), ("k3", (k["date"] >= tau) & who)):     # RV известен с τ, прогноз строки — с её дня d
        k.loc[m, c] = k.loc[m, c].to_numpy() * np.random.default_rng(9).uniform(2.0, 6.0, int(m.sum()))
    return k


def forecasts(k, var="A", clip=False, excl=None, rows_before=None):
    P = PA.build(k, TICKS)
    U = FI.run_config(P, P.k8, var, clip, excl)["U"]
    F = P.k8 * np.exp(U)
    msk = P.pres & (P.dd < np.datetime64(rows_before)) if rows_before is not None else P.pres
    return F, msk, P


def tau_star(k):
    ax = pd.DatetimeIndex(sorted(k["t_plus"].unique()))
    return ax[ax >= pd.Timestamp("2018-06-01")][10]


def _same_before(var, clip, mutate=None):
    k = synth()
    ts = tau_star(k)
    a, msk, _ = forecasts(k, var, clip, rows_before=ts)
    b, msk2, _ = forecasts(spoil(k, ts), var, clip, rows_before=ts)
    assert (msk == msk2).all() and msk.sum() > 1000
    return bool(np.array_equal(a[msk], b[msk]))


def test_future_data_does_not_change_past_forecasts_all_variants():
    for var, clip in (("A", False), ("A", True), ("B", True), ("C", False), ("D", False)):
        assert _same_before(var, clip), (var, clip)


def test_mutation_closed_plus_one_in_theta_is_caught():
    orig = FI.closed_n
    FI.closed_n = lambda axis, m0: orig(axis, m0) + 25
    try:
        assert not _same_before("B", False)
    finally:
        FI.closed_n = orig


def test_mutation_score_of_day_t_plus_in_update_is_caught():
    orig = FI.CO.run

    def leak(*a, **kw):
        U, S, X, acc = orig(*a, **kw)
        U2 = U.copy()
        U2[:-1] = U[1:]                                    # u строки τ берётся из обновления по s самого дня τ
        return U2, S, X, acc
    FI.CO.run = leak
    try:
        assert not _same_before("A", False)
    finally:
        FI.CO.run = orig


def _forecasts_whole_sample_graph(k, var, rows_before):
    """Мутация: W/λ по s̃ ВСЕЙ выборки данного набора (а не по прошлому)."""
    P0 = PA.build(k, TICKS)
    full = FI.run_config(P0, P0.k8, var, False)["X"]
    orig = GR.graph
    GR.graph = lambda v, X, excl=None: orig(v, full, excl)
    try:
        return forecasts(k, var, False, rows_before=rows_before)
    finally:
        GR.graph = orig


def test_mutation_w_lambda_from_whole_sample_is_caught():
    for var in ("C", "D"):
        k = synth()
        ts = tau_star(k)
        a, msk, _ = _forecasts_whole_sample_graph(k, var, ts)
        b, _, _ = _forecasts_whole_sample_graph(spoil(k, ts), var, ts)
        assert not np.array_equal(a[msk], b[msk]), var


def test_loto_excluded_ticker_does_not_enter_theta_w_lambda_and_mutation_is_caught():
    k = synth()
    j, name = 2, TICKS[2]
    bad = spoil(k, pd.Timestamp("2017-01-01"), [name])                       # бумага j испорчена целиком
    for var, clip in (("B", False), ("C", True), ("D", False)):
        a, msk, _ = forecasts(k, var, clip, excl=j)
        b, _, _ = forecasts(bad, var, clip, excl=j)
        keep = msk.copy()
        keep[:, j] = False
        assert np.array_equal(a[keep], b[keep])                              # остальные прогнозы побитно прежние
        a0, _, _ = forecasts(k, var, clip)
        b0, _, _ = forecasts(bad, var, clip)
        assert not np.array_equal(a0[keep], b0[keep])                        # без исключения бумага влияет (мутация «LOTO не исключает» ловится)


def test_e11_in_input_changes_nothing():
    k = synth()
    a, msk, _ = forecasts(k, "D", True)
    b, msk2, _ = forecasts(synth(extra=("E11",)), "D", True)                # лишняя бумага во входе: панель строится по списку 17/M16, не по кадру
    assert (msk == msk2).all() and np.array_equal(a[msk], b[msk2])


def test_check_closed_flags_row_whose_prior_score_is_not_closed():
    P = PA.build(synth(120), TICKS)
    assert PA.check_closed(P) == []
    P.dd[5, 0] = P.dd[4, 0] - np.timedelta64(3, "D")                          # день прогноза раньше даты цели предыдущей строки
    assert PA.check_closed(P) == [TICKS[0]]
