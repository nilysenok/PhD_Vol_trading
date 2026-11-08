"""V5 лист 2, G19: PIT, тождества и LOTO через боевой путь (`v5_sp_fit.run_config`, `v5_sp_har.run`) на синтетической панели.
Свойство: прогнозы строк с днём прогноза d < τ* не меняются при порче данных (полосы, формы дня, RV, прогнозы K1–K3) с даты τ* и позже. Мутации: «признаки дня d+1», «бары дня t⁺ в z», «закрыто + 1» в оценке, LOTO без исключения, межбумажный фактор без исключения.
Тождества: при θ = 0 SP2–SP4 = K8 побитно, при R = 1 SP1 = K8 побитно; E11 во входе не меняет прогнозов; прогнозы > 0 и конечны."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s12 import v5_gs_panel as PA, v5_sp_feat as FE, v5_sp_fit as FI, v5_sp_har as HAR

TICKS = ["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG"]
CFGS = ["SP1", "SP2", "SP3", "SP4"]


def synth(n: int = 760, seed: int = 3, extra=()):
    rng = np.random.default_rng(seed)
    ax = pd.bdate_range("2017-01-02", periods=n)
    days = pd.DatetimeIndex([ax[0] - pd.offsets.BDay(1)]).append(ax)
    f = np.cumsum(rng.normal(0, 0.05, n + 1)) * 0.3
    kr, br, fr = [], [], []
    for t in list(TICKS) + list(extra):
        lr = f + rng.normal(0, 0.4, n + 1)
        a = np.cumsum(rng.normal(0, 0.1, (n + 1, 3)), axis=0) * 0.2 + rng.normal(0, 0.3, (n + 1, 3))
        e = np.exp(lr)[:, None] * np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)
        rv = e.sum(axis=1)
        br.append(pd.DataFrame({"ticker": t, "date": days, "e_low": e[:, 0], "e_mid": e[:, 1], "e_high": e[:, 2], "rv": rv, "rv_close": rv * rng.uniform(0.1, 0.3, n + 1), "herf": rng.uniform(0.02, 0.1, n + 1)}))
        fr.append(pd.DataFrame({"ticker": t, "date": days, "firsth": rng.uniform(0.1, 0.3, n + 1)}))
        k = [np.exp(lr[:-1] * 0.5 + rng.normal(0, 0.25, n)) for _ in range(3)]
        kr.append(pd.DataFrame({"ticker": t, "date": days[:-1], "t_plus": ax, "group": "M16", "rv_tplus": rv[1:], "k1": k[0], "k2": k[1], "k3": k[2]}))
    return pd.concat(kr, ignore_index=True), pd.concat(br, ignore_index=True), pd.concat(fr, ignore_index=True)


def spoil(k, b, fh, tau, tickers=None):
    """Портит данные с даты tau (для tickers или всех): RV известен с τ, прогноз строки — с её дня d, признаки дня — с его даты."""
    k, b, fh = k.copy(), b.copy(), fh.copy()
    who = lambda df: df["ticker"].isin(tickers) if tickers else True
    u = lambda m: np.random.default_rng(9).uniform(2.0, 6.0, int(m.sum()))
    for c, m in (("rv_tplus", (k["t_plus"] >= tau) & who(k)), ("k1", (k["date"] >= tau) & who(k)), ("k2", (k["date"] >= tau) & who(k)), ("k3", (k["date"] >= tau) & who(k))):
        k.loc[m, c] = k.loc[m, c].to_numpy() * u(m)
    m = (b["date"] >= tau) & who(b)
    for c in ("e_low", "e_mid", "e_high", "rv_close", "herf"):
        b.loc[m, c] = b.loc[m, c].to_numpy() * u(m) * (np.random.default_rng(len(c)).uniform(0.5, 2.0, int(m.sum())) if c != "herf" else 1.0)
    b.loc[m, "rv"] = b.loc[m, ["e_low", "e_mid", "e_high"]].sum(axis=1)
    m = (fh["date"] >= tau) & who(fh)
    fh.loc[m, "firsth"] = fh.loc[m, "firsth"].to_numpy() * u(m)
    return k, b, fh


def forecasts(data, cfg, excl=None, before=None):
    k, b, fh = data
    P, F = PA.build(k, TICKS), FE.build(b, fh, TICKS)
    res = HAR.run(P, F) if cfg == "SP1" else FI.run_config(P, F, cfg, excl)
    msk = P.pres & (P.dd < np.datetime64(before)) if before is not None else P.pres
    return res["F"], msk, P


def tau_star(k):
    ax = pd.DatetimeIndex(sorted(k["t_plus"].unique()))
    return ax[ax >= pd.Timestamp("2018-08-01")][10]


def same_before(cfg):
    d = synth()
    ts = tau_star(d[0])
    a, msk, P = forecasts(d, cfg, before=ts)
    b, msk2, _ = forecasts(spoil(*d, ts), cfg, before=ts)
    assert (msk == msk2).all() and msk.sum() > 1500 and not np.array_equal(a[msk], P.k8[msk])      # прогноз не тождествен K8: проверка содержательна
    return bool(np.array_equal(a[msk], b[msk]))


def test_future_data_does_not_change_past_forecasts_all_configs():
    for cfg in CFGS:
        assert same_before(cfg), cfg


def _patched(mod, name, fn, cfgs):
    orig = getattr(mod, name)
    setattr(mod, name, fn)
    try:
        return [same_before(c) for c in cfgs]
    finally:
        setattr(mod, name, orig)


def test_mutation_closed_plus_one_in_estimation_is_caught():
    for mod, cfgs in ((FI, ["SP2", "SP3", "SP4"]), (HAR, ["SP1"])):
        orig = mod.closed_n
        assert not any(_patched(mod, "closed_n", lambda axis, m0, o=orig: o(axis, m0) + 25, cfgs))


def test_mutation_features_of_day_d_plus_one_and_target_day_bars_are_caught():
    orig = FE.panel_index

    def plus_one(P, F):
        di, ok = orig(P, F)
        return np.minimum(di + 1, len(F.D) - 1), ok

    def target_day(P, F):                                                 # бары дня цели t⁺ вместо дня прогноза d
        ti = np.minimum(np.searchsorted(F.D.to_numpy(), P.axis.to_numpy()), len(F.D) - 1)
        return np.broadcast_to(ti[:, None], P.pres.shape).copy(), P.pres
    for fn in (plus_one, target_day):
        assert not any(_patched(FE, "panel_index", fn, CFGS)), fn.__name__


def test_theta_zero_gives_k8_bitwise_and_r_one_gives_k8_bitwise(monkeypatch):
    d = synth(500)
    monkeypatch.setattr(FI, "MIN_DAYS", 10 ** 9)
    monkeypatch.setattr(HAR, "MIN_PAIRS", 10 ** 9)
    for cfg in CFGS:
        F, msk, P = forecasts(d, cfg)
        assert np.array_equal(F[msk], P.k8[msk]), cfg


def test_forecasts_positive_and_finite():
    for cfg in CFGS:
        F, msk, _ = forecasts(synth(), cfg)
        assert np.isfinite(F[msk]).all() and (F[msk] > 0).all(), cfg


def test_e11_in_input_changes_nothing():
    for cfg in ("SP1", "SP3", "SP4"):
        a, msk, _ = forecasts(synth(500), cfg)
        b, msk2, _ = forecasts(synth(500, extra=("E11",)), cfg)             # лишняя бумага во входе: панель и признаки строятся по списку, не по кадру
        assert (msk == msk2).all() and np.array_equal(a[msk], b[msk2]), cfg


def test_loto_excluded_ticker_does_not_enter_theta_and_factor_and_mutation_is_caught():
    d = synth(500)
    j = 2
    bad = spoil(*d, pd.Timestamp("2017-01-01"), [TICKS[j]])                 # бумага j испорчена целиком
    for cfg in ("SP2", "SP3", "SP4"):
        a, msk, _ = forecasts(d, cfg, excl=j)
        b, _, _ = forecasts(bad, cfg, excl=j)
        keep = msk.copy()
        keep[:, j] = False
        assert np.array_equal(a[keep], b[keep]), cfg
        a0, _, _ = forecasts(d, cfg)
        b0, _, _ = forecasts(bad, cfg)
        assert not np.array_equal(a0[keep], b0[keep]), cfg                  # без исключения бумага влияет


def test_mutation_cross_factor_ignores_exclusion_is_caught():
    d = synth(500)
    j = 2
    bad = spoil(*d, pd.Timestamp("2017-01-01"), [TICKS[j]])
    orig = FE.cross_factor
    FE.cross_factor = lambda F, keep: orig(F, np.ones(len(keep), bool))
    try:
        a, msk, _ = forecasts(d, "SP3", excl=j)
        b, _, _ = forecasts(bad, "SP3", excl=j)
    finally:
        FE.cross_factor = orig
    keep = msk.copy()
    keep[:, j] = False
    assert not np.array_equal(a[keep], b[keep])


def test_sp1_models_are_per_ticker_loto_equals_main():
    d = synth(500)
    a, msk, _ = forecasts(d, "SP1")
    bad = spoil(*d, pd.Timestamp("2017-01-01"), [TICKS[2]])
    b, _, _ = forecasts(bad, "SP1")
    keep = msk.copy()
    keep[:, 2] = False
    assert np.array_equal(a[keep], b[keep])
