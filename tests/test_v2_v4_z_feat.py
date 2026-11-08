"""V4-C, G15: признаки Z — известный ответ, PIT, обрезка, поперечные только по 17 бумагам, Intraday Shape из баров дня d, проверка сдвига Д1. Мутации идут через боевые функции (`own`, `cross`, `herf_by_day`, `profile`)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_z_build as ZB, v4_z_feat as ZF, v4_z_shape as SH


def synth(n=300, tickers=("AFLT", "ALRS", "HYDR", "IRAO"), seed=7):
    """P (ticker, date, t_plus, rv, h, k2, k3) и inputs {ticker: {daily, herf}}; дневной rv в t⁺ равен rv пары."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2017-01-02", periods=n + 1)
    rows, inp = [], {}
    for t in tickers:
        lr = np.zeros(n + 1)
        for i in range(1, n + 1):
            lr[i] = 0.9 * lr[i - 1] + rng.normal(0, 0.3)
        rv = np.exp(lr)
        h = np.exp(0.9 * lr[:-1] + rng.normal(0, 0.1, n))
        k2 = h * np.exp(rng.normal(0, 0.15, n))
        k3 = k2 * np.exp(rng.normal(0, 0.05, n))
        rows.append(pd.DataFrame({"ticker": t, "date": days[:-1], "t_plus": days[1:], "rv": rv[1:], "h": h, "k2": k2, "k3": k3}))
        u = rng.uniform(0.05, 0.4, n + 1)
        inp[t] = {"daily": pd.DataFrame({"rv": rv, "j": rv * u, "rv_close": rv * u / 2}, index=days), "herf": pd.Series(rng.uniform(0.05, 0.3, n + 1), index=days)}
    return pd.concat(rows, ignore_index=True), inp


def test_known_answer_own_features():
    """Три дня, ответ считается руками: ell = ln(M/H), lerr_h = ln(RV_d/H_{d−1}), dlrv, jshare, lasth, herf."""
    days = pd.to_datetime(["2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"])
    a = pd.DataFrame({"date": days[:3], "t_plus": days[1:], "rv": [2.0, 4.0, 8.0], "h": [1.0, 2.0, 5.0], "k2": [4.0, 2.0, 5.0], "k3": [1.0, 2.0, 5.0]})
    daily = pd.DataFrame({"rv": [1.0, 2.0, 4.0, 8.0], "j": [0.1, 0.2, 0.4, 0.8], "rv_close": [0.5, 0.5, 2.0, 2.0]}, index=days)
    herf = pd.Series([0.1, 0.2, 0.3, 0.4], index=days)
    z = ZF.own(a, daily, herf)
    assert z["ell"].iloc[0] == pytest.approx(np.log(2.0 / 1.0))                 # M = sqrt(4·1) = 2
    assert np.isnan(z["lerr_h"].iloc[0]) and z["lerr_h"].iloc[1] == pytest.approx(np.log(2.0 / 1.0))      # RV_d = 2 (пара с t⁺ = d), H_{d−1} = 1
    assert z["lerr_m"].iloc[2] == pytest.approx(np.log(4.0 / 2.0))              # M_{d−1} = sqrt(2·2) = 2
    assert z["dlrv"].iloc[1] == pytest.approx(np.log(2.0) - np.log(1.0)) and z["jshare"].iloc[2] == pytest.approx(0.1)
    assert z["lasth"].iloc[2] == pytest.approx(0.5) and z["herf"].iloc[1] == pytest.approx(0.2) and z["lrv"].iloc[3 - 1] == pytest.approx(np.log(4.0))


def test_ewqd_uses_only_closed_pairs():
    """EWMA строки d не видит пару с t⁺ = d+1 (цель строки d): подмена RV последней пары не меняет признаки строки d."""
    P, inp = synth(60)
    a = P[P["ticker"] == "AFLT"].reset_index(drop=True)
    b = a.copy()
    b.loc[b.index[-1], "rv"] *= 50
    assert np.array_equal(ZF.own(a, inp["AFLT"]["daily"], inp["AFLT"]["herf"]).iloc[-1].to_numpy(float), ZF.own(b, inp["AFLT"]["daily"], inp["AFLT"]["herf"]).iloc[-1].to_numpy(float), equal_nan=True)


def test_truncation_real_function():
    P, inp = synth()
    a = P[P["ticker"] == "ALRS"].reset_index(drop=True)
    assert all(ZB.truncation_ok(a, inp["ALRS"]["daily"], inp["ALRS"]["herf"], d) for d in ("2017-06-15", "2017-10-02"))


def test_mutation_lead_one_caught_by_truncation_and_shift_profile():
    """Мутация «признак дня d+1 вместо d» (`own(lead=1)`): профиль сдвига показывает утечку; honest — нет."""
    P, inp = synth(600, ("AFLT",))
    a, rv = P.reset_index(drop=True), np.log(P["rv"].to_numpy())
    ok = ZF.own(a, inp["AFLT"]["daily"], inp["AFLT"]["herf"])["lrv"].to_numpy()
    bad = ZF.own(a, inp["AFLT"]["daily"], inp["AFLT"]["herf"], lead=1)["lrv"].to_numpy()
    assert not ZB.leak(ZB.profile(ok, rv)) and ZB.leak(ZB.profile(bad, rv))
    full = ZF.own(a, inp["AFLT"]["daily"], inp["AFLT"]["herf"], lead=1)
    cut = ZF.own(a[a["date"] <= "2017-06-15"], inp["AFLT"]["daily"][inp["AFLT"]["daily"].index <= "2017-06-15"], inp["AFLT"]["herf"][inp["AFLT"]["herf"].index <= "2017-06-15"], lead=1)
    i = len(cut) - 1
    assert not np.allclose(full.iloc[i].to_numpy(float), cut.iloc[i].to_numpy(float), equal_nan=True)      # обрезка ловит мутацию


def test_cross_only_17_tickers_and_e11_rejected():
    P, inp = synth()
    zl = ZF.build_panel_z(P, inp)
    c = ZF.cross(zl, ["AFLT", "ALRS", "HYDR"])
    d = zl["date"].iloc[5]
    manual = zl[(zl["date"] == d) & zl["ticker"].isin(["AFLT", "ALRS", "HYDR"])]["lrv"].median()
    assert c.loc[d, "med_lrv"] == pytest.approx(manual)
    for bad in (["AFLT", "GCHE"], ["MRKC"]):        # E11 в поперечных признаках — отказ боевого пути
        with pytest.raises(ValueError):
            ZF.cross(zl.assign(ticker=zl["ticker"].replace({"IRAO": "GCHE"})), bad)


def test_loto_cross_over_16_excludes_held_out():
    """LOTO: поперечные признаки пересчитаны по остальным бумагам — собственное значение исключённой в медиану не входит."""
    P, inp = synth()
    zl = ZF.build_panel_z(P, inp)
    z3 = ZF.recross(zl, ["AFLT", "ALRS", "HYDR"])
    d = zl["date"].iloc[7]
    row = z3[(z3["ticker"] == "IRAO") & (z3["date"] == d)].iloc[0]
    assert row["med_lrv"] == pytest.approx(zl[(zl["date"] == d) & zl["ticker"].isin(["AFLT", "ALRS", "HYDR"])]["lrv"].median())
    assert row["med_lrv"] != pytest.approx(zl[zl["date"] == d]["lrv"].median())


def bars_day(day, q):
    """Бары основной сессии дня: 10:00…18:30, доходность i-го бара ≈ q[i] (log-цена задана накопленной суммой)."""
    begin = pd.date_range(day + " 10:00", periods=len(q), freq="10min")
    c = 100.0 * np.exp(np.cumsum(q))
    o = np.r_[100.0, c[:-1]]
    return pd.DataFrame({"begin": begin, "open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1, "value": 1.0})


def test_herf_known_answer_and_next_day_bars_mutation():
    """Херфиндаль дня d — только из баров дня d; мутация `lag=1` (бары дня d+1) отличается и ловится подменой баров после d."""
    q1, q2 = np.r_[0.01, np.zeros(9)], np.full(10, 0.01)       # день 1: всё в одном баре (H = 1); день 2: равномерно (H = 1/10)
    bars = pd.concat([bars_day("2020-01-06", q1), bars_day("2020-01-07", q2)], ignore_index=True)
    h = SH.herf_by_day(bars)
    assert h.iloc[0] == pytest.approx(1.0) and h.iloc[1] == pytest.approx(0.1)
    alt = pd.concat([bars_day("2020-01-06", q1), bars_day("2020-01-07", np.r_[0.02, np.zeros(9)])], ignore_index=True)
    assert SH.herf_by_day(alt).iloc[0] == h.iloc[0]                              # подмена баров после d не меняет значение дня d
    assert SH.herf_by_day(bars, lag=1).iloc[0] == pytest.approx(0.1) and SH.herf_by_day(alt, lag=1).iloc[0] != SH.herf_by_day(bars, lag=1).iloc[0]       # мутация: день d видит d+1


def test_last_hour_share_known_answer():
    d = pd.DataFrame({"rv": [4.0, 2.0], "rv_close": [1.0, 1.0]}, index=pd.to_datetime(["2020-01-06", "2020-01-07"]))
    s = SH.last_hour_share(d)
    assert s.iloc[0] == 0.25 and s.iloc[1] == 0.5 and SH.last_hour_share(d, lag=1).iloc[0] == 0.5
