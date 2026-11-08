"""V4-C лист C-b (G19/G20): PIT боевого пути C5–C10 (`v4_cb_run`): обрезка, мутации «закрыто + 1» (все шесть), «дерево C7 на парах года прогноза», γ ≥ 0 (C6), симплекс (C9), приор, LOTO, тождество с C1, C_YEARS."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_b_run as BR, v4_cb_fit as FB, v4_cb_run as R, v4_cb_years as Y, v4_c_run as CR, v4_z_feat as ZF
from tests.test_v2_v4_z_feat import synth

CFGS = R.CFGS


@pytest.fixture(scope="module")
def data():
    return synth(560)


def day_of(P, cfg):
    """Первый торговый день периода модели: месяц (2017-09) или год (2018) для C7 — к нему «закрыто + 1» уже видно."""
    d = P["date"]
    return d[d.dt.year == 2018].min() if R.SPECS[cfg][2] == "Y" else d[d.dt.to_period("M") == pd.Period("2017-09")].min()


@pytest.mark.parametrize("cfg", CFGS)
def test_cut_future_data_does_not_change_forecast(data, cfg):
    P, inp = data
    assert not R.pit_violation(cfg, P, inp, day_of(P, cfg))


@pytest.mark.parametrize("cfg", CFGS)
def test_mutation_closed_plus_one_is_caught_via_real_path(data, monkeypatch, cfg):
    P, inp = data
    monkeypatch.setattr(BR, "closed_mask", lambda t_plus, d0: np.asarray(t_plus) <= np.datetime64(d0) + np.timedelta64(3, "D"))
    assert R.pit_violation(cfg, P, inp, day_of(P, cfg))


def test_mutation_c7_tree_trained_on_pairs_of_forecast_year_is_caught(data, monkeypatch):
    P, inp = data
    monkeypatch.setattr(BR, "closed_mask", lambda t_plus, d0: np.asarray(t_plus) <= np.datetime64(d0) + np.timedelta64(366, "D"))
    assert R.pit_violation("C7", P, inp, pd.Timestamp("2018-06-01"))


def test_c7_is_refit_once_per_year_on_first_trading_day(data):
    P, inp = data
    _, w = R.forecast("C7", P, ZF.build_panel_z(P, inp))
    pan = w[w["ticker"] == "PANEL"]
    assert list(pan["month"]) == ["2017", "2018", "2019"] and pan["d0"].is_unique
    assert pan["n_pairs"].iloc[0] == 0 and pan["n_pairs"].iloc[1] > 60          # 2017 — приор (закрытых пар нет), 2018 — обучение на 2017


def test_prior_before_60_pairs_gives_equal_weights(data):
    P, inp = data
    zl = ZF.build_panel_z(P, inp)
    for cfg in CFGS:
        out = R.forecast(cfg, P, zl)[0]
        first = out[out["date"] < "2017-02-01"]                                  # январь: закрытых пар < 60
        M = np.sqrt(P.set_index(["ticker", "date"]).loc[list(zip(first["ticker"], first["date"])), ["k2", "k3"]].prod(axis=1).to_numpy())
        want = np.sqrt(first["h"].to_numpy() * M)                                  # w = 0,5: F = √(H·M) (B2 при w = 0,5)
        if cfg == "C9":                                                            # три эксперта: по 1/3 -> F = (H·K2·K3)^(1/3)
            want = np.cbrt(first["h"].to_numpy() * M ** 2)
        assert np.allclose(first["F"], want, rtol=1e-12) and np.allclose(first["w"], 0.5 if cfg != "C9" else 2 / 3), cfg


def test_c9_weights_on_simplex_in_boevoj_path(data):
    P, inp = data
    out = R.forecast("C9", P, ZF.build_panel_z(P, inp))[0]
    s = out[["wH", "wK2", "wK3"]]
    assert np.allclose(s.sum(1), 1.0, atol=1e-12) and (s >= 0).all().all() and np.allclose(out["w"], out["wK2"] + out["wK3"])


def test_c6_gamma_nonneg_boevoj_path_and_mutation_without_constraint_caught(data, monkeypatch):
    P, inp = data
    zl = ZF.build_panel_z(P, inp)
    zn = zl.assign(ewqd=-zl["ewqd"])                                            # знак ewqd инвертирован: одна из двух постановок имеет неограниченный γ < 0
    gam = lambda z: R.forecast("C6", P, z)[1].query("ticker == 'PANEL'")["b_ewqd"].dropna()
    assert min(gam(zl).min(), gam(zn).min()) >= -1e-12
    orig = FB.fit_lin
    monkeypatch.setattr(FB, "fit_lin", lambda *a, **k: orig(*a, **{**k, "nonneg": ()}))
    assert min(gam(zl).min(), gam(zn).min()) < -1e-3


def test_loto_excluded_ticker_has_panel_parameters_and_does_not_enter_cross_or_training(data):
    P, inp = data
    held, others = "IRAO", ["AFLT", "ALRS", "HYDR"]
    zl = ZF.build_panel_z(P, inp)
    P2 = P.copy()
    P2.loc[P2["ticker"] == held, "rv"] *= 7.3
    inp2 = {t: dict(v) for t, v in inp.items()}
    inp2[held] = {"daily": inp[held]["daily"].assign(rv=inp[held]["daily"]["rv"] * 7.3), "herf": inp[held]["herf"]}
    zl2 = ZF.build_panel_z(P2, inp2)
    for cfg in CFGS:
        a, wa = R.forecast(cfg, P, ZF.recross(zl, others), exclude=held)
        b, _ = R.forecast(cfg, P2, ZF.recross(zl2, others), exclude=held)
        assert np.array_equal(a[a["ticker"].isin(others)]["F"].to_numpy(), b[b["ticker"].isin(others)]["F"].to_numpy()), cfg
        h = wa[wa["ticker"] == held]
        assert (h["dev"] == 0).all() and (h["n_pairs"] == 0).all() and h["excluded"].all(), cfg


def test_linear_contour_reproduces_c1_and_eta_zero_is_c1(data):
    P, inp = data
    zl = ZF.build_panel_z(P, inp)
    ref = CR.forecast("C1", P, zl)[0]["F"].to_numpy()
    assert np.allclose(R.forecast("C1x", P, zl)[0]["F"].to_numpy(), ref, rtol=1e-8, atol=0)


def test_years_table_sum_of_parts_equals_crit_panel_and_mutation_caught():
    """crit_part по годам суммируется в hp_stats_sel.crit_panel; перепутанный год (сдвиг на +1) — другое число."""
    from models_v2.s10 import hp_stats_sel as S
    from models_v2.s11 import v4_io as IO
    rng = np.random.default_rng(8)
    rows = []
    for t in IO.M16[:4]:
        d = pd.bdate_range("2017-01-02", "2022-12-30")
        rv = np.exp(rng.normal(0, 0.4, len(d)) + np.repeat(rng.normal(0, 0.3, 6), [len(d[d.year == y]) for y in range(2017, 2023)]))
        rows.append(pd.DataFrame({"ticker": t, "date": d, "t_plus": d + pd.Timedelta(days=1), "group": "M16", "rv_tplus": rv, "k1": rv * np.exp(rng.normal(0, 0.2, len(d))),
                                  "d2": rv * np.exp(rng.normal(0, 0.3, len(d))), "A": rv * np.exp(rng.normal(0, 0.1, len(d)))}))
    k = pd.concat(rows, ignore_index=True)
    k["t_plus"] = k["date"]
    yt = Y.years_table(k, ["k1", "d2", "A"])
    crit = S.crit_panel(k, ["k1", "d2", "A"]).set_index("series")["crit_panel"]
    got = Y.crit_from_years(yt)
    assert all(abs(got[s] - crit[s]) < 1e-12 for s in crit.index)
    bad = yt.copy()
    bad.loc[bad["year"] == 2018, "crit_part"] = np.nan                             # мутация: потерян год — сумма теряет член
    assert any(abs(Y.crit_from_years(bad)[s] - crit[s]) > 1e-6 for s in crit.index)
    assert set(yt[yt["role"] == "описательно"]["year"]) == {2022} and len(yt[yt["role"] == "выбор"]) == 15
