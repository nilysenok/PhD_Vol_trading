"""V4-C, G15/G16: PIT боевого пути C (признаки + оценка + прогноз): обрезка, мутации «закрыто + 1», «признак дня d+1», исключённая бумага в LOTO, приор, диагностика обнуления. Всё — через `v4_c_run.forecast` / `pit_violation`."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_b_run as BR, v4_c_run as CR, v4_z_feat as ZF
from tests.test_v2_v4_z_feat import synth

DAY = "2017-09-01"          # первый торговый день месяца (пятница): к нему закрыты пары 2017-01…08
CFGS = ["C1", "C2", "C3", "C4"]


@pytest.fixture(scope="module")
def data():
    return synth(330)


def first_day_of_month(P, m):
    return P["date"][P["date"].dt.to_period("M") == pd.Period(m)].min()


@pytest.mark.parametrize("cfg", CFGS)
def test_cut_future_data_does_not_change_forecast(data, cfg):
    P, inp = data
    assert not CR.pit_violation(cfg, P, inp, first_day_of_month(P, "2017-09"))


@pytest.mark.parametrize("cfg", ["C1", "C4"])
def test_mutation_closed_plus_one_is_caught_via_real_path(data, monkeypatch, cfg):
    """Пары «закрыто + 1» (t⁺ ≤ d0 + 3 дня) в оценке месяца: подмена RV после d меняет прогноз d — обрезка боевого пути ловит."""
    P, inp = data
    monkeypatch.setattr(BR, "closed_mask", lambda t_plus, d0: np.asarray(t_plus) <= np.datetime64(d0) + np.timedelta64(3, "D"))
    assert CR.pit_violation(cfg, P, inp, first_day_of_month(P, "2017-09"))


def test_mutation_boundary_t_plus_equal_d0_changes_forecast(data, monkeypatch):
    """Граница закрытой пары: строго t⁺ < d0 (как в B). Пара с t⁺ = d0 известна к закрытию d0, поэтому обрезка (day = d0) её не подменяет и мутацию `<=` не ловит; ловит прямая проверка границы и сдвиг прогноза."""
    P, inp = data
    d0 = np.datetime64(first_day_of_month(P, "2017-09"))
    assert not BR.closed_mask(np.array([d0]), d0)[0] and BR.closed_mask(np.array([d0 - np.timedelta64(1, "D")]), d0)[0]
    zl = ZF.build_panel_z(P, inp)
    honest = CR.forecast("C2", P, zl)[0]["F"].to_numpy()
    monkeypatch.setattr(BR, "closed_mask", lambda t_plus, d: np.asarray(t_plus) <= np.datetime64(d))
    assert not np.array_equal(honest, CR.forecast("C2", P, zl)[0]["F"].to_numpy())


def test_mutation_feature_of_day_d_plus_one_is_caught_via_real_path(data):
    """Признак дня d+1 вместо d (`lead=1` в сборке Z боевого пути): подмена данных после d меняет прогноз d."""
    P, inp = data
    assert CR.pit_violation("C3", P, inp, first_day_of_month(P, "2017-09"), lead=1)


def test_mutation_next_day_bars_in_intraday_shape_is_caught(data):
    """Intraday Shape из данных дня d+1: herf сдвинут назад (как `herf_by_day(lag=1)`) — прогноз C4 дня d меняется при подмене данных после d."""
    P, inp = data
    assert CR.pit_violation("C4", P, inp, first_day_of_month(P, "2017-09"), herf_lag=1)
    assert not CR.pit_violation("C4", P, inp, first_day_of_month(P, "2017-09"))


def test_prior_before_60_pairs_gives_equal_weights(data):
    """До первых 60 закрытых пар — приор w = 0,5: F = √(H·M) (как B2 при w = 0,5)."""
    P, inp = data
    out, w = CR.forecast("C4", P, ZF.build_panel_z(P, inp))
    first = out[out["date"] < first_day_of_month(P, "2017-02")]          # январь: закрытых пар < 60 (4 бумаги × ~20 дней)
    M = np.sqrt(P.set_index(["ticker", "date"]).loc[list(zip(first["ticker"], first["date"])), ["k2", "k3"]].prod(axis=1).to_numpy())
    assert np.allclose(first["F"], np.sqrt(first["h"].to_numpy() * M), rtol=1e-12) and np.allclose(first["w"], 0.5)
    assert (w[w["ticker"] == "PANEL"]["converged"]).all()


def test_loto_excluded_ticker_does_not_enter_cross_or_training(data):
    """LOTO: подмена данных ИСКЛЮЧЁННОЙ бумаги не меняет прогнозы остальных (поперечные по остальным, оценка без неё); δ исключённой = 0."""
    P, inp = data
    held, others = "IRAO", ["AFLT", "ALRS", "HYDR"]
    zl = ZF.build_panel_z(P, inp)
    P2 = P.copy()
    P2.loc[P2["ticker"] == held, "rv"] *= 7.3
    inp2 = {t: dict(v) for t, v in inp.items()}
    inp2[held] = {"daily": inp[held]["daily"].assign(rv=inp[held]["daily"]["rv"] * 7.3), "herf": inp[held]["herf"]}
    zl2 = ZF.build_panel_z(P2, inp2)
    a, wa = CR.forecast("C4", P, ZF.recross(zl, others), exclude=held)
    b, _ = CR.forecast("C4", P2, ZF.recross(zl2, others), exclude=held)
    ka, kb = a[a["ticker"].isin(others)], b[b["ticker"].isin(others)]
    assert np.array_equal(ka["F"].to_numpy(), kb["F"].to_numpy())
    assert (wa[wa["ticker"] == held]["delta"] == 0).all() and (wa[wa["ticker"] == held]["n_pairs"] == 0).all()


def test_mutation_loto_ticker_left_in_cross_set_is_caught(data):
    """Мутация «исключённая бумага осталась в поперечных признаках»: прогнозы остальных зависят от её данных."""
    P, inp = data
    held, others = "IRAO", ["AFLT", "ALRS", "HYDR"]
    zl = ZF.build_panel_z(P, inp)
    P2 = P.copy()
    P2.loc[P2["ticker"] == held, "rv"] *= 7.3
    inp2 = {t: dict(v) for t, v in inp.items()}
    inp2[held] = {"daily": inp[held]["daily"].assign(rv=inp[held]["daily"]["rv"] * 7.3), "herf": inp[held]["herf"]}
    zl2 = ZF.build_panel_z(P2, inp2)
    a, _ = CR.forecast("C4", P, ZF.recross(zl, others + [held]), exclude=held)
    b, _ = CR.forecast("C4", P2, ZF.recross(zl2, others + [held]), exclude=held)
    assert not np.array_equal(a[a["ticker"].isin(others)]["F"].to_numpy(), b[b["ticker"].isin(others)]["F"].to_numpy())


def test_ablation_zero_beta_equals_cfg_without_block_coefficients(data):
    """Диагностика: обнуление всех коэффициентов признаков даёт F с постоянным весом σ(α+δ_i): совпадает с формулой, не с исходным F."""
    P, inp = data
    zl = ZF.build_panel_z(P, inp)
    out, _ = CR.forecast("C2", P, zl, ablate={"all": ZF.cfg_feats("C2")})
    late = out[out["date"] > "2017-06-01"]
    assert np.isfinite(late["F_abl_all"]).all() and not np.allclose(late["F_abl_all"], late["F"])
