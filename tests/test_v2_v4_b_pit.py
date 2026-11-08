"""V4-B, G10: PIT боевого пути walk-forward. Параметры месяца — только по парам t⁺ < первого дня месяца; обрезка (подмена RV после d не меняет прогноз d); мутация «закрыто + 1»; приор до 60 пар;
LOTO: исключённая бумага не входит в оценку панели (δ = 0)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_b_loto as LO, v4_b_run as R

CFGS = ["B1", "B2", "B3", "B4", "B5", "B6"]


def panel(n_days=330, tickers=("AFLT", "ALRS", "HYDR", "IRAO"), seed=5):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n_days)
    rows = []
    for t in tickers:
        k2 = np.exp(rng.normal(0, 0.4, n_days))
        k3 = k2 * np.exp(rng.normal(0, 0.1, n_days))
        h = k2 * np.exp(rng.normal(0, 0.2, n_days))
        rv = np.sqrt(h * k2) * np.exp(rng.normal(-0.04, 0.3, n_days))
        rows.append(pd.DataFrame({"ticker": t, "date": dates, "t_plus": list(dates[1:]) + [dates[-1] + pd.Timedelta(days=1)], "rv": rv, "h": h, "k2": k2, "k3": k3}))
    return pd.concat(rows, ignore_index=True)


@pytest.mark.parametrize("cfg", CFGS)
def test_cut_future_rv_does_not_change_forecast(cfg):
    assert not R.pit_violation(cfg, panel(), "2017-09")


@pytest.mark.parametrize("cfg", ["B2", "B6"])
def test_mutation_closed_plus_one_is_caught_via_real_path(monkeypatch, cfg):
    """Пары «закрыто + 1» (t⁺ ≤ d0 + 3 дня) в оценке параметров месяца: подмена RV будущего меняет прогноз — обрезка это ловит."""
    monkeypatch.setattr(R, "closed_mask", lambda t_plus, d0: np.asarray(t_plus) <= np.datetime64(d0) + np.timedelta64(3, "D"))
    assert R.pit_violation(cfg, panel(), "2017-09")


def test_mutation_boundary_t_plus_equal_d0_is_caught(monkeypatch):
    """Граница: пара с t⁺ = d0 (дата прошлого дня месяца -> d0) не закрыта до d0; нестрогое неравенство ловится."""
    monkeypatch.setattr(R, "closed_mask", lambda t_plus, d0: np.asarray(t_plus) <= np.datetime64(d0))
    assert R.pit_violation("B2", panel(), "2017-09")


def test_prior_before_first_closed_pairs_is_geometric_mean_with_w_half():
    P = panel(n_days=40, tickers=("AFLT",))      # < 60 закрытых пар в первом месяце -> приор
    out, w = R.forecast("B2", P)
    first = out[out["date"] < "2017-02-01"]
    assert np.allclose(first["F"], np.sqrt(first["h"] * np.sqrt(first["k2"] * first["k3"])), rtol=1e-12) and (w[w["month"] == "2017-01"]["w"] == 0.5).all()


def test_forecast_covers_all_rows_and_is_positive():
    P = panel()
    for cfg in CFGS:
        out, w = R.forecast(cfg, P)
        assert len(out) == len(P) and np.isfinite(out["F"]).all() and (out["F"] > 0).all() and w["converged"].all()


def loto_panel_ok(cfg, P, t) -> bool:
    """Панельные параметры LOTO(t) = параметры полного прогона на панели без бумаги t (исключённая не участвует в оценке)."""
    _, w_ex = R.forecast(cfg, P, exclude=t)
    _, w_wo = R.forecast(cfg, P[P["ticker"] != t])
    a = w_ex[w_ex["ticker"] == "PANEL"].sort_values("month")
    b = w_wo[w_wo["ticker"] == "PANEL"].sort_values("month")
    cols = ["wa", "wb", "wc"] if cfg == "B6" else ["w"]
    return bool(np.allclose(a[cols].to_numpy(), b[cols].to_numpy(), atol=1e-6))


@pytest.mark.parametrize("cfg", ["B2", "B6"])
def test_loto_excluded_ticker_not_in_panel_estimation(cfg):
    P = panel()
    assert loto_panel_ok(cfg, P, "ALRS")
    _, w = R.forecast(cfg, P, exclude="ALRS")
    assert LO.delta_zero(cfg, w, "ALRS")


def test_mutation_loto_ticker_left_in_panel_is_caught(monkeypatch):
    """Подмена боевого `_month_params`: exclude игнорируется (исключённая бумага остаётся в оценке) — проверка падает."""
    orig = R._month_params
    monkeypatch.setattr(R, "_month_params", lambda cfg, P, tix, d0, nT, exclude, prev: orig(cfg, P, tix, d0, nT, None, prev))
    assert not loto_panel_ok("B2", panel(), "ALRS")
