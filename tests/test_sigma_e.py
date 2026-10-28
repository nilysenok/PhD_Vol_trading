"""Семья E, ряды σ̂: ex-ante окно E-const, формула naive, детектор сдвига (§8 п. 1). Синтетика, сеть и данные ≥ 2022 не нужны."""
import numpy as np
import pandas as pd
import pytest

from fastbt import sigma_e as S


def _rv(n=700, tickers=("A", "B", "C"), seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-03", periods=n)
    rows = []
    for t in tickers:
        e = rng.normal(0, 0.5, n)
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.7 * x[i - 1] + e[i]
        rows.append(pd.DataFrame({"date": dates, "ticker": t, "rv": np.exp(-8.5 + x)}))
    return pd.concat(rows).sort_values(["ticker", "date"]).reset_index(drop=True)


def _const_with_current(rv):
    """Мутация: окно [d−251, d] — текущий день входит (нет shift(1))."""
    r = rv.sort_values(["ticker", "date"]).reset_index(drop=True)
    s = r.groupby("ticker")["rv"].transform(lambda x: np.sqrt(x).rolling(252, min_periods=252).mean())
    return S._out(r, s)


def _no_current_day(fn):
    """Возмущение RV в день d не должно менять σ̄_d."""
    rv = _rv()
    base = fn(rv).set_index(["date", "ticker"])["sigma"]
    d = rv[rv.ticker == "A"]["date"].iloc[400]
    pert = rv.copy()
    pert.loc[(pert.ticker == "A") & (pert.date == d), "rv"] *= 50.0
    new = fn(pert).set_index(["date", "ticker"])["sigma"]
    assert new[(d, "A")] == base[(d, "A")], "σ̄_d зависит от RV дня d"
    assert new[(d + pd.offsets.BDay(1), "A")] != base[(d + pd.offsets.BDay(1), "A")], "d+1 обязан видеть RV дня d"


def test_const_ne_vidit_tekushchiy_den():
    _no_current_day(S.sigma_const)


def test_mutaciya_const_s_tekushchim_dnem_lovitsya():
    with pytest.raises(AssertionError):
        _no_current_day(_const_with_current)


def test_const_formula_okno_252():
    rv = _rv()
    a = rv[rv.ticker == "B"].reset_index(drop=True)
    out = S.sigma_const(rv).query("ticker == 'B'").set_index("date")["sigma"]
    d = a["date"].iloc[300]
    assert out[d] == pytest.approx(np.sqrt(a["rv"].iloc[300 - 252:300]).mean())


def test_naive_okno_5_vklyuchaet_den_d():
    rv = _rv()
    a = rv[rv.ticker == "C"].reset_index(drop=True)
    out = S.sigma_naive(rv).query("ticker == 'C'").set_index("date")["sigma"]
    assert out[a["date"].iloc[50]] == pytest.approx(np.sqrt(a["rv"].iloc[46:51].mean()))


def _model_sigma(rv, shift=0):
    """Синтетический «прогноз»: функция RV до дня d включительно (последний известный день 0); shift>0 — утечка d+shift."""
    ln = np.log(rv["rv"])
    mu = ln.mean()
    d = rv[["date", "ticker"]].assign(sigma=np.exp(0.5 * (mu + 0.8 * (ln - mu))))
    if shift:
        d["sigma"] = d.groupby("ticker")["sigma"].shift(-shift)
    return d.dropna().reset_index(drop=True)


def test_detektor_sdvig_lovit_sdvig_plus1():
    """§8 п. 1: σ̂ сдвинута на +1 — результат меняется, детектор срабатывает (ok=False, argmax=+1)."""
    rv = _rv(n=900)
    ok0 = S.proverka_sdvig(_model_sigma(rv), rv, "model")
    bad = S.proverka_sdvig(_model_sigma(rv, shift=1), rv, "model")
    assert ok0["ok"] and ok0["argmax_uroven"] == 0
    assert not bad["ok"] and bad["argmax_uroven"] == 1
    assert bad["corr_k+1"] != ok0["corr_k+1"]


def test_detektor_sdvig_lovit_otstavanie_minus1():
    rv = _rv(n=900)
    lag = _model_sigma(rv, shift=-1)
    assert not S.proverka_sdvig(lag, rv, "model")["ok"]


def test_detektor_const_k_minus1_i_utechka_plus1_na_naive():
    rv = _rv(n=900)
    assert S.proverka_sdvig(S.sigma_const(rv), rv, "okno", (-1,))["ok"]
    leak = S.sigma_naive(rv.assign(rv=rv.groupby("ticker")["rv"].shift(-1)).dropna())
    assert not S.proverka_sdvig(leak, rv, "okno")["ok"]


def test_const_s_tekushchim_dnem_ne_prohodit_K_minus1():
    """Мутация «окно включает d» видна и детектору приращений: пик на 0, а не на заявленном −1."""
    rv = _rv(n=900)
    r = S.proverka_sdvig(_const_with_current(rv), rv, "okno", (-1,))
    assert not r["ok"] and r["argmax_prir"] == 0
