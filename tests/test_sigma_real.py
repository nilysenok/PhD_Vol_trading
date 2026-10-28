"""σ этапа 1 (Ж-047 п. 2): синтетика с известным ответом + мутации (σ после 18:50, σ d+1, λ, окно 22 → 21)."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import sigma_real as SR

ROOT = Path(__file__).resolve().parent.parent
BAD_MODULES = ("sigma_e", "vpred", "ch4_dev")
BAD_STRINGS = ("sigma_pred", "hyb", "harj", "ch4_dev")
STAGE1 = ("sigma_real", "etap1", "rezhim_e1", "sobytiya_e1")


def _bars(day: str, closes: list[float], start="09:50") -> pd.DataFrame:
    ts = pd.date_range(f"{day} {start}", periods=len(closes), freq="10min")
    return pd.DataFrame({"timestamp": ts, "close": closes})


def test_rv_known_answer_and_cut_1850():
    # бары 18:30, 18:40 внутри; 18:50 и 19:00 — вне окна
    b = _bars("2018-03-01", [100.0, 101.0], start="18:30")
    late = _bars("2018-03-01", [150.0, 90.0], start="18:50")
    rv = SR.rv_1850(pd.concat([b, late]))
    assert rv.iloc[0] == pytest.approx(np.log(101 / 100) ** 2, abs=1e-15)


def test_mutation_sigma_after_1850_caught():
    """Мутация: окно до 24:00 (σ видит вечер дня d) — RV меняется, тест обязан это увидеть."""
    b = pd.concat([_bars("2018-03-01", [100.0, 101.0], start="18:30"),
                   _bars("2018-03-01", [150.0, 90.0], start="19:00")])
    ok = SR.rv_1850(b).iloc[0]
    mut = SR.rv_1850(b, cut_min=24 * 60).iloc[0]
    assert ok == pytest.approx(np.log(1.01) ** 2) and mut > 100 * ok


def test_open_boundary_0950_and_no_overnight_return():
    d1 = _bars("2018-03-01", [100.0, 100.0], start="09:40")      # 09:40 вне окна
    d2 = _bars("2018-03-02", [200.0, 202.0], start="09:50")      # ночной гэп 100 → 200 не считается
    rv = SR.rv_1850(pd.concat([d1, d2]))
    assert len(rv) == 1 and rv.iloc[0] == pytest.approx(np.log(202 / 200) ** 2)


def _rv(n=100) -> pd.Series:
    return pd.Series(np.arange(1.0, n + 1), index=pd.bdate_range("2018-01-01", periods=n), name="rv")


def test_sigma_rv_windows_known_answer():
    rv = _rv()
    for name, w in SR.WINDOWS.items():
        s = SR.sigma_all(rv)[name]
        assert s.iloc[:w - 1].isna().all() and not s.iloc[w - 1:].isna().any()
        d = w + 3                                              # день d включён, d+1 — нет
        assert s.iloc[d] == pytest.approx(np.sqrt(rv.iloc[d - w + 1:d + 1].mean()))


def test_mutation_window_22_to_21_caught():
    rv = _rv()
    good = SR.sigma_rv(rv, 22)
    assert not good.equals(SR.sigma_rv(rv, 21))
    assert good.iloc[30] == pytest.approx(np.sqrt(rv.iloc[9:31].mean()))
    assert good.iloc[30] != pytest.approx(np.sqrt(rv.iloc[10:31].mean()))


def test_ewma_known_answer_and_mutation_lambda():
    rv = pd.Series([4.0, 1.0, 9.0] + [2.0] * 40, index=pd.bdate_range("2018-01-01", periods=43))
    v = [4.0]
    for x in rv.iloc[1:]:
        v.append(0.94 * v[-1] + 0.06 * x)
    s = SR.sigma_ewma(rv)
    assert s.iloc[:SR.EWMA_MIN - 1].isna().all()
    assert s.iloc[30] == pytest.approx(np.sqrt(v[30]), abs=1e-14)
    assert s.iloc[30] != pytest.approx(SR.sigma_ewma(rv, lam=0.97).iloc[30], rel=1e-6)
    assert SR.LAMBDA == 0.94


def _synth_rv(n=600, seed=3) -> pd.Series:
    r = np.random.default_rng(seed).lognormal(0.0, 0.7, n)
    return pd.Series(r, index=pd.bdate_range("2017-01-02", periods=n))


@pytest.mark.parametrize("name", SR.NAMES)
def test_sdvig_ok_and_plus_one_day_caught(name):
    rv = _synth_rv()
    s = SR.sigma_all(rv)[name]
    assert SR.proverka_sdvig(s, rv)["ok"]
    mut = s.shift(-1)                                          # σ_{d+1} на дне d: утечка (mutations.md № 1)
    r = SR.proverka_sdvig(mut, rv)
    assert not r["ok"] and r["argmax"] == 1


def test_sigma_frame_no_future_row():
    rv = pd.concat([_rv().rename("rv").reset_index().rename(columns={"index": "date"}).assign(ticker=t) for t in "AB"])
    cut = rv[rv["date"] <= rv["date"].iloc[40]]
    full = SR.sigma_frame(rv, "rv22").set_index(["ticker", "date"])["sigma"]
    part = SR.sigma_frame(cut, "rv22").set_index(["ticker", "date"])["sigma"]
    assert np.allclose(full.loc[part.index].values, part.values, rtol=0, atol=0)


def _forbidden(src: str) -> list[str]:
    bad = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            names = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            names = [n.module or ""] + [a.name for a in n.names]
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            bad += [n.value for k in BAD_STRINGS if k in n.value]
            continue
        else:
            continue
        bad += [x for x in names if any(m in x.split(".") for m in BAD_MODULES)]
    return bad


def test_static_no_forecast_imports():
    found = []
    for m in STAGE1:
        p = ROOT / "fastbt" / f"{m}.py"
        if p.exists():
            found += [(m, x) for x in _forbidden(p.read_text())]
    assert (ROOT / "fastbt" / "sigma_real.py").exists() and not found, found


@pytest.mark.parametrize("src", ["from fastbt.sigma_e import load", "import fastbt.sigma_e", "from fastbt import sigma_e",
                                 "x = 'sigma_pred_h1'", "p = 'data/regen/ch4_dev/x.parquet'", "from vpred import y"])
def test_mutation_forecast_import_caught(src):
    assert _forbidden(src)
