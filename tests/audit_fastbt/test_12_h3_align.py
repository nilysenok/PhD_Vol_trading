"""12. Выравнивание H3: экспозиция, выбранная на закрытии d, умножается на доходность бенчмарка
d→d+1 (как у МСП: r[d] = d→d+1, кэш за [d, d_next)); информация d+1 в выбор не входит; H3
сравнивает МСП-A и бенчмарк на одних и тех же периодах (решение владельца 26.10, 23:00).
"""
import numpy as np
import pandas as pd

from fastbt import bench

IDX = pd.bdate_range("2021-01-04", periods=260)
RNG = np.random.default_rng(7)
M = pd.Series(1000 * np.exp(np.cumsum(RNG.normal(0, 0.012, len(IDX)))), IDX)
B = pd.Series(500 * np.exp(np.cumsum(RNG.normal(0, 0.003, len(IDX)))), IDX)
RATE = pd.Series([0.05], pd.to_datetime(["2020-01-01"]))
# экспозиция на закрытии d знает доходность (d−1, d] — прошлое, не d+1
E = pd.Series((np.log(M).diff() > 0).astype(float).values * 0.5, IDX)


def _shift(k):
    orig = bench._frame

    def f(rate):
        df = orig(rate)
        for c in ("rm", "rb"):
            df[c] = df[c].shift(-k)
        return df.dropna()
    return f


MUTATIONS = {"bench_entry_plus1": lambda mp: mp.setattr(bench, "_frame", _shift(1)),
             "bench_entry_minus1": lambda mp: mp.setattr(bench, "_frame", _shift(-1))}


def _patch(monkeypatch):
    monkeypatch.setattr(bench, "_close", lambda p: M if p == bench.MCFTR else B)


def _fwd():
    rm = np.log(M.shift(-1) / M)
    cd = pd.Series(IDX, IDX).shift(-1).sub(pd.Series(IDX, IDX)).dt.days
    return rm, cd * 0.05 / 365


def test_exposure_times_next_day_return(monkeypatch):
    _patch(monkeypatch)
    out = bench.bench_series(RATE, E)
    rm, rf = _fwd()
    fr = out["MCFTR, экспозиция МСП-A"]
    j = fr.index
    assert len(j) == len(IDX) - 1 and j[-1] == IDX[-2]
    want = E[j] * rm[j] + (1 - E[j]) * rf[j]
    assert np.allclose(fr["total"], want)
    assert np.allclose(fr["total"] - fr["excess"], rf[j])
    # выбор по прошлому не даёт бенчмарку знания будущего: среднее близко к ставке
    assert abs((fr["excess"]).mean() * 252) < 0.15


def test_h3_pairs_same_periods(monkeypatch):
    """МСП-A, совпадающий с «MCFTR 16% + ставка» по периодам d→d+1, — ΔШарп ровно 0."""
    _patch(monkeypatch)
    rm, rf = _fwd()
    j = IDX[:-1]
    tot = 0.16 * rm[j] + 0.84 * rf[j]
    port = pd.DataFrame({"total": tot, "excess": tot - rf[j], "exposure": 0.16}, index=j)
    t = bench.h3_table(port, RATE, 50, "2021-01-01", "2021-12-31").set_index("portfolio")
    assert abs(t.loc["MCFTR 16% + ставка", "d_sharpe_msp_minus"]) < 1e-12
    assert t.loc["MCFTR 16% + ставка", "common_days"] == len(j)
