"""G7: утечки. Главный тест — прогноз на d не меняется при порче/обрезке всего после 18:50 дня d (побитно);
мутации (признаки +1 день; окно с парой (x_d, y_{d+1})) обязаны его ронять; допуски (i)–(iii) — из GATES.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import daily, har, prov  # noqa: E402

TAB = prov.ROOT / "docs/tables/v2_sber_h1/p1_leak.csv"


def damage(bars, d, mode):
    day = bars["begin"].dt.normalize()
    mask = (day > d) | ((day == d) & (bars["begin"].dt.strftime("%H:%M") >= "18:50"))
    if mode == "drop":
        return bars[~mask]
    b = bars.copy()
    rng = np.random.default_rng(7)
    for c in ("open", "high", "low", "close"):
        b.loc[mask, c] = rng.uniform(50, 500, int(mask.sum()))
    return b


def prod(p, i):
    return har.forecast_at(p, i)


def m1_shift_plus_one(p, i):
    X2 = np.vstack([p.X[1:], np.full((1, 4), np.nan)])
    return har.forecast_at(har.Panel(p.dates, p.seg, p.rv, X2), i)


def m2_pair_with_target(p, i):
    beta, s2, _ = har.ols(*har.pairs(p, i, har.COLS_HARJ, upto=i + 1))
    a, b = har.predict(beta, p.X[i, list(har.COLS_HARJ)], s2)
    return {"harj_a": a, "harj_b": b}


def run(bars, d, fc):
    dly, _ = daily.build_daily(bars)
    p = har.make_panel(dly)
    i = int(np.flatnonzero(dly.index == d)[0])
    return np.array(list(fc(p, i).values()), float).tobytes()


def invariant(bars, d, fc):
    """True, если прогноз на d побитно одинаков при порче и при обрезке; исключение = нарушение."""
    ref = run(bars, d, fc)
    try:
        return all(run(damage(bars, d, m), d, fc) == ref for m in ("garbage", "drop"))
    except IndexError:
        return False


def synth_bars(n_days=150, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-06", periods=n_days)
    lvl = np.exp(np.cumsum(rng.normal(0, 0.1, n_days)) * 0.3)
    frames, c0 = [], 100.0
    for k, day in enumerate(days):
        for start, m in (("10:00", 53), ("18:50", 4)):  # вечерние бары начинаются ровно с 18:50
            t0 = pd.Timestamp(str(day.date()) + " " + start)
            r = rng.normal(0, 0.001 * lvl[k], m)
            close = c0 * np.exp(np.cumsum(r))
            opn = np.concatenate([[c0], close[:-1]])
            frames.append(pd.DataFrame({"begin": [t0 + pd.Timedelta(minutes=10 * j) for j in range(m)], "open": opn,
                                        "high": np.maximum(opn, close), "low": np.minimum(opn, close),
                                        "close": close, "volume": 1, "value": 1.0}))
            c0 = close[-1]
    return pd.concat(frames, ignore_index=True), days


def test_synthetic_production_passes_and_mutants_caught():
    bars, days = synth_bars()
    d = days[100]
    assert invariant(bars, d, prod)
    assert not invariant(bars, d, m1_shift_plus_one)
    assert not invariant(bars, d, m2_pair_with_target)


def test_bar_exactly_1850_is_outside_and_window_mutation_caught():
    bars, days = synth_bars()
    d = days[100]
    b = bars.copy()
    m = b["begin"] == pd.Timestamp(str(d.date()) + " 18:50")
    assert m.sum() == 1
    for c in ("open", "high", "low", "close"):
        b.loc[m, c] = 777.0  # порча единственного бара с begin 18:50
    assert run(b, d, prod) == run(bars, d, prod)
    saved = daily.T_TO
    try:  # мутация: окно сессии до 18:50 включительно
        daily.T_TO = "18:50"
        assert run(b, d, prod) != run(bars, d, prod)
    finally:
        daily.T_TO = saved


@pytest.mark.parametrize("d", ["2017-06-15", "2020-04-02", "2021-12-28", "2022-06-01", "2025-12-10"])
def test_real_data_forecast_unchanged_by_bars_after_1850(d):
    bars = pd.read_parquet(prov.BARS)
    assert invariant(bars, pd.Timestamp(d), prod)


@pytest.mark.parametrize("d", ["2020-04-02", "2022-06-01"])
def test_real_data_mutants_caught(d):
    bars = pd.read_parquet(prov.BARS)
    assert not invariant(bars, pd.Timestamp(d), m1_shift_plus_one)
    assert not invariant(bars, pd.Timestamp(d), m2_pair_with_target)


def crit_i(work, oracle):
    return oracle <= 0.5 * work


def crit_ii(work, full):
    return full <= work + 1e-12


def test_criteria_catch_no_effect_mutants():
    assert not crit_i(0.3, 0.3) and crit_i(0.3, 0.0)  # цель назад не улучшила — тест падает
    assert not crit_ii(0.3, 0.31)


def _q():
    t = pd.read_csv(TAB, comment="#")
    return t.pivot(index="model", columns="variant", values="qlike"), float(t["qlike_rv_m"].iloc[0])


MODELS = ["har_a", "har_b", "harj_a", "harj_b"]
RED_II = "G7(ii) КРАСНЫЙ (объявлено до счёта): у (а) обучение на всей выборке не лучше рабочего; см. отчёт"
RED_III = "G7(iii) КРАСНЫЙ (объявлено до счёта): перемешанные (б) далеки от RV_m (±30 %); см. отчёт"


@pytest.mark.parametrize("m", MODELS)
def test_i_oracle_improves_sharply(m):
    q, _ = _q()
    assert crit_i(q.loc[m, "work"], q.loc[m, "oracle"])


@pytest.mark.parametrize("m", [pytest.param("har_a", marks=pytest.mark.xfail(strict=True, reason=RED_II)),
                               "har_b",
                               pytest.param("harj_a", marks=pytest.mark.xfail(strict=True, reason=RED_II)),
                               "harj_b"])
def test_ii_full_sample_not_worse(m):
    q, _ = _q()
    assert crit_ii(q.loc[m, "work"], q.loc[m, "full"])


@pytest.mark.parametrize("m", ["har_a", "harj_a"])
def test_iii_permuted_a_not_better_than_work(m):
    q, _ = _q()
    assert q.loc[m, "perm"] >= q.loc[m, "work"]


@pytest.mark.parametrize("m", [pytest.param("har_b", marks=pytest.mark.xfail(strict=True, reason=RED_III)),
                               pytest.param("harj_b", marks=pytest.mark.xfail(strict=True, reason=RED_III))])
def test_iii_permuted_b_near_rv_m(m):
    q, rvm = _q()
    assert abs(q.loc[m, "perm"] / rvm - 1) <= 0.30
