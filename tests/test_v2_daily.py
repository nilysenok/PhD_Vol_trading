"""G1, G8: дневной ряд SBER (RV/BV/J на ручном примере, окно сессии, порог 30 баров, разрыв 2022)."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import daily  # noqa: E402

A = 0.001
A2 = (A * 100) ** 2  # ожидаемые RV/BV/J в %^2


def mk_day(date, rets, start="10:00", open1=100.0):
    """Бары по 10 минут; r_i заданы: close_i = close_{i-1}*exp(r_i), open_1 = open1, close_0 = open1."""
    t0 = pd.Timestamp(date + " " + start)
    close = open1 * np.exp(np.cumsum(rets))
    opn = np.concatenate([[open1], close[:-1]])
    return pd.DataFrame({"begin": [t0 + pd.Timedelta(minutes=10 * k) for k in range(len(rets))],
                         "open": opn, "high": np.maximum(opn, close), "low": np.minimum(opn, close),
                         "close": close, "volume": 1, "value": 1.0})


def test_hand_example_alternating_has_no_jump():
    rets = np.array([A if k % 2 == 0 else -A for k in range(30)])
    d, ex = daily.build_daily(mk_day("2020-03-02", rets))
    row = d.iloc[0]
    assert math.isclose(row.rv, 30 * A2, rel_tol=1e-9)
    assert math.isclose(row.bv, math.pi / 2 * 29 * A2, rel_tol=1e-9)
    assert row.j == 0.0 and len(ex) == 0


def test_hand_example_with_jump():
    rets = np.full(30, A)
    rets[14] = 10 * A  # бар 15
    row = daily.build_daily(mk_day("2020-03-02", rets))[0].iloc[0]
    assert math.isclose(row.rv, 129 * A2, rel_tol=1e-9)
    assert math.isclose(row.bv, math.pi / 2 * 47 * A2, rel_tol=1e-9)
    assert math.isclose(row.j, 129 * A2 - math.pi / 2 * 47 * A2, rel_tol=1e-9)


def test_overnight_gap_not_in_rv():
    a = mk_day("2020-03-02", np.full(30, A), open1=100.0)
    b = mk_day("2020-03-03", np.full(30, A), open1=150.0)  # открытие +50 %
    d, _ = daily.build_daily(pd.concat([a, b]))
    assert np.allclose(d["rv"], 30 * A2, rtol=1e-9)


def test_session_window_and_evening_ignored():
    day = mk_day("2020-03-02", np.full(53, A))  # 10:00..18:40
    assert day["begin"].iloc[-1].strftime("%H:%M") == "18:40"
    junk = mk_day("2020-03-02", np.array([0.5] * 4), start="18:50", open1=100.0)
    early = mk_day("2020-03-02", np.array([0.5] * 2), start="09:40", open1=100.0)
    d, _ = daily.build_daily(pd.concat([early, day, junk]))
    assert d["n_bars"].iloc[0] == 53
    assert math.isclose(d["rv"].iloc[0], 53 * A2, rel_tol=1e-9)


def test_threshold_30_bars():
    bars = pd.concat([mk_day("2020-03-02", np.full(29, A)), mk_day("2020-03-03", np.full(30, A))])
    d, ex = daily.build_daily(bars)
    assert list(d.index) == [pd.Timestamp("2020-03-03")]
    assert ex.to_dict("records") == [{"date": pd.Timestamp("2020-03-02"), "n_bars_main": 29, "reason": "bars<30"}]


def test_weekend_sessions_excluded_from_2025_but_workday_saturday_before_kept():
    bars = pd.concat([mk_day("2024-11-30", np.full(53, A)),   # суббота до 2025 — перенесённый рабочий день
                      mk_day("2025-03-01", np.full(53, A)),   # суббота 2025 — выходная сессия, даже с 53 барами
                      mk_day("2025-03-03", np.full(53, A))])
    d, ex = daily.build_daily(bars)
    assert list(d.index) == [pd.Timestamp("2024-11-30"), pd.Timestamp("2025-03-03")]
    assert ex["reason"].tolist() == ["weekend_2025"]


def test_day_inside_stop_excluded_even_with_bars():
    bars = pd.concat([mk_day("2022-03-01", np.full(53, A)), mk_day("2022-03-31", np.full(53, A))])
    d, ex = daily.build_daily(bars)
    assert list(d.index) == [pd.Timestamp("2022-03-31")]
    assert ex["reason"].tolist() == ["stop_2022"]


def _series():
    dates = [x for x in pd.bdate_range("2022-01-03", "2022-06-30") if not (daily.GAP[0] <= x <= daily.GAP[1])]
    return pd.DataFrame({"rv": np.arange(1, len(dates) + 1, dtype=float)}, index=pd.DatetimeIndex(dates))


def test_windows_do_not_cross_the_stop():
    d = daily.add_windows(_series())
    post = d[d["seg"] == 1]
    assert post["rv_w"].iloc[:4].isna().all() and post["rv_w"].iloc[4:].notna().all()
    assert post["rv_m"].iloc[:21].isna().all() and post["rv_m"].iloc[21:].notna().all()
    assert post["rv_w"].iloc[4] == post["rv"].iloc[:5].mean()
    assert d[d["seg"] == 0]["rv_w"].iloc[4:].notna().all()


def test_mutation_glued_windows_is_caught():
    s = _series()
    good = daily.add_windows(s)
    first_post = np.flatnonzero(good["seg"].to_numpy() == 1)[0]
    assert np.isnan(good["rv_w"].iloc[first_post])
    saved = daily.GAP
    try:  # настоящая мутация производственной функции: разрыв «не виден», сегмент один (склейка)
        daily.GAP = (saved[0], pd.Timestamp("2100-01-01"))
        glued = daily.add_windows(s)
    finally:
        daily.GAP = saved
    assert (glued["seg"] == 0).all() and not np.isnan(glued["rv_w"].iloc[first_post])
    assert not good["rv_w"].equals(glued["rv_w"])  # тест остановки обязан падать на этой мутации


def test_windows_include_day_d_and_exact_mean():
    d = daily.add_windows(_series())
    assert d["rv_w"].iloc[4] == np.mean([1, 2, 3, 4, 5]) and d["rv_m"].iloc[21] == np.mean(np.arange(1, 23))
