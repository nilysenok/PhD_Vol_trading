"""V6-2, G24: признаки качества измерения Q и индекс u. Известные ответы: ручной расчёт q1…q7 на шести барах; броуновский день без шума — RV20/RV10 ≈ 1 и СКО RV30 по сдвигам → 0 при росте числа баров; q8 = 1 для полного дня и PIT-мода года;
сетки собраны только из баров своего дня. Мутации (через боевой `features`): бары дня t⁺ в Q дня d; мода q8 по всему году — строки ≤ d меняются при замене баров/чисел баров после d."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s13 import v6_1_feat as FE, v6_2_q as QF

X = np.array([0.1, -0.2, 0.0, 0.3, 0.1, -0.2])
V = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 2.0])


def make_bars(seed: int = 3, days: int = 70, nb=53, sigma: float = 0.002) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, px = [], 100.0
    for k in range(days):
        day = pd.Timestamp("2016-01-04") + pd.offsets.BDay(k)
        n = nb(k) if callable(nb) else nb
        for j in range(n):
            o = px
            px = px * np.exp(rng.normal(0, sigma))
            rows.append((day + pd.Timedelta(minutes=600 + 10 * j), o, max(o, px), min(o, px), px, float(rng.integers(100, 1000))))
    return pd.DataFrame(rows, columns=["begin", "open", "high", "low", "close", "volume"])


def daily_of(bars):
    return D.build_daily(bars, index_days=None)[0]


def test_hand_calculation_q1_to_q7():
    q = QF.day_q(X, V)
    rv = 0.01 + 0.04 + 0.0 + 0.09 + 0.01 + 0.04                           # 0,19
    rv20 = (0.11 + 0.20) / 2                                                 # s = 0: блоки (-.1; .3; -.1) = .11; s = 1: блоки (-.2; .4) = .20
    g30 = [(-0.1) ** 2 + 0.2 ** 2, 0.1 ** 2, 0.4 ** 2]                       # s = 0: блоки (-0,1; 0,2) = .05; s = 1: (0,1) = .01; s = 2: (0,4) = .16
    m30 = sum(g30) / 3
    sd30 = (sum((v - m30) ** 2 for v in g30) / 3) ** 0.5
    want = [np.log(rv20 / rv), np.log(m30 / rv), sd30 / m30, 1 / 6, (0.1 ** 4 + 0.2 ** 4 + 0.3 ** 4 + 0.1 ** 4 + 0.2 ** 4) / rv ** 2, (0.1 * -0.2 + -0.2 * 0 + 0 * 0.3 + 0.3 * 0.1 + 0.1 * -0.2) / rv, (1 + 4 + 9 + 16 + 0 + 4) / 144.0]
    assert np.allclose(q, want, atol=1e-12), (q, want)
    assert np.allclose(QF.grid_rv(X, 2), [0.11, 0.20]) and np.allclose(QF.grid_rv(X, 3), [0.05, 0.01, 0.16])


def test_brownian_day_ratio_near_one_and_q3_shrinks():
    rng = np.random.default_rng(1)
    ratio = np.mean([np.exp(QF.day_q(rng.normal(0, 0.001, 600), np.ones(600))[0]) for _ in range(300)])
    assert abs(ratio - 1.0) < 0.02, ratio                                  # E[RV20/RV10] ≈ 1 (неполное покрытие 599/600)
    q3 = {n: np.mean([QF.day_q(rng.normal(0, 0.001, n), np.ones(n))[2] for _ in range(100)]) for n in (60, 600, 6000)}
    assert q3[60] > q3[600] > q3[6000] and q3[6000] < 0.03, q3            # СКО RV30 по сдвигам → 0 при росте числа баров


def test_q8_full_day_is_one_and_pit_mode():
    bars = make_bars(days=40, nb=lambda k: 50 if k == 30 else 53)
    d = daily_of(bars)
    q = QF.features(bars, d)
    ok = q["q8"].drop(d.index[30])
    assert np.allclose(ok, 1.0) and np.isclose(q["q8"].iloc[30], 50 / 53)
    n = pd.Series([53, 53, 50, 50, 50, 53.0], index=pd.bdate_range("2016-03-01", periods=6))
    assert np.allclose(QF.mode_share(n).to_numpy(), [1, 1, 50 / 53, 50 / 53, 1, 1])        # мода накапливается; при равенстве частот — большее n (4-й и 6-й дни: 2:2 и 3:3)


def test_mode_is_taken_from_sessions_up_to_d_only():
    n = pd.Series([53.0] * 4 + [40.0] * 10, index=pd.bdate_range("2016-03-01", periods=14))
    m = QF.mode_share(n)
    assert np.isclose(m.iloc[4], 40 / 53) and np.isclose(m.iloc[7], 40 / 53) and np.isclose(m.iloc[8], 1.0)   # 5-й день: 53 (4) против 40 (1); 8-й: 4:4 -> 53; 9-й: 40 (5) против 53 (4)
    n2 = n.copy()
    n2.iloc[10:] = 53.0
    assert np.allclose(QF.mode_share(n2).iloc[:10].to_numpy(), m.iloc[:10].to_numpy())          # замена чисел баров после d не меняет q8 до d


def leaks(day_idx: int = 45, field: str = "bars") -> bool:
    bars = make_bars()
    day = daily_of(bars).index[day_idx]
    ref = QF.features(bars, daily_of(bars))
    b = bars.copy()
    late = b["begin"].dt.normalize() > day
    rng = np.random.default_rng(99)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, int(late.sum()))))
    b.loc[late, "close"], b.loc[late, "open"] = px, np.r_[px[:1], px[:-1]]
    b.loc[late, "volume"] = rng.integers(1, 5000, int(late.sum())).astype(float)
    b = b[~(late & (b["begin"].dt.minute == 40) & (b["begin"].dt.hour == 18))]               # и другое число баров после d
    new = QF.features(b, daily_of(bars))
    return not np.allclose(new.loc[:day].to_numpy(), ref.loc[:day].to_numpy(), equal_nan=True)


def test_no_leak_and_mutations_caught(monkeypatch):
    assert leaks() is False
    real_table = QF.day_table
    monkeypatch.setattr(QF, "day_table", lambda bars: real_table(bars).shift(-1))                       # МУТАЦИЯ: бары дня t⁺ в Q дня d
    assert leaks() is True
    monkeypatch.setattr(QF, "day_table", real_table)
    assert leaks() is False


def test_mode_whole_year_mutation_differs_from_pit():
    n = pd.Series([53.0] * 4 + [40.0] * 10, index=pd.bdate_range("2016-03-01", periods=14))
    whole = n / n.groupby(n.index.year).transform(lambda s: s.mode().max())
    assert not np.allclose(whole.iloc[:9], QF.mode_share(n).iloc[:9])                                   # мода по всему году (40) ≠ PIT-мода на ранних днях (53)


def test_u_components_and_norm_on_closed_rows_only():
    rng = np.random.default_rng(2)
    Qm = rng.normal(size=(300, 8))
    tr = np.arange(200)
    u = QF.u_values(Qm, tr)
    C = QF.components(Qm)
    assert np.allclose(C[:, 1], np.abs(Qm[:, 5])) and np.allclose(C[:, 0], Qm[:, 2]) and np.allclose(C[:, 2], Qm[:, 0]) and np.allclose(C[:, 3], Qm[:, 1])
    assert abs(u[tr].mean()) < 1e-12 and abs(u[200:].mean()) > 1e-6                                     # z на закрытых парах: среднее u на них 0
    Qm2 = Qm.copy()
    Qm2[200:] = rng.normal(size=(100, 8)) * 9
    assert np.array_equal(QF.u_values(Qm2, tr)[tr], u[tr]) and QF.u_threshold(QF.u_values(Qm2, tr), tr) == QF.u_threshold(u, tr)   # будущие строки не меняют нормировку и порог


def test_real_q_no_nan_on_panel_days():
    bars, daily = FE.load_inputs()
    q = QF.features(bars, daily)
    ix = daily.index[(daily.index >= "2015-07-07") & (daily.index <= "2026-02-26")]
    assert list(q.columns) == QF.Q_NAMES and len(ix) >= 2649 and not q.loc[ix].isna().any().any()
    assert (q["q8"] > 0).all() and (q["q7"] > 0).all() and (q["q4"] >= 0).all() and (q["q5"] >= 1 / 60).all()
