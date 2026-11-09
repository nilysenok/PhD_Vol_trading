"""V6-3, G30: признаки шока и восстановления ρ1…ρ6. Известные ответы на синтетических днях (падение утром с восстановлением; стабильное утро и падение перед закрытием; день без отрицательных баров; серия всплесков), счёт «руками».
Правило слияния с Σ: |корреляция| на закрытых парах 2015-07-07 … 2016-12-29, порог 0,8 (мутация «окно — вся выборка» ловится). Мутация «бары дня t⁺ в R дня d» ловится через боевой `features` (замена баров после d не меняет R дней ≤ d)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s13 import v6_3_merge as MG, v6_3_r as RF

ONES = np.ones(10)


def test_morning_fall_with_recovery_known_answer():
    r = np.array([0.001, -0.01, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001])
    rho, fl = RF.day_rho(r, ONES)
    rv = float((r * r).sum())
    assert np.isclose(rho[0], 0.8) and np.isclose(rho[1], 0.013 / 0.009)
    assert np.isclose(rho[2], np.log((23e-6 / 8) / (1.01e-4 / 2))) and np.isclose(rho[3], 0.8)
    assert np.isclose(rho[4], np.log(1e-12 * rv / 1e-4)) and rho[5] == 0 and fl == (False, False, False, False, True)         # RS⁻ второй половины пуст -> ε


def test_calm_morning_fall_before_close_known_answer():
    r = np.array([0.001] * 8 + [-0.02, 0.001])
    rho, fl = RF.day_rho(r, ONES)
    assert np.isclose(rho[0], 0.1) and np.isclose(rho[3], 0.1)
    assert np.isclose(rho[1], 0.001 / 0.012)                   # P_n − P_k* = −0,011 + 0,012; P_0 − min P = 0,012 (в единицах 1/√RV)
    assert np.isclose(rho[2], np.log(1e-6 / ((8e-6 + 4e-4) / 9))) and rho[2] < 0           # после шока (1 бар) спокойнее, чем до: ρ3 < 0
    assert rho[4] > 0 and fl == (False, False, False, False, True)            # всё RS⁻ во второй половине: первая половина пуста -> ε
    assert rho[5] == 1.0                                       # |−0,02| > 3·√(RV/n) ≈ 0,0192


def test_day_without_negative_bars_edge_cases():
    r = np.array([0.001, 0.002, 0.003, 0.004, 0.001, 0.002, 0.003, 0.004])
    rho, fl = RF.day_rho(r, np.ones(8))
    assert fl[0] is True and rho[1] == 0.0 and rho[4] == 0.0 and fl[4] is True         # нет падения ниже открытия: ρ2 = 0; обе половины пусты: ε/ε -> ρ5 = 0
    assert np.isclose(rho[0], 7 / 8)                                                     # шок — первый минимум: бар 1 (r = 0,001)
    last = np.array([0.002, 0.001, 0.003, -0.01])
    rho2, fl2 = RF.day_rho(last, np.ones(4))
    assert rho2[0] == 0.0 and fl2[1] is True and rho2[2] == 0.0 and np.isclose(rho2[3], 0.0)       # шок на последнем баре: «после» пусто -> ρ3 = 0, ρ4 = 0


def test_burst_series_counts_spikes():
    r = np.full(50, 1e-4)
    r[[5, 17, 30, 31, 44]] = [0.01, -0.01, 0.01, -0.01, 0.01]
    rho, _ = RF.day_rho(r, np.ones(50))
    assert rho[5] == 5.0
    r2 = np.full(50, 1e-4)
    r2[[5, 6, 7, 8, 9, 10]] = 0.01                         # шесть равных всплесков: порог 3·√(RV/n) выше 0,01 — ни один не «всплеск»
    assert RF.day_rho(r2, np.ones(50))[0][5] == 0.0


def test_zero_volume_day_falls_back_to_bar_share():
    rho, fl = RF.day_rho(np.array([0.001, -0.01, 0.002, 0.001]), np.zeros(4))
    assert fl[3] is True and np.isclose(rho[3], rho[0])


def make_bars(seed: int = 3, days: int = 70, nb: int = 53) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, px = [], 100.0
    for k in range(days):
        day = pd.Timestamp("2016-01-04") + pd.offsets.BDay(k)
        for j in range(nb):
            o = px
            px = px * np.exp(rng.normal(0, 0.002))
            rows.append((day + pd.Timedelta(minutes=600 + 10 * j), o, max(o, px), min(o, px), px, float(rng.integers(100, 1000))))
    return pd.DataFrame(rows, columns=["begin", "open", "high", "low", "close", "volume"])


def daily_of(bars):
    return D.build_daily(bars, index_days=None)[0]


def corrupt_after(bars, day):
    b = bars.copy()
    late = b["begin"].dt.normalize() > day
    rng = np.random.default_rng(99)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, int(late.sum()))))
    b.loc[late, "close"], b.loc[late, "open"] = px, np.r_[px[:1], px[:-1]]
    b.loc[late, "volume"] = rng.integers(1, 5000, int(late.sum())).astype(float)
    return b


def leaks(day_idx: int = 45) -> bool:
    bars = make_bars()
    day = daily_of(bars).index[day_idx]
    b2 = corrupt_after(bars, day)
    a, c = RF.features(bars, daily_of(bars)), RF.features(b2, daily_of(b2))
    return not np.allclose(a[a.index <= day].to_numpy(), c[c.index <= day].to_numpy(), equal_nan=True, rtol=0, atol=1e-12)


def test_r_uses_only_bars_of_day_d_and_mutation_t_plus_caught(monkeypatch):
    assert not leaks()
    orig = RF.day_table
    monkeypatch.setattr(RF, "day_table", lambda ar: orig(ar).shift(-1))             # МУТАЦИЯ: бары дня t⁺ в R дня d
    assert leaks()


def test_features_have_no_nan_on_synthetic_panel():
    bars = make_bars()
    f = RF.features(bars, daily_of(bars))
    assert not f[RF.RHO_NAMES].isna().any().any() and list(f.columns) == RF.RHO_NAMES + RF.FLAG_NAMES


def synth_merge(shift_after: bool = False):
    idx = pd.bdate_range("2015-07-07", "2029-12-31")
    rng = np.random.default_rng(0)
    sig = pd.DataFrame(rng.normal(size=(len(idx), 3)), index=idx, columns=["a", "b", "c"])
    R = pd.DataFrame(rng.normal(size=(len(idx), 6)), index=idx, columns=RF.RHO_NAMES)
    R["rho1"] = sig["a"] + 0.2 * rng.normal(size=len(idx))                          # дубль признака Σ: |корр.| ≈ 0,98
    if shift_after:
        R["rho2"] = np.where(idx > "2016-12-29", sig["b"], R["rho2"])               # совпадение только ПОСЛЕ закрытого окна
    return R, sig


def test_merge_excludes_duplicate_above_08_only_on_closed_window():
    R, sig = synth_merge()
    t, kept = MG.merge_table(R, sig)
    assert "rho1" not in kept and set(kept) == set(RF.RHO_NAMES) - {"rho1"} and len(t) == 18
    assert t.loc[t["rho"] == "rho1", "abs_corr"].max() > 0.8 and (t["n"] == len(pd.bdate_range("2015-07-07", "2016-12-29"))).all()
    R2, sig2 = synth_merge(True)
    _, kept2 = MG.merge_table(R2, sig2)
    assert "rho2" in kept2                                                              # окно закрытых пар — только до 2016-12-29
    _, mut = MG.merge_table(R2, sig2, win=("2015-07-07", "2029-12-31"))                 # МУТАЦИЯ: окно — вся выборка (включая будущие пары)
    assert "rho2" not in mut


def test_merge_threshold_is_strict_08():
    idx = pd.bdate_range("2015-07-07", "2016-12-29")
    rng = np.random.default_rng(2)
    xs = pd.Series(rng.normal(size=len(idx)), index=idx)
    R = pd.DataFrame({n: rng.normal(size=len(idx)) for n in RF.RHO_NAMES}, index=idx)
    R["rho3"] = xs + 0.75 * rng.normal(size=len(idx))                # |корр.| с x ≈ 0,8
    c = abs(float(R["rho3"].corr(xs)))
    assert 0.7 < c < 0.9
    assert "rho3" in MG.merge_table(R, pd.DataFrame({"x": xs}), thr=c)[1]                 # строго «больше»: на самом пороге не исключается
    assert "rho3" not in MG.merge_table(R, pd.DataFrame({"x": xs}), thr=c - 1e-9)[1]
    assert MG.THRESHOLD == 0.8 and MG.WIN == ("2015-07-07", "2016-12-29")                   # параметры плана
