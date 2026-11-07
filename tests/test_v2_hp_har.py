"""P0, G4 (R2, R3): HAR-J K1 на синтетике через боевой путь + сверка SBER с Ж-214 на реальных входах (пропуск, если файлов нет).
Мутации: «HAR-J по полной выборке» (класс № 104) и «цель на календарный t+1 вместо следующего собственного дня» (gap)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2 import har as H
from models_v2.s10 import hp_daily as HD, hp_har as HH, hp_paths as P, hp_synth as SS

HOLES_ROWS = list(range(500, 508)) + [600, 700]        # 8 подряд идущих будних дней (разрыв > 5 сут.) и одиночные дыры


@pytest.fixture(scope="module")
def world():
    probe = SS.weekdays(900)
    skip = [probe[i] for i in HOLES_ROWS]
    bars, imo, cal = SS.world(900, ("A", "B"), skip={"A": skip})
    return bars["A"], skip


def test_cel_sledujushchij_sobstvennyj_den_i_gap(world):
    bars, skip = world
    d = HD.core(bars)
    k = HH.from_bars(bars)
    ix = d.index
    assert not (set(skip) & set(ix))                                            # дыры действительно в ряду бумаги
    assert (ix.get_indexer(pd.DatetimeIndex(k["target_date"])) == ix.get_indexer(pd.DatetimeIndex(k["d"])) + 1).all()
    assert np.array_equal(k["rv"].to_numpy(), d["rv"].reindex(pd.DatetimeIndex(k["target_date"])).to_numpy())
    assert (k["gap"] == ((k["target_date"] - k["d"]).dt.days > 5)).all() and k["gap"].sum() == 1   # одна пара с разрывом 8 будних дней
    assert (k["target_date"] - k["d"]).dt.days.max() >= 10


def test_mutaciya_celi_na_kalendarnyj_t_plus_1_lovitsya(world, monkeypatch):
    bars, _ = world
    d = HD.core(bars)

    def calendar_next(p, rows):
        td = p.dates[rows] + np.timedelta64(1, "D")
        return td, d["rv"].reindex(pd.DatetimeIndex(td)).to_numpy()
    monkeypatch.setattr(HH, "target_of", calendar_next)
    k = HH.from_bars(bars)
    ix = d.index
    own_next = (ix.get_indexer(pd.DatetimeIndex(k["target_date"])) == ix.get_indexer(pd.DatetimeIndex(k["d"])) + 1).all()
    assert not own_next                       # мутант: на пятницу цель — суббота (нет собственного дня), через дыру — несуществующий день


def _trunc_diff(bars: pd.DataFrame, d0: pd.Timestamp) -> tuple:
    a = HH.from_bars(bars).set_index("d")["sigma2"]
    b = HH.from_bars(HH.corrupt_after(bars, d0)).set_index("d")["sigma2"]
    assert a.index.equals(b.index) or len(a.loc[:d0]) == len(b.loc[:d0])
    return float(np.abs(a.loc[:d0] - b.loc[:d0]).max()), float(np.abs(a.loc[d0 + pd.Timedelta(days=1):] - b.loc[d0 + pd.Timedelta(days=1):]).max())


def test_k1_obrezka_buduschego_boevoj_put(world):
    bars, _ = world
    d0 = HD.core(bars).index[750]
    before, after = _trunc_diff(bars, d0)
    assert before == 0.0
    assert after > 0.0               # порча после d0 реально меняет прогнозы после d0 (тест не пустой)


def test_mutaciya_har_po_polnoj_vyborke_lovitsya(world, monkeypatch):
    """Класс № 104: оценка по всей выборке даёт прогноз на d, зависящий от баров после d."""
    bars, _ = world

    def full_sample(p, i):
        Xt, yt = H.pairs(p, len(p.rv) - 1, H.COLS_HARJ)
        return H.ols(Xt, yt)
    d0 = HD.core(bars).index[750]
    monkeypatch.setattr(HH, "fit_at", full_sample)
    before, _ = _trunc_diff(bars, d0)
    assert before > 0.0


def test_k1_znanyj_otvet_nezavisimyj_ols(world):
    """Независимая реализация: пары (s, s⁺), s⁺ <= d, один сегмент, lstsq по [1, ln RV, ln RV_w, ln RV_m, ln(1+J)]; σ̂² = exp(f + s²/2)."""
    bars, _ = world
    d = HD.core(bars)
    k = HH.from_bars(bars)
    i = len(k) // 2
    day = k["d"].iloc[i]
    r = d.index.get_loc(day)
    X = np.column_stack([np.ones(len(d)), np.log(d["rv"]), np.log(d["rv_w"]), np.log(d["rv_m"]), np.log1p(d["j"])])
    y = np.log(d["rv"].to_numpy())
    rows = [s for s in range(r) if d["seg"].iloc[s] == d["seg"].iloc[s + 1] and np.isfinite(X[s]).all()]
    beta = np.linalg.lstsq(X[rows], y[np.array(rows) + 1], rcond=None)[0]
    res = y[np.array(rows) + 1] - X[rows] @ beta
    s2 = (res ** 2).sum() / (len(rows) - 5)
    want = np.exp(X[r] @ beta + s2 / 2)
    assert k["sigma2"].iloc[i] == pytest.approx(want, rel=1e-9)
    assert k["n_pairs"].iloc[i] == len(rows) and (k["n_pairs"] >= P.MIN_PAIRS).all()


def test_sber_raven_zh214_pobitno():
    """Реальные входы: K1(SBER) = components.HAR, M10-величины = m10_daily Ж-214 (max_abs_diff = 0)."""
    if not (P.bars_path("SBER").exists() and P.COMP.exists() and P.M10_SBER.exists()):
        pytest.skip("нет реальных входов")
    imo = HD.read_bars("IMOEX")
    t = HD.ticker_daily(HD.read_bars("SBER"), imo)
    ref = pd.read_parquet(P.M10_SBER)
    r = HD.max_abs_diff(t, ref, [c for c in ref.columns if c in t.columns and c != "grid_d"])
    assert r["same_index"] and r["max"] == 0.0
    comp = pd.read_parquet(P.COMP)
    k = HH.series(t)
    m = k.merge(comp[["d", "target_date", "HAR"]], on=["d", "target_date"])
    assert len(m) == len(comp) and np.array_equal(m["sigma2"].to_numpy(), m["HAR"].to_numpy())
