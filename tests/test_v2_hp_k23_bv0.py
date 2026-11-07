"""У-18 (GATES_PANEL.md, Р-1 находка 2): день с BV = 0 (ненулевая только одна 10-минутная доходность) даёт ln BV = −inf в замороженном `s2.features.build`;
обёртка `hp_k23_world.build_world` обязана заменить нефинитное значение пропуском (NaN), конечные оставить побитно. Мутация «замена снята» — тест падает.
Синтетика, данных проекта не читает."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s2 import features as ft
from models_v2.s4 import mkt
from models_v2.s10 import hp_k23 as K, hp_k23_world as KW, hp_synth as SS
from models_v2.s4.fit import HP

DAY = 500
I2 = {Y: HP(2, 30.0, 2.0, 0.1, 0.8, 0.5, 40) for Y in KW.YEARS}


@pytest.fixture(scope="module")
def sw():
    bars, imo, base = SS.world(900, ("A", "B", "C"))
    return bars, imo, base


def freeze_day(b: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Все бары дня после первого стоят на цене закрытия первого: ненулевая доходность одна -> BV = 0 при RV > 0 (как RGSS 26.01.2024)."""
    b = b.copy()
    m = (b["begin"].dt.normalize() == day).to_numpy()
    idx = np.flatnonzero(m)
    c0 = float(b["close"].iloc[idx[0]])
    for col in ("open", "close", "high", "low"):
        b.iloc[idx[1:], b.columns.get_loc(col)] = c0
    b.iloc[idx[0], b.columns.get_loc("high")] = max(float(b["open"].iloc[idx[0]]), c0)
    b.iloc[idx[0], b.columns.get_loc("low")] = min(float(b["open"].iloc[idx[0]]), c0)
    return b


def test_syntetic_day_bv0_syroj_dizajn_daet_minus_inf_a_mir_propusk(sw):
    bars, imo, base = sw
    day = base[DAY]
    bad = freeze_day(bars["A"], day)
    F = ft.build(bad)
    k = int(np.flatnonzero(F.dates == np.datetime64(day))[0])
    assert np.isneginf(F.X[k, KW.LN_BV]), "синтетика не воспроизвела BV = 0 (сырой дизайн должен дать −inf)"
    w = KW.build_world(bad, imo, [mkt.paper(bars[t]) for t in ("A", "B", "C")])
    assert np.isnan(w["F"].X[k, KW.LN_BV]) and not np.isinf(w["F"].X).any() and not np.isinf(w["dz0"].X).any()


def test_konechnye_znacheniya_pobitno_prezhnie(sw):
    bars, imo, _ = sw
    w = KW.build_world(bars["A"], imo, [mkt.paper(bars[t]) for t in ("A", "B", "C")])
    raw = ft.build(bars["A"])
    assert np.isfinite(raw.X[:, KW.LN_BV][~np.isnan(raw.X[:, KW.LN_BV])]).all()
    assert np.array_equal(w["F"].X, raw.X, equal_nan=True)


def test_mutaciya_zamena_snyata_lovitsya(sw, monkeypatch):
    """Мутант: `_bv_to_nan` возвращает столбец как есть -> −inf остаётся в F и в дизайне K2 (проверка из первого теста падает)."""
    bars, imo, base = sw
    monkeypatch.setattr(KW, "_bv_to_nan", lambda c: c)
    w = KW.build_world(freeze_day(bars["A"], base[DAY]), imo, [mkt.paper(bars[t]) for t in ("A", "B", "C")])
    assert np.isinf(w["F"].X).any()


def test_boevoj_put_k2_k3_konechny_na_dne_bv0(sw):
    bars, imo, base = sw
    pool = [mkt.paper(bars[t]) for t in ("A", "B", "C")]
    bad = freeze_day(bars["A"], base[DAY])
    w = KW.build_world(bad, imo, pool)
    g = w["grid"]
    rows = g[::40]
    k = int(np.flatnonzero(w["F"].dates == np.datetime64(base[DAY]))[0])
    rows = np.unique(np.r_[rows, g[np.searchsorted(g, k)]])
    fr = K.KC.forecast_frame(w, I2, rows, 2)
    ok = fr["hp_year"].notna()
    assert ok.sum() > 5 and np.isfinite(fr.loc[ok, ["sigma2_k2", "sigma2_k3"]]).all().all()
