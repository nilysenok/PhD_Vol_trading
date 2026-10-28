# -*- coding: utf-8 -*-
"""К-в1: периоды В1 (train 2016–2019, прогрев 2015). Синтетика, данных и счёта нет.
Умолчание (train 2018–2019) = старый путь побайтно; мутации: целевая по 2 годам из 4, прогрев в метриках, чтение ≥ 2020."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastbt import etap1_data as D
from fastbt import etap1_eval as E
from fastbt import etap1_poisk_vorota as V
from fastbt import sweep_drivers as sd

Y4, Y2 = (2016, 2017, 2018, 2019), (2018, 2019)


def _res(msr: dict, sdelok: dict | None = None, n: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return dict(msr_by_year=msr, sdelok_y=sdelok or {y: 40 for y in msr}, vyrozhdena=False,
                excess_d={y: list(rng.normal(0, 0.01, n)) for y in msr})


def _ledger(period: str, years: tuple, k: int = 6) -> V.Ledger:
    led = V.Ledger(lambda p, per: _res({y: 0.1 * (p["i"] + 1) for y in years}, seed=p["i"]))
    for i in range(k):
        led({"i": i}, period)
    return led


# --- умолчание = старый путь побайтно -----------------------------------------------------------------------------
def test_umolchaniya_ne_izmenilis():
    assert E.PERIODS["train"] == (2018, 2019) and D.START == pd.Timestamp("2017-01-01") and V.NBLK == 8
    for per, (a, b) in E.PERIODS.items():
        assert V.PERIOD_YEARS[per] == tuple(range(a, b + 1)) == E.years_of(per)
    assert E.PERIODS["train4"] == (2016, 2019) and 2015 not in V.PERIOD_YEARS["train4"]    # 2015 — только прогрев


def test_golden_train_pobaytno_staryy_put():
    led = _ledger("train", Y2)
    old = np.column_stack([np.r_[np.asarray(r["excess_d"][2018]), np.asarray(r["excess_d"][2019])]
                          for (_, per), (_, r) in led.cache.items() if per == "train"])
    M = V.matrix(led)
    assert M.tobytes() == old.tobytes() and V.matrix(led, "train").tobytes() == old.tobytes()
    assert V.pbo(M) == sd.pbo_cscv(M.tolist(), 8)["pbo"] == V.pbo(M, 8)
    r = _res({2018: 0.3, 2019: -0.1})
    assert V.obj(r) == V.obj(r, "train") == -0.1 and repr(V.po_godam(r)) == repr(V.po_godam(r, "train"))
    assert sorted(V.po_godam(r)) == sorted(f"{n}_{y}" for n in ("sharpe", "msr", "storon", "sdelok") for y in Y2)


def test_split_years_train_bez_progreva_pobaytno():
    x, yrs = np.arange(10.0), np.array([2018] * 5 + [2019] * 5)
    xs, yy = E.split_years(x, yrs, E.years_of("train"))
    assert xs.tobytes() == x.tobytes() and sorted(yy) == [2018, 2019] and yy[2019].tobytes() == x[5:].tobytes()


# --- В1: четыре года ------------------------------------------------------------------------------------------------
def test_matrix_po_godam_perioda():
    led = _ledger("train4", Y4)
    M = V.matrix(led, "train4")
    assert M.shape == (80, 6) and V.matrix(led).shape == (0, 0)             # ключ ledger = период; train пуст
    r = led({"i": 0}, "train4")
    assert np.array_equal(M[:, 0], np.concatenate([r["excess_d"][y] for y in Y4]))
    assert V.matrix(_ledger("train4", Y2), "train4").shape == (0, 0)        # не хватает годов — столбца нет


def test_celevaya_min_po_chetyrem_godam():
    r = _res({2016: -0.2, 2017: 0.5, 2018: 0.4, 2019: 0.3})
    assert V.obj(r, "train4") == -0.2
    assert np.isnan(V.obj(_res({2018: 0.4, 2019: 0.3}), "train4"))          # 2 года из 4 — не целевая В1
    assert np.isnan(V.obj(_res({2016: 0.1, 2017: float("nan"), 2018: 0.4, 2019: 0.3}), "train4"))


def test_mutaciya_celevaya_po_dvum_godam_lovitsya(monkeypatch):
    r = _res({2016: -0.2, 2017: 0.5, 2018: 0.4, 2019: 0.3})
    good = V.obj(r, "train4")
    monkeypatch.setitem(V.PERIOD_YEARS, "train4", Y2)                       # мутация: целевая по 2 годам
    assert V.obj(r, "train4") != good
    monkeypatch.undo()
    assert V.obj(r, "train4") == good


def test_vorota_sdelok_v_kazhdom_godu():
    ok = {2016: 40, 2017: 31, 2018: 40, 2019: 30}
    assert V.sdelok_ok(ok, "train4")
    for bad in ({**ok, 2016: 29}, {**ok, 2017: float("nan")}, {y: v for y, v in ok.items() if y != 2016}, {}):
        assert not V.sdelok_ok(bad, "train4")
    assert not V.sdelok_ok({2018: 40, 2019: 40}, "train4") and V.sdelok_ok({2018: 40, 2019: 40}, "train")


def test_gates_i_dopustima_berut_period():
    ax = {"a": [0, 1, 2], "k_s": [1.0, 2.0, 3.0], "R": [1.0, 2.0, 3.0], "tau": [1, 3, 5]}
    pt = {"a": 1, "k_s": 2.0, "R": 2.0, "tau": 3}
    led = V.Ledger(lambda p, per: _res({y: 0.2 for y in V.PERIOD_YEARS[per]}, {y: 5 if y == 2016 else 40 for y in V.PERIOD_YEARS[per]}))
    assert V.gates(led, pt, ax, ["a"], 0.0, lambda p: True, "train")["G2_sdelki"]
    assert not V.gates(led, pt, ax, ["a"], 0.0, lambda p: True, "train4")["G2_sdelki"]     # 2016: 5 < 30
    yak = {**pt, "a": 0}
    assert V.dopustima(led, pt, yak, "train") and not V.dopustima(led, pt, yak, "train4")


def test_nblk_parametr():
    M = np.random.default_rng(1).normal(0, 0.01, (48, 5))
    assert sd.pbo_cscv(M.tolist(), 12)["n_splits"] == 924
    assert V.pbo(M, 12) == sd.pbo_cscv(M.tolist(), 12)["pbo"] and V.pbo(M) == V.pbo(M, 8)
    assert sd.pbo_cscv(M.tolist(), 8)["n_splits"] == 70


# --- ctrl2019 / Моррис на одном году: старое поведение побайтно ---------------------------------------------------
def _legacy_obj_morris(res: dict) -> float:
    v = [float(x) for x in (res.get("msr_by_year") or {}).values()]
    o = min(v) if v and all(np.isfinite(x) for x in v) else float("nan")
    return o if np.isfinite(o) else 0.0


def test_obj_odin_god_kak_ranshe():
    r = {"msr_by_year": {2019: 0.7}}
    assert V.obj(r) == V.obj(r, "train") == 0.7 and V.obj_morris(r) == 0.7 == _legacy_obj_morris(r)
    assert np.isnan(V.obj(r, "train4")) and V.obj_morris(r, "train4") == 0.0       # В1 с одним годом — нейтрально


def test_morris_2019_effekty_nenulevye_i_ravny_rodителю(tmp_path, monkeypatch):
    from test_etap1_poisk import csv, run
    _, w = run(tmp_path / "new", morris_2019=True)
    m = csv(tmp_path / "new", "morris")
    assert set(w.calls[:20]) == {"ctrl2019"} and m[m.os == "a"].mu_star.item() > 0 and (m.mu_star > 0).any()
    import fastbt.etap1_poisk as P
    monkeypatch.setattr(P, "obj_morris", _legacy_obj_morris)
    run(tmp_path / "old", morris_2019=True)
    old = csv(tmp_path / "old", "morris")
    assert m[["os", "rang", "mu_star", "sigma", "n"]].to_csv() == old[["os", "rang", "mu_star", "sigma", "n"]].to_csv()


# --- границы окна: первый день метрик, h по прогреву -----------------------------------------------------------------
def _dates():
    return pd.bdate_range("2014-06-09", "2019-12-31").to_numpy("datetime64[ns]")


def test_window_idx_default_i_v1():
    from fastbt import etap1
    d = _dates()
    s, a, i = etap1.window_idx(d, D.START, 2018, "2020-01-01")
    assert d[s] >= np.datetime64("2017-01-01") > d[s - 1] and d[a] >= np.datetime64("2018-01-01") > d[a - 1] and i == len(d)
    s, a, i = etap1.window_idx(d, D.START_V1, 2016, "2020-01-01")
    assert d[s] >= np.datetime64("2015-01-01") > d[s - 1] and d[a] >= np.datetime64("2016-01-01") > d[a - 1]
    assert str(d[a])[:10] == "2016-01-01" and str(d[s - 1])[:4] == "2014"        # первый торговый день 2016; 2015 — прогрев


def _ev_h(a_idx: int):
    """Evaluator без данных: позиции — сделки по 2 дня в прогреве (до 40), по 10 дней в train (после 40)."""
    from types import SimpleNamespace
    pos = np.zeros(120)
    for s in range(0, 40, 8):
        pos[s:s + 2] = 1.0
    for s in range(40, 120, 10):
        pos[s:s + 8] = 1.0
    ev = object.__new__(E.Evaluator)
    ev.inp = SimpleNamespace(meta={"T": dict(s0=0, a18=a_idx, i1=119)})
    ev._positions = lambda *a: pos
    return ev


def test_h_po_progrevu_a_ne_po_train():
    assert _ev_h(40)._h({}, 0.0) == 2.0
    assert _ev_h(120)._h({}, 0.0) > 2.0                                  # мутация: медиана захватывает train — ловится


# --- прогрев 2015 не входит в метрики -------------------------------------------------------------------------------
def test_progrev_2015_ne_vkhodit_v_metriki():
    x, yrs = np.arange(12.0), np.repeat([2015, 2016, 2017, 2018], 3)
    xs, yy = E.split_years(x, yrs, E.years_of("train4"))
    assert sorted(yy) == [2016, 2017, 2018] and len(xs) == 9 and 2015 not in yy and xs.min() == 3.0


def test_mutaciya_progrev_v_metrikakh_lovitsya(monkeypatch):
    x, yrs = np.arange(12.0), np.repeat([2015, 2016, 2017, 2018], 3)
    def invariant():
        xs, yy = E.split_years(x, yrs, E.years_of("train4"))
        assert 2015 not in yy and len(xs) == 9
    invariant()
    monkeypatch.setattr(E, "years_of", lambda p: (2015, 2016, 2017, 2018, 2019))      # мутация: прогрев входит
    with pytest.raises(AssertionError):
        invariant()


# --- граница периода: чтение ≥ 2020 в train В1 --------------------------------------------------------------------
class _Rec:
    calls: list = []

    @classmethod
    def load(cls, t, start, end, window, src=None):
        cls.calls.append((start, end))
        return pd.DataFrame(dict(date=pd.to_datetime([]), open=[], high=[], low=[], close=[], pusto=pd.Series([], dtype=bool)))


def _build(end: str, **kw):
    _Rec.calls = []
    return D.build_days("T", np.array([np.datetime64("2019-12-30")]), np.array([1.0]), 0, 0, pd.Timestamp(end), **kw)


def test_start_v1_i_umolchanie(monkeypatch):
    monkeypatch.setattr(D, "load_bars5m", _Rec.load)
    _build("2020-01-01", last_year=2019)
    assert _Rec.calls == [("2017-01-01", "2019-12-31")]                      # умолчание START — как раньше
    _build("2020-01-01", start=D.START_V1, last_year=2019)
    assert _Rec.calls == [("2015-01-01", "2019-12-31")]


def test_chtenie_za_2020_v_train_v1_lovitsya(monkeypatch):
    monkeypatch.setattr(D, "load_bars5m", _Rec.load)
    for end in ("2020-01-02", "2021-01-01"):
        with pytest.raises(PermissionError):
            _build(end, start=D.START_V1, last_year=2019)
    with pytest.raises(PermissionError):
        _build("2022-01-02")                                                 # ≥ LIMIT и без last_year
    assert _Rec.calls == []                                                  # до чтения не дошло

    def invariant(**kw):                                                     # «ни одного чтения после 2019» в train В1
        _build("2021-01-01", start=D.START_V1, **kw)
        assert all(e <= "2019-12-31" for _, e in _Rec.calls)
    with pytest.raises(PermissionError):
        invariant(last_year=2019)
    with pytest.raises(AssertionError):                                      # мутация: last_year потерян — чтение 2020–2021
        invariant()
