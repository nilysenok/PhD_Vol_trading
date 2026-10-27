"""Э1-а (Ж-031): приёмка движка перебора на синтетике. Каждая «мутация» делается в памяти и обязана менять исход."""
import json

import numpy as np
import pandas as pd
import pytest

from fastbt import sweep_stat as ss
from fastbt.sweep import Spec, Sweep, SignalCache
from fastbt.sweep_synth import SynthEvaluator

GRIDS = {"a": [3, 5, 10, 20, 40]}


def mk(ev=None, mode="oat", period="train", **kw):
    spec = Spec(obj={"kind": "Synth"}, base={"a": 5, "w": 1.0}, grids=GRIDS, period=period, mode=mode, **kw)
    return Sweep(spec, ev or SynthEvaluator())


def varies(sw):
    t = sw.run()["table"]
    return t[t.comm_pct == 0.035]["sharpe"].nunique() > 1


def test_param_changes_result(monkeypatch):
    assert varies(mk())
    orig = SynthEvaluator.signals
    monkeypatch.setattr(SynthEvaluator, "signals", lambda self, sp: orig(self, {"a": 5}))
    assert not varies(mk())          # мутация: параметр сигнала игнорируется -> тест видит


def test_param_pnl_side_changes_result():
    sw = mk()
    a = sw.objective({"a": 5, "w": 1.0})
    assert sw.objective({"a": 5, "w": 2.0}) != a


def test_validation_not_in_objective():
    a, b = mk(SynthEvaluator(late=1.0)), mk(SynthEvaluator(late=6.0))
    assert a.objective({"a": 5, "w": 1.0}) == b.objective({"a": 5, "w": 1.0})
    assert a.run()["table"].equals(b.run()["table"])
    with pytest.raises(PermissionError):
        a._df({"a": 5, "w": 1.0}, 0.0004, (2018, 2020))


def test_mixing_2020_into_train_changes_objective():
    """Мутация: 2020 подмешан в настройку -> цель зависит от validation-данных (тест её ловит)."""
    per = ss.Periods(setup=(2018, 2020), control=(2019,), validation=(2021,))
    obj = [mk(SynthEvaluator(late=k), periods=per)._df({"a": 5, "w": 1.0}, 0.0004, (2018, 2020)) for k in (1, 6)]
    assert not obj[0].equals(obj[1])
    with pytest.raises(ValueError):
        ss.Periods(setup=(2018, 2020), validation=(2020, 2021))


def test_periods_disjoint():
    p = ss.Periods()
    assert p.train == (2018, 2019) and p.validation == (2020, 2021)
    assert not set(p.setup) & set(p.control) and not set(p.train) & set(p.validation)
    for bad in (dict(control=(2018,)), dict(validation=(2019, 2020)), dict(validation=(2021, 2022)),
                dict(setup=(2020,), validation=(2018,))):
        with pytest.raises(ValueError):
            ss.Periods(**bad)
    with pytest.raises(ValueError):
        Spec(obj={}, base={}, grids={}, period="validation", mode="oat")


def test_commission_both_sides():
    sk = pytest.importorskip("fastbt.sweep_sk")
    dates = pd.bdate_range("2019-03-04", periods=6)
    first = np.array([True] + [False] * 5)
    for pos, n_sides in (([0, 1, 1, 0, 0, 0], 2), ([0, 0, 0, 0, 1, 1], 2), ([1, -1, 0, 0, 0, 0], 4)):
        eng = sk._Mini(dates, np.arange(6), first, np.zeros(6))
        c = 0.0004
        net = eng.port(np.array(pos, float), comm=c)["r"].sum()
        assert net == pytest.approx(-n_sides * c, abs=1e-12)    # вход + выход; на баре открытия в конце окна — плата за выход
        assert eng.port(np.array(pos, float), comm=0.0)["r"].sum() == 0.0


def test_commission_lowers_sharpe_sweep():
    t = mk().run()["table"]
    lo, hi = t[t.comm_pct == 0.035].sharpe.values, t[t.comm_pct == 0.04].sharpe.values
    assert (hi < lo).all()


def test_signal_cache_separate_from_pnl():
    ev = SynthEvaluator()
    sw = mk(ev)
    sw.run()
    assert ev.signal_calls == 5                 # 5 значений a; две комиссии и w не пересчитывают сигналы
    sw.objective({"a": 5, "w": 2.0})
    assert ev.signal_calls == 5


def test_cache_key_depends_on_params_series_code():
    base = ("o", {"a": 5}, "d", "c")
    k = SignalCache.key(*base)
    assert k == SignalCache.key(*base)
    assert SignalCache.key("o", {"a": 6}, "d", "c") != k
    assert SignalCache.key("o", {"a": 5}, "d2", "c") != k
    assert SignalCache.key("o", {"a": 5}, "d", "c2") != k
    assert SynthEvaluator(seed=1).series_digest() != SynthEvaluator(seed=2).series_digest()


def test_reopen_validation_needs_note_and_doubles_n(tmp_path):
    out = str(tmp_path)
    tr = Spec(obj={"kind": "Synth"}, base={"a": 5, "w": 1.0}, grids=GRIDS, out=out)
    n_train = Sweep(tr, SynthEvaluator()).run()["N"]
    ledger0 = json.loads((tmp_path / "ledger.json").read_text())

    def val(note=None):
        return Sweep(Spec(obj={"kind": "Synth"}, base={"a": 5, "w": 1.0}, grids=GRIDS, period="validation",
                          mode="validation", points=[{"a": 10}], out=out, reopen_note=note), SynthEvaluator())
    r1 = val().run()
    assert r1["N"] == n_train and r1["table"]["dsr"].notna().all()
    assert json.loads((tmp_path / "validation_flag.json").read_text())["opens"][0]["text"].startswith("validation открыта: ")
    with pytest.raises(PermissionError):
        val().run()                                   # повторно без записи об изменении
    r2 = val("сменили порог по записи").run()
    assert r2["N"] == 2 * n_train                     # удвоение N
    assert json.loads((tmp_path / "ledger.json").read_text()) == ledger0     # validation в пробы не пишется
