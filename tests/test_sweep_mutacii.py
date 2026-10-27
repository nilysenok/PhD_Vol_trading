"""Э1-а (Ж-034): подложенные ошибки в оценщике в памяти (monkeypatch); дерево `fastbt/` не меняется.
Для каждой: детектор на правильном коде True (тесты в engine/drivers), на мутации — False / сработала защита движка.
Мутации модуля владельца — `test_sweep_modul_mut.py`. Именование: test_mut_<N>_<что подложено>."""
import json

import numpy as np
import pandas as pd
import pytest

from fastbt import metriki_e as me
from fastbt import sweep_stat as ss
from fastbt.sweep import Sweep
from fastbt.sweep_spec import SignalCache
from fastbt.sweep_synth import SynthEvaluator
from test_sweep_drivers import (ok_cache_keys, ok_cache_miss_end_to_end, ok_cache_separate_from_pnl,
                                ok_dsr_from_ledger, ok_ledger_only_selection_comm)
from test_sweep_engine import (Tiny, ok_2022_closed, ok_commission_lowers_sharpe, ok_commission_two_sides, ok_ddof1,
                               ok_end_cut, ok_param_pnl, ok_param_signal, ok_periods_raise, ok_reopen_doubles_n,
                               ok_validation_closed, ok_validation_independent)


def caught(fn) -> bool:
    """Мутация поймана: свойство нарушено (False) либо сработала защита движка (PermissionError)."""
    try:
        return not fn()
    except PermissionError:
        return True


def _open_guard(monkeypatch):
    orig = Sweep.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self._val_open = True
    monkeypatch.setattr(Sweep, "__init__", init)


def test_mut_01_param_signala_ignoriruetsya(monkeypatch):
    assert ok_param_signal()
    orig = SynthEvaluator.signals
    monkeypatch.setattr(SynthEvaluator, "signals", lambda self, sp: orig(self, {"a": 5}))
    assert not ok_param_signal()


def test_mut_02_param_pnl_ignoriruetsya(monkeypatch):
    assert ok_param_pnl()
    orig = SynthEvaluator.pnl
    monkeypatch.setattr(SynthEvaluator, "pnl", lambda self, s, p, c, e: orig(self, s, {**p, "w": 1.0}, c, e))
    assert not ok_param_pnl()


@pytest.mark.parametrize("guard_off", [False, True])
def test_mut_03_2020_podmeshan_v_train(monkeypatch, guard_off):
    assert ok_validation_independent()
    monkeypatch.setattr(ss.Periods, "train", property(lambda self: tuple(self.setup) + tuple(self.control) + (2020,)))
    if guard_off:
        _open_guard(monkeypatch)      # без защиты движка строка меняется от доходностей 2020 — детектор это видит
    assert caught(ok_validation_independent)
    if guard_off:
        assert not ok_validation_independent()


def test_mut_04_zashchita_validation_snyata(monkeypatch):
    assert ok_validation_closed() and ok_periods_raise(validation=(2019, 2020))
    monkeypatch.setattr(ss.Periods, "__post_init__", lambda self: None)
    assert not ok_periods_raise(validation=(2019, 2020))          # периоды пересекаются, а ошибки нет
    _open_guard(monkeypatch)
    assert not ok_validation_closed()


def test_mut_05_granitsa_2022_sdvinuta(monkeypatch):
    assert ok_2022_closed()
    monkeypatch.setattr(ss, "END_ALLOWED", 2022)
    assert not ok_2022_closed()


def test_mut_06_granitsa_train_zakhvatyvaet_2020(monkeypatch):
    assert ok_end_cut()
    monkeypatch.setattr(ss.Periods, "years", lambda self, name: (2018, 2019, 2020) if name == "train" else (2020, 2021))
    assert caught(ok_end_cut)
    _open_guard(monkeypatch)
    assert not ok_end_cut()


def test_mut_07_komissiya_odna_storona_v_dvizhke(monkeypatch):
    assert ok_commission_two_sides()
    orig = Sweep.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.comm = [c / 2 for c in self.comm]                    # % -> доля потеряна вдвое
    monkeypatch.setattr(Sweep, "__init__", init)
    assert not ok_commission_two_sides()


def test_mut_08_komissiya_tolko_na_vkhode(monkeypatch):
    orig = Tiny.pnl
    monkeypatch.setattr(Tiny, "pnl", lambda self, sigs, p, comm, end: orig(self, sigs, p, comm / 2, end))
    assert not ok_commission_two_sides()                          # выход бесплатен: 2 стороны из 4


def test_mut_09_komissiya_sk_odna_storona(monkeypatch):
    sk = pytest.importorskip("fastbt.sweep_sk")
    dates = pd.bdate_range("2019-03-04", periods=6)

    def net(eng):
        return eng.port(np.array([0, 1, 1, 0, 0, 0], float), comm=0.0004)["r"].sum()
    eng = sk._Mini(dates, np.arange(6), np.array([True] + [False] * 5), np.zeros(6))
    assert net(eng) == pytest.approx(-2 * 0.0004, abs=1e-12)    # вход + выход
    orig = sk._Mini.port
    monkeypatch.setattr(sk._Mini, "port", lambda self, pos, comm=0.0, **k: orig(self, pos, comm=comm / 2, **k))
    assert net(eng) != pytest.approx(-2 * 0.0004, abs=1e-12)


def test_mut_10_komissiya_argument_ignoriruetsya(monkeypatch):
    assert ok_commission_lowers_sharpe()
    orig = Sweep.evaluate
    monkeypatch.setattr(Sweep, "evaluate", lambda self, params, period="train", comm_pct=None: orig(self, params, period, None))
    assert not ok_commission_lowers_sharpe()


def test_mut_12_ddof0_v_sharpe(monkeypatch):
    assert ok_ddof1()
    monkeypatch.setattr(me, "sharpe", lambda x, *a, **k: float(np.mean(x) / np.std(x, ddof=0) * np.sqrt(252)))
    assert not ok_ddof1()


def test_mut_13_ledger_ne_nakaplivaet_probi(monkeypatch):
    assert ok_dsr_from_ledger()
    orig = Sweep.trials_sr
    monkeypatch.setattr(Sweep, "trials_sr", lambda self: orig(self)[-1:])          # N схлопнулось до 1
    assert not ok_dsr_from_ledger()


@pytest.mark.parametrize("drop", ["obj", "params", "digest", "code"])
def test_mut_16_klyuch_kesha_teryaet_vkhod(monkeypatch, drop):
    assert ok_cache_keys()
    orig = SignalCache.key
    cut = {"obj": lambda o, s, d, c: ("-", s, d, c), "params": lambda o, s, d, c: (o, {}, d, c),
           "digest": lambda o, s, d, c: (o, s, "-", c), "code": lambda o, s, d, c: (o, s, d, "-")}[drop]
    monkeypatch.setattr(SignalCache, "key", staticmethod(lambda *a: orig(*cut(*a))))
    assert not ok_cache_keys()


@pytest.mark.parametrize("kw,drop", [(dict(seed=1), "digest"), (dict(code="v2"), "code")])
def test_mut_17_kesh_ne_promakhivaetsya_pri_podmene(tmp_path, monkeypatch, kw, drop):
    assert ok_cache_miss_end_to_end(tmp_path / "ok", **kw)
    orig = SignalCache.key
    monkeypatch.setattr(SignalCache, "key", staticmethod(
        lambda o, s, d, c: orig(o, s, "-" if drop == "digest" else d, "-" if drop == "code" else c)))
    assert not ok_cache_miss_end_to_end(tmp_path / "bad", **kw)


def test_mut_18_kesh_signalov_zavisit_ot_pnl_parametra(monkeypatch):
    assert ok_cache_separate_from_pnl()
    monkeypatch.setattr(SynthEvaluator, "sig_names", {"a", "w"})
    assert not ok_cache_separate_from_pnl()


def test_mut_25_ledger_pishet_pri_lyuboy_komissii(monkeypatch):
    assert ok_ledger_only_selection_comm()
    orig = Sweep.evaluate

    def ev(self, params, period="train", comm_pct=None):
        r = orig(self, params, period, comm_pct)          # запись пробы на любом периоде и при любой комиссии
        self.ledger[json.dumps(sorted({**self.spec.base, **params}.items()), default=str)] = r["sr_d"]
        return r
    monkeypatch.setattr(Sweep, "evaluate", ev)
    assert not ok_ledger_only_selection_comm()


def test_mut_24_povtornoe_otkrytie_ne_udvaivaet_N(tmp_path, monkeypatch):
    assert ok_reopen_doubles_n(tmp_path / "ok")
    monkeypatch.setattr(Sweep, "n_mult", lambda self: 1)
    assert not ok_reopen_doubles_n(tmp_path / "bad")
