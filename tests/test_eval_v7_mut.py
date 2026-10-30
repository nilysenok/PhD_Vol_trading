"""E1 v7, тестировщик #900: мутации через monkeypatch (ключ без семьи; S7 в сетке; S1 через v7-ветку с другим исполнением; sha панели) и пробелы."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import eval_v7
from fastbt.eval_v7 import EvaluatorV7
from fastbt.eval_v6 import EvaluatorV6
from fastbt.signals import v7_impuls
from fastbt.signals.interface import WARMUP, get_family, register_family
from test_eval_v7 import SP, _ev, _ist7, _td


def test_mut_klyuch_bez_semyi_lovitsya(monkeypatch):
    k = lambda s: _ev(EvaluatorV7, s)._fkey  # noqa: E731
    assert k("S10") != k("S11")
    monkeypatch.setattr(EvaluatorV7, "_k7", lambda self: ())
    assert k("S10") == k("S11")                                     # мутация: семьи различить нельзя — тест выше (!=) её роняет


def test_mut_s7_v_setke_lovitsya(monkeypatch):
    with pytest.raises(KeyError):
        get_family("S7", setka="v7")
    from fastbt.signals import interface
    monkeypatch.setitem(interface._REGISTRY_V7, "S7", lambda ist7: None)
    assert get_family("S7", setka="v7") is not None                 # мутация: S7 вошла в реестр — проверка KeyError выше её роняет


def test_mut_put_s1_cherez_v7_s_drugim_ispolneniem_lovitsya(monkeypatch):
    s6 = _ev(EvaluatorV6)._signals5(SP)["T"]
    orig = EvaluatorV6._signals5

    def sdvig(self, sp):
        u = orig(self, sp)
        for t in u:
            u[t].sigs = np.roll(u[t].sigs, 1)                        # исполнение на день позже
        return u

    monkeypatch.setattr(EvaluatorV6, "_signals5", sdvig)
    s7 = _ev(EvaluatorV7)._signals5(SP)["T"]
    assert s6.sigs.tobytes() != s7.sigs.tobytes()                   # побайтное сравнение G1 такое отличие видит


def _panel(**izm):
    ix = pd.bdate_range("2018-01-02", periods=5)
    p = dict(c=pd.DataFrame({"A": np.arange(5.0)}, index=ix), imoex=pd.Series(np.arange(5.0), index=ix),
             metki=np.zeros((5, 1)), sigma=np.ones((5, 1)), ok=np.ones((5, 1), bool))
    p.update(izm)
    return SimpleNamespace(**p)


@pytest.mark.parametrize("pole,znach", [("c", pd.DataFrame({"A": np.arange(5.0) + 1}, index=pd.bdate_range("2018-01-02", periods=5))),
                                        ("sigma", np.full((5, 1), 2.0)), ("ok", np.zeros((5, 1), bool)), ("metki", np.ones((5, 1)))])
def test_sha_paneli_menyaetsya_ot_kazhdogo_polya(pole, znach):
    a = _ev(EvaluatorV7, "S8", ist7=eval_v7.Ist7(panel=_panel()))._fkey
    assert a != _ev(EvaluatorV7, "S8", ist7=eval_v7.Ist7(panel=_panel(**{pole: znach})))._fkey      # Д5: подмена входа — другой ключ
    assert a != _ev(EvaluatorV7, "S9", ist7=eval_v7.Ist7(panel=_panel()))._fkey


def test_signal_v_progreve_s10_kak_est(monkeypatch):
    """Прогрев S10–S12: сигнал и выход режет сам v7_impuls (обёртка не обнуляет, в отличие от S8/S9); тест фиксирует факт."""
    td, ind = _td()
    ryad = pd.Series(0.0, index=pd.DatetimeIndex(td.dates))
    ryad.iloc[WARMUP - 5] = 1.0
    monkeypatch.setattr(v7_impuls.I, "signal_dnya", lambda b10, sid, t_sig="18:30": ryad)
    monkeypatch.setattr(v7_impuls.I, "sila_dnya", lambda b10, sid, t_sig="18:30": ryad.abs() + 1.0)
    ss = get_family("S10", setka="v7")(eval_v7.Ist7(b10={"T": pd.DataFrame({"date": pd.DatetimeIndex([])})}))(td, ind, [dict(tau=3, vhod="d1", T="17:00")])
    assert not ss.sig[0, :WARMUP].any() and not ss.exit_l[0, :WARMUP].any()    # v7_impuls сам режет прогрев: сигнала и выхода там нет
