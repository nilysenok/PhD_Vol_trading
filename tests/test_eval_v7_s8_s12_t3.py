"""Лист B s8_s12 (часть 3): РЕАЛЬНЫЙ путь флагов выхода S10 — SignalSet.exit_l/exit_s → Unit.exl → numba-ядро → sweep_post.positions → filtr_signala → exec_e.run
(без заглушек EvaluatorV6.evaluate_v2/_rows). Сигнал держится 10 дней подряд, τ = 2: сделки закрываются ровно на d+τ по close, а не когда сигнал упадёт в 0.
Мутация «флаг выхода не передан» (ss.exit_l = ss.exit_s = False) → позиция держится до конца серии → тест падает.
python3 -m pytest tests/test_eval_v7_s8_s12_t3.py -q -s"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import eval_v7, eval_v7_impuls as SI, etap1, exec_e
from fastbt.eval_v6 import EvaluatorV6, IstochnikiV6
from fastbt.eval_v7 import EvaluatorV7
from fastbt.exec_e_types import Day, Params
from fastbt.indicators import compute_base
from fastbt.signals import v7_impuls
from fastbt.signals.interface import SignalSet, TickerData
from fastbt.sweep_sk import SIG_GRIDS

NT, D0, DER = 400, 310, 10
DATES = pd.bdate_range("2018-01-02", periods=NT)
MODS = np.arange(590, 1140, 5)


def _ev(monkeypatch, tau, vhod, T="18:00", bez_flagov=False):
    n = np.arange(float(NT))
    x = 100 + n + 0.5 * np.sin(n)
    td = TickerData("T", DATES.to_numpy("datetime64[ns]"), x - 0.1, x + 0.3, x - 0.3, x, 100 + n, np.zeros(NT))   # объём растёт: база S3 держит маски adx/объём
    e = object.__new__(EvaluatorV7)
    e.sid, e.proxy, e.ist7 = "S3", "S3", eval_v7.Ist7(b10={"T": pd.DataFrame({"date": pd.DatetimeIndex([])})})
    e._fk, e.v6, e.ist, e._rel, e._dan, e.fdr, e.semya, e._sigs, e._pos = (), {}, IstochnikiV6(), {}, {}, None, "v6", {}, {}
    e._fkey, e.r5, e._okup_k = (), None, None
    e.sk = SimpleNamespace(_td={"T": (td, compute_base(td.close, td.high, td.low, td.open, td.volume))}, tickers=["T"], sig_names=())
    names = set(SIG_GRIDS["S3"]) | set(SI.ZAGLUSHKA)                   # как EvaluatorV5.__init__: sk.sig_names ∪ SIGNAL[sid]
    e._names5 = tuple(sorted(names))
    e._base5 = {k: v for k, v in etap1.base_point("S3").items() if k not in names}
    ryad = pd.Series(0.0, index=DATES)
    ryad.iloc[D0:D0 + DER] = 1.0                                       # импульс держится 10 дней подряд
    monkeypatch.setattr(v7_impuls.I, "signal_dnya", lambda b, sid, t_sig="18:30": ryad)
    monkeypatch.setattr(v7_impuls.I, "sila_dnya", lambda b, sid, t_sig="18:30": ryad.abs() + 1.0)
    e._sg = dict(tau=tau, vhod=vhod, T=T)
    e._sem = "S10"
    if bez_flagov:                                                     # мутация: флаг выхода не передан в Unit
        orig = eval_v7._kontrakt_impuls
        monkeypatch.setattr(eval_v7, "_kontrakt_impuls", lambda fn: (lambda td_, ind, grid: (lambda s: SignalSet(
            s.sig, np.zeros_like(s.exit_l), np.zeros_like(s.exit_s), s.strength, s.params, split=True, extra=s.extra))(orig(fn)(td_, ind, grid))))
    return e


def _trades(mp, e, vhod):
    """Реальный путь: Params.ext — те, что ставит SemyaMixin._rows (захват на месте вызова EvaluatorV6._rows); sig — positions как в EvaluatorV5._rows
    (sig[-1] = 0, фильтр оборота тривиален); дальше exec_e.run с теми же Params."""
    cap = {}
    mp.setattr(EvaluatorV6, "_rows", lambda self, pt, period, anchor: cap.update(ext=dict(self.r5.ext)) or [])
    e.r5 = SimpleNamespace(nov={}, ext={}, sp={})
    e._rows({}, "train4", None)
    sig = np.asarray(e._positions("T", dict(SI.ZAGLUSHKA), 0.0, False, NT), float).copy()
    sig[-1] = 0.0
    days = [Day(o := 100.0 + k + (MODS - 590) / 1000.0, o + 0.2, o - 0.2, o + 0.1, False, MODS) for k in range(NT)]
    cena = np.array([d.o[98] for d in days]) if vhod == "T" else None
    return exec_e.run("T", days, sig, np.full(NT, 0.01), replace(Params(comm=0.0), **cap["ext"]), cena_T=cena).trades


# ручной счёт (τ = 2, сигнал 310…319): d1 — берутся 310, 312, …, 318 (вход d+1, выход close d+2); T — 310, 313, 316, 319 (вход в T дня d, выход close d+2)
ZHDEM = {"d1": ([311, 313, 315, 317, 319], [312, 314, 316, 318, 320]), "T": ([310, 313, 316, 319], [312, 315, 318, 321])}


@pytest.mark.parametrize("vhod", ["d1", "T"])
def test_g1_vyhod_ravno_d_plus_tau_na_realnom_puti(monkeypatch, vhod):
    tr = _trades(monkeypatch, _ev(monkeypatch, 2, vhod), vhod)
    assert ([t.entry_day for t in tr], [t.exit_day for t in tr]) == ZHDEM[vhod]
    assert all(t.exit_px == t.exit_px and t.exit_bar == len(MODS) - 1 for t in tr), "выход по close (последний бар дня)"


@pytest.mark.parametrize("vhod", ["d1", "T"])
def test_g3_mutaciya_flag_vyhoda_ne_peredan_lovitsya(monkeypatch, vhod):
    tr = _trades(monkeypatch, _ev(monkeypatch, 2, vhod, bez_flagov=True), vhod)
    assert ([t.entry_day for t in tr], [t.exit_day for t in tr]) != ZHDEM[vhod]


@pytest.mark.parametrize("vhod", ["d1", "T"])
def test_g3_mutaciya_vremya_stop_ne_peredan_lovitsya(monkeypatch, vhod):
    monkeypatch.setattr(SI, "tau_vyhoda", lambda *a: None)
    tr = _trades(monkeypatch, _ev(monkeypatch, 2, vhod), vhod)
    assert ([t.entry_day for t in tr], [t.exit_day for t in tr]) != ZHDEM[vhod]
