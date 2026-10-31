"""Лист A s8_s12: точка S8–S12 в EvaluatorV7 (SemyaMixin, fastbt/eval_v7_impuls.py). Синтетика, без сети; данные не читает.
G1 S8: сигнал оценщика = сторона панели по рангам d−1, нейтральность лонг-шорт; S10 d1: вход открытием d+1 (Д2), число сделок/знак PnL по ручному расчёту;
G2 M-d1 (панель с днём d+1 / импульс со сдвигом) ловится; G3 `eval_v5.run_rows` не подменяется (класс № 43), cena_T — явный параметр.
Вход в T по close и S11/S12 — лист B. python3 -m pytest tests/test_eval_v7_s8_s12.py -q"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import eval_v5, eval_v7, exec_e
from fastbt import eval_v7_impuls as SI
from fastbt import kross_v7 as kv
from fastbt.eval_v6 import EvaluatorV6, IstochnikiV6
from fastbt.eval_v7 import EvaluatorV7
from fastbt.exec_e_types import Day, Params
from fastbt.indicators import compute_base
from fastbt.signals import v7_impuls, v7_kross as vk
from fastbt.signals.interface import WARMUP, TickerData

ORIG_RUN_ROWS = eval_v5.run_rows
N, NT = 8, 320
IDX = pd.bdate_range("2017-01-02", periods=NT)
COLS = [f"T{i}" for i in range(N)]
METKI = np.array(["A", "A", "A", "B", "B", "B", "C", ""] * NT, dtype=object).reshape(NT, N)
MODS = np.arange(590, 1140, 5)
PT8 = dict(rebal="nedelya", porog=0.4, bufer=0.1, cap=2, tau=3)
PT10 = dict(tau=3, vhod="d1", T="17:00")


def _panel():
    r = np.random.default_rng(1)
    c = pd.DataFrame(100 * np.exp(np.cumsum(r.normal(0, 0.01, (NT, N)), axis=0)), index=IDX, columns=COLS)
    m = pd.Series(1000 * np.exp(np.cumsum(r.normal(0, 0.005, NT))), index=IDX)
    return vk.Panel(IDX.to_numpy("datetime64[ns]"), COLS, c, m, METKI, np.ones((NT, N)), np.ones((NT, N), bool))


def _td(x, name, idx=IDX):
    x = np.asarray(x, float)
    return TickerData(name, idx.to_numpy("datetime64[ns]"), x, x, x, x, np.ones(len(x)), np.zeros(len(x)))


def _ev(sid, ist7, tds):
    e = object.__new__(EvaluatorV7)
    e.sid, e.proxy, e.ist7, e._fk, e.v6, e.ist, e._rel, e._dan = sid, eval_v7.PROXY[sid], ist7, (), {}, IstochnikiV6(), {}, {}
    e.sk = SimpleNamespace(_td={t: (td, compute_base(td.close, td.high, td.low, td.open, td.volume)) for t, td in tds.items()}, tickers=list(tds))
    e.r5, e._okup_k, e.fdr = None, None, None
    return e


def _probe(mp, ev, pt):
    """evaluate_v2 с заглушкой EvaluatorV6: сигналы семьи через настоящий _signals5, настройки _rows — через настоящий SemyaMixin._rows."""
    cap = {}

    def evaluate(self, q, period, vozm=None):
        cap.update(sid_vnutri=self.sid, q=dict(q), sg=dict(self._sg), ss=self._signals5(dict(q)))
        self.r5 = SimpleNamespace(nov={}, ext={}, sp=dict(q))
        self._rows(q, period, None)
        return "ok"

    def rows(self, pt_, period, anchor):
        cap.update(ext=dict(self.r5.ext), nov=dict(self.r5.nov), fdr_kw=self.fdr_kw(), run_rows_vnutri=eval_v5.run_rows)
        return []

    mp.setattr(EvaluatorV6, "evaluate_v2", evaluate)
    mp.setattr(EvaluatorV6, "_rows", rows)
    assert ev.evaluate_v2(dict(pt), "train4") == "ok"
    cap["sid_posle"] = ev.sid
    return cap


# ---------------- S8: сечение ----------------
def _s8(mp, mutaciya=False):
    pn = _panel()
    pn.ozhid = _ozhidaemo(pn)                                      # до подмены: эталон не видит мутацию
    if mutaciya:                                                   # M-d1: скор панели берёт закрытие d+1
        orig = kv.panel
        mp.setattr(kv, "panel", lambda c, im, me, sid: orig(c.shift(-1).ffill(), im, me, sid))
    ev = _ev("S8", eval_v7.Ist7(panel=pn), {t: _td(pn.c[t], t) for t in COLS})
    cap = _probe(mp, ev, PT8)
    return pn, cap, np.array([cap["ss"][t].ss.sig[0] for t in COLS])


def _ozhidaemo(pn):
    """Независимо от оценщика, из ряда закрытий без d+1: kross_v7.storona по панели S8 (как test_signals_v7_kross::test_g2)."""
    sk = kv.rebal(kv.panel(pn.c, pn.imoex, pn.metki, "S8"), pn.daty, "nedelya")
    e = kv.storona(sk, pn.sigma, pn.metki, pn.ok, 0.4, 0.1, 2).copy()
    e[:WARMUP] = 0
    return e.T                                                     # [тикер, день], как сигналы оценщика


def test_g1_s8_oborot_tochki_i_signal_ravny_paneli_d_minus_1(monkeypatch):
    pn, cap, sig = _s8(monkeypatch)
    assert cap["sid_vnutri"] == "S3" and cap["sid_posle"] == "S8"                   # прокси на время расчёта, sid восстановлен
    assert cap["sg"] == dict(porog=0.4, bufer=0.1, cap=2, rezhim="nedelya")
    assert not set(cap["sg"]) & set(cap["q"]) and cap["q"]["stop_nach"] == SI.STOP_S89 and cap["q"]["tau"] == 3
    assert (sig != 0).sum() > 20
    assert np.array_equal(sig, pn.ozhid)
    assert eval_v5.run_rows is ORIG_RUN_ROWS and cap["run_rows_vnutri"] is ORIG_RUN_ROWS


def test_g1_s8_nejtralnost_long_short(monkeypatch):
    _, _, sig = _s8(monkeypatch)
    n_long, n_short = (sig == 1).sum(axis=0), (sig == -1).sum(axis=0)
    assert n_long.sum() > 0 and n_short.sum() > 0
    assert (n_long == n_short).all(), "лонг и шорт сечения должны быть равны по числу имён (сумма весов 0)"


def test_g2_s8_mutaciya_d_plus_1_lovitsya(monkeypatch):
    pn, _, sig = _s8(monkeypatch, mutaciya=True)
    assert not np.array_equal(sig, pn.ozhid)


# ---------------- S10: импульс, вход d1 ----------------
DNI = {310: 1, 330: -1, 350: 1}


def _ryad(shift=0):
    s = pd.Series(0.0, index=IDX.__class__(pd.bdate_range("2018-01-02", periods=400)))
    for i, z in DNI.items():
        s.iloc[i + shift] = z
    return s


def _ist10(mp, shift=0):
    mp.setattr(v7_impuls.I, "signal_dnya", lambda b10, sid, t_sig="18:30": _ryad(shift))
    mp.setattr(v7_impuls.I, "sila_dnya", lambda b10, sid, t_sig="18:30": _ryad(shift).abs() + 1.0)
    return eval_v7.Ist7(b10={"T": pd.DataFrame({"date": pd.DatetimeIndex([])})})


def _days(n=400):
    out = []
    for k in range(n):
        o = 100.0 + k + (MODS - 590) / 1000.0                  # цена растёт: лонг всегда в плюс, шорт всегда в минус
        out.append(Day(o, o + 0.2, o - 0.2, o + 0.1, False, MODS))
    return out


def _s10(mp, shift=0):
    dates = pd.bdate_range("2018-01-02", periods=400)
    ev = _ev("S10", _ist10(mp, shift), {"T": _td(100 + np.arange(400.0), "T", dates)})
    cap = _probe(mp, ev, PT10)
    sig = cap["ss"]["T"].ss.sig[0].astype(float)
    p = replace(Params(comm=0.0, tau=3), **cap["ext"])
    return cap, exec_e.run("T", _days(), sig, np.full(400, 0.01), p).trades


def test_g1_s10_d1_vhod_d_plyus_1_chislo_i_znak(monkeypatch):
    cap, tr = _s10(monkeypatch)
    assert cap["ext"].get("vhod_T") is None and cap["ext"]["k"] is None
    assert [t.d_sig for t in tr] == [310, 330, 350]                                 # ручной счёт: 3 сигнала, все взяты
    assert [t.entry_day for t in tr] == [311, 331, 351], "вход не раньше d+1 (Д2)"
    assert [t.side for t in tr] == [1, -1, 1]
    assert all(t.entry_bar == 2 and t.entry_px == _days()[t.entry_day].o[2] for t in tr), "вход по open бара 10:00 d+1"
    pnl = [t.side * (t.exit_px - t.entry_px) for t in tr]
    assert [p > 0 for p in pnl] == [True, False, True]                              # цена растёт: лонг +, шорт −


def test_g2_s10_mutaciya_impuls_so_sdvigom_lovitsya(monkeypatch):
    _, tr = _s10(monkeypatch, shift=1)                                              # импульс известен на день позже / раньше
    assert [t.d_sig for t in tr] != [310, 330, 350]


# ---------------- cena_T — явный параметр, подмены модуля нет (класс № 43) ----------------
def _b10(dates, slot_open):
    rows = [(d, s, v, v + 1.0) for d in dates for s, v in slot_open.items()]
    return pd.DataFrame(rows, columns=["date", "slot", "open", "close"])


def test_cena_t_open_bara_T_i_close_dlya_18_40_nan_bez_bara():
    d = pd.bdate_range("2019-03-04", periods=3)
    b = _b10(d[:2], {(17 * 60 - 600) // 10: 50.0, (18 * 60 + 40 - 600) // 10: 70.0})
    c = SI.cena_t(b, d, "17:00")
    assert c[0] == c[1] == 50.0 and np.isnan(c[2])
    assert SI.cena_t(b, d, "18:40")[0] == 71.0


def test_run_rows_cena_T_yavnyj_parametr_ravno_exec_e():
    n, T = 3, "12:00"
    days = [Day(o := 100.0 * (k + 1) + (MODS - 590) / 100.0, o + 0.5, o - 0.5, o + 0.25, False, MODS) for k in range(n)]
    cena = np.array([100.0 * (k + 1) + (12 * 60 - 590) / 100.0 for k in range(n)])
    p, sig = Params(vhod_T=T), np.array([0, 1, 0])
    ref = exec_e.run("T", days, sig, np.full(n, 0.01), p, cena_T=cena).trades[0]
    for ct in (cena, lambda t, dates: cena):
        r = eval_v5.run_rows("T", days, sig, np.full(n, 0.01), p, win=(0, n), dates=None, cena_T=ct)
        assert (r.trades[0].entry_day, r.trades[0].entry_bar, r.trades[0].entry_px) == (ref.entry_day, ref.entry_bar, ref.entry_px) == (1, 26, cena[1])


def test_g3_vhod_T_cena_cherez_parametr_run_rows_ne_podmenyaetsya(monkeypatch):
    d = pd.bdate_range("2018-01-02", periods=400)
    b10 = _b10(d, {(17 * 60 - 600) // 10: 50.0})
    ev = _ev("S10", eval_v7.Ist7(b10={"T": b10}), {"T": _td(100 + np.arange(400.0), "T", d)})
    monkeypatch.setattr(v7_impuls.I, "signal_dnya", lambda b, sid, t_sig="18:30": _ryad())
    monkeypatch.setattr(v7_impuls.I, "sila_dnya", lambda b, sid, t_sig="18:30": _ryad().abs() + 1.0)
    cap = _probe(monkeypatch, ev, dict(tau=3, vhod="T", T="17:00"))
    assert cap["ext"]["vhod_T"] == "17:00" and cap["ext"]["k"] is None and not cap["ext"].get("vhod_open")
    assert cap["run_rows_vnutri"] is ORIG_RUN_ROWS and eval_v5.run_rows is ORIG_RUN_ROWS
    assert (cap["fdr_kw"]["cena_T"]("T", d) == 50.0).all()
    assert "cena_T" not in ev.fdr_kw()                                              # вне расчёта точки — прежний словарь
