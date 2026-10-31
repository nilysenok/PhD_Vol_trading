"""Независимая проверка #922 (S8–S12 в EvaluatorV7, лист A): тестировщик, не автор. Синтетика, без сети, данные не читает.
S8: нейтральность (лонг = шорт каждый день), сигнал дня d не зависит от цен после d (M-d1); S10 d1: вход open d+1 (Д2), выход d_sig+τ, τ ∈ {2,3,5};
ключ точки (Д5) различает семьи и оси, детерминирован в двух процессах; eval_v5.run_rows после оценки — исходный (класс № 43).
Мутации в памяти: ранги с днём d+1, вход в день сигнала, τ не доходит до исполнения, S8 без нейтральности — каждая роняет проверку."""
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import eval_v5, eval_v7, exec_e, kross_v7 as kv
from fastbt.eval_v6 import EvaluatorV6, IstochnikiV6
from fastbt.eval_v7 import EvaluatorV7
from fastbt.exec_e_types import Day, Params
from fastbt.indicators import compute_base
from fastbt.signals import v7_impuls, v7_kross as vk
from fastbt.signals.interface import TickerData

ORIG, NT, N = eval_v5.run_rows, 300, 8
IDX = pd.bdate_range("2017-01-02", periods=NT)
D18 = pd.bdate_range("2018-01-02", periods=400)
COLS = [f"X{i}" for i in range(N)]
MODS = np.arange(590, 1140, 5)
PT8 = dict(rebal="nedelya", porog=0.4, bufer=0.1, cap=2, tau=3)
KS = range(255, 262)                                               # после прогрева скора (252 дня)
PT8D = dict(rebal="den", porog=0.1, bufer=0.0, cap=3, tau=3)        # решение каждый день, много сигналов: лаг виден по дням
DNI = {310: 1, 330: -1, 350: 1}                                    # после прогрева WARMUP


def _panel(seed=5, c=None):
    r = np.random.default_rng(seed)
    c = c if c is not None else pd.DataFrame(100 * np.exp(np.cumsum(r.normal(0, 0.01, (NT, N)), axis=0)), index=IDX, columns=COLS)
    m = pd.Series(1000 * np.exp(np.cumsum(r.normal(0, 0.005, NT))), index=IDX)
    metki = np.array(["A", "A", "A", "B", "B", "B", "C", ""] * NT, dtype=object).reshape(NT, N)
    return vk.Panel(IDX.to_numpy("datetime64[ns]"), COLS, c, m, metki, np.ones((NT, N)), np.ones((NT, N), bool))


def _td(x, name, idx):
    x = np.asarray(x, float)
    return TickerData(name, idx.to_numpy("datetime64[ns]"), x, x, x, x, np.ones(len(x)), np.zeros(len(x)))


def _ev(sid, ist7, tds):
    e = object.__new__(EvaluatorV7)
    e.sid, e.proxy, e.ist7, e._fk, e.v6, e.ist, e._rel, e._dan = sid, eval_v7.PROXY[sid], ist7, (), {}, IstochnikiV6(), {}, {}
    e.sk = SimpleNamespace(_td={t: (td, compute_base(td.close, td.high, td.low, td.open, td.volume)) for t, td in tds.items()}, tickers=list(tds))
    e.r5, e._okup_k, e.fdr = None, None, None
    return e


def _run(mp, ev, pt):
    """evaluate_v2 с заглушкой EvaluatorV6; сигналы — настоящий _signals5, настройки и ключ — настоящий SemyaMixin."""
    cap = {}

    def evaluate(self, q, period, vozm=None):
        cap.update(key=self._fkey, ss=self._signals5(dict(q)))
        self.r5 = SimpleNamespace(nov={}, ext={}, sp=dict(q))
        self._rows(q, period, None)
        return "ok"

    def rows(self, pt_, period, anchor):
        cap.update(ext=dict(self.r5.ext), rr=eval_v5.run_rows)
        return []

    mp.setattr(EvaluatorV6, "evaluate_v2", evaluate)
    mp.setattr(EvaluatorV6, "_rows", rows)
    assert ev.evaluate_v2(dict(pt), "train4") == "ok"
    return cap


def _sig8(mp, pn, shift_future=False, pt=PT8):
    if shift_future:                                               # M-d1: скор панели берёт закрытие d+1
        orig = kv.panel
        mp.setattr(kv, "panel", lambda c, im, me, sid: orig(c.shift(-1).ffill(), im, me, sid))
    ev = _ev("S8", eval_v7.Ist7(panel=pn), {t: _td(pn.c[t], t, IDX) for t in COLS})
    cap = _run(mp, ev, pt)
    return np.array([cap["ss"][t].ss.sig[0] for t in COLS])        # [тикер, день]


def _bez_k(k, hvost, gorizont, **kw):
    """Цены портятся (hvost: все дни после k, иначе только день k); сигнал на днях < k + gorizont обязан остаться прежним."""
    c2 = _panel().c.copy()
    sl = slice(k + 1, None) if hvost else slice(k, k + 1)
    c2.iloc[sl] *= np.random.default_rng(9).uniform(0.5, 1.5, c2.iloc[sl].shape)
    with pytest.MonkeyPatch.context() as m1:
        a = _sig8(m1, _panel(), pt=PT8D, **kw)
    with pytest.MonkeyPatch.context() as m2:
        b = _sig8(m2, _panel(c=c2), pt=PT8D, **kw)
    return np.array_equal(a[:, :k + gorizont], b[:, :k + gorizont]) and (a != 0).sum() > 20


def _nejtralno(sig) -> bool:
    return bool((sig == 1).sum() > 0 and ((sig == 1).sum(0) == (sig == -1).sum(0)).all())


def test_s8_nejtralnost_kazhdyj_den_i_mutaciya(monkeypatch):
    sig = _sig8(monkeypatch, _panel())
    assert (sig != 0).sum() > 20 and _nejtralno(sig)
    j = int(np.flatnonzero((sig == -1).any(0))[0])
    sig[np.flatnonzero(sig[:, j] == -1)[0], j] = 0                  # S8 без нейтральности: шорт дня снят — обязан быть пойман
    assert not _nejtralno(sig)


def test_s8_signal_dnya_d_ne_zavisit_ot_cen_posle_d():
    """Скор S8 = 12−1 моментум (сдвиг 21 день): цена дня k не видна до k+21; цены после k не видны сигналу дня ≤ k."""
    assert all(_bez_k(k, True, 1) and _bez_k(k, False, 21) for k in KS)


def test_s8_mutaciya_rangi_s_dnyom_d_plus_1_lovitsya():
    assert not all(_bez_k(k, False, 21, shift_future=True) for k in KS)


# ---------------- S10: вход d1 ----------------
def _ryad():
    s = pd.Series(0.0, index=D18)
    for i, z in DNI.items():
        s.iloc[i] = z
    return s


def _ist10(mp):
    mp.setattr(v7_impuls.I, "signal_dnya", lambda b, sid, t_sig="18:30": _ryad())
    mp.setattr(v7_impuls.I, "sila_dnya", lambda b, sid, t_sig="18:30": _ryad().abs() + 1.0)
    return eval_v7.Ist7(b10={"X": pd.DataFrame({"date": pd.DatetimeIndex([])})})


def _s10(mp, tau, mut=None):
    ev = _ev("S10", _ist10(mp), {"X": _td(100 + np.arange(400.0), "X", D18)})
    cap = _run(mp, ev, dict(tau=tau, vhod="d1", T="17:00"))
    ss = cap["ss"]["X"].ss
    if mut == "tau_ne_dohodit":
        ss.exit_l[:], ss.exit_s[:] = False, False                  # флаги выхода не доходят до исполнения
    held, cur = np.zeros(400), 0                                   # позиция: взятый сигнал держится до флага выхода (close d+τ)
    for i in range(400):
        cur = 0 if cur and (ss.exit_l[0, i] if cur > 0 else ss.exit_s[0, i]) else (cur or int(ss.sig[0, i]))
        held[i] = cur
    days = [Day(o := 100.0 + k + (MODS - 590) / 1000.0, o + 0.2, o - 0.2, o + 0.1, False, MODS) for k in range(400)]
    p = replace(Params(comm=0.0, tau=tau), **cap["ext"])
    if mut == "vhod_v_den_signala":
        p = replace(p, vhod_open=False, k=None)                    # вход по close дня сигнала
    return exec_e.run("X", days, held, np.full(400, 0.01), p).trades, days


@pytest.mark.parametrize("tau", [2, 3, 5])
def test_s10_vhod_d_plus_1_open_vyhod_d_plus_tau(monkeypatch, tau):
    tr, days = _s10(monkeypatch, tau)
    assert [t.d_sig for t in tr] == [310, 330, 350] and [t.side for t in tr] == [1, -1, 1]
    assert [t.entry_day for t in tr] == [311, 331, 351]            # Д2: не раньше d+1
    assert all(t.entry_bar == 2 and t.entry_px == days[t.entry_day].o[2] for t in tr)
    assert [t.exit_day for t in tr] == [310 + tau, 330 + tau, 350 + tau]
    assert [t.side * (t.exit_px - t.entry_px) > 0 for t in tr] == [True, False, True]
    assert eval_v5.run_rows is ORIG                                 # класс № 43: подмены не осталось


def test_s10_mutacii_lovyatsya(monkeypatch):
    assert [t.entry_day for t in _s10(monkeypatch, 3, "vhod_v_den_signala")[0]] != [311, 331, 351]
    assert [t.exit_day for t in _s10(monkeypatch, 3, "tau_ne_dohodit")[0]] != [313, 333, 353]


# ---------------- ключ точки (Д5) ----------------
def _key(mp, sid, pt):
    ev = _ev(sid, _ist10(mp), {"X": _td(100 + np.arange(400.0), "X", D18)})
    return _run(mp, ev, pt)["key"]


def test_klyuch_razlichaet_semyi_i_osi(monkeypatch):
    base = dict(tau=3, vhod="d1", T="17:00")
    keys = {s: _key(monkeypatch, s, base) for s in ("S10", "S11", "S12")}
    for nm, ch in (("tau", dict(tau=5)), ("vhod", dict(vhod="T")), ("T", dict(T="18:00"))):
        keys[nm] = _key(monkeypatch, "S10", {**base, **ch})
    assert len(set(keys.values())) == len(keys), "ключи двух разных точек совпали"
    assert _key(monkeypatch, "S10", base) == keys["S10"]


def test_klyuch_s8_menyaetsya_ot_osej_i_panely(monkeypatch):
    def k8(pt, pn):
        ev = _ev("S8", eval_v7.Ist7(panel=pn), {COLS[0]: _td(pn.c[COLS[0]], COLS[0], IDX)})
        return _run(monkeypatch, ev, pt)["key"]
    a = k8(PT8, _panel())
    assert a == k8(PT8, _panel()) and a != k8({**PT8, "porog": 0.5}, _panel()) and a != k8({**PT8, "cap": 1}, _panel()) and a != k8(PT8, _panel(seed=6))


SKRIPT = """import pandas as pd
from fastbt import eval_v7
e = object.__new__(eval_v7.EvaluatorV7)
b = pd.DataFrame({"date": pd.to_datetime(["2018-01-02"]), "slot": [1], "open": [1.0]})
e.ist7, e._fk, e._okup_k, e._sem, e._sg = eval_v7.Ist7(b10={"X": b}), (), None, "S10", {"tau": 3, "vhod": "T", "T": "17:00"}
print(repr(e._fkey))
"""


def test_klyuch_determinirovan_v_dvuh_processah():
    root = str(Path(__file__).resolve().parents[1])
    out = [subprocess.run([sys.executable, "-c", SKRIPT], capture_output=True, text=True, check=True, cwd=root,
                          env={"PYTHONHASHSEED": s, "PATH": "/usr/bin:/bin", "PYTHONPATH": root}).stdout for s in ("1", "2")]
    assert out[0] == out[1] and "S10" in out[0] and "17:00" in out[0]
