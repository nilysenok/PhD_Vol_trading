"""Лист B s8_s12: S10 со входом vhod=T (T ∈ 17:00, 18:00, 18:40) в EvaluatorV7. Синтетика, без сети, данные не читает.
G1 вход в T того же дня по cena_T (open 10-мин бара T, 18:40 — close), выход close d+τ, число сделок и знак PnL по ручному счёту;
G3 мутации M-T (цена после T; вход в T по close, 18:40 по open; вход по close дня), M-d1 (импульс на день позже), сигнал по 18:30 вместо T ловятся.
python3 -m pytest tests/test_eval_v7_s8_s12_t.py -q"""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import eval_v7, exec_e
from fastbt import eval_v7_impuls as SI
from fastbt.eval_v6 import EvaluatorV6, IstochnikiV6
from fastbt.eval_v7 import EvaluatorV7
from fastbt.exec_e_types import Day, Params
from fastbt.indicators import compute_base
from fastbt.signals import v7_impuls
from fastbt.signals.interface import TickerData

NT = 400
DATES = pd.bdate_range("2018-01-02", periods=NT)
MODS = np.arange(590, 1140, 5)
DNI = {310: 1, 330: -1, 350: 1}
TS = ("17:00", "18:00", "18:40")
MIN = {"17:00": 1020, "18:00": 1080, "18:40": 1120}


def _px(k, m):
    """Цена в минуту m дня k: единая для 5-мин баров и 10-минуток (рост день ото дня, лонг в плюс, шорт в минус)."""
    return 100.0 + k + (m - 590) / 1000.0


GAP = 0.15        # close 10-мин бара ≠ open следующего: цена сигнала (close бара, закрывшегося в T) отличима от цены входа


def _b10(skoro=1.0, do_slota=54):
    rows = [(d, s, skoro * _px(k, 600 + 10 * s), skoro * (_px(k, 610 + 10 * s) + GAP)) for k, d in enumerate(DATES) for s in range(do_slota)]
    return pd.DataFrame(rows, columns=["date", "slot", "open", "close"])


def _days(do_bara=len(MODS)):
    return [Day(o := _px(k, MODS[:do_bara]), o + 0.2, o - 0.2, o + 0.1, False, MODS[:do_bara]) for k in range(NT)]


def _ryad(shift=0):
    s = pd.Series(0.0, index=DATES)
    for i, z in DNI.items():
        s.iloc[i + shift] = z
    return s


def _td(name="T"):
    x = 100 + np.arange(float(NT))
    return TickerData(name, DATES.to_numpy("datetime64[ns]"), x, x, x, x, np.ones(NT), np.zeros(NT))


def _ev(sid="S10", ist7=None):
    e = object.__new__(EvaluatorV7)
    e.sid, e.proxy, e.ist7, e._fk, e.v6, e.ist, e._rel, e._dan = sid, eval_v7.PROXY[sid], ist7 or eval_v7.Ist7(b10={"T": _b10()}), (), {}, IstochnikiV6(), {}, {}
    td = _td()
    e.sk = SimpleNamespace(_td={"T": (td, compute_base(td.close, td.high, td.low, td.open, td.volume))}, tickers=["T"])
    e.r5, e._okup_k, e.fdr = None, None, None
    return e


def _fake(mp, shift=0, zapis=None, sig_fn=None):
    def f(b10, sid, t_sig="18:30"):
        zapis is None or zapis.append(t_sig)
        return sig_fn(t_sig) if sig_fn else _ryad(shift)
    mp.setattr(v7_impuls.I, "signal_dnya", f)
    mp.setattr(v7_impuls.I, "sila_dnya", lambda b, sid, t_sig="18:30": f(b, sid, t_sig).abs() + 1.0)


def _probe(mp, pt, ist7=None):
    cap, ev = {}, _ev(ist7=ist7)

    def evaluate(self, q, period, vozm=None):
        self.r5 = SimpleNamespace(nov={}, ext={}, sp=dict(q))
        cap.update(ss=self._signals5(dict(q)), fkey=self._fkey)
        self._rows(q, period, None)
        return "ok"

    def rows(self, pt_, period, anchor):
        cap.update(ext=dict(self.r5.ext), cena=self.fdr_kw().get("cena_T"))
        return []

    mp.setattr(EvaluatorV6, "evaluate_v2", evaluate)
    mp.setattr(EvaluatorV6, "_rows", rows)
    assert ev.evaluate_v2(dict(pt), "train4") == "ok"
    return cap


def _derzhanie(ss):
    """Состояние позиции из флагов семьи (ручная модель positions): сигнал открывает, флаг exit на баре i обнуляет бар i (выход по close i)."""
    st, out = 0.0, np.zeros(NT)
    for i in range(NT):
        st = 0.0 if ss.exit_l[0][i] else (float(ss.sig[0][i]) or st)
        out[i] = st
    return out


def _trades(cap, cena=None, days=None):
    sig = _derzhanie(cap["ss"]["T"].ss)
    p = replace(Params(comm=0.0, tau=3), **cap["ext"])
    return exec_e.run("T", days or _days(), sig, np.full(NT, 0.01), p, cena_T=cap["cena"]("T", DATES) if cena is None else cena).trades


def _cena_verna(cena, T):
    """Независимо от оценщика: open 10-мин бара, начинающегося в T (Л6.3, и для 18:40, пока бар есть), цена единой формулы _px."""
    ozhid = np.array([_px(k, MIN[T]) for k in range(NT)])
    return bool(np.allclose(cena, ozhid, atol=1e-12, rtol=0))


@pytest.mark.parametrize("T", TS)
def test_g1_vhod_T_chislo_znak_cena_T_vyhod_close(monkeypatch, T):
    zapis = []
    _fake(monkeypatch, zapis=zapis)
    cap = _probe(monkeypatch, dict(tau=3, vhod="T", T=T))
    assert cap["ext"]["vhod_T"] == T and cap["ext"]["k"] is None and not cap["ext"].get("vhod_open")
    assert set(zapis) == {T}, "сигнал решается в момент T, а не в 18:30"
    assert _cena_verna(cap["cena"]("T", DATES), T)
    tr = _trades(cap)
    assert [t.d_sig for t in tr] == [310, 330, 350] and [t.side for t in tr] == [1, -1, 1]       # ручной счёт: 3 сигнала, все взяты
    assert [t.entry_day for t in tr] == [310, 330, 350], "вход в T того же дня"
    assert all(_days()[t.entry_day].t[t.entry_bar] == MIN[T] for t in tr)
    assert all(t.entry_px == cap["cena"]("T", DATES)[t.entry_day] for t in tr)
    assert [t.exit_day for t in tr] == [313, 333, 353] and all(t.exit_px == _days()[t.exit_day].c[-1] for t in tr)
    assert [t.side * (t.exit_px - t.entry_px) > 0 for t in tr] == [True, False, True]


def test_g1_d1_signal_po_18_30_vhod_ne_v_T(monkeypatch):
    zapis = []
    _fake(monkeypatch, zapis=zapis)
    cap = _probe(monkeypatch, dict(tau=3, vhod="d1", T="17:00"))
    assert set(zapis) == {"18:30"} and cap["ext"]["vhod_T"] is None and cap["cena"] is None


@pytest.mark.parametrize("T", TS)
def test_g3_mutaciya_cena_posle_T_lovitsya(monkeypatch, T):
    _fake(monkeypatch)
    orig = SI.cena_t
    monkeypatch.setattr(SI, "cena_t", lambda b10, dates, t: orig(b10, dates, f"{(MIN[t] + 10) // 60}:{(MIN[t] + 10) % 60:02d}" if t != "18:40" else "18:30"))
    cap = _probe(monkeypatch, dict(tau=3, vhod="T", T=T))
    assert not _cena_verna(cap["cena"]("T", DATES), T)


@pytest.mark.parametrize("T", TS)
def test_g3_mutaciya_vhod_po_close_bara_zakryvayushchegosya_v_T_lovitsya(monkeypatch, T):
    """Вход в T по close бара, закрывающегося в T (цена сигнала, Д2), вместо open бара, начинающегося в T."""
    _fake(monkeypatch)
    orig = SI.cena_t
    monkeypatch.setattr(SI, "cena_t", lambda b10, dates, t: orig(b10.assign(open=b10.groupby("date")["close"].shift(1).fillna(b10["open"])), dates, t))
    cap = _probe(monkeypatch, dict(tau=3, vhod="T", T=T))
    assert not _cena_verna(cap["cena"]("T", DATES), T)


def test_g1_18_40_bez_bara_vhod_po_close_dnya_na_poslednem_bare(monkeypatch):
    """Бара с началом ≥ 18:40 нет в данных: вход в тот же день по close дня основной сессии, на последнем 5-мин баре; сделок столько же."""
    _fake(monkeypatch)
    ist7 = eval_v7.Ist7(b10={"T": _b10(do_slota=52)})                          # 10-мин слоты до 18:30 включительно (слот 51)
    days = _days(do_bara=MODS.tolist().index(1120))                            # 5-мин бары до 18:15 включительно
    cap = _probe(monkeypatch, dict(tau=3, vhod="T", T="18:40"), ist7=ist7)
    cena = cap["cena"]("T", DATES)
    assert np.allclose(cena, [_px(k, 1120) + GAP for k in range(NT)], atol=1e-12, rtol=0)      # close слота 51 = close дня
    tr = _trades(cap, days=days)
    assert [t.entry_day for t in tr] == [310, 330, 350] and all(t.entry_bar == len(days[0].c) - 1 and t.entry_px == cena[t.entry_day] for t in tr)
    monkeypatch.setattr(SI, "AUKCION", 10 ** 6)                                # мутация: «нет бара → нет сделки» (прежняя семантика)
    cap2 = _probe(monkeypatch, dict(tau=3, vhod="T", T="18:40"), ist7=ist7)
    assert np.isnan(cap2["cena"]("T", DATES)).all() and _trades(cap2, days=days) == []


def test_g3_ispolnenie_po_realnoj_lente_a_ne_skorrektirovannoj(monkeypatch):
    """Цена входа — из b10_real (без дивидендной поправки, #370), даже если скорректированная b10 отличается."""
    _fake(monkeypatch)
    ist7 = eval_v7.Ist7(b10={"T": _b10(skoro=0.9)}, b10_real={"T": _b10()})
    cap = _probe(monkeypatch, dict(tau=3, vhod="T", T="17:00"), ist7=ist7)
    assert _cena_verna(cap["cena"]("T", DATES), "17:00")
    mut = eval_v7.Ist7(b10={"T": _b10(skoro=0.9)}, b10_real={"T": _b10(skoro=0.9)})      # мутация: исполнительная лента = скорректированная
    assert not _cena_verna(_probe(monkeypatch, dict(tau=3, vhod="T", T="17:00"), ist7=mut)["cena"]("T", DATES), "17:00")


def test_g3_mutaciya_vhod_po_close_dnya_lovitsya(monkeypatch):
    _fake(monkeypatch)
    cap = _probe(monkeypatch, dict(tau=3, vhod="T", T="17:00"))
    close_dnya = np.array([d.c[-1] for d in _days()])
    assert not _cena_verna(close_dnya, "17:00")
    tr = _trades(cap, cena=close_dnya)
    assert [t.entry_px for t in tr] != [_px(t.entry_day, MIN["17:00"]) for t in tr]


@pytest.mark.parametrize("shift", (1, -1))
def test_g3_mutaciya_d1_impuls_so_sdvigom_lovitsya(monkeypatch, shift):
    _fake(monkeypatch, shift=shift)
    tr = _trades(_probe(monkeypatch, dict(tau=3, vhod="T", T="18:00")))
    assert [t.d_sig for t in tr] != [310, 330, 350] and [t.entry_day for t in tr] != [310, 330, 350]


def test_g3_signal_zavisit_ot_t_sig_i_mutaciya_18_30_lovitsya(monkeypatch):
    """Сигнал дня существует только на момент T = 17:00; решение по 18:30 для vhod=T — утечка после T (M-T)."""
    ryad = lambda t: _ryad() if t == "17:00" else _ryad(7)          # noqa: E731
    _fake(monkeypatch, sig_fn=ryad)
    tr = _trades(_probe(monkeypatch, dict(tau=3, vhod="T", T="17:00")))
    assert [t.d_sig for t in tr] == [310, 330, 350]
    orig = v7_impuls.I.signal_dnya
    monkeypatch.setattr(v7_impuls.I, "signal_dnya", lambda b, sid, t_sig="18:30": orig(b, sid, "18:30"))
    tr = _trades(_probe(monkeypatch, dict(tau=3, vhod="T", T="17:00")))
    assert [t.d_sig for t in tr] != [310, 330, 350]
