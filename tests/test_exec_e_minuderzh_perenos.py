"""V7-4b (Ж-153 §1.1): ПЕРЕНОС раннего выхода на N-й день — fastbt/exec_e_minuderzh.py. Синтетика, без сети.
Сигнал выхода (или τ), пришедший раньше дня N, запоминается; в день N выход обязателен (по цене обычного выхода дня N),
если стоп/тейк до того не сработали. Разворот в T (RunT, причина "razvorot") переносится так же: закрытие в день N по cena_T,
новая позиция — по правилам входа дня N (_decide: есть сигнал — вход в T). G2 перенос; G3 мутация «забыт»; G4 разворот в T + мутация.
pytest tests/test_exec_e_minuderzh_perenos.py -q"""
import numpy as np
import pytest

from fastbt import exec_e, exec_e_minuderzh as M
from fastbt.exec_e_types import Day, Params

SG = 0.01
MODS = list(range(590, 1140, 5))
BT = MODS.index(720)


def _day(c, lo=None, hi=None):
    return Day(np.array([c, c]), np.array([c + 0.1 if hi is None else hi] * 2), np.array([c - 0.1 if lo is None else lo] * 2), np.array([c, c]))


def _run(sig, p, days=None, **kw):
    days = days or [_day(100.0 + i) for i in range(len(sig))]
    return exec_e.run("X", days, np.array(sig), np.full(len(days), SG), p, **kw)


def _tr(r, k=0):
    t = r.trades[k]
    return t.exit_day, t.exit_px, t.reason


# ---------- G2: перенос ----------
def test_g2_signal_vernulsya_vyhod_vse_ravno_na_den_2():
    sig = [1, 0, 1, 1, 1, 1]
    assert _tr(_run(sig, Params())) == (1, 101.0, "signal")                       # v6: день 1
    r = _run(sig, Params(min_uderzhanie=2))
    assert _tr(r) == (2, 102.0, "signal") and r.trades[0].entry_day == 0          # перенос: день 2, close дня 2


def test_g2_tau_do_dnya_N_perenositsya_na_N():
    assert _tr(_run([1] * 6, Params(tau=1, min_uderzhanie=3))) == (3, 103.0, "vremya")


def test_g2_signal_na_dne_N_ne_nuzhen_pamyat_ne_dubliruet():
    """Сигнал ушёл и остался ушедшим: ровно один выход на дне N, вторая сделка не от «памяти»."""
    r = _run([1, 0, 0, 0, 0, 0], Params(min_uderzhanie=2))
    assert len(r.trades) == 1 and _tr(r) == (2, 102.0, "signal")


def test_g2_stop_do_dnya_N_snimaet_otlozhennyj_vyhod_i_ne_techet_v_novuyu_poziciyu():
    """Ранний сигнал (день 1) запомнен, но стоп дня 1 закрыл раньше. Новая позиция (вход день 2) отложенный выход не наследует."""
    days = [_day(100.0), _day(101.0, lo=90.0)] + [_day(100.0 + i) for i in range(2, 7)]
    r = _run([1, 0, 1, 1, 1, 1, 1], Params(b=2.0, min_uderzhanie=2), days)
    assert _tr(r) == (1, 98.0, "stop")
    assert len(r.trades) == 2 and r.trades[1].entry_day == 2 and r.trades[1].exit_day is None


# ---------- G3: мутация «отложенный выход забыт» ----------
def _bez_pamyati(s, i, j, vrem):
    """Мутация: ранний сигнал/τ не запоминается — на дне N смотрится только текущее состояние."""
    now = "vremya" if vrem else ("signal" if s._sig_exit(i) else None)
    return None if M.rano(s, j) else now


def test_g3_mutatsiya_otlozhennyj_vyhod_zabyt_lovitsya(monkeypatch):
    sig = [1, 0, 1, 1, 1, 1]
    assert _tr(_run(sig, Params(min_uderzhanie=2))) == (2, 102.0, "signal")        # правка
    monkeypatch.setattr(M, "reshenie", _bez_pamyati)
    assert _run(sig, Params(min_uderzhanie=2)).trades[0].exit_day is None          # мутация держит позицию → тест выше упал бы


def test_g3_mutatsiya_pamyat_ne_privyazana_k_sdelke_lovitsya(monkeypatch):
    days = [_day(100.0), _day(101.0, lo=90.0)] + [_day(100.0 + i) for i in range(2, 7)]
    p, sig = Params(b=2.0, min_uderzhanie=2), [1, 0, 1, 1, 1, 1, 1]
    assert _run(sig, p, days).trades[1].exit_day is None
    def glob(s, i, j, vrem):
        now = "vremya" if vrem else ("signal" if s._sig_exit(i) else None)
        if M.rano(s, j):
            s.mu = s.mu or (None, now) if now else s.mu
            return None
        return now or (s.mu[1] if s.mu else None)
    monkeypatch.setattr(M, "reshenie", glob)
    assert _run(sig, p, days).trades[1].exit_day is not None                       # чужая память выводит новую позицию


# ---------- G4: разворот в T ----------
def _t_dni(n=7):
    def day(k):
        o = np.array([100.0 * (k + 1) + (m - 590) / 100.0 for m in MODS])
        return Day(o, o + 0.5, o - 0.5, o + 0.25, False, np.array(MODS))
    ds = [day(k) for k in range(n)]
    return ds, np.array([float(d.o[BT]) for d in ds])


def _run_t(sig, p):
    days, cena = _t_dni(len(sig))
    return exec_e.run("T", days, np.array(sig), np.full(len(days), SG), p, cena_T=cena), cena


def test_g4_razvorot_v_T_do_dnya_N_perenositsya_a_vykl_den_1():
    r, cena = _run_t([1, -1, -1, -1, -1, -1], Params(vhod_T="12:00"))
    assert (r.trades[0].exit_day, r.trades[0].reason) == (1, "razvorot")           # v6: разворот день 1
    r, cena = _run_t([1, -1, -1, -1, -1, -1], Params(vhod_T="12:00", min_uderzhanie=2))
    t0, t1 = r.trades[0], r.trades[1]
    assert (t0.exit_day, t0.exit_px, t0.reason) == (2, cena[2], "razvorot")
    assert (t1.entry_day, t1.side, t1.entry_px) == (2, -1, cena[2])


def test_g4_razvorot_zapomnen_signal_vernulsya_zakrytie_v_den_N_i_vhod_po_pravilam():
    """Сигнал дня 1 против позиции, на дне 2 — снова за неё: закрытие дня 2 по cena_T обязательно; _decide открывает по сигналу дня 2."""
    r, cena = _run_t([1, -1, 1, 1, 1, 1], Params(vhod_T="12:00", min_uderzhanie=2))
    t0, t1 = r.trades[0], r.trades[1]
    assert (t0.exit_day, t0.exit_px, t0.reason) == (2, cena[2], "razvorot")
    assert (t1.entry_day, t1.side, t1.entry_px) == (2, 1, cena[2])


def test_g4_mutatsiya_razvorot_ne_perenositsya_lovitsya(monkeypatch):
    sig, p = [1, -1, -1, -1, -1, -1], Params(vhod_T="12:00", min_uderzhanie=2)
    assert _run_t(sig, p)[0].trades[0].exit_day == 2
    monkeypatch.setattr(M, "rano", lambda s, j: False)                              # перенос выключен целиком
    assert _run_t(sig, p)[0].trades[0].exit_day == 1


def test_g4_mutatsiya_razvorot_zabyt_lovitsya(monkeypatch):
    sig, p = [1, -1, 1, 1, 1, 1], Params(vhod_T="12:00", min_uderzhanie=2)
    r, cena = _run_t(sig, p)
    assert _tr(r) == (2, cena[2], "razvorot")
    monkeypatch.setattr(M, "zapomnit_razvorot", lambda s: None)
    r, cena = _run_t(sig, p)
    assert _tr(r) != (2, cena[2], "razvorot")                                       # без памяти закрытие уходит на close, причина signal


def test_g4_vykl_razvorot_t_pobajtno_kak_v6():
    sig = [1, -1, 1, -1, -1, 1, -1]
    a, _ = _run_t(sig, Params(vhod_T="12:00"))
    for m in (0, 1):
        b, _ = _run_t(sig, Params(vhod_T="12:00", min_uderzhanie=m))
        assert repr(a.trades) == repr(b.trades) and a.gross.tobytes() == b.gross.tobytes() and a.cost.tobytes() == b.cost.tobytes()
