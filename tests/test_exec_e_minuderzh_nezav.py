"""V7-4 (#917), независимая проверка: min_uderzhanie = N (Ж-153 §1.1). Известный ответ на синтетике (вход d+1 и вход в T) и мутации в памяти.
Правило: выход по сигналу/τ/развороту в T раньше N-го дня от ВХОДА переносится на N-й и исполняется там обязательно; стоп/тейк — с 1-го дня;
тренд (S3, S4) не затронут; N ≤ 1 — путь v6. Без сети. pytest tests/test_exec_e_minuderzh_nezav.py -q"""
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import exec_e, exec_e_minuderzh as M, exec_e_run
from fastbt.etap1_eval_v2 import NOVYE
from fastbt.eval_v6 import EvaluatorV6
from fastbt.eval_v7 import EvaluatorV7
from fastbt.exec_e_types import Day, Params

SG = 0.01


def _day(c, lo=None, hi=None, o=None):
    o0 = c if o is None else o
    return Day(np.array([o0, c]), np.array([c + 0.1 if hi is None else hi, c + 0.1]),
               np.array([c - 0.1 if lo is None else lo, c - 0.1]), np.array([c, c]))


def _run(sig, p, days, **kw):
    return exec_e.run("X", days, np.array(sig), np.full(len(days), SG), p, **kw)


def _tr(res, k=0):
    t = res.trades[k]
    return t.exit_day, None if t.exit_px is None else round(float(t.exit_px), 6), t.reason


def _dni_d1():
    """Сигнал дня 0, лимит d+1 (k = 0,5σ, L = 99,5) исполняется на дне 1; close дня k = 100 + k после дня 1."""
    return [_day(100.0), _day(100.0, lo=99.0, o=100.0)] + [_day(100.0 + i) for i in range(2, 8)]


def _dni_T():
    mods = list(range(590, 1140, 5))
    def day(k):
        o = np.array([100.0 * (k + 1) + (m - 590) / 100.0 for m in mods])
        return Day(o, o + 0.5, o - 0.5, o + 0.25, False, np.array(mods))
    return [day(k) for k in range(7)], np.array([100.0 * (k + 1) + 1.3 for k in range(7)])


# ---------- известный ответ: вход d+1 ----------
def test_vhod_d1_signal_dnya_1_perenositsya_na_den_vhoda_plus_2():
    sig, p = [1, 1, 0, 0, 0, 0, 0, 0], Params(k=0.5, min_uderzhanie=2)
    r = _run(sig, p, _dni_d1())
    assert (r.trades[0].d_sig, r.trades[0].entry_day) == (0, 1)
    assert _tr(r) == (3, 103.0, "signal")                       # сигнал ушёл на дне 2 (1 день удержания) → исполнение на дне 3
    assert _tr(_run(sig, Params(k=0.5), _dni_d1())) == (2, 102.0, "signal")        # N выкл.: день 2


def test_vhod_d1_signal_vernulsya_vse_ravno_vyhod_na_n():
    """Сигнал ушёл на дне 2 и вернулся на дне 3: исполнение дня 3 обязательно (по цене close дня 3)."""
    assert _tr(_run([1, 1, 0, 1, 1, 1, 1, 1], Params(k=0.5, min_uderzhanie=2), _dni_d1())) == (3, 103.0, "signal")


def test_vhod_d1_signal_na_n_ne_perenositsya_dalshe():
    """Сигнал ушёл только на дне 3 (≥ N): без переноса, день 3."""
    assert _tr(_run([1, 1, 1, 0, 0, 0, 0, 0], Params(k=0.5, min_uderzhanie=2), _dni_d1())) == (3, 103.0, "signal")


def test_vhod_d1_tau_perenositsya():
    r = _run([1] * 8, Params(k=0.5, tau=1, min_uderzhanie=2), _dni_d1())
    assert _tr(r) == (3, 103.0, "vremya")
    assert _tr(_run([1] * 8, Params(k=0.5, tau=1), _dni_d1())) == (2, 102.0, "vremya")


# ---------- известный ответ: вход в T ----------
def test_vhod_T_signal_perenositsya():
    days, cena = _dni_T()
    p = Params(vhod_T="12:00", min_uderzhanie=2)
    r = _run([0, 1, 0, 0, 0, 0, 0], p, days, cena_T=cena)
    assert (r.trades[0].d_sig, r.trades[0].entry_day) == (1, 1)
    assert r.trades[0].exit_day == 3 and r.trades[0].reason == "signal"
    assert _run([0, 1, 0, 0, 0, 0, 0], Params(vhod_T="12:00"), days, cena_T=cena).trades[0].exit_day == 2


def test_razvorot_v_T_perenositsya_i_obyazatelen():
    days, cena = _dni_T()
    sig = [0, 1, -1, 1, 1, 1, 1]                                 # день 2: разворот (1 день удержания) → день 3; на дне 3 сигнал вернулся за позицию
    r = _run(sig, Params(vhod_T="12:00", min_uderzhanie=2), days, cena_T=cena)
    assert _tr(r) == (3, 401.3, "razvorot")
    r0 = _run(sig, Params(vhod_T="12:00"), days, cena_T=cena)
    assert _tr(r0) == (2, 301.3, "razvorot")


def test_razvorot_v_T_na_n_ne_perenositsya():
    days, cena = _dni_T()
    r = _run([0, 1, 1, -1, -1, -1, -1], Params(vhod_T="12:00", min_uderzhanie=2), days, cena_T=cena)
    assert _tr(r) == (3, 401.3, "razvorot")


# ---------- стоп и тейк с первого дня ----------
def _dni_stop():
    return [_day(100.0), _day(101.0, o=101.0, lo=90.0)] + [_day(100.0 + k) for k in range(2, 6)]


def test_stop_s_pervogo_dnya_s_konstantoy():
    p = Params(b=2.0, min_uderzhanie=2)
    assert _tr(_run([1] * 6, p, _dni_stop())) == (1, 98.0, "stop")


def test_tejk_s_pervogo_dnya_s_konstantoy():
    days = [_day(100.0), _day(101.0, o=101.0, hi=110.0)] + [_day(100.0 + k) for k in range(2, 6)]
    assert _tr(_run([1] * 6, Params(a=2.0, min_uderzhanie=2), days)) == (1, 102.0, "take")


def test_otlozhennyj_vyhod_ne_techet_v_novuyu_poziciyu():
    """Ранний сигнал (день 1) отложен, но стоп дня 1... стоп раньше: позиция закрыта стопом; новая сделка не наследует отложенный выход."""
    days = [_day(100.0), _day(100.0), _day(100.0, lo=90.0)] + [_day(100.0) for _ in range(5)]
    r = _run([1, 0, 1, 1, 1, 1, 1, 1], Params(b=2.0, min_uderzhanie=2), days)
    assert _tr(r, 0) == (2, 98.0, "stop")                         # сигнал дня 1 отложен на день 2, но стоп дня 2 сработал внутри бара раньше выхода
    assert len(r.trades) >= 2 and r.trades[1].entry_day >= 2 and r.trades[1].exit_day is None


# ---------- мутации ----------
def _mut_perenos_zabyt(monkeypatch):
    """Перенос забыт: ранний сигнал просто игнорируется без памяти; на дне N смотрится только текущий сигнал."""
    def reshenie(s, i, j, vrem):
        now = "vremya" if vrem else ("signal" if s._sig_exit(i) else None)
        return None if M.rano(s, j) else now
    monkeypatch.setattr(M, "reshenie", reshenie)


def _mut_stop_perenositsya(monkeypatch):
    real = exec_e_run.Run._levels
    def f(s, j):
        lv = list(real(s, j))
        return (None, None) + tuple(lv[2:]) if M.rano(s, j) else tuple(lv)
    monkeypatch.setattr(exec_e_run.Run, "_levels", f)


def _mut_ot_dnya_signala(monkeypatch):
    monkeypatch.setattr(M, "rano", lambda s, j: s.p.min_uderzhanie > 1 and j - s.ix.index(s.tr.d_sig) < s.p.min_uderzhanie)


def _mut_razvorot_zabyt(monkeypatch):
    monkeypatch.setattr(M, "zapomnit_razvorot", lambda s: None)


def test_mut_perenos_zabyt_lovitsya(monkeypatch):
    _mut_perenos_zabyt(monkeypatch)
    with pytest.raises(AssertionError):
        test_vhod_d1_signal_vernulsya_vse_ravno_vyhod_na_n()


def test_mut_razvorot_zabyt_lovitsya(monkeypatch):
    _mut_razvorot_zabyt(monkeypatch)
    with pytest.raises(AssertionError):
        test_razvorot_v_T_perenositsya_i_obyazatelen()


def test_mut_stop_perenositsya_lovitsya(monkeypatch):
    _mut_stop_perenositsya(monkeypatch)
    with pytest.raises(AssertionError):
        test_stop_s_pervogo_dnya_s_konstantoy()


def test_mut_ot_dnya_signala_lovitsya(monkeypatch):
    _mut_ot_dnya_signala(monkeypatch)
    with pytest.raises(AssertionError):
        test_vhod_d1_signal_dnya_1_perenositsya_na_den_vhoda_plus_2()


# ---------- тренд и ключ кэша ----------
@pytest.fixture
def _bez_dannyh(monkeypatch):
    monkeypatch.setattr(EvaluatorV6, "__init__", lambda self, *a, **k: None)


def _ev(sid, n):
    ev = EvaluatorV7(sid, None, {}, min_uderzhanie=n)
    ev.inp, ev.comm, ev._fk = SimpleNamespace(tick={"X": 0.01}), 0.0004, ()
    return ev


def _pt():
    return dict(k_s="net", tau="net", vhod="rynok", R="net", k_tr="net", bezubytok="net", proverka="bary")


def _n_v_params(sid):
    return _ev(sid, 2)._params(_pt(), "X", 5.0, False, dict(NOVYE), {}).min_uderzhanie


def _proverka_trend():
    assert [_n_v_params(s) for s in ("S1", "S2", "S5", "S6")] == [2, 2, 2, 2]
    assert [_n_v_params(s) for s in ("S3", "S4")] == [0, 0]


def test_trend_ne_zatronut(_bez_dannyh):
    _proverka_trend()


def test_mut_perenos_u_trenda_lovitsya(_bez_dannyh, monkeypatch):
    monkeypatch.setattr(M, "kontrtrend", lambda sid: True)
    with pytest.raises(AssertionError):
        _proverka_trend()


def test_n_menshe_dvuh_put_v6(_bez_dannyh):
    for n in (0, 1, -3):
        assert _ev("S1", n)._fkey == () and _n_v_params("S1") == 2
        assert _ev("S1", n)._params(_pt(), "X", 5.0, False, dict(NOVYE), {}).min_uderzhanie == 0


def _proverka_klyucha():
    k0, k2, k3 = (_ev("S1", n)._fkey for n in (0, 2, 3))
    assert k0 != k2 and k2 != k3 and k2 == _ev("S1", 2)._fkey
    assert _ev("S3", 2)._fkey == _ev("S3", 0)._fkey


def test_min_uderzhanie_v_klyuche_kesha(_bez_dannyh):
    _proverka_klyucha()


def test_mut_min_uderzhanie_ne_v_klyuche_lovitsya(_bez_dannyh, monkeypatch):
    real = EvaluatorV7._k7
    def bez(self):
        s, self.min_uder = self.min_uder, 0
        try:
            return real(self)
        finally:
            self.min_uder = s
    monkeypatch.setattr(EvaluatorV7, "_k7", bez)
    with pytest.raises(AssertionError):
        _proverka_klyucha()
