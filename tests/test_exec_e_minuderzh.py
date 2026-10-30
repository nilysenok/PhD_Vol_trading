"""V7-4 (Ж-153 §1.1): минимальное удержание контртренда 2 дня — Params.min_uderzhanie (fastbt/exec_e_minuderzh.py).
Выход по сигналу и по τ раньше 2-го дня удержания (дней от дня входа) переносится на 2-й; стоп и тейк — с первого дня.
Синтетика, без сети. Отсчёт — от дня ВХОДА (close d, лимит d+1, T того же дня). G1 выкл. (0, 1) = v6 побайтно; G2 известный ответ;
G3 мутация «стоп/тейк переносятся»; G4 «перенос тренду»; G5 «от дня сигнала» / «вход всегда d+1». pytest tests/test_exec_e_minuderzh.py -q"""
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import exec_e, exec_e_minuderzh as M, exec_e_run
from fastbt.etap1_eval import params_of
from fastbt.etap1_eval_v2 import NOVYE
from fastbt.eval_v6 import EvaluatorV6
from fastbt.eval_v7 import EvaluatorV7
from fastbt.exec_e_types import Day, Params

SG = 0.01


def _day(c, lo=None, hi=None, o=None):
    """Два бара: цены около c; lo/hi — минимум/максимум дня (бар 0), o — open бара 0."""
    o0 = c if o is None else o
    return Day(np.array([o0, c]), np.array([c + 0.1 if hi is None else hi, c + 0.1]),
               np.array([c - 0.1 if lo is None else lo, c - 0.1]), np.array([c, c]))


def _run(sig, p, days=None, **kw):
    days = days or [_day(100.0 + i) for i in range(len(sig))]
    return exec_e.run("X", days, np.array(sig), np.full(len(days), SG), p, **kw)


def _tr(res):
    t = res.trades[0]
    return t.exit_day, t.exit_px, t.reason


# ---------- G2: известный ответ ----------
@pytest.mark.parametrize("side", [1, -1])
def test_g2_signal_dnya_1_perenositsya_na_den_2(side):
    sig = [1, 0, 0, 0, 0]
    r0, r2 = _run(sig, Params(side=side)), _run(sig, Params(side=side, min_uderzhanie=2))
    assert _tr(r0) == (1, 101.0, "signal")                    # без константы — день 1, close дня 1
    assert _tr(r2) == (2, 102.0, "signal")                    # с константой — день 2, close дня 2 (та же ценовая точка)
    assert len(r2.trades) == 1 and r2.trades[0].entry_day == 0


def test_g2_tau_1_vyhod_na_den_2_a_tau_3_ne_menyaetsya():
    s = [1] * 6
    assert _tr(_run([1, 1, 0, 0, 0], Params(min_uderzhanie=2))) == (2, 102.0, "signal")      # сигнал дня 2 — без переноса
    assert _tr(_run(s, Params(tau=1))) == (1, 101.0, "vremya")
    assert _tr(_run(s, Params(tau=1, min_uderzhanie=2))) == (2, 102.0, "vremya")
    assert _tr(_run(s, Params(tau=3, min_uderzhanie=2))) == (3, 103.0, "vremya")


def test_g2_signal_vernulsya_na_den_2_pozitsiya_derzhitsya():
    """Выбор реализации (❓ в PR): не выходим по сигналу раньше 2-го дня; на 2-й день сигнал смотрится заново."""
    assert _run([1, 0, 1, 1, 1, 1], Params(min_uderzhanie=2)).trades[0].exit_day is None
    assert _tr(_run([1, 0, 1, 1, 1, 1], Params())) == (1, 101.0, "signal")


# ---------- G1: выкл. = v6 побайтно ----------
def _sluchajnyj(seed):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80)))
    days = [_day(float(x), lo=float(x) * (1 - abs(rng.normal(0, 0.012))), hi=float(x) * (1 + abs(rng.normal(0, 0.012)))) for x in c]
    return days, rng.integers(-1, 2, 80)


def _b(r):
    return repr(r.trades).encode() + r.gross.tobytes() + r.cost.tobytes() + r.pos.tobytes() + r.free.tobytes()


@pytest.mark.parametrize("seed", range(4))
def test_g1_vykl_ravno_v6_pobaytno(seed):
    days, sig = _sluchajnyj(seed)
    base = dict(b=1.5, a=1.5, tau=4)
    eta = _b(exec_e.run("X", days, sig, np.full(80, SG), Params(**base)))
    for m in (0, 1):
        assert _b(exec_e.run("X", days, sig, np.full(80, SG), Params(min_uderzhanie=m, **base))) == eta
    on = _b(exec_e.run("X", days, sig, np.full(80, SG), Params(min_uderzhanie=2, **base)))
    assert on != eta                                          # контроль: на этом ряду константа что-то меняет


def test_g1_novoe_pole_otdayot_tochku_etalonu():
    from fastbt import fast_etap1
    assert fast_etap1._slow_only(Params(min_uderzhanie=2), None) and not fast_etap1._slow_only(Params(), None)


# ---------- G3: стоп и тейк не переносятся ----------
def _off_early(real, i):
    """Мутация: уровень i (0 — стоп, 1 — тейк) в первые 2 дня снят, т. е. «переехал на день 2»."""
    def f(s, j):
        lv = list(real(s, j))
        lv[i] = None if j - s.ej < 2 else lv[i]
        return tuple(lv)
    return f


@pytest.mark.parametrize("i,kw,hi_lo,ожидание", [
    (0, dict(b=2.0), dict(lo=90.0), (1, 98.0, "stop")), (1, dict(a=2.0), dict(hi=110.0), (1, 102.0, "take"))])
def test_g3_mutatsiya_stop_tejk_perenosyatsya_lovitsya(monkeypatch, i, kw, hi_lo, ожидание):
    """G2: стоп S = 100·(1 − 2·0,01) = 98 (пробой дня 1 → min(open, S)), тейк T = 102 (high 110 → max(open, T)): день 1 и с константой, и без."""
    days = [_day(100.0), _day(101.0, o=101.0, **hi_lo)] + [_day(100.0 + k) for k in range(2, 5)]
    assert _tr(_run([1] * 5, Params(**kw), days)) == ожидание
    assert _tr(_run([1] * 5, Params(min_uderzhanie=2, **kw), days)) == ожидание          # правка проходит
    monkeypatch.setattr(exec_e_run.Run, "_levels", _off_early(exec_e_run.Run._levels, i))
    assert _tr(_run([1] * 5, Params(min_uderzhanie=2, **kw), days)) != ожидание          # мутация падает


# ---------- G4: тренду (S3/S4) не применяется ----------
def _pt():
    return dict(k_s="net", tau="net", vhod="rynok", R="net", k_tr="net", bezubytok="net", proverka="bary")


@pytest.fixture(autouse=True)
def _bez_dannyh(monkeypatch):
    monkeypatch.setattr(EvaluatorV6, "__init__", lambda self, *a, **k: None)       # EvaluatorV7 без данных: нужен только _params


def _ev(sid, n):
    ev = EvaluatorV7(sid, None, {}, min_uderzhanie=n)
    ev.inp, ev.comm, ev._fk = SimpleNamespace(tick={"X": 0.01}), 0.0004, ()
    return ev


def _params(sid, n):
    return _ev(sid, n)._params(_pt(), "X", 5.0, False, dict(NOVYE), {})


@pytest.mark.parametrize("sid,ozhid", [("S1", 2), ("S2", 2), ("S5", 2), ("S6", 2), ("S3", 0), ("S4", 0)])
def test_g4_konstanta_tolko_kontrtrendu(sid, ozhid):
    assert _params(sid, 2).min_uderzhanie == ozhid


@pytest.mark.parametrize("sid", ["S1", "S3"])
def test_g4_bez_konstanty_params_i_klyuch_kak_v6(sid):
    assert _params(sid, 0) == params_of(_pt(), 0.01, 5.0, False, 0.0004) and _ev(sid, 0)._fkey == ()
    assert _ev(sid, 2)._fkey == ((M.KLYUCH, 2),) * (sid == "S1")        # Д5: константа — в ключе кэша контртренда; тренду ключ прежний


def test_g4_mutatsiya_perenos_tomu_zhe_trendu_lovitsya(monkeypatch):
    monkeypatch.setattr(M, "kontrtrend", lambda sid: True)
    assert _params("S3", 2).min_uderzhanie == 2                          # мутация действует; тест выше на ней упал бы
    with pytest.raises(AssertionError):
        test_g4_konstanta_tolko_kontrtrendu("S3", 0)


# ---------- G5: отсчёт от дня входа ----------
def _ot_dnya_signala(dobavka):
    def f(s, j):
        return j - (s.ix.index(s.tr.d_sig) + dobavka) < s.p.min_uderzhanie
    return f


def _lim_dni():
    """Сигнал дня 0, лимит d+1 (k = 0,5σ: L = 99,5) исполняется на дне 1 баром 0; дальше ровно."""
    return [_day(100.0), _day(100.0, lo=99.0, o=100.0)] + [_day(100.0 + i) for i in range(2, 7)]


def test_g5_vhod_d_plus_1_otschet_ot_dnya_vhoda(monkeypatch):
    sig, p = [1, 1, 0, 0, 0, 0, 0], Params(k=0.5, min_uderzhanie=2)
    r = _run(sig, p, _lim_dni())
    assert (r.trades[0].entry_day, r.trades[0].d_sig) == (1, 0)
    assert _tr(r) == (3, 103.0, "signal")                     # вход день 1, выход день 2 → перенос на день 3
    monkeypatch.setattr(M, "rano", _ot_dnya_signala(0))
    assert _tr(_run(sig, p, _lim_dni())) == (2, 102.0, "signal")         # мутация считает от дня сигнала → выходит раньше


def _t_dni():
    mods = list(range(590, 1140, 5))
    def day(k):
        o = np.array([100.0 * (k + 1) + (m - 590) / 100.0 for m in mods])
        return Day(o, o + 0.5, o - 0.5, o + 0.25, False, np.array(mods))
    return [day(k) for k in range(6)], np.array([100.0 * (k + 1) + (720 - 590) / 100.0 for k in range(6)])


def test_g5_vhod_v_T_otschet_ot_dnya_vhoda(monkeypatch):
    days, cena = _t_dni()
    sig, p = [0, 1, 0, 0, 0, 0], Params(vhod_T="12:00", min_uderzhanie=2)
    r = _run(sig, p, days, cena_T=cena)
    assert (r.trades[0].entry_day, r.trades[0].d_sig) == (1, 1)
    assert r.trades[0].exit_day == 3                            # вход день 1 (в T), выход по сигналу дня 2 → день 3
    assert _run(sig, Params(vhod_T="12:00"), days, cena_T=cena).trades[0].exit_day == 2
    monkeypatch.setattr(M, "rano", _ot_dnya_signala(1))        # мутация «вход всегда d+1»: в T считает на день меньше
    assert _run(sig, p, days, cena_T=cena).trades[0].exit_day == 4


def test_g5_tau_ot_dnya_vhoda_pri_t_vyhod_1500(monkeypatch):
    """Run2._scan режет τ на баре 15:00 дня истечения — константа переносит и его (день 2, 15:00)."""
    mods = np.array([600, 900, 1000])
    def day(c):
        o = np.array([c, c, c])
        return Day(o, o + 0.1, o - 0.1, o, False, mods)
    days = [day(100.0 + i) for i in range(5)]
    p0 = Params(tau=1, t_vyhod="1500")
    assert _run([1] * 5, p0, days).trades[0].exit_day == 1
    r = _run([1] * 5, Params(tau=1, t_vyhod="1500", min_uderzhanie=2), days)
    assert (r.trades[0].exit_day, r.trades[0].exit_bar, r.trades[0].exit_px) == (2, 1, 102.0)
