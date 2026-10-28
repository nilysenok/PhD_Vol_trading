"""Ж-074: добор с плечом внутри дня — синтетика с известным ответом и 4 подложенные ошибки (мутации читают исходник
модуля, подменяют одну строку и прогоняют тот же сценарий: правка проходит, мутация обязана упасть).
Нигде не читаются данные и не используется сеть."""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from fastbt import diag_zh074 as z
from fastbt.exec_e_types import Day

SRC = Path(z.__file__).read_text(encoding="utf-8")
COMM, RATE, CAL = 0.0004, 0.10, 3


def day(rows):
    a = np.array(rows, float)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


# бары (open, high, low, close); вход лонг по 100 на баре 0, σ_d = 1 %, x = 0,5 → цель log(цена/100) ≥ 0,5 %
LONG = [(100.0, 100.2, 99.9, 100.1), (100.1, 100.4, 100.0, 100.3), (100.4, 100.8, 100.4, 100.6),
        (100.6, 100.7, 100.5, 99.0), (99.0, 99.5, 98.5, 99.2), (99.2, 99.5, 99.0, 99.3), (99.3, 99.6, 99.1, 99.4)]
# шорт зеркально: вход 100, цель — вниз; бар срабатывания 2 (low 99,2), продажа лимиткой по его close 99,4
SHORT = [(100.0, 100.1, 99.8, 99.9), (99.9, 100.0, 99.6, 99.7), (99.6, 99.7, 99.2, 99.4),
         (99.4, 99.5, 99.3, 101.0), (101.0, 101.5, 100.5, 101.2), (101.2, 101.5, 101.0, 101.3), (101.3, 101.6, 101.1, 101.4)]


def mk(mod, bars, side, xpx, last_ok=6):
    return mod.Tr(side, 0.01, 100.0, 0, day(bars), last_ok, xpx, 0.0, COMM, None, CAL, RATE, "take")


def load(old=None, new=None):
    """Модуль из исходника; с old/new — с подложенной ошибкой (строка обязана существовать)."""
    src = SRC
    if old is not None:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    mod = types.ModuleType("diag_zh074_mut")
    mod.__file__ = z.__file__
    sys.modules[mod.__name__] = mod                                              # dataclass ищет свой модуль
    exec(compile(src, "diag_zh074_mut", "exec"), mod.__dict__)
    return mod


def scenario(mod):
    """Известный ответ. Лонг: срабатывание на баре 2, добор по 100,6 исполняется на баре 3 (low 100,5 < 100,6)."""
    t = mk(mod, LONG, 1, 101.0)
    assert mod.trigger(t, 12, 0.5) == 2
    assert mod.trigger(t, 2, 0.5) is None and mod.trigger(t, 3, 0.5) == 2        # окно N баров от бара входа
    m, px = mod.fill(t, 2)
    assert (m, px) == (3, 100.6)                                                 # не в баре срабатывания, не по open бара 2
    one = 101.0 / 100.6 - 1.0 - 2 * COMM
    mg = RATE * CAL / 365.0
    assert mod.add_pnl(t, px, 2.0) == pytest.approx(one - mg, abs=1e-12)         # маржа — на часть сверх 100 % (L−1 = 1)
    assert mod.add_pnl(t, px, 1.5) == pytest.approx(0.5 * (one - mg), abs=1e-12)
    assert mod.add_pnl(t, px, 1.0) == 0.0
    s = mk(mod, SHORT, -1, 99.0)
    k = mod.trigger(s, 12, 0.5)
    assert k == 2                                                                # шорт: «в сторону позиции» — вниз
    m, px = mod.fill(s, k)
    assert (m, px) == (3, 99.4)                                                  # продажа: high 99,5 > 99,4 на баре 3, не в баре 2
    assert mod.add_pnl(s, px, 2.0) == pytest.approx(-(99.0 / 99.4 - 1.0) - 2 * COMM - mg, abs=1e-12)


def test_scenario_pravka_prohodit():
    scenario(load())


def test_mutaciya_1_usloviye_proveryaet_bary_posle_srabatyvaniya_lovitsya():
    mut = load("            return k\n    return None\n\n\ndef fill",
               "            return None if any(s * (d.c[q] - t.px) < 0 for q in range(k, min(t.eb + N, len(d.c)))) else k\n"
               "    return None\n\n\ndef fill")
    with pytest.raises(AssertionError):
        scenario(mut)


def test_mutaciya_2_dobor_v_bare_srabatyvaniya_lovitsya():
    mut = load("for m in range(k + 1, t.last_ok + 1)", "for m in range(k, t.last_ok + 1)")
    with pytest.raises(AssertionError):
        scenario(mut)


def test_mutaciya_3_marzha_so_vsei_pozicii_lovitsya():
    mut = load("- w * t.rate * t.cal / 365.0", "- L * t.rate * t.cal / 365.0")
    with pytest.raises(AssertionError):
        scenario(mut)


def test_mutaciya_4_short_kak_long_lovitsya():
    mut = load("d, s, need = t.day, t.side, x * t.sg", "d, s, need = t.day, 1, x * t.sg")
    with pytest.raises(AssertionError):
        scenario(mut)


def test_plohoy_close_do_srabatyvaniya_blokiruet():
    bars = [(100.0, 100.2, 99.9, 100.1), (100.1, 100.2, 99.0, 99.5), (99.5, 101.5, 99.5, 101.0)]
    assert z.trigger(mk(z, bars, 1, 101.0, 2), 12, 0.5) is None                  # бар 1 закрылся ниже входа


def test_dobor_ne_ispolnen_esli_cena_ne_vernulas_i_posle_vykhoda():
    bars = [(100.0, 100.2, 99.9, 100.1), (100.1, 101.0, 100.1, 100.9), (100.9, 101.5, 100.9, 101.2)]
    t = mk(z, bars, 1, 101.0, 2)
    assert z.trigger(t, 12, 0.5) == 1 and z.fill(t, 1) is None                   # low бара 2 не ниже 100,9
    t2 = mk(z, LONG, 1, 101.0, z.last_ok_bar(7, True, 3, "stop"))                # стоп на баре 3: жив до бара 2
    assert t2.last_ok == 2 and z.fill(t2, 2) is None


def test_last_ok_bar():
    assert z.last_ok_bar(60, False, 10, "stop") == 59
    assert z.last_ok_bar(60, True, 10, "take") == 9
    assert z.last_ok_bar(60, True, 59, "signal") == 59


def test_l1_pnl_ravno_summe_zhurnala_i_plus_vnutri_dnya():
    a, b = mk(z, LONG, 1, 101.0), mk(z, SHORT, -1, 99.0)
    c = mk(z, LONG, 1, 99.5)
    a.reason, b.reason, c.reason = "take", "stop", "stop"
    trs = [a, b, c]
    for N in z.GRID_N:
        for x in z.GRID_X:
            assert math.isclose(z.pnl_at(trs, N, x, 1.0), sum(t.pnl for t in trs), abs_tol=1e-9)
    assert z.pnl_at(trs, 12, 0.5, 2.0) > z.pnl_at(trs, 12, 0.5, 1.0) - 1                 # плечо считается
    n, s = z.plus_intraday(trs)
    assert n == 2 and s == pytest.approx(a.pnl + b.pnl)                              # c — в минус, не входит
    assert len(z.dobor_rows("S1", "0,04", trs)[0]) == len(z.C_DOB) and len(z.day_rows("S1", "0,04", trs)[0]) == len(z.C_DAY)
