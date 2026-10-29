"""Т-v611 (тестировщик): добавка к test_exec_e_stop_v6osi.py — дробный шаг трейлинга и зеркала шорта с известным ответом.
Лонг: вход 102, 1σ = 1,02. Шорт: вход 98, 1σ = 0,98. Уровни действуют со следующего бара; цена выхода — худшая из open и уровня.
Мутации: step_level floor -> ceil (экстремум 1,5 шага даёт 2 шага вместо 1); знак в plus_level.
python3 -m pytest tests/test_exec_e_stop_v6osi_t.py -q"""
import pytest
from test_exec_e_stop_osi import LONG, SHORT, P
from test_exec_e_stop_v6osi import RANG, _go

ENT = (101, 102.4, 101.5, 102.2)                    # бар входа лонга: high 0,39 шага, low выше начального стопа 100,98


def test_step_s_poltora_shaga_long_floor():
    """Экстремум 103,53 = 1,5 шага (1,02): целых шагов 1 -> стоп 100,98 + 1,02 = 102,0 (ceil дал бы 103,02)."""
    t = _go([[ENT], [(102.2, 103.53, 102.6, 103.4), (103.4, 103.4, 101.9, 102)]], P(b=1.0, step_s=1.0)).trades[0]
    # ceil даёт «фантомный» уровень 102,0 уже на баре входа (экстремум = вход -> n=1): выход в день входа, а не на баре 1 дня 16
    assert (t.reason, t.exit_px, t.exit_day, t.exit_bar) == ("trail", pytest.approx(102.0, abs=1e-9), t.entry_day + 1, 1)


def test_step_s_poltora_shaga_short_floor():
    """Шорт: экстремум 96,53 = 1,5 шага (0,98): 1 шаг -> стоп 98,98 - 0,98 = 98,0 (ceil дал бы 97,02)."""
    ent_s = [(98.5, 98.5, 97.6, 97.8)]                # sell-stop 98; high ниже начального стопа 98,98 (у SHORT high 100 выбил бы его)
    t = _go([ent_s, [(98, 98.2, 96.53, 96.8), (97, 98.1, 96.9, 97.5)]], P(b=1.0, step_s=1.0)).trades[0]
    assert (t.side, t.reason, t.exit_px, t.exit_day) == (-1, "trail", pytest.approx(98.0, abs=1e-9), t.entry_day + 1)


def test_step_s_menshe_shaga_net_trejla():
    """0,98 шага: уровня нет, выход по начальному стопу 100,98, не trail."""
    t = _go([LONG, [(102.2, 102.99, 102.0, 102.5), (102.5, 102.5, 100.0, 100.5)]], P(b=1.0, step_s=1.0)).trades[0]
    assert (t.reason, t.exit_px) == ("stop", pytest.approx(100.98, abs=1e-9))


@pytest.mark.parametrize("p,kw", [(P(b=5.0, b0=1.0), {}), (P(b=5.0, b0_c=2.0), dict(rang=RANG))])
def test_bezubytok_short(p, kw):
    """Проход -1σ (97,02) в баре 1; в баре 2 возврат к входу 98: выход по уровню 98,0 (open 97 лучше уровня)."""
    bars = [SHORT, [(98, 98.2, 96.9, 97), (97, 98.1, 97, 97.5)]]
    t = _go(bars, p, **kw).trades[0]
    assert (t.side, t.reason, t.exit_px, t.exit_bar) == (-1, "bu", 98.0, 1)
    assert _go(bars, P(b=5.0)).trades[0].reason is None


def test_bezubytok_short_gep_po_open():
    t = _go([SHORT, [(98, 98.2, 96.9, 97), (99, 99.5, 98.8, 99)]], P(b=5.0, b0=1.0)).trades[0]
    assert (t.reason, t.exit_px) == ("bu", 99.0)       # open 99 хуже уровня 98


def test_stop_v_plyus_short():
    """Шорт, sp=(1;0,5): после -1σ стоп 98·(1 - 0,005) = 97,51; бар 2: open 97,3, high 97,6 -> выход 97,51."""
    t = _go([SHORT, [(98, 98.2, 96.9, 97), (97.3, 97.6, 96.8, 97.0)]], P(b=5.0, sp=(1.0, 0.5))).trades[0]
    assert (t.side, t.reason, t.exit_px) == (-1, "trail", pytest.approx(98 * 0.995, abs=1e-9))


def test_stop_v_plyus_short_ne_v_tom_zhe_bare():
    """Уровень со следующего бара: в баре прохода high 98,2 выше 97,51, но выхода нет."""
    t = _go([SHORT, [(98, 98.2, 96.9, 97)]], P(b=5.0, sp=(1.0, 0.5))).trades[0]
    assert t.reason is None
