"""Ж-086 п. 4: выходы open2 / 1500 / 1700 — по open 5-минутного бара, начинающегося в 10:00 / 15:00 / 17:00.

Синтетика, без данных. Бары дня: 09:50 (590) … 18:45, шаг 5 мин → индекс 10:00 = 2, 15:00 = 62, 17:00 = 86.
Нет бара ровно в hhmm — первый после; после hhmm баров нет — open бара 10:00 следующего дня (нет его — первый после 10:00).
Стоп и тейк — по барам строго до бара выхода. Лимитный вход k_e не меняется.
Мутации (каждая обязана ронять свой тест): open2 по первому бару; 15:00 по close 14:55; 17:00 по open 16:55;
τ при 1500 не по 15:00; вход k_e со скана с 10:00.
"""
import numpy as np
import pytest

from fastbt import exec_e2, exec_e_vremya
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

T_START, N = 590, 108
SIG = (1, 1, 0, 0, 0)          # вход по close дня 0; сигнал снят на дне 2 → выход на дне 3 (x+1)


def day(over=None, t=None):
    t = np.arange(T_START, T_START + 5 * N, 5) if t is None else np.asarray(t)
    a = np.full((len(t), 4), 100.0)
    for b, v in (over or {}).items():
        a[b] = v
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], False, t)


def rn(days, sig=SIG, **kw):
    kw.setdefault("comm", 0.0)
    return run("T", days, list(sig), np.full(len(days), 0.02), Params(**kw))


def five(d3, d4=None, **kw):
    return rn([day(), day(), day(), d3, d4 or day()], **kw)


def prev_bar(price):
    """Мутация: выход на бар раньше (b−1), цена — open или close того бара."""
    orig = exec_e2.Run2._exit_t

    def f(s, i, j, b, px, reason):
        d = s.days[i]
        return orig(s, i, j, b - 1, d.c[b - 1] if price == "c" else d.o[b - 1], reason)
    return f


D_OPEN2 = {0: (100.7, 100.9, 100.6, 100.8), 1: (100.5, 100.6, 100.4, 100.55), 2: (101.2, 101.5, 101.0, 101.4)}
D_1500 = {61: (100.1, 100.3, 99.9, 99.0), 62: (101.3, 101.6, 101.0, 105.0)}
D_1700 = {85: (100.1, 100.3, 99.9, 100.2), 86: (102.4, 102.6, 102.3, 102.5)}


def one(r):
    t = r.trades[0]
    return t.exit_day, t.exit_bar, t.exit_px, t.reason, r.x_t


def test_open2_bar_1000_not_first():
    assert one(five(day(D_OPEN2), t_vyhod="open2")) == (3, 2, 101.2, "signal", 1)


def test_open2_net_bara_1000_pervyj_pozzhe():
    t = [590, 595, 605] + list(range(610, 590 + 5 * N, 5))
    over = {0: (100.7, 100.9, 100.6, 100.8), 1: (100.5, 100.6, 100.4, 100.55), 2: (101.2, 101.5, 101.0, 101.4)}
    assert one(five(day(over, t), t_vyhod="open2")) == (3, 2, 101.2, "signal", 1)


def test_1500_open_baru_1500_ne_close_1455():
    assert one(five(day(D_1500), t_vyhod="1500")) == (3, 62, 101.3, "signal", 1)


def test_1700_open_bara_1700_ne_open_1655():
    assert one(five(day(D_1700), t_vyhod="1700")) == (3, 86, 102.4, "signal", 1)


def test_net_bara_posle_hhmm_open_1000_sleduyushchego_dnya():
    d3 = day(t=np.arange(T_START, 960, 5))                          # день кончается в 16:00 — 17:00 нет
    d4 = day({0: (97.0, 97.2, 96.8, 97.1), 2: (103.0, 103.2, 102.8, 103.1)})   # open бара 10:00, не первого бара
    assert one(five(d3, d4, t_vyhod="1700")) == (4, 2, 103.0, "signal", 1)
    t4 = [600 + 5 * k for k in range(60)]                           # утренней сессии нет, 10:00 есть → первый бар = 10:00
    d4b = day({0: (104.0, 104.2, 103.8, 104.1)}, t4)
    assert one(five(d3, d4b, t_vyhod="1700")) == (4, 0, 104.0, "signal", 1)
    t4 = [590, 595] + [605 + 5 * k for k in range(60)]              # бара 10:00 нет → первый после 10:00
    d4c = day({2: (105.0, 105.2, 104.8, 105.1)}, t4)
    assert one(five(d3, d4c, t_vyhod="1700")) == (4, 2, 105.0, "signal", 1)


def test_stop_do_bara_vyhoda_srabatyvaet_a_v_bare_vyhoda_net():
    s = one(five(day({61: (100, 100, 97.5, 99.0), 62: (101, 101, 97, 100)}), t_vyhod="1500", b=1.0))
    assert s == (3, 61, 98.0, "stop", 0)                            # стоп 98 — на 14:55, строго до бара выхода
    s = one(five(day({62: (101, 101, 97, 100)}), t_vyhod="1500", b=1.0))
    assert s == (3, 62, 101.0, "signal", 1)                         # low бара выхода не проверяется: выход по его open


def test_tau_1500_po_open_1500_dnya_x():
    days = [day(), day(), day(D_1500), day()]
    assert one(rn(days, sig=(1, 1, 1, 1), t_vyhod="1500", tau=2)) == (2, 62, 101.3, "vremya", 1)


def test_vhod_k_e_skan_s_pervogo_bara():
    d1 = day({1: (99.5, 99.6, 98.9, 99.2)})                         # 09:55: low ≤ L = 100·(1 − 0.5·0.02) = 99
    r = rn([day(), d1, day(), day(), day()], sig=(1, 1, 1, 1, 1), k=0.5, t_vyhod="open2")
    t = r.trades[0]
    assert (t.entry_day, t.entry_bar, t.entry_px) == (1, 1, 99.0)


def test_close_yakor_ne_menyaetsya():
    t = five(day(D_1500), t_vyhod="close").trades[0]
    assert (t.exit_day, t.exit_px, t.reason) == (2, 100.0, "signal")


# --- мутации ---
def test_mut_open2_po_pervomu_baru(monkeypatch):
    orig = exec_e_vremya.bar_at
    monkeypatch.setattr(exec_e_vremya, "bar_at", lambda d, h: 0 if h == 600 else orig(d, h))
    with pytest.raises(AssertionError):
        test_open2_bar_1000_not_first()


def test_mut_1500_po_close_1455(monkeypatch):
    monkeypatch.setattr(exec_e2.Run2, "_exit_t", prev_bar("c"))
    try:
        test_1500_open_baru_1500_ne_close_1455()
    except AssertionError:
        return
    except ValueError as e:     # инвариант #617: исполнение вне [low; high] — тоже поймана
        assert "вне [low; high]" in str(e), f"иная ValueError: {e}"
        return
    pytest.fail("мутант не пойман: проверка прошла")


def test_mut_1700_po_open_1655(monkeypatch):
    monkeypatch.setattr(exec_e2.Run2, "_exit_t", prev_bar("o"))
    with pytest.raises(AssertionError):
        test_1700_open_bara_1700_ne_open_1655()


def test_mut_tau_1500_ne_po_1500(monkeypatch):
    monkeypatch.setattr(exec_e_vremya, "vyhod_plan", lambda tv, pr: (1, exec_e_vremya.HHMM[tv]))
    with pytest.raises(AssertionError):
        test_tau_1500_po_open_1500_dnya_x()


def test_mut_vhod_k_e_skan_s_1000(monkeypatch):
    orig = exec_e2.limit_mask

    def f(sd, h, l, L, tick):
        m = np.array(orig(sd, h, l, L, tick))
        m[:2] = False                                               # бары до 10:00 не сканируются
        return m
    monkeypatch.setattr(exec_e2, "limit_mask", f)
    with pytest.raises((AssertionError, IndexError)):
        test_vhod_k_e_skan_s_pervogo_bara()
