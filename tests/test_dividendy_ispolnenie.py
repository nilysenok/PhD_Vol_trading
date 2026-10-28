"""Дивиденды в слое исполнения exec_e (Ж-057 п. 1, Ж-060): PnL позиции через отсечку и сдвиг стопа на D.
Синтетика, данные не читаются. Числа: σ = 1/16, b = 1: стоп лонга 93,75, шорта 106,25 от цены входа 100.
python3 -m pytest tests/test_dividendy_ispolnenie.py -q"""
import numpy as np
import pytest

from fastbt import exec_e, exec_e_run
from test_zh047_pomoshch import F, P, SG, D

DIV = 10.0


def _days(p_after=90.0):
    """День 0: вход по close 100. День 1 — гэп: цена упала на D (100 → 90). День 2 — плоский 90 (сигнал 0 → выход)."""
    return [D(F(100.0)), D(F(p_after)), D(F(p_after, 2))]


def _run(side, div, nalog=1.0, days=None, p=None, sig=(1, 1, 0)):
    p = p or P(R=None, b=None, side=side)
    return exec_e.run("T", days or _days(), list(sig), np.full(3, SG), p, div=div, nalog=nalog)


GAP = np.array([0.0, DIV, 0.0])


@pytest.mark.parametrize("side", [1, -1])
def test_poziciya_cherez_otsechku_daet_nol_oba_znaka(side):
    assert _run(side, GAP).gross.sum() == pytest.approx(0.0, abs=1e-12)


def test_bez_dividenda_staryy_pnl_longa_minus_log():
    r = exec_e.run("T", _days(), [1, 1, 0], np.full(3, SG), P(R=None, b=None))
    assert r.gross.sum() == pytest.approx(np.log(0.9))


def test_nalog_087_i_znak():
    lg = _run(1, GAP, 0.87).gross.sum()
    assert lg == pytest.approx(np.log((90 + 0.87 * DIV) / 100))
    assert _run(-1, GAP, 0.87).gross.sum() == pytest.approx(-lg)


def test_den_gepa_bez_barov_perenositsya_na_sleduyushchiy_torgovyy():
    pusto = exec_e.Day(*(np.empty(0),) * 4, True)
    days = [D(F(100.0)), pusto, D(F(90.0)), D(F(90.0, 2))]
    r = exec_e.run("T", days, [1, 1, 1, 0], np.full(4, SG), P(R=None, b=None), div=np.array([0, DIV, 0, 0.0]), nalog=1.0)
    assert r.gross.sum() == pytest.approx(0.0, abs=1e-12)


def test_vhod_v_den_gepa_dividenda_ne_poluchaet():
    """Вход на день гэпа (цена уже ex-div): дивиденд не начисляется."""
    days = [D(F(100.0)), D(F(90.0, 2)), D(F(90.0, 2))]
    r = exec_e.run("T", days, [0, 1, 0], np.full(3, SG), P(R=None, b=None), div=GAP, nalog=1.0)
    assert r.gross.sum() == pytest.approx(0.0, abs=1e-12) and len(r.trades) == 1


def test_mutatsiya_den_zapisi_vmesto_gepa_lovitsya():
    assert abs(_run(1, np.roll(GAP, -1)).gross.sum()) > 1e-3


def test_mutatsiya_dividend_ne_nachislen_lovitsya():
    assert abs(_run(1, np.zeros(3)).gross.sum()) > 1e-3


def test_mutatsiya_znak_dlya_shorta_lovitsya(monkeypatch):
    """Подложить: шорт тоже получает D (знак не домножен) — сумма шорта перестаёт быть нулём."""
    orig = exec_e_run.Run.step

    def bad(s, j, i):
        orig(s, j, i)
        if s.tr is not None and s.tr.side < 0:
            s.gross[i] += 2 * np.log((s.close_of(i) + DIV) / s.close_of(i)) * (s.div[i] > 0)
    monkeypatch.setattr(exec_e_run.Run, "step", bad)
    assert abs(_run(-1, GAP).gross.sum()) > 1e-3


# --- Ж-060: стоп открытой позиции через отсечку сдвигается вниз на D ---
def test_stop_longa_sdvigaetsya_vniz_na_D():
    """Лонг от 100, стоп 93,75; после отсечки D = 10 цена 92 (скорр. 102): стоп сдвинут до 83,75 — выхода нет."""
    r = _run(1, GAP, days=[D(F(100.0)), D(F(92.0)), D(F(92.0, 2))], p=P(R=None, b=1.0, side=1))
    assert r.trades[0].reason == "signal"


def test_stop_shorta_sdvigaetsya_vniz_na_D():
    """Шорт от 100, стоп 106,25 → 96,25; цена 97 (скорр. 107 > 106,25) — реальный стоп, выход по open гэпа 97."""
    r = _run(-1, GAP, days=[D(F(100.0)), D(F(97.0)), D(F(97.0, 2))], p=P(R=None, b=1.0, side=-1))
    t = r.trades[0]
    assert (t.reason, t.exit_px) == ("stop", 97.0)
    assert r.gross.sum() == pytest.approx(-np.log((97 + DIV) / 100))


def test_stop_ne_sdvigaetsya_bez_otsechki():
    r = _run(1, np.zeros(3), days=[D(F(100.0)), D(F(92.0)), D(F(92.0, 2))], p=P(R=None, b=1.0, side=1))
    assert r.trades[0].reason == "stop"


def test_mutatsiya_stop_ne_sdvinut_lovitsya(monkeypatch):
    monkeypatch.setattr(exec_e_run.Run, "_stop_shift", lambda s, S: S)
    r = _run(1, GAP, days=[D(F(100.0)), D(F(92.0)), D(F(92.0, 2))], p=P(R=None, b=1.0, side=1))
    assert r.trades[0].reason == "stop"                      # без сдвига выбивает ложно: тест «reason == signal» упал бы
    r = _run(-1, GAP, days=[D(F(100.0)), D(F(97.0)), D(F(97.0, 2))], p=P(R=None, b=1.0, side=-1))
    assert r.trades[0].reason != "stop"                      # шорт: реальный стоп пропущен
