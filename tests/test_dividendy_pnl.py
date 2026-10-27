"""Дивиденды в PnL (Ж-054 п. 2, расследование): s9.Engine.port считает pos·lr по ценам закрытия ряда 18:50 без поправки
на дивиденды (fastbt/data.py: lr = log(c[t+1]/c[t]); data/corporate_actions.csv — только сплиты). В день отсечки цена падает на D:
лонг теряет D (должен получить D обратно, PnL = 0), шорт получает D (должен заплатить D, PnL = 0).
Синтетика, данные не читаются. Падающие до правки проверки помечены xfail(strict=True): после правки они обязаны пройти
(XPASS станет красным и потребует снять метку).
python3 -m pytest tests/test_dividendy_pnl.py -q
"""
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import s9

P0, D = 100.0, 5.0            # цена до отсечки; дивиденд (5 %)
N, EX = 30, 15                # дней в ряду; индекс бара отсечки: close[EX] = P0 − D (сырая цена, не скорректирована)


def _engine(lr):
    """Одна пара, одна дата на строку, ставка 0: доходность позиции = pos·lr без финансирования."""
    n = len(lr)
    kr = np.zeros(n)
    ns = types.SimpleNamespace(
        lr_row=np.asarray(lr, float), first=np.r_[True, np.zeros(n - 1, bool)], cnt=np.ones(n), kr=kr, rf=kr,
        cal=np.ones(n), r=dict(date=np.arange(n)), dates=pd.date_range("2018-01-01", periods=n, freq="B"))
    ns._finish = lambda *a: s9.Engine._finish(ns, *a)
    return ns


def _raw_lr():
    """Сырые цены: плоско P0, на баре EX — P0 − D (отсечка), дальше плоско; lr = log(c[t+1]/c[t]), последний 0 (как в data.load_ticker)."""
    c = np.full(N, P0)
    c[EX:] = P0 - D
    lr = np.zeros(N)
    lr[:-1] = np.log(c[1:] / c[:-1])
    return lr


def _pnl(lr, side):
    """Накопленный PnL позиции side (±1), открытой на баре 5 и закрытой на баре 25, до комиссий (comm = 0)."""
    pos = np.zeros(N)
    pos[5:25] = side
    eng = _engine(lr)
    return float(s9.Engine.port(eng, pos, comm=0.0)["gross"].sum())


def _with_dividend(lr):
    """Правильный вход движка: в день отсечки доходность пробегает по цене с дивидендом, lr += log(1 + D / c[EX])."""
    out = lr.copy()
    out[EX - 1] += np.log(1 + D / (P0 - D))
    return out


@pytest.mark.xfail(strict=True, reason="дивиденды не учитываются: лонг через отсечку теряет D, движок даёт −log(1−D/P0)")
def test_long_cherez_otsechku_pnl_ravno_nulyu():
    assert _pnl(_raw_lr(), +1) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.xfail(strict=True, reason="дивиденды не учитываются: шорт через отсечку получает D вместо того, чтобы платить")
def test_short_cherez_otsechku_pnl_ravno_nulyu():
    assert _pnl(_raw_lr(), -1) == pytest.approx(0.0, abs=1e-9)


def test_dvizhok_na_skorrektirovannom_vhode_daet_nol_oba_znaka():
    """Контроль оракула: если вход несёт дивиденд, арифметика port даёт 0 и для лонга, и для шорта (знак не перепутан)."""
    lr = _with_dividend(_raw_lr())
    assert _pnl(lr, +1) == pytest.approx(0.0, abs=1e-9)
    assert _pnl(lr, -1) == pytest.approx(0.0, abs=1e-9)


def test_symptom_sdvig_znaka_i_vne_otsechki():
    """Сегодняшнее поведение зафиксировано числом: лонг −5,13 %, шорт +5,13 % (log(0,95)); вне отсечки позиция даёт 0."""
    lr = _raw_lr()
    assert _pnl(lr, +1) == pytest.approx(np.log((P0 - D) / P0))
    assert _pnl(lr, -1) == pytest.approx(-np.log((P0 - D) / P0))
    flat = np.zeros(N)
    assert _pnl(flat, +1) == 0.0 and _pnl(flat, -1) == 0.0


def test_mutatsiya_oshibochnyy_znak_popravki_ne_daet_nol():
    """Подложенная ошибка: поправка с обратным знаком (lr −= …) не обнуляет PnL — оракул её ловит."""
    bad = _raw_lr().copy()
    bad[EX - 1] -= np.log(1 + D / (P0 - D))
    assert abs(_pnl(bad, +1)) > 1e-3 and abs(_pnl(bad, -1)) > 1e-3


def test_corporate_actions_ne_soderzhit_primenyaemyh_dividendov():
    """Механизм: в data/corporate_actions.csv применяются только сплиты/консолидации; дивидендные строки — question."""
    ca = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "corporate_actions.csv")
    ap = ca[ca["status"] == "applied"]
    assert len(ap) > 0
    assert not ap["kind"].str.contains("дивиденд", case=False).any()
