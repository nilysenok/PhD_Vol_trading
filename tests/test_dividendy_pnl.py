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


def _lr_data(tmp_path, monkeypatch, nalog=1.0, cum_shift=0):
    """lr из боевого модуля fastbt.dividendy на сырых ценах: таблица в tmp, last_cum_date = день бара EX−1 (+ cum_shift — мутация)."""
    from fastbt import dividendy
    dates = pd.date_range("2018-01-01", periods=N, freq="B")
    cum = dates[EX - 1 + cum_shift].date()
    monkeypatch.setattr(dividendy, "_calendar", lambda: dates.values.astype("datetime64[ns]"))   # T+2 по календарю баров (PR div-daty-t2)
    rec = dates[EX + 1 + cum_shift].date()
    f = tmp_path / "div.csv"
    f.write_text(f"ticker,record_date,last_cum_date,announce_date,dps_rub\nT,{rec},{cum},{cum},{D}\n")
    monkeypatch.setattr(dividendy, "TABLE", f)
    c = np.full(N, P0)
    c[EX:] = P0 - D
    return dividendy.adjust_lr("T", dates.values, c, _raw_lr(), nalog=nalog)


def test_long_cherez_otsechku_pnl_ravno_nulyu(tmp_path, monkeypatch):
    """Было xfail (Ж-054 п. 2); после правки Ж-057 п. 1 (полный D, nalog = 1) лонг через отсечку даёт 0."""
    assert _pnl(_lr_data(tmp_path, monkeypatch), +1) == pytest.approx(0.0, abs=1e-9)


def test_short_cherez_otsechku_pnl_ravno_nulyu(tmp_path, monkeypatch):
    assert _pnl(_lr_data(tmp_path, monkeypatch), -1) == pytest.approx(0.0, abs=1e-9)


def test_osnovnoy_variant_nalog_087_ostavlyaet_chast_poteri(tmp_path, monkeypatch):
    """D×0,87: лонг получает 0,87·D, т.е. PnL = log((c+0,87D)/c0) = log(0,9935…) < 0; шорт — зеркально; сумма лонг+шорт = 0."""
    lg, sh = _pnl(_lr_data(tmp_path, monkeypatch, 0.87), +1), _pnl(_lr_data(tmp_path, monkeypatch, 0.87), -1)
    assert lg == pytest.approx(np.log((P0 - D + 0.87 * D) / P0)) and lg < 0
    assert sh == pytest.approx(-lg)


def test_mutatsiya_dividend_ne_nachislen_lovitsya(tmp_path, monkeypatch):
    """Мутация владельца: adjust_lr ничего не добавляет — тест «лонг через отсечку = 0» обязан упасть."""
    from fastbt import dividendy
    monkeypatch.setattr(dividendy, "gap_divs", lambda t, d: np.zeros(len(d)))
    assert abs(_pnl(_lr_data(tmp_path, monkeypatch), +1)) > 1e-3


def test_mutatsiya_den_zapisi_vmesto_dnya_gepa_lovitsya(tmp_path, monkeypatch):
    """Мутация: дивиденд поставлен на день раньше (на день записи/cum, а не на день гэпа) — PnL через отсечку ≠ 0."""
    assert abs(_pnl(_lr_data(tmp_path, monkeypatch, cum_shift=-1), +1)) > 1e-3


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
