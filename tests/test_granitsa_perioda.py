"""Расследование: доходность последнего дня окна берёт цену первого бара ПОСЛЕ окна.

fastbt/data.py:97-98: lr[:-1] = log(c[1:] / c[:-1]) по всему ряду тикера — lr[i] есть
log(c[i+1]/c[i]); для последнего дня окна (31.12.Y) это close первого бара года Y+1.
Синтетика, временный каталог, реальные данные не читаются (ни 2022, ни другие).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fastbt.data as D
from fastbt.stats import sharpe_window


def _frame(first_year: int, last: str, seed: int = 0) -> pd.DataFrame:
    d = pd.bdate_range(f"{first_year}-01-01", last)
    rng = np.random.default_rng(seed)
    c = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(d))))
    return pd.DataFrame({"ticker": "SYN", "date": d, "open": c, "high": c * 1.01,
                         "low": c * 0.99, "close": c, "volume": 1000.0})


def _load(monkeypatch, df: pd.DataFrame):
    monkeypatch.setattr(D, "daily_df", lambda: df)
    monkeypatch.setattr(D, "align_sigma", lambda dates, t: {})
    D.load_ticker.cache_clear()
    try:
        return D.load_ticker("SYN")
    finally:
        D.load_ticker.cache_clear()


def _window_sharpe(td, year: int) -> float:
    s, e = D.year_bounds(td.dates, year)
    pos = np.ones(len(td.close))          # всегда в лонге: пересекает границу окна
    return sharpe_window(pos, td.log_ret, s, e, 0.0004)


# xfail(strict): дефект воспроизведён, пока тест красный. Когда правку внесут — XPASS станет
# ошибкой и отметку надо будет снять.
@pytest.mark.xfail(strict=True, reason="lr последнего дня окна читает close первого бара после окна "
                   "(data.py:97-98): мусор после конца окна меняет Шарп окна")
def test_okno_ne_zavisit_ot_tsen_posle_kontsa(monkeypatch):
    base = _frame(2018, "2019-03-29")
    td0 = _load(monkeypatch, base)
    junk = base.copy()
    m = junk["date"] >= "2019-01-01"
    junk.loc[m, ["open", "high", "low", "close"]] *= 7.0   # мусор строго после окна 2018
    td1 = _load(monkeypatch, junk)
    assert _window_sharpe(td0, 2018) == pytest.approx(_window_sharpe(td1, 2018), abs=1e-12)


def test_defekt_vosproizveden_priamo(monkeypatch):
    """Не xfail: фиксирует механизм — lr[31.12.2018] = log(c[2019-01-0x]/c[2018-12-31])."""
    base = _frame(2018, "2019-03-29")
    td = _load(monkeypatch, base)
    s, e = D.year_bounds(td.dates, 2018)
    assert td.dates[e - 1] == np.datetime64("2018-12-31")
    assert td.dates[e] > np.datetime64("2018-12-31")               # бар вне окна
    assert td.log_ret[e - 1] == pytest.approx(np.log(td.close[e] / td.close[e - 1]))


def test_ryad_do_2021_ne_chitaet_2022(monkeypatch):
    """Н6: ряд кончается 31.12.2021 → lr последнего бара = 0, цены 2022 нет и читать нечего.
    (Граница validation→test в продуктовом ряду не течёт; течёт только внутри 2018–2021.)"""
    td = _load(monkeypatch, _frame(2018, "2021-12-31"))
    assert td.dates[-1] <= np.datetime64("2021-12-31")
    assert td.log_ret[-1] == 0.0
    # Если ряд продлить (тестовый набор), lr 31.12.2021 станет зависеть от первого бара 2022:
    ext = _frame(2018, "2022-01-14")
    tde = _load(monkeypatch, ext)
    i = int(np.searchsorted(tde.dates, np.datetime64("2021-12-31")))
    assert tde.log_ret[i] != 0.0


def test_otsenka_vliyaniya_odnogo_dnya(monkeypatch):
    """Порядок величины (синтетика, σ_день = 1%, окно ~260 дней): день ±3σ на границе
    сдвигает годовой Шарп окна на десятые доли — сопоставимо с шагом сетки отбора."""
    base = _frame(2018, "2019-03-29")
    td0 = _load(monkeypatch, base)
    jump = base.copy()
    jump.loc[jump["date"] >= "2019-01-01", ["open", "high", "low", "close"]] *= np.exp(0.03)
    td1 = _load(monkeypatch, jump)
    d = abs(_window_sharpe(td1, 2018) - _window_sharpe(td0, 2018))
    assert 0.05 < d < 0.5
