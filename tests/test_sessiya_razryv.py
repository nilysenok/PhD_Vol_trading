"""Д5: сокращённая сессия не входит в RV (sigma_real.korotkie_dni) и заявка не переживает остановку торгов (Params.razryv).
Умолчания дирижёра 29.10: порог 0,5 медианы за прошлые 252 дня (≤ d−1); разрыв > 5 календарных дней. Позиция через разрыв держится.
Мутации: сокращённый день как полный; медиана с днём d (заглядывание); заявка не снимается; порог разрыва ≥ вместо >."""
import numpy as np
import pandas as pd
import pytest

from fastbt import exec_e, sigma_real
from fastbt.exec_e import Params, run
from test_exec_e import D, F
from test_ostanovka_2022 import test_sokrashchennaya_sessiya_ne_zanizhaet_sigma as scenariy


# --- (1) сокращённый день ---
def test_korotkie_dni_izvestnyj_otvet():
    n = pd.Series([48] * 10 + [23, 24, 48, 0])                  # порог 0,5·48 = 24: строго меньше
    assert list(sigma_real.korotkie_dni(n)[10:]) == [True, False, False, True]


def test_korotkie_dni_malo_istorii_ne_sudim():
    assert not sigma_real.korotkie_dni(pd.Series([48, 48, 2])).any()           # < SHORT_MIN наблюдений в прошлом


def test_mut_mediana_s_dnem_d_lovitsya():
    n = pd.Series([48] * 10 + [20] * 10 + [16])                  # прошлая медиана 34: 16 < 17 — сокращённый
    assert sigma_real.korotkie_dni(n).iloc[-1]
    assert not sigma_real.korotkie_dni(n, sdvig=0).iloc[-1]       # с днём d медиана 20 → не сокращённый: заглядывание видно


def test_mut_sokrashchennyj_kak_polnyj_lovitsya(monkeypatch):
    scenariy()                                                  # штатно — проходит
    monkeypatch.setattr(sigma_real, "korotkie_dni", lambda n, *a, **k: n < 0)
    with pytest.raises(AssertionError):
        scenariy()


# --- (2) заявка через остановку торгов ---
def _days():
    return [D(F(100.0)), D([(99, 99, 95, 96)]), D(F(100.0)), D(F(100.0))]       # сигнал дня 0 → лимит 98 на дне 1 (low 95)


def _dates(gap):
    return pd.DatetimeIndex(["2022-02-25", pd.Timestamp("2022-02-25") + pd.Timedelta(days=gap), "2022-03-30", "2022-03-31"])


def go(gap, razryv, sig=(1, 0, 0, 0)):
    return run("T", _days(), list(sig), np.full(4, 0.02), Params(k=1.0, comm=0.0, razryv=razryv), dates=_dates(gap))


def test_zayavka_snimaetsya_cherez_ostanovku():
    r = go(26, 5)
    assert r.trades == [] and r.n_orders == 1 and r.n_fills == 0                # поставлена вечером d, снята, не исполнена


@pytest.mark.parametrize("gap,razryv,fill", [(5, 5, True), (6, 5, False), (3, 5, True), (26, None, True)])
def test_porog_razryva_strogo_bolshe(gap, razryv, fill):
    assert bool(go(gap, razryv).trades) is fill                     # (5, 5): не «>=» (мутация); (26, None): без правила — прежнее


def test_mut_zayavka_ne_snimaetsya_lovitsya():
    assert len(go(26, None).trades) == 1 and go(26, 5).trades == []   # без снятия заявка исполняется по гэпу через 26 дней


def test_novaya_zayavka_posle_razryva_stavitsya():
    r = go(26, 5, sig=(1, 1, 0, 0))                                 # сигнал жив на дне 1: вечером d+1 ставится новая заявка на дне 2
    assert r.n_orders == 2 and r.n_fills == 0 and r.trades == []    # день 2: low 100 > 98 — не исполнена, но поставлена заново


def test_razryv_bez_dates_otkaz():
    with pytest.raises(ValueError):
        run("T", _days(), [1, 0, 0, 0], np.full(4, 0.02), Params(k=1.0, razryv=5))


def test_pozitsiya_derzhitsya_cherez_razryv_i_zakryvaetsya_po_open():
    days = [D(F(100.0)), D(F(100.0)), D([(90, 91, 89, 90)]), D(F(90.0))]
    dates = pd.DatetimeIndex(["2022-02-24", "2022-02-25", "2022-03-24", "2022-03-25"])
    kw = dict(comm=0.0, b=1.0)
    r = run("T", days, [1] * 4, np.full(4, 0.02), Params(razryv=5, **kw), dates=dates)
    t = r.trades[0]                                                 # k=None: вход по close 100 дня 0; стоп 98, open 90 → по open
    assert (t.reason, t.exit_px, t.exit_day) == ("stop", 90.0, 2)
    base = run("T", days, [1] * 4, np.full(4, 0.02), Params(**kw))
    assert np.array_equal(r.gross, base.gross) and np.array_equal(r.cost, base.cost)


def test_bez_razryvov_pobaytno_prezhnee():
    rng = np.random.default_rng(7)
    days = [D([(c, c * 1.01, c * 0.99, c * 1.002)] * 3) for c in 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))]
    sig = rng.integers(0, 2, 60)
    dates = pd.bdate_range("2021-01-11", periods=60)                # разрывы ≤ 3 дней
    p = dict(k=0.5, comm=0.0004, a=1.0, b=1.0, tau=5)
    a = exec_e.run("T", days, sig, np.full(60, 0.02), Params(**p))
    b = exec_e.run("T", days, sig, np.full(60, 0.02), Params(razryv=5, **p), dates=dates)
    assert a.trades == b.trades and np.array_equal(a.gross, b.gross) and np.array_equal(a.cost, b.cost)
