"""К-wf4: проверка «прогон до t = префикс полного прогона» (scripts/proverka_prefiksa_wf.py). Синтетика + дневной ряд ≤ 2021.
Мутации: (1) признак с процентилем по всему ряду; (2) таблица дивидендов целиком в префиксе (артефакт обрезки)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pwf", ROOT / "scripts" / "proverka_prefiksa_wf.py")
P = importlib.util.module_from_spec(spec)
sys.modules["pwf"] = P
spec.loader.exec_module(P)

IDX = pd.bdate_range("2016-01-01", periods=400)
X = pd.Series(np.random.default_rng(7).normal(0, 1, 400), index=IDX)


def test_prichinnyj_priznak_prefiks_raven():
    f = lambda s: s - s.expanding(30).mean()                              # noqa: E731
    assert P.max_delta(f(X.iloc[:250]).dropna(), f(X).dropna(), hvost=0) <= P.TOL


def test_mutaciya_procentil_po_vsemu_ryadu_lovitsya():
    f = lambda s: s / s.abs().quantile(0.9)                               # noqa: E731 процентиль по всему ряду
    assert P.max_delta(f(X.iloc[:250]), f(X), hvost=0) > 1e-3


def test_pustoe_i_raznye_daty_inf():
    assert P.max_delta(X.iloc[:2], X, hvost=3) == float("inf")             # пустое после отсечения хвоста
    assert P.max_delta(X.iloc[:50], X.iloc[100:], hvost=0) == float("inf")


def test_kvartaly_bez_poslednego():
    q = P.kvartaly(pd.Series(1.0, index=IDX), "2017-06-30")
    assert q.index[-1] == pd.Period("2017-1", "Q") and (q > 50).all()


def test_tochki_vosproizvodimy():
    assert P.tochki("S2", 5) == P.tochki("S2", 5) and len(P.tochki("S5", 20)) == 20


def test_defekt_ravenstvo_close_sma_menyaetsya_ot_masshtaba():
    """Зафиксированный дефект (fastbt/signals/s12.py:66, :59): точное равенство close == SMA в сырых ценах после умножения
    ряда на константу (обратная корректировка) решается округлением. Когда сравнение получит допуск — тест перевернуть."""
    base = np.r_[100 - np.arange(1, 13) * 0.5, 100 + np.arange(1, 13) * 0.5, 100.0]       # среднее = 100 = последняя цена
    assert base.mean() == 100.0 and base[-1] <= pd.Series(base).rolling(25).mean().iloc[-1]
    flips = [f for f in np.linspace(0.5, 0.99, 400)
             if not (base[-1] * f <= pd.Series(base * f).rolling(25).mean().iloc[-1])]
    assert flips, "ни один масштаб не сломал равенство: дефект исчез — перевернуть тест"


def _dannye():
    from fastbt import data
    if not Path(data.OHLCV).exists():
        pytest.skip("нет дневного ряда 18:50")


def test_obrezka_dobavlyaet_lozhnyj_dividend_v_naivnom_rezhime():
    _dannye()
    from fastbt import dividendy, etap1
    out = {}
    for zn in (True, False):
        with P.ryad_do("2019-12-31", zn):
            td = etap1.load_ticker_nf("SBER")
            out[zn] = dividendy.gap_divs("SBER", td.dates)[-3:]
    assert not out[True].any(), "событие после cut не должно быть известно префиксу"
    assert out[False].any(), "наивная обрезка: дивиденд после cut попадает на последние бары"


def test_prefiks_s1_ravny_na_dannyh():
    _dannye()
    r = P.sravnenie("S1", 3, True)
    assert r["2019-12-31"]["dni"] <= P.TOL and r["2019-12-31"]["top_ok"]
