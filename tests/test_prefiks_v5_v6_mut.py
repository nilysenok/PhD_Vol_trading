"""Тестировщик PR #757: мутации к tests/test_prefiks_v5_v6.py, которых там нет.
(в) шаг цены, зависящий от длины окна, подставлен на входы оценщика (быстрый слой берёт шаг не из build_days);
(г) ключ кэша (Д5) не меняется при смене шага — E1Inputs.digest и fast_etap1._digest_fast.
(б) «год start+1 включительно» на реальных данных: do сдвинут на год вперёд — префикс нарушается."""
import numpy as np
import pandas as pd
import pytest

from fastbt import etap1, etap1_data as D, fast_etap1
from fastbt.etap1 import E1Inputs
import test_prefiks_v5_v6 as P


def _inp(tick: float) -> E1Inputs:
    idx = pd.date_range("2018-01-01", periods=3)
    return E1Inputs(sk=None, sigma={}, bars={}, close_n6={}, rezhim={}, sobytiya={}, tick={"A": tick},
                    rate=pd.Series([0.1, 0.1, 0.1], index=idx), meta={"A": dict(dates=idx.values)})


def _tracks_tick(make) -> bool:
    return make(0.01).digest() != make(0.02).digest() and make(0.01).digest() == make(0.01).digest()


def test_digest_tick_v_klyuche():
    assert _tracks_tick(_inp)


def test_digest_tick_v_klyuche_bystryj_sloj():
    with fast_etap1.patched():
        assert _tracks_tick(_inp)


def test_mutaciya_digest_bez_tika_lovitsya():
    assert not _tracks_tick(lambda t: _inp(0.01))              # мутация: шаг не доходит до ключа


def _ne_zavisit_ot_bars_ot_do(tick) -> bool:
    """Шаг не меняется, если менять бары на/после do (причинность по дням окна, а не только префикс ≥ start+1)."""
    a = P._bars({2015: 0.1, 2016: 0.1, 2017: 0.1})
    b = P._bars({2015: 0.1, 2016: 0.3, 2017: 0.3})             # 2016+ другой шаг; до do=2016-01-01 то же
    return tick(a, P.DO) == tick(b, P.DO)


def test_tick_ne_zavisit_ot_bars_ot_do():
    assert _ne_zavisit_ot_bars_ot_do(D.tick_of)


def test_mutaciya_tick_god_start_plus_1_vklyuchitelno_lovitsya():
    # префикс-тест автора её не видит (год start+1 есть в обоих окнах) — ловит только этот тест
    assert P._prefix_ok(lambda b, do: D.tick_of(b, do + pd.DateOffset(years=1)))
    assert not _ne_zavisit_ot_bars_ot_do(lambda b, do: D.tick_of(b, do + pd.DateOffset(years=1)))


def test_mutaciya_tick_iz_drugogo_mesta_lovitsya_na_realnyh(monkeypatch):
    orig = etap1._build

    def build(sid, ev, end, period):                           # шаг зависит от длины окна, минуя tick_of
        inp = orig(sid, ev, end, period)
        for t, m in inp.meta.items():
            inp.tick[t] = inp.tick[t] * (1.0 + 1e-3 * len(m["dates"]))
        return inp

    monkeypatch.setattr(etap1, "_build", build)
    dd, dq = P._prefiks("S3", "v5")
    assert dd > 1e-12 or dq > 1e-12
