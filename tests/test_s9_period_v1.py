"""Ж-075 п. 1: s9 на периоде train4 (В1: 2016–2019, прогрев 2015). Синтетика, данных s9 не читает (кроме ставки ЦБ).
Умолчание (train) — побайтно прежнее; B/C/D на train4 — отказ с текстом.
python3 -m pytest tests/test_s9_period_v1.py -q"""
import numpy as np
import pandas as pd
import pytest

from fastbt import ekviv_s9, s9


@pytest.fixture(autouse=True)
def _restore():
    old = s9.PERIOD, s9.YEARS, s9.VAL_START
    yield
    s9.PERIOD, s9.YEARS, s9.VAL_START = old


def _cache(y0=2015, y1=2019):
    d = pd.bdate_range(f"{y0}-01-01", f"{y1}-12-31")
    rng = np.random.default_rng(1)
    lr = rng.normal(0, 0.01, len(d))
    pos = np.where(rng.random(len(d)) > 0.5, 1, -1).astype(np.int8)
    return {"SBER": dict(dates=d.values, lr=lr, A={s: ({Y: pos for Y in range(2016, 2020)} if s == "S1" else {})
                                                   for s in s9.SIDS})}


def test_umolchanie_ne_izmeneno():
    assert s9.YEARS == (2018, 2019, 2020, 2021) and s9.VAL_START == 2017 and s9.PERIOD == "train"
    assert ekviv_s9.WINDOWS == {2018: (2018,), 2019: (2018, 2019)}
    assert ekviv_s9.windows() is ekviv_s9.WINDOWS


def test_train4_A_schitaetsya_2016_2019_bez_sigma(monkeypatch):
    monkeypatch.setattr(s9, "SOURCES", {})                 # σ-источников нет вообще: путь A их не читает
    s9.set_period("train4")
    assert s9.YEARS == (2016, 2017, 2018, 2019) and s9.VAL_START == 2015
    ys = ekviv_s9.windows("train4")[2019]
    assert ys == (2016, 2017, 2018, 2019)
    eng = ekviv_s9.s9_engine(_cache(), ys, "train4")
    ex = eng.port(eng.A_row, comm=0.0004)["excess"]
    assert sorted(set(ex.index.year)) == [2016, 2017, 2018, 2019] and len(ex) > 900
    assert s9.YEARS == (2016, 2017, 2018, 2019)          # контекст вернул глобалы


def test_s9_engine_umolchanie_okno_2017():
    eng = ekviv_s9.s9_engine(_cache(2017, 2019), (2018,))
    assert s9.VAL_START == 2017 and eng.year_of.min() == 2018


def test_train4_BCD_otkaz_s_tekstom(monkeypatch, tmp_path):
    """Ж-075/Ж-080: на train4 build_cache строит A (σ-источники не читаются), B/C/D (σ-источники) отказывают с текстом."""
    monkeypatch.setattr(s9, "_job", lambda args: (args[0], {"A": {}}))
    monkeypatch.setattr(s9, "OUT", tmp_path)
    monkeypatch.setattr(s9, "cache_key", lambda: "k4")
    s9.set_period("train4")
    assert s9.build_cache(workers=1).name == "cache_k4.pkl"          # A строится, а не отказ
    for f in (lambda: s9.load_source("hyb_h1"), s9.build_sources):
        with pytest.raises(NotImplementedError, match="train4"):
            f()


def test_train4_tolko_dev_i_izvestnye_periody():
    with pytest.raises(ValueError):
        s9.set_period("train5")
    with pytest.raises(ValueError):
        ekviv_s9.windows("train5")


def test_mutatsiya_progrev_ne_2015_lovitsya(monkeypatch):
    """Прогрев 2016 вместо 2015: окно валидации 2016 пустое → ValueError, а не молчаливый счёт."""
    monkeypatch.setitem(s9.PERIODS_S9, "train4", ((2016, 2017, 2018, 2019), 2016))
    with pytest.raises(ValueError, match="пустое окно"):
        ekviv_s9.s9_engine(_cache(), (2016, 2017, 2018, 2019), "train4")


def test_mutatsiya_years_po_umolchaniyu_sdvinuty_lovitsya():
    """Тест-страж: сдвиг YEARS/VAL_START по умолчанию (2017–2020 / 2016) обязан падать на нашей проверке."""
    assert s9.YEARS != (2017, 2018, 2019, 2020) and s9.VAL_START != 2016
    assert s9.PERIODS_S9["train"] == (s9.YEARS, s9.VAL_START)
