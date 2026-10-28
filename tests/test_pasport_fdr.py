"""К-Ф4: вклад ФДР в паспорте. Синтетика: сделка по close — вклад T+0 = 0; внутридневная — > 0; T+1 ниже T+0; мутация mg ≠ 0."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import fdr as fdr_mod
from fastbt import pasport_fdr as pf
from fastbt.s9 import Engine

ND = 8


def make_engine() -> Engine:
    """Одна пара, 8 торговых дней 2018 (дальше — по календарю), ставка 7,5 % годовых, cal = 1."""
    e = Engine.__new__(Engine)
    e.dates = pd.date_range("2018-01-09", periods=ND, freq="D")
    e.r = {"date": np.arange(ND), "pair": np.zeros(ND, dtype=int)}
    e.first = np.r_[True, np.zeros(ND - 1, dtype=bool)]
    e.cnt = np.ones(ND)
    e.lr_row = np.full(ND, 0.001)
    e.rf, e.cal, e.kr = np.full(ND, 0.075 / 365), np.ones(ND), 0.1
    return e


def pos_close_only() -> np.ndarray:
    return np.array([0, 1, 1, 1, 0, 0, 0, 0], float)         # вход по close дня 1, выход по close дня 4


def test_close_only_t0_ravno_baze_pobaytno():
    cols = pf.kolonki_pos(make_engine(), pos_close_only())
    pf.g3(cols, "синтетика")                                   # не падает
    assert pf.vklad_god(cols["+ФДР T+0"], (2018,)) == pf.vklad_god(cols["база"], (2018,))
    assert pf.vklad_god(cols["+ФДР T+0"], (2018,)) > 0          # ставка на свободный капитал начисляется (мутация «ноль» не проходит)


def test_close_only_t1_nizhe_t0():
    cols = pf.kolonki_pos(make_engine(), pos_close_only())
    assert pf.vklad_god(cols["+ФДР T+1"], (2018,)) < pf.vklad_god(cols["+ФДР T+0"], (2018,))


def test_g3_lovit_rashozhdenie():
    cols = pf.kolonki_pos(make_engine(), pos_close_only())
    cols["+ФДР T+0"] = cols["+ФДР T+1"]
    with pytest.raises(SystemExit):
        pf.g3(cols, "синтетика")


def intraday(eng):
    """Внутридневная сделка в день 2: доля баров в рынке 0,3, вес 1; entry_day = exit_day."""
    tr = [SimpleNamespace(entry_day=2, exit_day=2, d_sig=1)]
    g = np.zeros(ND)
    expo = np.zeros(ND)
    expo[2] = 0.3
    base = eng.port_rows(g, np.zeros(ND), expo, COMM0)
    out = {"база": base}
    for name, rez in zip(pf.KOLONKI[1:], fdr_mod.REZHIMY):
        noch = fdr_mod.noch_iz_sdelok(tr, {1: 1.0}, ND, rez)
        out[name] = eng.port_rows(g, np.zeros(ND), expo, COMM0, noch=noch)
    return out


COMM0 = 0.0004


def test_vnutridnevnaya_vklad_polozhitelen():
    cols = intraday(make_engine())
    d = pf.vklad_god(cols["+ФДР T+0"], (2018,)) - pf.vklad_god(cols["база"], (2018,))
    assert d == pytest.approx(100 * 0.3 * 0.075 / 365)        # день 2 целиком свободен вместо 0,7


def test_vnutridnevnaya_t1_nizhe_t0():
    cols = intraday(make_engine())
    assert pf.vklad_god(cols["+ФДР T+1"], (2018,)) < pf.vklad_god(cols["+ФДР T+0"], (2018,))


def plechi(eng):
    return eng.port(np.array([0, 2, 2, 2, 0, 0, 0, 0], float), comm=COMM0)


def test_mg_ne_nol_ostanavlivaet():
    with pytest.raises(SystemExit):
        pf.vklad_god(plechi(make_engine()), (2018,))


def test_mutaciya_vklad_pri_mg_ne_nol_lovitsya(monkeypatch):
    """Мутация: проверка mg убрана — вклад считается молча при плече; тест выше (SystemExit) на ней красный."""
    df = plechi(make_engine())
    assert float(df["exposure"].max()) > 1.0
    monkeypatch.setattr(pf, "proverka_mg", lambda d: None)
    v = pf.vklad_god(df, (2018,))                              # не падает: число есть, но это fdr − mg, не ФДР
    e = make_engine()
    fdr_true = 100 * (np.maximum(0.0, 1.0 - df["exposure"].to_numpy()) * e.rf * e.cal).sum()
    assert v != pytest.approx(fdr_true)                        # мутация даёт неверное число, а штатный код остановился бы
