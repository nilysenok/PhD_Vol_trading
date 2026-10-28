"""Проверка PR #488 (Ж-092): atr_m / chast_1r. Синтетика, без данных. Красные до правки автора (Ж-081):
1) быстрый слой молча считает atr_m / chast_1r как без режима; 2) run_e1/fdr считают дочернюю «part» отдельной сделкой."""
import numpy as np
import pytest

from fastbt import exec_e, fast_etap1
from fastbt.etap1_run import run_e1
from fastbt.exec_e import Params
from fastbt.fdr import noch_iz_sdelok
from test_trail_atr import day, flat, lenta  # noqa: E402  (tests/ в sys.path при запуске pytest)


def _p(**kw):
    return Params(b=2.0, comm=0.001, reentry=False, **kw)


@pytest.mark.parametrize("kw", [dict(atr_m=2.0), dict(chast_1r=True)])
def test_bystryj_sloy_uhodit_na_medlennyj(kw):
    d14 = [(100, 100, 100, 100), (100, 110, 100, 110), (110, 110, 105, 105)] + flat(105, 2)
    days, sig = lenta(d14, flat(100, 5))
    sg = np.full(len(days), 0.01)
    a = fast_etap1.run("T", days, sig, sg, _p(**kw))
    b = exec_e.run("T", days, sig, sg, _p(**kw))
    assert [t.reason for t in a.trades] == [t.reason for t in b.trades]
    assert a.gross.sum() == pytest.approx(b.gross.sum())


def _lenta_part():
    """Вход d13 (close 100), +1R=102 пройден на d15, стоп 98 на d16: родитель 13→16, дочерняя 13→15."""
    d14 = [(100, 100, 100, 100), (100, 101, 100, 101)] + flat(101, 3)
    d15 = [(101, 103, 101, 103)] + flat(103, 4)
    d16 = [(103, 103, 97, 98)] + flat(98, 4)
    days = [day([(100, 101, 99, 100)]) for _ in range(14)] + [day(d14), day(d15), day(d16)]
    return days, [0] * 13 + [1] * 4, np.full(len(days), 0.01)


def test_chast_dva_trade_odin_vhod():
    days, sig, sg = _lenta_part()
    r = exec_e.run("T", days, sig, sg, _p(chast_1r=True))
    assert sorted(t.reason for t in r.trades) == ["part", "stop"]
    assert sum(t.fee_in for t in r.trades) == pytest.approx(0.001)   # вход оплачен один раз
    assert sum(t.frac for t in r.trades) == pytest.approx(1.0)


def test_run_e1_chislo_sdelok():
    days, sig, sg = _lenta_part()
    e = run_e1("T", days, sig, sg, _p(chast_1r=True))
    assert e.sides[13] == pytest.approx(1.0) and e.cnt[13] == pytest.approx(1.0)


def test_fdr_zanyatost_ne_bolshe_edinitsy():
    days, sig, sg = _lenta_part()
    r = exec_e.run("T", days, sig, sg, _p(chast_1r=True))
    noch = noch_iz_sdelok(r.trades, {13: 1.0}, len(days), "t0")
    assert noch.max() <= 1.0 + 1e-12
