"""Тесты таблицы разрыва предзащиты (scripts/revers/r1_tabl_razryv.py): сумма шагов = разрыв, мутации.
Читают готовые CSV из docs/tables/revers_r1, модели не считают, в сеть не ходят.
Запуск: python3 -m pytest tests/test_r1_tabl_razryv.py -q"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
import r1_tabl_razryv as R  # noqa: E402


def _d(x: float) -> dict:
    return {p: x for p in R.PERIODS}


def test_sintetika_summa_ravna_razryvu():
    r = R.razlozhit(_d(2.0), _d(-1.0), [_d(-2.5)], [_d(0.25)], [_d(-0.5), _d(-0.25)])
    for p in R.PERIODS:
        assert r[p]["chestnoe"] - r[p]["pechat"] == pytest.approx(-3.0)
        assert r[p]["utechka"] + r[p]["dividendy"] + r[p]["ispolnenie"] == pytest.approx(-3.0)
        assert abs(r[p]["ostatok"]) < 1e-12


def test_sintetika_propushchen_shag_lovitsya():
    with pytest.raises(ValueError):
        R.razlozhit(_d(2.0), _d(-1.0), [_d(-2.5)], [_d(0.25)], [_d(-0.5)])  # шаг −0,25 пропущен


def test_znak_pereputan_lovitsya():
    with pytest.raises(ValueError):
        R.razlozhit(_d(2.0), _d(-1.0), [_d(2.5)], [_d(0.25)], [_d(-0.5), _d(-0.25)])


@pytest.mark.parametrize("obj", ["A", "C", "D"])
def test_realnye_tablicy_ostatok_nol(obj):
    res = R.sobrat(obj)
    for p in R.PERIODS:
        r = res[p]
        assert abs(r["ostatok"]) < R.OSTATOK_MAX
        summa = r["utechka"] + r["dividendy"] + r["ispolnenie"] + r["ostatok"]
        assert r["chestnoe"] - r["pechat"] == pytest.approx(summa, abs=1e-9)


@pytest.mark.parametrize("obj", ["A", "C", "D"])
@pytest.mark.parametrize("shag", ["P0", "P4", "P7", "P10", "E5", "P1", "P6"])
def test_realnye_propushchen_shag_lovitsya(obj, shag):
    """Мутация: шаг лестницы 5а выкинут — остаток ≥ 1e-3 хотя бы в одном периоде (ValueError).
    Шаг с нулевым вкладом (A: P7 = 0 по построению, ладдер 5а) выкинуть незаметно — это не дефект, проверяем его нулевость."""
    vklad = R._per(R._steps5a()[(obj, shag)], "delta")
    if all(abs(v) < R.OSTATOK_MAX for v in vklad.values()):
        R.sobrat(obj, skip=(shag,))
        return
    with pytest.raises(ValueError):
        R.sobrat(obj, skip=(shag,))


def test_chestnoe_sovpadaet_s_a_cd_test1():
    res = R.sobrat("C")
    assert res["2022–2024"]["chestnoe"] == pytest.approx(-0.525349713666074)
    assert res["2022"]["pechat"] == pytest.approx(2.859614367466887)


def test_utechka_pervym_shagom_znak_otricatelnyj():
    for obj in ("A", "C", "D"):
        assert all(R.sobrat(obj)[p]["utechka"] < -1.0 for p in R.PERIODS)
