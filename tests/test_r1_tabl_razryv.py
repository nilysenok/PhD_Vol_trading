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
    cols = {"utechka_okna": [_d(-2.5)], "div_2224": [_d(0.25)], "ispolnenie": [_d(-0.5), _d(-0.25)]}
    r = R.razlozhit(_d(2.0), _d(-1.0), cols)
    for p in R.PERIODS:
        assert r[p]["chestnoe"] - r[p]["pechat"] == pytest.approx(-3.0)
        assert sum(r[p][c] for c in R.COLS) == pytest.approx(-3.0)
        assert abs(r[p]["ostatok"]) < 1e-12


def test_sintetika_propushchen_shag_lovitsya():
    cols = {"utechka_okna": [_d(-2.5)], "div_2224": [_d(0.25)], "ispolnenie": [_d(-0.5)]}  # −0,25 пропущен
    with pytest.raises(ValueError):
        R.razlozhit(_d(2.0), _d(-1.0), cols)


def test_znak_pereputan_lovitsya():
    cols = {"utechka_okna": [_d(2.5)], "div_2224": [_d(0.25)], "ispolnenie": [_d(-0.5), _d(-0.25)]}
    with pytest.raises(ValueError):
        R.razlozhit(_d(2.0), _d(-1.0), cols)


@pytest.mark.parametrize("obj", ["A", "C", "D"])
def test_realnye_tablicy_ostatok_nol(obj):
    res = R.sobrat(obj)
    for p in R.PERIODS:
        r = res[p]
        assert abs(r["ostatok"]) < R.OSTATOK_MAX
        assert r["chestnoe"] - r["pechat"] == pytest.approx(sum(r[c] for c in R.COLS) + r["ostatok"], abs=1e-9)


@pytest.mark.parametrize("obj", ["A", "C", "D"])
@pytest.mark.parametrize("shag", ["P0", "P4", "P7", "P10", "E5", "P1", "P6", "5б", "6", "3"])
def test_realnye_propushchen_shag_lovitsya(obj, shag):
    """Мутация: шаг выкинут — остаток ≥ 1e-3 хотя бы в одном периоде (ValueError).
    Шаг с нулевым вкладом (например A:P7 = 0 по построению) выкинуть незаметно — проверяем его нулевость.
    Шаги 5б, 6, 3 есть только у C/D (у A их роль играют разности табличных Шарпов)."""
    pechat, honest, cols = R._chain(obj, ())
    ключи = {k.split("/")[-1] for v in cols.values() for _, k, _ in v}
    if shag not in ключи:
        pytest.skip("у этого объекта шага нет")
    vklad = next(d for v in cols.values() for _, k, d in v if k.split("/")[-1] == shag)
    if all(abs(x) < R.OSTATOK_MAX for x in vklad.values()):
        R.sobrat(obj, skip=(shag,))
        return
    with pytest.raises(ValueError):
        R.sobrat(obj, skip=(shag,))


def test_chestnoe_sovpadaet_s_a_cd_test1():
    res = R.sobrat("C")
    assert res["2022–2024"]["chestnoe"] == pytest.approx(-0.525349713666074)
    assert res["2022"]["pechat"] == pytest.approx(2.859614367466887)


def test_div_2224_tolko_etot_shag():
    """A: только разность A_zashch и A без дивидендов 2022–2024 (≈ −0,09, ожидание владельца); C: шаг 6 CD-лестницы."""
    assert R.sobrat("A")["2022–2024"]["div_2224"] == pytest.approx(-0.9145861306377668 + 0.8205093123756493)
    assert R.sobrat("C")["2022–2024"]["div_2224"] == pytest.approx(0.024742663639502394)


def test_div_progrev_eto_P6():
    assert R.sobrat("A")["2022–2024"]["div_progrev"] == pytest.approx(-0.18937080401360654 - 0.0, abs=1e-9)


def test_utechka_pervym_shagom_znak_otricatelnyj():
    for obj in ("A", "C", "D"):
        assert all(R.sobrat(obj)[p]["utechka_okna"] < -1.0 for p in R.PERIODS)


def test_istochniki_pokryvayut_vse_yacheiki():
    for obj in ("A", "C", "D"):
        rows = R.istochniki(obj)
        res = R.sobrat(obj)
        for p in R.PERIODS:
            for c in R.COLS:
                s = sum(float(r[5]) for r in rows if r[1] == p and r[2] == c)
                assert s == pytest.approx(res[p][c], abs=1e-5)
