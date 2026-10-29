"""Лист R-в: знаменатель «процентов в год» берётся из дат окна, а не константой 4. Известный ответ: ровно +1 % за год за 3 года → 1 %/год.
Мутация: god = 4.0 на 3-летнем окне (прежняя ошибка) → ловится (test_mutaciya_god_4_na_test1_lovitsya). Колонки в долях не зависят от god."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts" / "revers"), str(ROOT / "scripts" / "analysis")]
import overnajt as O  # noqa: E402
import r1_v as R  # noqa: E402

PCT = ("valovaya_doh_pct_god", "doh_noch_pct_god", "doh_den_pct_god")


def dates(y0, y1):
    return np.arange(f"{y0}-01-01", f"{y1 + 1}-01-01", dtype="datetime64[D]")


def kross(n=300):
    I, N = np.full(n, 0.01 / n * 3), np.full(n, 0.0)         # за окно сумма I = 0.03 (3 %) — при 3 годах 1 %/год
    x = np.linspace(-0.001, 0.001, n)
    return {"S7": dict(I=I, N=N, x=x, d=np.ones(n, bool))}


def test_chislo_let_iz_dat():
    assert R.chislo_let(dates(2022, 2024)) == 3.0 and R.chislo_let(dates(2016, 2019)) == 4.0
    assert R.chislo_let(dates(2022, 2022)) == 1.0


def test_procenty_v_god_test1_ravny_odnomu():
    god = R.chislo_let(dates(2022, 2024))
    for r in O.stroki_kross(kross(), god):
        assert abs(r["doh_den_pct_god"] - 1.0) < 1e-9 and abs(r["valovaya_doh_pct_god"] - 1.0) < 1e-9


def test_mutaciya_god_4_na_test1_lovitsya():
    ok = O.stroki_kross(kross(), R.chislo_let(dates(2022, 2024)))[0]
    bad = O.stroki_kross(kross(), 4.0)[0]               # прежнее поведение
    assert abs(bad["doh_den_pct_god"] - 1.0) > 0.2 and abs(ok["doh_den_pct_god"] - 1.0) < 1e-9
    assert all(np.isclose(ok[c], bad[c] * 4 / 3) for c in PCT)


def test_umolchanie_train4_ne_izmenilos():
    a, b = O.stroki_kross(kross())[0], O.stroki_kross(kross(), 4.0)[0]
    assert a == b or all(np.isclose(a[k], b[k], equal_nan=True) for k in a if isinstance(a[k], float))
