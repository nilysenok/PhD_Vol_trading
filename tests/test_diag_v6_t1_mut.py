"""Тестировщик #864: мутации таблицы (1) — пороги правила дословно по плану §3 п. (1), граница порога 0,05, «не измерено» ≠ 0."""
from __future__ import annotations

import pytest

from fastbt import diag_v6_t1 as t1
from test_diag_v6_t1 import OKNA, _zapisat, _stroka


def _vhod(tmp_path, osi_s):
    root = tmp_path / "m"
    _zapisat(root, "S1", osi_s, [])
    g = tmp_path / "g.csv"
    g.write_text("sid,os,razdel_kuba,gruppa,spornaya,prichina\n" + "".join(f"S1,{o},{k},{gg},net,\n" for o, k, gg, *_ in osi_s), encoding="utf-8")
    return root, g


# сигнал: max μ* = 0,04 (между 0,03 и 0,05) и он первый по Σμ*; контекст слабее. Единственная причина слабого места — max < 0,05.
S_GRAN = [("a", "signal", "signal", 25, (0.04,) * 3), ("b", "signal", "kontekst", 25, (0.03,) * 3)]


def test_porogi_dosloveno_po_planu_par3_p1():
    assert t1.T1_PRAVILO == dict(okno="<=2018", top_n=2, ne_izmereno_dolya_max=0.5, max_mu_min=0.05)


def test_granica_005_slaboe_mesto(tmp_path):
    assert t1.tablica_1(*_vhod(tmp_path, S_GRAN))[1].endswith("Sk: S1")


def test_mutaciya_porog_003_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setitem(t1.T1_PRAVILO, "max_mu_min", 0.03)
    assert t1.tablica_1(*_vhod(tmp_path, S_GRAN))[1] == "не слабое место"  # мутант расходится с эталоном выше


def test_ravno_005_ne_slaboe_mesto(tmp_path):
    osi = [("a", "signal", "signal", 25, (0.05,) * 3), ("b", "signal", "kontekst", 25, (0.01,) * 3)]
    assert t1.tablica_1(*_vhod(tmp_path, osi))[1] == "не слабое место"


def test_ne_izmereno_nan_i_nol_aktivnyh_ne_nol(tmp_path):
    """Ось без шагов (n_akt=0, μ*=nan) — не измерена, а не «измерена с 0»: считается в ne_izmereno, не в izmereno, не в медиане."""
    osi = [("a", "signal", "signal", 25, (0.2,) * 3), ("b", "signal", "signal", 0, (None,) * 3)]
    df, _ = t1.tablica_1(*_vhod(tmp_path, osi))
    s = _stroka(df, "S1", "<=2018", "signal")
    assert (s.osej, s.izmereno, s.ne_izmereno, s.mediana_mu) == (2, 1, 1, 0.2)


def test_mutaciya_ne_izmereno_kak_nol_lovitsya(tmp_path, monkeypatch):
    """Мутант: «ось без шагов» считается измеренной с 0 (порог n_akt снят, nan → 0) — медиана падает до 0,1."""
    osi = [("a", "signal", "signal", 25, (0.2,) * 3), ("b", "signal", "signal", 0, (None,) * 3)]
    monkeypatch.setattr(t1, "N_AKT_MIN", 0)
    monkeypatch.setattr(t1.math, "isfinite", lambda v: True)
    monkeypatch.setattr(t1.pd.Series, "median", lambda self, *a, **k: float(self.fillna(0).mean()), raising=False)
    df, _ = t1.tablica_1(*_vhod(tmp_path, osi))
    s = _stroka(df, "S1", "<=2018", "signal")
    assert (s.izmereno, s.ne_izmereno) != (1, 1)


@pytest.mark.parametrize("okno", ["<=2016", "<=2017"])
def test_vyvod_tolko_po_oknu_2018(tmp_path, okno):
    """Вывод идёт по окну ≤ 2018: сигнал силён в ≤2016/≤2017, но слаб в ≤2018 — слабое место."""
    osi = [("a", "signal", "signal", 25, (0.9, 0.9, 0.01)), ("b", "signal", "kontekst", 25, (0.1, 0.1, 0.1))]
    assert okno in OKNA
    assert t1.tablica_1(*_vhod(tmp_path, osi))[1].startswith("слабое место")
