"""Ж-040 п. 2: сверка оценщика перебора с боевым путём s9 на 20 фиксированных точках (fastbt/designs/sweep/ekviv_20.yaml).

ДОПУСК ОБЪЯВЛЕН ДО СЧЁТА (proverka.md п. 3):
  - позиции (по тикерам, бары окна) — побайтно: число расходящихся баров == 0;
  - дневной PnL (excess) — max |оценщик − s9| ≤ 1e-12 (PNL_TOL); Шарп считается по тем же рядам;
  - окна: 2018 (end 2019-01-01) и 2019 (end 2020-01-01, срез 2019); комиссии 0,04 % и 0,035 %.
Класс «нет эталона» (√RV-выходы, маски после кэша, trail_n вне TREND_TN) — не подменяется: в таблице статус «нет эталона»,
в зачёт «совпало» он не идёт; смена класса точки (yaml `etalon`) роняет тест.
Мутация (№ 20 mutations.md): «двойной знак» в sweep_post (результат exec_layer снова умножен на знак сырой позиции) обязана
уронить точки (б) и не трогать центры (а).
Данные: validation 2020–2021 не читается (окна ≤ 2019, A из кэша s9 обрезана по 2019). В сеть не ходит.
Запуск (≈ 10–20 с на реальных данных): FASTBT_MAIN=$PWD pytest tests/test_ekviv_sweep_s9.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import ekviv_s9 as ek
from fastbt import s9, sweep_post

DATA_OK = (s9.OUT / "cache_latest_v2.txt").exists() and (Path(s9.MAIN) / "data" / "regen" / "features_pit_v3.parquet").exists()
real = pytest.mark.skipif(not DATA_OK, reason="нет данных проекта (кэш s9 v2, features_pit_v3)")
PTS = {x["id"]: x for x in ek.load_points()["tochki"]}
IDS_B = [i for i, x in PTS.items() if x["klass"] == "ispolnenie"]
IDS_A = [i for i, x in PTS.items() if x["klass"] == "tsentr"]
EXEC = ("consec_entry", "min_strength", "exec_cooldown", "exec_min_hold")
TRIV = dict(consec_entry=1, min_strength=0.0, exec_cooldown=0, exec_min_hold=0)


def double_sign(monkeypatch):
    """Подложенная ошибка Ж-038: exec_layer уже возвращает знак, оценщик умножает ещё раз."""
    orig = sweep_post.exec_layer
    monkeypatch.setattr(sweep_post, "exec_layer", lambda pos, *a, **k: orig(pos, *a, **k) * np.where(pos < 0, -1.0, 1.0))


@pytest.fixture(scope="module")
def tablica():
    return ek.run_all()


# ---------- структура 20 точек (без данных) ----------

def test_dvadtsat_tochek_po_tri_na_strategiyu_plyus_a():
    assert len(PTS) == 20
    for k in range(1, 7):
        assert sorted(x["klass"] for x in PTS.values() if x["sid"] == f"S{k}") == ["ispolnenie", "kandidat", "tsentr"]
    assert sorted(i for i, x in PTS.items() if x["sid"] == "A") == ["A_2018", "A_2019"]


def test_tochka_b_imeet_nenulevoy_parametr_ispolneniya_a_tsentr_net():
    for i in IDS_B:
        assert any(PTS[i]["p"][k] != TRIV[k] for k in EXEC), i
    for i in IDS_A:
        assert all(PTS[i]["p"][k] == TRIV[k] for k in EXEC), i


def test_dopusk_obyavlen_v_yaml_i_v_kode():
    assert ek.load_points()["dopusk"] == {"pozitsii": "pobaytno", "pnl_abs": 1e-12} and ek.PNL_TOL == 1e-12


@pytest.mark.parametrize("upd", [{"rm_edinitsa": "sqrt_rv"}, {"adx_min": 25}, {"q_sily": 0.5}, {"vyhod_dev": 0.75}])
def test_net_etalona_raspoznayotsya(upd):
    with pytest.raises(ek.NoRef):
        ek.rm_row("S4", {**PTS["S4_a"]["p"], **upd})
    assert ek.rm_row("S4", PTS["S4_a"]["p"])["trail_n"] == 15          # центр — эталон есть


def test_row_lovit_raznitsu_pnl_vyshe_dopuska_i_pustuyu_pozitsiyu():
    idx = pd.date_range("2018-01-02", periods=5, freq="B")
    a = pd.Series([0.01, -0.02, 0.0, 0.01, 0.0], idx)
    pt = dict(id="x", klass="k")
    assert ek.row(pt, 2018, 0.0004, a, a.copy())["status"] == "совпало"
    assert ek.row(pt, 2018, 0.0004, a, a + 1e-13)["status"] == "совпало"
    assert ek.row(pt, 2018, 0.0004, a, a + 1e-9)["status"] == "РАСХОЖДЕНИЕ"
    assert ek.row(pt, 2018, 0.0004, a, a.copy(), npos=1)["status"] == "РАСХОЖДЕНИЕ"
    assert ek.row(pt, 2018, 0.0004, a, a.iloc[:4])["status"] == "РАСХОЖДЕНИЕ"       # другой набор дней
    assert ek.row(pt, 2018, 0.0004, a, None)["status"] == "нет эталона"


# ---------- сверка на реальных данных ----------

@real
def test_vse_tochki_s_etalonom_sovpali(tablica):
    bad = tablica[(tablica.etalon == "da") & (tablica.status != "совпало")]
    assert bad.empty, bad.to_string()
    assert tablica.max_dpnl.max() <= ek.PNL_TOL and (tablica.bar_poz_razl == 0).all()


@real
def test_klass_net_etalona_sovpadaet_s_yaml(tablica):
    got = tablica.groupby("tochka").etalon.first().to_dict()
    assert got == {i: x["etalon"] for i, x in PTS.items()}
    assert (tablica[tablica.etalon == "net"].status == "нет эталона").all()


@real
def test_trail_n_vne_trend_tn_eto_net_etalona():
    from fastbt.data import load_ticker
    from fastbt.indicators import compute_base
    td = load_ticker("SBER")
    with pytest.raises(ek.NoRef, match="trail_n"):
        ek.boevoy_pos("S3", PTS["S3_v"]["p"], "SBER", td, compute_base(td.close, td.high, td.low, td.open, td.volume))


@real
def test_pokrytie_16_iz_20(tablica):
    ok = tablica[tablica.status == "совпало"]
    assert ok.tochka.nunique() == 16 and len(ok) == 14 * 4 + 2 * 2   # Sk: 2 года × 2 комиссии; A_2018 / A_2019: свой год × 2 комиссии
    assert set(tablica.tochka) == set(PTS)


@real
def test_mutatsiya_dvoynoy_znak_lovitsya_na_tochkah_b(monkeypatch):
    double_sign(monkeypatch)
    t = ek.run_all(IDS_B)
    for i in IDS_B:
        r = t[(t.tochka == i) & (t.god == 2018)]
        assert (r.status == "РАСХОЖДЕНИЕ").all() and (r.bar_poz_razl > 0).all(), i


@real
def test_mutatsiya_ne_trogaet_tsentry(monkeypatch):
    """Центр не включает слой исполнения — двойной знак на него не действует (мутация бьёт именно по пути исполнения)."""
    double_sign(monkeypatch)
    t = ek.run_all(IDS_A)
    assert (t.status == "совпало").all()
