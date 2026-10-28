"""Оборот фазы 18:45–18:50: синтетика с известным ответом и мутации (чтение ≥ 2020, окно до 18:50)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import obyom_fazy_zakrytiya as ofz  # noqa: E402
from fastbt import bars5m  # noqa: E402

COLS = ["begin", "open", "high", "low", "close", "volume", "value"]


def mk(day, hm, val):
    return [pd.Timestamp(f"{day} {hm}"), 1.0, 1.0, 1.0, 1.0, 1, float(val)]


def write(src: Path, t: str, y: int, rows):
    src.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_csv(src / f"{t}_{y}.csv", index=False)


def synth(tmp_path):
    src = tmp_path / "src"
    write(src, "AAA", 2018, [
        mk("2018-03-01", "18:44", 7), mk("2018-03-01", "18:45", 100), mk("2018-03-01", "18:47", 50),
        mk("2018-03-01", "18:50", 999),                                  # вне фазы
        mk("2018-03-02", "18:49", 300),                                  # доп. фаза: нет 18:45
    ])
    write(src, "BBB", 2018, [mk("2018-03-01", "10:00", 1), mk("2018-03-02", "10:00", 1), mk("2018-03-05", "10:00", 1)])
    return src


def test_sintetika_oborot_dop_nuli(tmp_path):
    dni = ofz.sobrat(synth(tmp_path), ["AAA", "BBB"])
    a = dni["AAA"]
    assert list(a["oborot"]) == [150.0, 300.0, 0.0]                      # 18:50 и 18:44 не входят; 05.03 — нет баров
    assert list(a["dop"]) == [False, True, False]
    s = ofz.stroka("AAA", "2018", a, 0.1)
    assert s["mediana_rub"] == 150.0 and s["p10_rub"] == pytest.approx(30.0)
    assert s["dolya_nulevyh"] == pytest.approx(1 / 3) and s["dolya_dop"] == pytest.approx(1 / 3)
    assert s["poz_200"] == pytest.approx(2e7) and s["flag_p10_lt_poz200"] is True


def test_dop_faza_tolko_minuta_1849(tmp_path):
    src = tmp_path / "s"
    write(src, "AAA", 2018, [mk("2018-03-01", "18:47", 40), mk("2018-03-01", "18:49", 300)])   # нет 18:45
    assert ofz.sobrat(src, ["AAA"])["AAA"]["oborot"].tolist() == [300.0]


def test_ves_mediana_odin_na_cnt(tmp_path):
    src = synth(tmp_path)
    dni = ofz.sobrat(src, ["AAA", "BBB"])                                # cnt: 01.03 → 2, 02.03 → 2, 05.03 → 1
    diag = tmp_path / "diag"
    diag.mkdir()
    pd.DataFrame({"data_vhoda": ["2018-03-01", "2018-03-05", "2018-03-05"]}).to_csv(diag / "S1_sdelki_posle370.csv", index=False)
    assert ofz.ves_pozitsii(dni, diag) == pytest.approx(1.0)             # веса 0.5, 1, 1 → медиана 1


def test_ves_otkazyvaet_za_granitsey(tmp_path):
    dni = ofz.sobrat(synth(tmp_path), ["AAA", "BBB"])
    diag = tmp_path / "diag"
    diag.mkdir()
    pd.DataFrame({"data_vhoda": ["2020-01-09"]}).to_csv(diag / "S1_sdelki_posle370.csv", index=False)
    with pytest.raises(AssertionError):
        ofz.ves_pozitsii(dni, diag)


def test_fayly_2020_ne_otkryvayutsya(tmp_path):
    src = synth(tmp_path)
    (src / "AAA_2020.csv").write_text("мусор, который нельзя читать")      # открытие упало бы
    assert len(ofz.chitat("AAA", src)) == 5


def test_mutatsiya_chteniya_za_predelami_lovitsya(tmp_path, monkeypatch):
    """Мутация: _clip возвращает всё. Строка 2020 в файле 2019 обязана вызвать отказ."""
    src = tmp_path / "src"
    write(src, "AAA", 2019, [mk("2019-12-30", "18:45", 1), mk("2020-01-09", "18:45", 9)])
    assert ofz.chitat("AAA", src)["begin"].max() < pd.Timestamp("2020-01-01")   # исходный код проходит
    monkeypatch.setattr(bars5m, "_clip", lambda d, s, e: d)
    with pytest.raises(AssertionError):
        ofz.chitat("AAA", src)


def test_mutatsiya_okno_do_1850_lovitsya(tmp_path, monkeypatch):
    """Мутация: фаза включает минуту 18:50 (L=18:50) → оборот 150 перестаёт совпадать."""
    src = synth(tmp_path)
    assert ofz.sobrat(src, ["AAA"])["AAA"]["oborot"].iloc[0] == 150.0
    monkeypatch.setattr(ofz, "L", 18 * 60 + 50)
    assert ofz.sobrat(src, ["AAA"])["AAA"]["oborot"].iloc[0] != 150.0


def test_pustaya_lenta(tmp_path):
    src = tmp_path / "e"
    src.mkdir()
    r = ofz.po_dnyam(ofz.chitat("ZZZ", src))
    assert r.empty
