"""Реверс, тест-1 без дивидендов (Ж-117, вариант б): FASTBT_Z2_DIV=NET — явный режим, не пустая таблица (Д4).
Синтетика, данные ≥ 2022 не читаются. Мутации (mutations.md № 38–40): NET молча = пустая таблица без пометки; режим не в ключе кэша.
python3 -m pytest tests/test_z2_bez_dividendov.py -q"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import dividendy, zona

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/revers"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
DATES = pd.bdate_range("2022-01-03", "2024-12-30").to_numpy("datetime64[ns]")


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    cal = pd.DataFrame({"date": pd.bdate_range("2014-06-02", "2024-12-31")})
    cal.to_parquet(tmp_path / "daily.parquet")
    tab = tmp_path / "div.csv"
    tab.write_text("ticker,record_date,dps_rub\nSBER,2023-07-20,25.0\n", encoding="utf-8")
    monkeypatch.setenv("FASTBT_Z2_DAILY", str(tmp_path / "daily.parquet"))
    monkeypatch.delenv("FASTBT_Z2_DIV", raising=False)
    monkeypatch.setattr(dividendy, "TABLE", tmp_path / "dev.csv")
    monkeypatch.setattr(dividendy, "TABLE_2015", tmp_path / "dev2015.csv")
    for p in (dividendy.TABLE, dividendy.TABLE_2015):
        p.write_text("ticker,record_date,dps_rub\n", encoding="utf-8")
    dividendy._calendar.cache_clear()
    monkeypatch.setattr(zona, "_zone", "dev")
    yield tab
    zona.set_zone("dev")
    dividendy._calendar.cache_clear()


def _narusheniya(tab, monkeypatch, tmp_path) -> list[str]:
    """Что нарушено в режиме NET (пустой список — всё верно)."""
    bad = []
    with zona.zone_ctx("test1"):
        monkeypatch.setenv("FASTBT_Z2_DIV", str(tab))
        d_tab, key_tab = dividendy.gap_divs("SBER", DATES), dividendy.digest()
        monkeypatch.setenv("FASTBT_Z2_DIV", "NET")
        if dividendy.gap_divs("SBER", DATES).any():
            bad.append("D != 0")
        if not dividendy.bez_dividendov():
            bad.append("bez_dividendov() False")
        key_net = dividendy.digest()
        if key_net == key_tab:
            bad.append("ключ NET = ключ с таблицей")
        if "дивиденды: НЕТ" not in zona.shapka_dividendy():
            bad.append("нет строки шапки")
    monkeypatch.delenv("FASTBT_Z2_DIV")
    if key_net == dividendy.digest():
        bad.append("ключ NET = ключ dev")
    assert d_tab.sum() == 25.0                                   # контроль: таблица действительно даёт D
    return bad


def test_bez_peremennoy_file_not_found():
    with zona.zone_ctx("test1"), pytest.raises(FileNotFoundError):
        dividendy.events("SBER")


def test_net_d_nol_klyuch_i_shapka(_env, monkeypatch, tmp_path):
    assert _narusheniya(_env, monkeypatch, tmp_path) == []
    assert not dividendy.bez_dividendov()


def test_net_vne_test1_otkaz(monkeypatch):
    monkeypatch.setenv("FASTBT_Z2_DIV", "NET")
    with pytest.raises(ValueError):
        dividendy.bez_dividendov()
    with pytest.raises(ValueError):
        dividendy.digest()


def test_shapka_lista_soderzhit_stroku(monkeypatch, tmp_path):
    import r1_d
    monkeypatch.setattr(r1_d, "ROOT", tmp_path)
    cfg = dict(out="o", shapka="# реверс\n")
    df = pd.DataFrame({"a": [1]})
    monkeypatch.setenv("FASTBT_Z2_DIV", "NET")
    r1_d.zapis(df, "x.csv", cfg)
    head = (tmp_path / "o" / "x.csv").read_text(encoding="utf-8").splitlines()
    assert head[1].startswith("# дивиденды: НЕТ") and head[2] == "a"
    monkeypatch.delenv("FASTBT_Z2_DIV")
    r1_d.zapis(df, "y.csv", cfg)
    assert "дивиденды" not in (tmp_path / "o" / "y.csv").read_text(encoding="utf-8")


def test_mutatsiya_net_molcha_pustaya_tablitsa_lovitsya(_env, monkeypatch, tmp_path):
    """Подложить: NET = пустая таблица без пометки (режим не виден, не в ключе, шапки нет) — проверка обязана найти нарушения."""
    monkeypatch.setattr(zona, "div_net", lambda: False)
    monkeypatch.setattr(zona, "shapka_dividendy", lambda: "")
    pusto = tmp_path / "pusto.csv"
    pusto.write_text("ticker,record_date,dps_rub\n", encoding="utf-8")
    old = zona.path

    def path(kind, default=None):
        if kind != "div":
            return old(kind, default)
        v = os.environ["FASTBT_Z2_DIV"]
        return pusto if v == "NET" else Path(v)
    monkeypatch.setattr(zona, "path", path)
    bad = _narusheniya(_env, monkeypatch, tmp_path)
    assert {"bez_dividendov() False", "нет строки шапки"} <= set(bad)
