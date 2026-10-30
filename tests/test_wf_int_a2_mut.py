"""Тестировщик PR #794: мутации, не пойманные tests/test_wf_int_a2.py (проверены в отдельной копии).

Не пойманы были: (1) зона не сбрасывается при ПРЯМОМ set_period wf → train4 (автор проверял возврат только через period_ctx);
(2) граница календаря WF_KALENDAR_DO сдвинута (проходит 2026-01-01, тест автора проверял только 2021-12-30 и 2026-02-27).
Находки первой версии (бары ≥ END_WF в календаре wf, пустой ряд) автор исправил; тесты стали обычными.
Состав 2022+ в зоне dev: на базе и на ветке ValueError (устаревание 120 дней, Д4) — моя первая находка была ошибкой тестировщика."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850, dividendy as dv, period_v2 as pv, s9, vselennaya as vs, zona


@pytest.fixture(autouse=True)
def _chistka():
    dv._calendar.cache_clear()
    yield
    zona.set_zone("dev")
    s9.set_period("train")
    dv._calendar.cache_clear()


def _parquet(tmp_path, monkeypatch, do):
    p = tmp_path / "daily.parquet"
    pd.DataFrame({"date": pd.bdate_range("2014-06-02", do)}).to_parquet(p)
    monkeypatch.setattr(daily_1850, "out_path", lambda main=None: p)


@pytest.mark.parametrize("vozvrat", ["train", "train4", "v2"])
def test_pryamoj_set_period_posle_wf_vozvrashchaet_dev(vozvrat):
    s9.set_period("wf")
    assert zona.name() == "wf"
    s9.set_period(vozvrat)
    assert zona.name() == "dev" and s9.PERIOD == vozvrat
    assert zona.end() == zona.END_DEV


def test_kalendar_wf_granica_20_fevralya(tmp_path, monkeypatch):
    with zona.zone_ctx("wf"):
        _parquet(tmp_path, monkeypatch, "2026-02-19")
        dv._calendar.cache_clear()
        with pytest.raises(FileNotFoundError, match="ряд wf"):
            dv._calendar()
        _parquet(tmp_path, monkeypatch, "2026-02-20")
        dv._calendar.cache_clear()
        assert dv._calendar()[-1] == np.datetime64("2026-02-20")


def test_kalendar_wf_pustoj_otkaz(tmp_path, monkeypatch):
    """Пустой ряд — отказ с причиной (FileNotFoundError/ValueError), а не IndexError."""
    p = tmp_path / "daily.parquet"
    pd.DataFrame({"date": pd.to_datetime([])}).to_parquet(p)
    monkeypatch.setattr(daily_1850, "out_path", lambda main=None: p)
    with zona.zone_ctx("wf"):
        with pytest.raises((FileNotFoundError, ValueError)):
            dv._calendar()


def test_kalendar_wf_ne_vyhodit_za_end_wf(tmp_path, monkeypatch):
    _parquet(tmp_path, monkeypatch, "2026-03-13")
    with zona.zone_ctx("wf"):
        try:
            cal = dv._calendar()
        except (FileNotFoundError, PermissionError, ValueError):
            return
        assert cal[-1] < np.datetime64(zona.END_WF.date())


def test_sostav_2022_ne_vidno_v_zone_dev():
    """Д4: в dev дата за порогом устаревания — ValueError (таблица 2022+ не видна, перенос вперёд не молчит)."""
    assert zona.name() == "dev"
    with pytest.raises(ValueError, match="Д4"):
        vs.sostav("2024-06-25")


def test_sostav_v_predelah_poroga_dev_posledniaya_stroka():
    old = vs._sostav_table(str(vs.SOSTAV), vs.SOSTAV_COL)
    last = [tk for de, tk in old if de <= pd.Timestamp("2021-12-30")][-1]
    assert vs.sostav("2021-12-30") == last == vs.sostav("2022-03-01")
    with zona.zone_ctx("wf"):
        assert vs.sostav("2021-12-30") == last
