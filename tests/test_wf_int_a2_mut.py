"""Тестировщик PR #794: мутации, не пойманные tests/test_wf_int_a2.py (проверены в отдельной копии).

Не пойманы были: (1) зона не сбрасывается при ПРЯМОМ set_period wf → train4 (автор проверял возврат только через period_ctx);
(2) граница календаря WF_KALENDAR_DO сдвинута (проходит 2026-01-01, тест автора проверял только 2021-12-30 и 2026-02-27).
Найдено, не мутация: календарь wf принимает бары ≥ 2026-03-01; состав 2022+ виден в зоне dev — оба теста xfail(strict)."""
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


@pytest.mark.xfail(strict=True, reason="мелкая находка #794: пустой ряд даёт IndexError cal[-1], а не отказ с причиной (не тихо — не блокер)")
def test_kalendar_wf_pustoj_otkaz(tmp_path, monkeypatch):
    """Пустой ряд — отказ с причиной (FileNotFoundError/ValueError), а не IndexError."""
    p = tmp_path / "daily.parquet"
    pd.DataFrame({"date": pd.to_datetime([])}).to_parquet(p)
    monkeypatch.setattr(daily_1850, "out_path", lambda main=None: p)
    with zona.zone_ctx("wf"):
        with pytest.raises((FileNotFoundError, ValueError)):
            dv._calendar()


@pytest.mark.xfail(strict=True, reason="находка #794: календарь wf не обрезается и не отвергает бары ≥ END_WF (2026-03-01)")
def test_kalendar_wf_ne_vyhodit_za_end_wf(tmp_path, monkeypatch):
    _parquet(tmp_path, monkeypatch, "2026-03-13")
    with zona.zone_ctx("wf"):
        try:
            cal = dv._calendar()
        except (FileNotFoundError, PermissionError, ValueError):
            return
        assert cal[-1] < np.datetime64(zona.END_WF.date())


@pytest.mark.xfail(strict=True, reason="находка #794: состав 2022+ из новой таблицы виден в зоне dev (sostav не смотрит на zona)")
def test_sostav_2022_ne_vidno_v_zone_dev():
    old = vs._sostav_table(str(vs.SOSTAV), vs.SOSTAV_COL)
    assert zona.name() == "dev"
    assert vs.sostav("2024-06-25") == old[-1][1]
