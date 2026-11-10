"""Лист ZE (Ж-247 п. 4), реестр: ZE, ZEp дописаны («после просмотра Y1–Y4, Z, ZD; Ж-247»), прежние строки побайтно прежние, запись ДО прогноза и только после предусловия; незарегистрированная конфигурация (Y5, Y6, любая сверх двух) и запись позже прогноза — отказ.
Реестр только дополняется: тесты не требуют, чтобы строки ZE были последними (класс № 164)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_reg as R, x1_ze_reg as ZR


def test_registry_has_two_ze_rows_with_marks_and_earlier_rows_present() -> None:
    reg = pd.read_csv(R.TRIALS, comment="#", dtype=str, keep_default_na=False).set_index("id")
    assert list(reg.index[:6]) == ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6"] and {"ZA", "ZB", "ZB0", "ZC", "ZCp"} <= set(reg.index)
    for i, n, f, p, st in ZR.ROWS:
        assert tuple(reg.loc[i, ["name", "formula", "params", "status"]]) == (n, f, p, st) and "после просмотра Y1–Y4, Z, ZD (Ж-247)" in reg.at[i, "status"] and reg.at[i, "status"].startswith("зарегистрирована")
    assert "контроль" in reg.at["ZEp", "status"] and tuple(reg.index).count("ZE") == 1


def test_registry_before_any_forecast_and_after_precondition_when_written() -> None:
    cp = json.loads(R.REP.joinpath("CHECKPOINT.json").read_text(encoding="utf-8"))
    pre = cp.get("ZE.precondition")
    assert pre is None or pre["ok"] is True
    if "ZE" not in cp:
        pytest.skip("прогнозы ZE ещё не записаны")
    reg = pd.read_csv(R.TRIALS, comment="#", dtype=str, keep_default_na=False).set_index("id")
    for c in ZR.IDS:
        ZR.guard(c, computed_at=cp["ZE"]["computed_at"])
        assert R._parse(pre["written_at"]) <= R._parse(reg.at[c, "registered_at"]) <= R._parse(cp["ZE"]["computed_at"])


def test_register_appends_only_missing_rows_and_keeps_existing_bytes(tmp_path) -> None:
    p = tmp_path / "ALL_TRIALS.csv"
    head = R.TRIALS.read_text(encoding="utf-8").split("\nZE,")[0] + "\n"
    p.write_text(head, encoding="utf-8")
    assert ZR.register("01.01.2030 00:00:00", p) == 2
    assert p.read_text(encoding="utf-8").startswith(head) and ZR.register("02.01.2030 00:00:00", p) == 0
    assert (pd.read_csv(p, comment="#", dtype=str).set_index("id").loc[list(ZR.IDS), "registered_at"] == "01.01.2030 00:00:00").all()


def test_guard_refuses_unregistered_extra_late_and_bad_status(tmp_path) -> None:
    for c in ("Y5", "Y6", "Y1", "ZX", "ZE2", "ZEq", "ZA", "ZB", "H1"):
        with pytest.raises(SystemExit):
            ZR.guard(c)
    p = tmp_path / "t.csv"
    p.write_text(R.TRIALS.read_text(encoding="utf-8").split("\nZE,")[0] + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):                                  # ещё не записана ⇒ отказ
        ZR.guard("ZE", trials=p)
    ZR.register("10.11.2026 12:00:00", p)
    ZR.guard("ZE", trials=p)
    ZR.guard("ZE", computed_at="10.11.2026 12:00:01", trials=p)
    with pytest.raises(SystemExit):                                  # запись позже прогноза
        ZR.guard("ZE", computed_at="10.11.2026 11:59:59", trials=p)
    p.write_text(p.read_text(encoding="utf-8").replace("зарегистрирована; после", "отозвана; после"), encoding="utf-8")
    with pytest.raises(SystemExit):
        ZR.guard("ZE", trials=p)


def test_registry_refused_without_ok_precondition() -> None:
    assert not ZR.precondition_ok({}) and not ZR.precondition_ok({"ZE.precondition": {"ok": False}}) and ZR.precondition_ok({"ZE.precondition": {"ok": True}})
