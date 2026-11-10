"""Этап Z (Ж-246), реестр: пять строк Z дописаны (ZB, ZC — с пометкой «после просмотра Y1–Y4»), строки Y1–Y6 не изменены, запись ДО прогноза; незарегистрированная конфигурация (в т. ч. Y5/Y6, шестая) и запись позже прогноза — отказ."""
from __future__ import annotations

import json

import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_reg as R, x1_z_reg as ZR, x1_z_stage as S


def test_registry_has_five_z_rows_with_post_view_marks_and_y_rows_unchanged() -> None:
    reg = pd.read_csv(R.TRIALS, comment="#", dtype=str, keep_default_na=False).set_index("id")
    assert list(reg.index[:6]) == ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6"] and list(reg.index[6:11]) == list(ZR.IDS) == ["ZA", "ZB", "ZB0", "ZC", "ZCp"]      # строки после Z (листы H1, Hc, H2…) допустимы
    for i, n, f, p, st in ZR.ROWS:                                    # прежние поля и время записи Z (10.11.2026 16:12:23, до прогноза)
        assert tuple(reg.loc[i, ["name", "formula", "params", "status"]]) == (n, f, p, st) and reg.at[i, "registered_at"] == "10.11.2026 16:12:23"
    for i, n, f, p, st in R.ROWS:                                     # строки Y1–Y6 — как в x1_reg.ROWS
        assert tuple(reg.loc[i, ["name", "formula", "params", "status"]]) == (n, f, p, st)
    assert "после просмотра Y1–Y4" in reg.at["ZB", "status"] and "после просмотра Y1–Y4" in reg.at["ZC", "status"]
    assert all(reg.at[c, "status"].startswith("зарегистрирована") for c in ZR.IDS) and "после просмотра" not in reg.at["ZA", "status"]
    assert len(set(reg["registered_at"].iloc[6:])) >= 1 and all(r > "" for r in reg["registered_at"])


def test_registry_before_any_forecast_when_forecast_written() -> None:
    cp = json.loads((S.REP / "CHECKPOINT.json").read_text(encoding="utf-8")).get("Z")
    if cp is None:
        pytest.skip("прогнозы Z ещё не записаны")
    for c in ZR.IDS:
        ZR.guard(c, computed_at=cp["computed_at"])
    reg = pd.read_csv(R.TRIALS, comment="#", dtype=str, keep_default_na=False).set_index("id")
    assert all(R._parse(reg.at[c, "registered_at"]) <= R._parse(cp["computed_at"]) for c in ZR.IDS)


def test_register_appends_only_missing_rows_and_keeps_existing_bytes(tmp_path) -> None:
    p = tmp_path / "ALL_TRIALS.csv"
    head = R.TRIALS.read_text(encoding="utf-8").split("\nZA,")[0] + "\n"
    p.write_text(head, encoding="utf-8")
    assert ZR.register("01.01.2030 00:00:00", p) == 5
    assert p.read_text(encoding="utf-8").startswith(head) and ZR.register("02.01.2030 00:00:00", p) == 0
    assert (pd.read_csv(p, comment="#", dtype=str).set_index("id").loc[list(ZR.IDS), "registered_at"] == "01.01.2030 00:00:00").all()


def test_guard_refuses_unregistered_extra_late_and_bad_status(tmp_path) -> None:
    for c in ("Y5", "Y6", "Y1", "ZX", "ZD", "Z6", "ZA2"):
        with pytest.raises(SystemExit):
            ZR.guard(c)
    p = tmp_path / "t.csv"
    p.write_text(R.TRIALS.read_text(encoding="utf-8"), encoding="utf-8")
    ZR.guard("ZA", trials=p)
    t = pd.read_csv(p, comment="#", dtype=str, keep_default_na=False)
    t.loc[t["id"] == "ZB", "status"] = "условная"
    t.loc[t["id"] == "ZC", "registered_at"] = "31.12.2099 00:00:00"
    p.write_text("#\n" + t.to_csv(index=False, lineterminator="\n"), encoding="utf-8")
    with pytest.raises(SystemExit, match="статус"):
        ZR.guard("ZB", trials=p)
    with pytest.raises(SystemExit, match="позже прогноза"):
        ZR.guard("ZC", computed_at="10.11.2026 16:30:00", trials=p)
    q = tmp_path / "empty.csv"
    q.write_text("#\nid,name,formula,params,status,registered_at\n", encoding="utf-8")
    for c in ZR.IDS:
        with pytest.raises(SystemExit, match="не зарегистрирована"):
            ZR.guard(c, trials=q)
    for st in ("Y5", "Y6"):
        with pytest.raises(SystemExit):
            S.main(["--stage", st])
