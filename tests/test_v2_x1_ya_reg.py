"""Y-A (Ж-243), реестр Y1–Y6: регистрация до прогноза; незарегистрированная, записанная позже прогноза, Y5/Y6 — отказ; реестр только дополняется. Реальный файл — только чтение."""
from __future__ import annotations

import pandas as pd
import pytest

from models_v2.s16 import x1_reg as R


def test_register_rows_statuses_and_append_only(tmp_path) -> None:
    p = tmp_path / "ALL_TRIALS.csv"
    assert R.register("10.11.2026 04:00:00", p) == 6
    t = pd.read_csv(p, comment="#", dtype=str)
    assert list(t["id"]) == ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6"]
    assert set(t.loc[:3, "status"]) == {"зарегистрирована"} and set(t.loc[4:, "status"]) == {R.COND}
    raw = p.read_bytes()
    assert R.register("10.11.2026 09:00:00", p) == 0 and p.read_bytes() == raw          # повтор ничего не переписывает
    p.write_text(raw.decode().replace("\nY3,", "\nY3x,"), encoding="utf-8")            # потеряна строка Y3 -> дописывается только она
    assert R.register("10.11.2026 10:00:00", p) == 1 and "10.11.2026 04:00:00" in p.read_text(encoding="utf-8")


def test_guard_refuses_unregistered_late_and_conditional(tmp_path) -> None:
    p = tmp_path / "ALL_TRIALS.csv"
    with pytest.raises(SystemExit, match="не зарегистрирована"):
        R.guard("Y1", trials=p)                               # реестра нет
    R.register("10.11.2026 04:00:00", p)
    R.guard("Y1", computed_at="10.11.2026 04:00:01", trials=p)
    R.guard("Y2", computed_at="10.11.2026 04:00:00", trials=p)          # ровно в ту же секунду — допустимо
    with pytest.raises(SystemExit, match="позже прогноза"):
        R.guard("Y1", computed_at="10.11.2026 03:59:59", trials=p)      # запись реестра позже прогноза
    for y in ("Y5", "Y6"):
        with pytest.raises(SystemExit, match="отдельного разрешения"):
            R.guard(y, trials=p)                              # всегда, даже зарегистрированные
    with pytest.raises(SystemExit):
        R.guard("Y7", trials=p)
    q = tmp_path / "other.csv"
    q.write_text(p.read_text(encoding="utf-8").replace("зарегистрирована,10.11.2026 04:00:00\nY3", "отозвана,10.11.2026 04:00:00\nY3", 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="статус"):
        R.guard("Y2", trials=q)


def test_real_registry_written_before_forecasts() -> None:
    t = R.read().set_index("id")
    assert list(t.index) == ["Y1", "Y2", "Y3", "Y4", "Y5", "Y6"]
    assert t.loc["Y1":"Y4", "status"].eq("зарегистрирована").all() and t.loc["Y5":"Y6", "status"].eq(R.COND).all()
    for y in ("Y1", "Y2"):
        R.guard(y)                                            # реальный реестр пропускает Y1, Y2
    with pytest.raises(SystemExit):
        R.guard("Y5")
