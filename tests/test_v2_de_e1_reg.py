"""E1: реестр 5 мест — запись до прогноза, отказы бегуна (синтетика, временный каталог)."""
import pytest

from models_v2.s15 import de0_io as IO, de_reg as R, run_de


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "REP", tmp_path)
    monkeypatch.setattr(R, "TRIALS", tmp_path / "ALL_TRIALS.csv")
    monkeypatch.setattr(IO, "CP_FILE", tmp_path / "CHECKPOINT.json")
    return tmp_path


def test_register_five_places_append_only(reg):
    assert R.register("01.01.2020 10:00:00") == 5
    t = R.read()
    assert list(t["id"]) == ["X1", "X2", "X3", "X4", "X5"] and set(t["status"]) == {"зарегистрирована"}
    assert "6.2" in t.loc[t["id"] == "X5", "condition"].iloc[0]
    txt = R.TRIALS.read_text(encoding="utf-8")
    assert R.register("02.02.2021 11:00:00") == 0 and R.TRIALS.read_text(encoding="utf-8") == txt      # повтор ничего не меняет


def test_unregistered_configuration_is_refused(reg):
    with pytest.raises(SystemExit, match="не зарегистрирована"):
        R.guard("X1")
    R.register("01.01.2020 10:00:00")
    R.guard("X1")
    with pytest.raises(SystemExit):
        R.guard("X6")


def test_registry_written_after_forecast_is_caught(reg):
    R.register("09.11.2026 23:30:00")
    R.guard("X1", computed_at="09.11.2026 23:31:00")
    with pytest.raises(SystemExit, match="позже прогноза"):
        R.guard("X1", computed_at="09.11.2026 23:29:00")


def test_x5_refused_without_condition_and_allowed_with_it(reg):
    R.register("01.01.2020 10:00:00")
    with pytest.raises(SystemExit, match="условие"):
        R.guard("X5", cp={})
    with pytest.raises(SystemExit, match="условие"):
        R.guard("X5", cp={"E5": {"condition": False}})
    R.guard("X5", cp={"E5": {"condition": True}})


def test_runner_stage_refusals(reg, capsys, monkeypatch):
    with pytest.raises(SystemExit, match="не зарегистрирована"):
        run_de.main(["--stage", "E1"])
    R.register("01.01.2020 10:00:00")
    with pytest.raises(SystemExit, match="условие"):
        run_de.main(["--stage", "E5"])                       # X5 без условия
    monkeypatch.setattr(run_de, "STAGES", {})                # стадия без подключённого счёта — отказ; реальный счёт из теста не запускается
    with pytest.raises(SystemExit, match="не реализован"):
        run_de.main(["--stage", "E2"])
    with pytest.raises(SystemExit):
        run_de.main(["--stage", "reg", "--check"])
