"""V6-1b, G18/G19/G21/G22: `--stage 1b --check` ничего не пишет (суммарный хэш выходов до = после); модуль DX-A (`v6_1b_dx*`) не импортируется путём прогноза (AST; разрешён только оркестратор `run_v6.py`); реестр: A0, A1, A2 (+ A3 по правилу), без дублей;
DX-A на закрытых парах до 2021-12-30. Мутации ловятся (rc 1): служебный файл при проверке, подмена таблицы / parquet / DXA / CHECKPOINT / ALL_TRIALS на диске."""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from models_v2.s13 import run_v6, v6_0_cp as CP, v6_0_io as IO, v6_1_reg as RG, v6_1b_dxa as DXA, v6_1b_stage as ST

ROOT = Path(__file__).resolve().parents[1]
PREF = ("v6_1b_dx",)
ALLOWED = {"run_v6.py"}


def imports_dx(source: str) -> bool:
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import) and any(any(p in a.name for p in PREF) for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and (any(p in (n.module or "") for p in PREF) or any(a.name.startswith(PREF) for a in n.names)):
            return True
    return False


def test_dxa_not_imported_by_forecast_path():
    bad = [p.name for p in sorted((ROOT / "models_v2/s13").glob("*.py")) if not p.name.startswith(PREF) and p.name not in ALLOWED and imports_dx(p.read_text(encoding="utf-8"))]
    assert bad == [] and imports_dx((ROOT / "models_v2/s13/run_v6.py").read_text(encoding="utf-8"))


def test_mutation_dxa_import_in_forecast_path_caught():
    assert imports_dx("from models_v2.s13 import v6_1b_dxa\n") and imports_dx("import models_v2.s13.v6_1b_dxa as d\n") and imports_dx("from models_v2.s13.v6_1b_dxa import compute\n")
    assert not imports_dx("from models_v2.s13 import v6_1b_stage\n")


@pytest.fixture(scope="module")
def result():
    return ST.compute()


@pytest.fixture(scope="module")
def dx(result):
    return DXA.compute(result["tab"], result["r0"]["S"].set_index("d"), result["sigfe"].set_index("d"))


def env(tmp_path, monkeypatch, result, dx):
    shutil.copytree(IO.REP, tmp_path / "rep")
    shutil.copytree(IO.OUT, tmp_path / "out")
    monkeypatch.setattr(IO, "REP", tmp_path / "rep")
    monkeypatch.setattr(IO, "OUT", tmp_path / "out")
    monkeypatch.setattr(CP, "CP", tmp_path / "rep" / "CHECKPOINT.json")
    monkeypatch.setattr(CP, "ALL_TRIALS", tmp_path / "rep" / "ALL_TRIALS.csv")
    monkeypatch.setattr(ST, "compute", lambda: result)
    monkeypatch.setattr(DXA, "compute", lambda *a: dx)


def test_registry_a0_a1_a2_and_a3_by_rule(result):
    t = RG.trial_text(result["ids"]).splitlines()
    keys = [x.split(",", 1)[0] for x in t[1:]]
    assert keys[:3] == ["Ж-227:A0", "Ж-227:A1", "Ж-227:A2"] and (("Ж-227:A3" in keys) == result["dec"]["compute"]) and len(keys) == len(set(keys)) and len(keys) in (3, 4)
    assert RG.trials_bad(CP.ALL_TRIALS.read_text(encoding="utf-8")) == [] and RG.trials_ok(result["ids"])
    assert not any("sens" in k.lower() for k in keys)                                            # соседние точки в реестр не входят


def test_levels_table_five_levels_per_config_and_level1_technical(result):
    lv = result["levels"]
    assert len(lv) == 5 * len(result["ids"]) and set(lv["config"]) == set(result["ids"]) and set(lv["passed"]) <= {"да", "нет"}
    assert (lv[lv["level"] == 1]["passed"] == "да").all() and lv["basis"].str.len().min() > 20


def test_dxa_closed_pairs_until_2021_and_header(result, dx):
    assert set(dx["model"]) == {"S + спектр + Σ", "S + Σ (без спектра)", "S + спектр (без Σ)"} and (dx["last_t_plus"] <= "2021-12-30").all() and (dx["first_d"] == "2015-07-07").all()
    w = dx[dx["test"].str.startswith("Wald:")]
    assert ((w["p"] >= 0) & (w["p"] <= 1)).all() and (w["df"].isin([12, 2])).all()
    assert "диагностика задним числом" in (IO.REP / "DXA.csv").read_text(encoding="utf-8").splitlines()[0] and (IO.REP / "DXA.csv").exists()


def test_check_changes_no_file_and_passes(tmp_path, monkeypatch, capsys, result, dx):
    env(tmp_path, monkeypatch, result, dx)
    before = IO.outputs_digest()
    assert run_v6.main(["--stage", "1b", "--check"]) == 0 and IO.outputs_digest() == before
    assert "ИТОГ: OK" in capsys.readouterr().out


def test_mutation_check_writes_service_file_caught(tmp_path, monkeypatch, capsys, result, dx):
    env(tmp_path, monkeypatch, result, dx)
    from models_v2.s13 import v6_1b_io as IO1B
    real = IO1B.check

    def writing(r, extra=None):
        (IO.REP / "CHECK_runtime_sec.txt").write_text("1.0")
        return real(r, extra)
    monkeypatch.setattr(IO1B, "check", writing)
    assert run_v6.main(["--stage", "1b", "--check"]) == 1 and "изменил файлы" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["A_INFO.csv", "DXA.csv", "A2_TESTS.csv"])
def test_mutation_table_on_disk_changed_caught(tmp_path, monkeypatch, capsys, result, dx, name):
    env(tmp_path, monkeypatch, result, dx)
    p = IO.REP / name
    lines = p.read_text(encoding="utf-8").splitlines()
    lines[3] = lines[3].replace(",", ",9", 1)                                       # второе поле строки данных искажено (не в хвосте десятичной дроби)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert run_v6.main(["--stage", "1b", "--check"]) == 1 and name in capsys.readouterr().out


def test_mutation_forecast_parquet_changed_caught(tmp_path, monkeypatch, capsys, result, dx):
    env(tmp_path, monkeypatch, result, dx)
    a = pd.read_parquet(IO.OUT / "A2.parquet")
    a.loc[5, "F"] *= 1.001
    a.to_parquet(IO.OUT / "A2.parquet", index=False)
    assert run_v6.main(["--stage", "1b", "--check"]) == 1 and "A2.parquet" in capsys.readouterr().out


def test_mutation_checkpoint_and_trials_changed_caught(tmp_path, monkeypatch, capsys, result, dx):
    env(tmp_path, monkeypatch, result, dx)
    cp = json.loads(CP.CP.read_text())
    cp["1b"]["qlike"]["A2"]["all"] += 0.001
    CP.CP.write_text(json.dumps(cp, ensure_ascii=False))
    assert run_v6.main(["--stage", "1b", "--check"]) == 1 and "CHECKPOINT ключ 1b" in capsys.readouterr().out
    cp["1b"]["qlike"]["A2"]["all"] -= 0.001
    CP.CP.write_text(json.dumps(cp, ensure_ascii=False))
    CP.ALL_TRIALS.write_text(CP.ALL_TRIALS.read_text(encoding="utf-8") + "Ж-227:A2_trees_x2,Ж-227,x,y,z,trial\n", encoding="utf-8")     # соседняя точка записана как конфигурация
    assert run_v6.main(["--stage", "1b", "--check"]) == 1 and "ALL_TRIALS" in capsys.readouterr().out


def test_checkpoint_1b_numbers_and_prior_keys_intact(result):
    cp = CP.read()
    assert cp["1b"]["used"] in (3, 4) and cp["1b"]["budget"] == 18 and cp["1b"]["n_dates"] == 2271 and (cp["1b"]["N"], cp["1b"]["N_wide"]) == (722, 3122)
    assert (cp["0"]["N"], cp["0"]["N_wide"]) == (722, 3122) and cp["0"]["used"] == 0 and cp["1a"]["used"] == 1
    assert ("посчитана" in cp["1b"]["A3"]) == result["dec"]["compute"] and cp["1b"]["code_hash_v6_1b"] == ST.code_hash()
