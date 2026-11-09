"""V6-1a, G15/G16: `--stage 1a --check` ничего не пишет (суммарный хэш выходов до = после); диагностика задним числом (`v6_0_dx*`, `v6_dx*`, `v6_1_dx*`) не импортируется путём прогноза (AST, v6_1_*);
реестр: ровно одна строка `Ж-227:A0`, чувствительность в реестр не входит; выходы на диске = пересчёт. Мутации: служебный файл при проверке, подмена таблицы / parquet / CHECKPOINT / ALL_TRIALS на диске — ловятся (rc 1)."""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from models_v2.s13 import run_v6, v6_0_cp as CP, v6_0_io as IO, v6_1_reg as RG, v6_1_stage as S1

ROOT = Path(__file__).resolve().parents[1]
PREF = ("v6_0_dx", "v6_dx", "v6_1_dx")


def imports_dx(source: str) -> bool:
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import) and any(any(p in a.name for p in PREF) for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and (any(p in (n.module or "") for p in PREF) or any(a.name.startswith(PREF) for a in n.names)):
            return True
    return False


def test_dx_not_imported_by_v6_1_forecast_path():
    bad = [p.name for p in sorted((ROOT / "models_v2/s13").glob("*.py")) if not p.name.startswith(PREF) and imports_dx(p.read_text(encoding="utf-8"))]
    assert bad == []


def test_mutation_dx_import_caught():
    assert imports_dx("from models_v2.s13 import v6_1_dx_a\n") and imports_dx("import models_v2.s13.v6_1_dx_b as x\n") and imports_dx("from models_v2.s13.v6_dx_c import f\n")
    assert not imports_dx("from models_v2.s13 import v6_1_fit\n")


def test_registry_has_exactly_a0_and_budget_ok():
    t = RG.trial_text(["A0"]).splitlines()
    assert t[0] == IO.HEADER_TRIALS.strip() and len(t) == 2 and t[1].startswith("Ж-227:A0,") and len(RG.CONFIGS) <= 18
    assert "λ" not in t[1] and "sens" not in t[1].lower()
    assert CP.ALL_TRIALS.read_text(encoding="utf-8").startswith(RG.trial_text(["A0"]))          # диск = реестр (с листа 1b после строки A0 идут A1–A3: правка по G18)


@pytest.fixture(scope="module")
def result():
    return S1.compute()


def env(tmp_path, monkeypatch, result):
    shutil.copytree(IO.REP, tmp_path / "rep")
    shutil.copytree(IO.OUT, tmp_path / "out")
    monkeypatch.setattr(IO, "REP", tmp_path / "rep")
    monkeypatch.setattr(IO, "OUT", tmp_path / "out")
    monkeypatch.setattr(CP, "CP", tmp_path / "rep" / "CHECKPOINT.json")
    monkeypatch.setattr(CP, "ALL_TRIALS", tmp_path / "rep" / "ALL_TRIALS.csv")
    monkeypatch.setattr(S1, "compute", lambda: result)


def test_check_changes_no_file_and_passes(tmp_path, monkeypatch, capsys, result):
    env(tmp_path, monkeypatch, result)
    before = IO.outputs_digest()
    assert run_v6.main(["--stage", "1a", "--check"]) == 0 and IO.outputs_digest() == before
    assert "ИТОГ: OK" in capsys.readouterr().out


def test_mutation_check_writes_service_file_caught(tmp_path, monkeypatch, capsys, result):
    env(tmp_path, monkeypatch, result)
    real = S1.check

    def writing(r):
        (IO.REP / "CHECK_runtime_sec.txt").write_text("1.0")
        return real(r)
    monkeypatch.setattr(S1, "check", writing)
    assert run_v6.main(["--stage", "1a", "--check"]) == 1 and "изменил файлы" in capsys.readouterr().out


def test_mutation_table_on_disk_changed_caught(tmp_path, monkeypatch, capsys, result):
    env(tmp_path, monkeypatch, result)
    p = IO.REP / "A0_METRICS.csv"
    lines = p.read_text(encoding="utf-8").splitlines()
    lines[3] = lines[3].replace(",0", ",1", 1)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert run_v6.main(["--stage", "1a", "--check"]) == 1 and "A0_METRICS.csv" in capsys.readouterr().out


def test_mutation_forecast_parquet_changed_caught(tmp_path, monkeypatch, capsys, result):
    env(tmp_path, monkeypatch, result)
    a = pd.read_parquet(IO.OUT / "A0.parquet")
    a.loc[5, "F"] *= 1.001
    a.to_parquet(IO.OUT / "A0.parquet", index=False)
    assert run_v6.main(["--stage", "1a", "--check"]) == 1 and "A0.parquet" in capsys.readouterr().out


def test_mutation_checkpoint_and_trials_changed_caught(tmp_path, monkeypatch, capsys, result):
    env(tmp_path, monkeypatch, result)
    cp = json.loads(CP.CP.read_text())
    cp["1a"]["qlike_all"] += 0.001
    CP.CP.write_text(json.dumps(cp, ensure_ascii=False))
    assert run_v6.main(["--stage", "1a", "--check"]) == 1 and "CHECKPOINT ключ 1a" in capsys.readouterr().out
    cp["1a"]["qlike_all"] -= 0.001
    CP.CP.write_text(json.dumps(cp, ensure_ascii=False))
    CP.ALL_TRIALS.write_text(RG.trial_text(["A0"]) + "Ж-227:A0_sens_up,Ж-227,x,y,z,trial\n", encoding="utf-8")        # чувствительность записана как конфигурация
    assert run_v6.main(["--stage", "1a", "--check"]) == 1 and "ALL_TRIALS" in capsys.readouterr().out


def test_checkpoint_1a_numbers_and_stage0_key_intact(result):
    cp = CP.read()
    assert cp["1a"]["used"] == 1 and cp["1a"]["budget"] == 18 and cp["1a"]["n_dates"] == 2271 and (cp["1a"]["N"], cp["1a"]["N_wide"]) == (722, 3122)
    assert (cp["0"]["N"], cp["0"]["N_wide"]) == (722, 3122) and cp["0"]["used"] == 0
