"""V6-0, G8: `--stage 0 --check` не меняет файлов (хэш до/после); диагностика задним числом (`v6_0_dx*`, `v6_dx*`) не импортируется путём прогноза (AST); CHECKPOINT: N = 722, N_wide = 3122, ALL_TRIALS — только заголовок.
Мутации: «проверка пишет служебный файл» ловится; подмена таблицы на диске ловится; импорт диагностики путём прогноза ловится; N, подменённый на 1, ловится."""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

from models_v2.s13 import run_v6, v6_0_cp as CP, v6_0_io as IO, v6_0_stage as ST

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"v6_0_stage.py", "run_v6.py"}


def imports_dx(source: str) -> bool:
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import) and any("v6_0_dx" in a.name or "v6_dx" in a.name for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and ("v6_0_dx" in (n.module or "") or "v6_dx" in (n.module or "") or any(a.name.startswith(("v6_0_dx", "v6_dx")) for a in n.names)):
            return True
    return False


def violators() -> list:
    bad = []
    for d in ("s10", "s11", "s12", "s13"):
        for p in sorted((ROOT / "models_v2" / d).glob("*.py")):
            if p.name in ALLOWED or p.name.startswith(("v6_0_dx", "v6_dx", "v5_0_dx", "v5_dx1_")):
                continue
            if imports_dx(p.read_text(encoding="utf-8")):
                bad.append("%s/%s" % (d, p.name))
    return bad


def test_dx_not_imported_by_forecast_paths():
    assert violators() == []


def test_mutation_dx_import_in_forecast_path_is_caught():
    assert imports_dx("from models_v2.s13 import v6_0_dx_x\n") and imports_dx("import models_v2.s13.v6_dx_y as x\n") and imports_dx("from models_v2.s13.v6_0_dx_z import f\n")
    assert not imports_dx("from models_v2.s13 import v6_0_io\n")


def _tmp_env(tmp_path, monkeypatch):
    shutil.copytree(IO.REP, tmp_path / "rep")
    if IO.OUT.exists():
        shutil.copytree(IO.OUT, tmp_path / "out")
    monkeypatch.setattr(IO, "REP", tmp_path / "rep")
    monkeypatch.setattr(IO, "OUT", tmp_path / "out")
    monkeypatch.setattr(CP, "CP", tmp_path / "rep" / "CHECKPOINT.json")
    monkeypatch.setattr(CP, "ALL_TRIALS", tmp_path / "rep" / "ALL_TRIALS.csv")
    monkeypatch.setattr(ST, "MAP", tmp_path / "rep" / "MAP_DONE_V6.md")


def test_check_changes_no_file(tmp_path, monkeypatch, capsys):
    _tmp_env(tmp_path, monkeypatch)
    before = IO.outputs_digest()
    rc = run_v6.main(["--stage", "0", "--check"])
    assert IO.outputs_digest() == before and rc == 0
    assert "ИТОГ: OK" in capsys.readouterr().out


def test_mutation_check_writes_service_key_is_caught(tmp_path, monkeypatch, capsys):
    _tmp_env(tmp_path, monkeypatch)
    real_check = ST.check

    def writing_check(r):
        (IO.REP / "CHECK_runtime_sec.txt").write_text("1.0")                       # служебный ключ, как DX_runtime_sec в V4
        return real_check(r)

    monkeypatch.setattr(ST, "check", writing_check)
    assert run_v6.main(["--stage", "0", "--check"]) == 1
    assert "изменил файлы" in capsys.readouterr().out


def test_mutation_table_on_disk_changed_is_caught(tmp_path, monkeypatch, capsys):
    _tmp_env(tmp_path, monkeypatch)
    p = IO.REP / "RESID_SBER.csv"
    p.write_text(p.read_text(encoding="utf-8").replace("2016-12-30", "2016-12-31", 1), encoding="utf-8")
    assert run_v6.main(["--stage", "0", "--check"]) == 1
    assert "RESID_SBER.csv: диск ≠ пересчёт" in capsys.readouterr().out


def test_checkpoint_n_and_all_trials_header_only(tmp_path, monkeypatch):
    _tmp_env(tmp_path, monkeypatch)
    e = CP.read()["0"]
    assert (e["N"], e["N_wide"], e["budget_V6"], e["used"]) == (722, 3122, 18, 0) and e["go_series"] and "written_at" in e
    assert CP.check(e) == []
    assert CP.ALL_TRIALS.read_text(encoding="utf-8") == IO.HEADER_TRIALS
    assert any("N = 1" in b for b in CP.check({**e, "N": 1}))                      # мутация: N, подменённый на 1
    CP.ALL_TRIALS.write_text(IO.HEADER_TRIALS + "k,Ж-227,x,A,{},trial\n", encoding="utf-8")
    assert any("ALL_TRIALS" in b for b in CP.check(e))                              # мутация: trial записан до первого прогноза


def test_thresholds_written_before_any_forecast_numbers_are_finite():
    e = CP.read()["0"]
    th = e["thresholds"]
    for k in ("cond1_crit", "cond2_crit_no2017", "cond3_crit_no2017_18", "cond4_crit_trim1"):
        assert 0 < th[k]["threshold"] < 1 and abs(th[k]["threshold"] - (th[k]["min_over_go"] - 0.0005)) < 1e-15
    assert sorted(th["cond5_annual_best_k23"]) == ["2017", "2018", "2019", "2020", "2021"]
