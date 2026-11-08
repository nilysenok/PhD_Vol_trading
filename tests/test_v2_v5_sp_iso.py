"""V5 лист 2, G19/G22: модули прогноза SP не импортируют диагностику, оценку и стадию (AST); диагностика листа 0 и модули `v5_sp_eval/stage/main` не импортируются путём прогноза; `--stage SP --check` не меняет файлов (хэш до/после); мутации ловятся."""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

from models_v2.s12 import run_v5, v5_0_io as IO

ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "models_v2" / "s12"
FORECAST = ["v5_sp_feat.py", "v5_sp_har.py", "v5_sp_fit.py", "v5_sp_run.py"]
DIAG = ("v5_0_dx", "v5_dx1", "v5_gs_eval", "v5_gs_enc", "v5_gs_stage", "v5_gs_main", "v5_sp_eval", "v5_sp_stage", "v5_sp_main", "run_v5")


def imports_diag(source: str) -> bool:
    for n in ast.walk(ast.parse(source)):
        names = [a.name for a in n.names] if isinstance(n, ast.Import) else ([n.module or ""] + [a.name for a in n.names]) if isinstance(n, ast.ImportFrom) else []
        if any(x.split(".")[-1].startswith(DIAG) for x in names):
            return True
    return False


def test_forecast_path_does_not_import_diagnostics_or_estimation():
    assert [f for f in FORECAST if imports_diag((S12 / f).read_text(encoding="utf-8"))] == []
    assert all((S12 / f).exists() for f in FORECAST)


def test_mutation_diagnostic_import_in_forecast_path_is_caught():
    for src in ("from models_v2.s12 import v5_0_dx_spec", "import models_v2.s12.v5_sp_eval", "from models_v2.s12 import v5_sp_stage as ST", "from models_v2.s12.v5_gs_eval import tests"):
        assert imports_diag(src), src
    assert not imports_diag("from models_v2.s12 import v5_sp_feat as FE, v5_0_bars as BR")


def test_no_s10_s11_module_imports_sp():
    bad = [p.name for d in ("s10", "s11") for p in (ROOT / "models_v2" / d).glob("*.py") if "v5_sp" in p.read_text(encoding="utf-8")]
    assert bad == []


def test_mutation_check_writes_service_key_is_caught(tmp_path, monkeypatch, capsys):
    if not (IO.REP / "SP_COMPARISON.csv").exists():
        pytest.skip("стадия SP ещё не записана")
    for src, name in ((IO.REP, "rep"), (IO.OUT, "out")):
        shutil.copytree(src, tmp_path / name)
    monkeypatch.setattr(IO, "REP", tmp_path / "rep")
    monkeypatch.setattr(IO, "OUT", tmp_path / "out")
    from models_v2.s12 import v5_sp_stage as ST
    real = ST.check

    def writing_check(r):
        (IO.REP / "CHECK_runtime_sec.txt").write_text("1.0")            # служебный ключ, как DX_runtime_sec в V4
        return real(r)
    monkeypatch.setattr(ST, "check", writing_check)
    assert run_v5.main(["--stage", "SP", "--check"]) == 1
    assert "изменил файлы" in capsys.readouterr().out


def test_check_changes_no_file():
    if not (IO.REP / "SP_COMPARISON.csv").exists():
        pytest.skip("стадия SP ещё не записана")
    before = IO.outputs_digest()
    rc = run_v5.main(["--stage", "SP", "--check"])
    assert rc == 0 and IO.outputs_digest() == before
