"""V5 лист 1a, G15: модули диагностики и оценки GS не импортируются путём прогноза (AST); `--stage GS --check` не меняет файлов (хэш до/после); мутации ловятся."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from models_v2.s12 import run_v5, v5_0_io as IO

ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "models_v2" / "s12"
FORECAST = ["v5_gs_core.py", "v5_gs_panel.py", "v5_gs_graph.py", "v5_gs_fit.py", "v5_gs_run.py"]
DIAG = ("v5_0_dx", "v5_dx1", "v5_gs_eval", "v5_gs_enc", "v5_gs_stage", "v5_gs_main", "run_v5")


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
    for src in ("from models_v2.s12 import v5_0_dx_score", "import models_v2.s12.v5_gs_eval", "from models_v2.s12.v5_dx1_moments import f", "from models_v2.s12 import v5_gs_stage as ST"):
        assert imports_diag(src), src
    assert not imports_diag("from models_v2.s12 import v5_gs_core as CO, v5_0_io as IO")


def test_no_s10_s11_module_imports_gs():
    bad = []
    for d in ("s10", "s11"):
        for p in (ROOT / "models_v2" / d).glob("*.py"):
            if "v5_gs" in p.read_text(encoding="utf-8"):
                bad.append(p.name)
    assert bad == []


def test_check_changes_no_file():
    if not (IO.REP / "GS_COMPARISON.csv").exists():
        pytest.skip("стадия GS ещё не записана")
    before = IO.outputs_digest()
    rc = run_v5.main(["--stage", "GS", "--check"])
    assert rc == 0 and IO.outputs_digest() == before
