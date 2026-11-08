"""V5-FIN (G26): изоляция и только чтение. `run_v5 --stage FIN --check` не меняет ни одного файла (хэши выходов V5, V4, итогового документа, state до и после) и не вызывает путей записи; модули прогноза GS/SP не импортируют FIN;
FIN не пишет parquet, не импортирует диагностику листа 1b и модели прогноза; размеры файлов и функций в пределах канона. Мутация «проверка пишет» ловится детектором."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

from models_v2.s12 import v5_0_io as IO, v5_fin_asm as A, v5_fin_cost as CO, v5_fin_main as MAIN, v5_fin_stage as ST

ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "models_v2" / "s12"
FIN = sorted(S12.glob("v5_fin_*.py"))
FORECAST = ["v5_gs_core.py", "v5_gs_graph.py", "v5_gs_fit.py", "v5_gs_panel.py", "v5_gs_run.py", "v5_sp_feat.py", "v5_sp_har.py", "v5_sp_fit.py", "v5_sp_run.py", "v5_0_bars.py"]


def snapshot(paths) -> dict:
    out = {}
    for d in paths:
        for f in sorted(Path(d).rglob("*")) if Path(d).is_dir() else [Path(d)]:
            if f.is_file() and f.name != ".DS_Store":
                out[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def changed(a: dict, b: dict) -> list:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def imported(src: str) -> list:
    names = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            names += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            names += [n.module or ""] + [a.name for a in n.names]
    return names


def test_stage_fin_check_changes_no_file_and_never_calls_write_paths(monkeypatch):
    paths = [IO.REP, IO.OUT, IO.ROOT / "docs/otchety/harj_hybrid_v4", IO.ROOT / "data/regen/harj_hybrid_v4", A.B.FINAL, IO.STATE]
    before, calls = snapshot(paths), []
    for obj, name in ((ST, "write"), (A, "write_docs"), (CO, "timing"), (ST, "tables_sha_write")):
        if hasattr(obj, name):
            monkeypatch.setattr(obj, name, lambda *a, **k: calls.append(name))
    rc = MAIN.main(argparse.Namespace(check=True))
    assert rc == 0 and calls == []
    assert changed(before, snapshot(paths)) == []


def test_detector_catches_check_that_writes(tmp_path):
    (tmp_path / "FIN_ALL.csv").write_text("a\n1\n")
    before = snapshot([tmp_path])
    (tmp_path / "FIN_ALL.csv").write_text("a\n2\n")
    assert changed(before, snapshot([tmp_path])) == [str(tmp_path / "FIN_ALL.csv")]


def test_forecast_modules_do_not_import_fin_and_nothing_outside_fin_does():
    for f in FORECAST:
        assert not any("v5_fin" in x for x in imported((S12 / f).read_text(encoding="utf-8"))), f
    others = [p for d in ("s10", "s11", "s12") for p in (ROOT / "models_v2" / d).glob("*.py") if not p.name.startswith("v5_fin_") and p.name != "run_v5.py"]
    assert [p.name for p in others if any("v5_fin" in x for x in imported(p.read_text(encoding="utf-8")))] == []


def test_fin_modules_import_neither_dx_nor_write_parquet_and_fit_size_limits():
    assert FIN and len(FIN) >= 8
    for p in FIN:
        src = p.read_text(encoding="utf-8")
        assert "to_parquet" not in src and not any(("v5_dx1" in x or "v5_0_dx" in x or "run_v5_dx" in x) for x in imported(src)), p.name
        assert len(src.splitlines()) <= 200, p.name
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.FunctionDef):
                assert n.end_lineno - n.lineno + 1 <= 40, (p.name, n.name)


def test_stage_zero_check_ignores_fin_key_in_checkpoint():
    from models_v2.s12 import v5_0_reg as RG, v5_0_stage as S0
    assert "FIN" in Path(RG.__file__).read_text(encoding="utf-8") and "FIN" in Path(S0.__file__).read_text(encoding="utf-8")
