"""V5, лист 1b, G18: модули DX1 не импортируются путём прогноза (AST, включая `v5_gs_*`); `--check` ничего не пишет; лист 1b не трогает `CHECKPOINT.json`, `ALL_TRIALS.csv`, `state/`. Мутации: импорт DX1 путём прогноза; проверка пишет служебный файл."""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

from models_v2.s12 import run_v5_dx, v5_0_io as IO, v5_dx1_stage as ST

ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "models_v2" / "s12"
ALLOWED = {"run_v5_dx.py"}                        # запускающий модуль; сами v5_dx1_* друг друга импортируют
FORBIDDEN_NAMES = ("CHECKPOINT", "ALL_TRIALS", "write_checkpoint", "v5_0_reg", "v5_gs_")


def imports_dx1(source: str) -> bool:
    for n in ast.walk(ast.parse(source)):
        if isinstance(n, ast.Import) and any("v5_dx1" in a.name or "run_v5_dx" in a.name for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and ("v5_dx1" in (n.module or "") or "run_v5_dx" in (n.module or "") or any(a.name.startswith(("v5_dx1", "run_v5_dx")) for a in n.names)):
            return True
    return False


def violators() -> list:
    bad = []
    for d in ("s10", "s11", "s12"):
        for p in sorted((ROOT / "models_v2" / d).glob("*.py")):
            if p.name in ALLOWED or p.name.startswith("v5_dx1"):
                continue
            if imports_dx1(p.read_text(encoding="utf-8")):
                bad.append("%s/%s" % (d, p.name))
    return bad


def test_dx1_not_imported_by_forecast_paths_including_v5_gs():
    assert violators() == []            # перебирает все файлы s10–s12, в т. ч. `v5_gs_*` и `run_v5.py`, если они есть


def test_mutation_dx1_import_in_forecast_path_is_caught():
    assert imports_dx1("from models_v2.s12 import v5_dx1_mom\n") and imports_dx1("import models_v2.s12.v5_dx1_jump as j\n") and imports_dx1("from models_v2.s12.run_v5_dx import main\n")
    assert not imports_dx1("from models_v2.s12 import v5_0_io\n")


def code_words(source: str) -> str:
    """Имена, атрибуты, импорты и строковые константы вне docstring."""
    tree = ast.parse(source)
    doc = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef)) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    w = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            w.append(n.id)
        elif isinstance(n, ast.Attribute):
            w.append(n.attr)
        elif isinstance(n, ast.ImportFrom):
            w += [n.module or ""] + [a.name for a in n.names]
        elif isinstance(n, ast.Import):
            w += [a.name for a in n.names]
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc:
            w.append(n.value)
    return "\n".join(w)


def test_dx1_modules_do_not_touch_sheet_1a_files():
    mods = list(S12.glob("v5_dx1_*.py")) + [S12 / "run_v5_dx.py"]
    assert len(mods) >= 7
    for p in mods:
        code = code_words(p.read_text(encoding="utf-8"))
        assert not any(f in code for f in FORBIDDEN_NAMES), p.name


def test_mutation_touching_checkpoint_is_caught():
    assert "CHECKPOINT" in code_words("p = ROOT / 'CHECKPOINT.json'\n") and "write_checkpoint" in code_words("RG.write_checkpoint(x)\n")


def _tmp_env(tmp_path, monkeypatch):
    (tmp_path / "rep").mkdir()
    (tmp_path / "out").mkdir()
    for f in ST.own_files():
        shutil.copy2(f, tmp_path / ("rep" if f.parent == IO.REP else "out") / f.name)
    monkeypatch.setattr(IO, "REP", tmp_path / "rep")
    monkeypatch.setattr(IO, "OUT", tmp_path / "out")


def test_check_changes_no_file_and_exit_zero(tmp_path, monkeypatch, capsys):
    _tmp_env(tmp_path, monkeypatch)
    before = ST.digest()
    assert run_v5_dx.main(["--stage", "DX1", "--check"]) == 0
    assert ST.digest() == before and len(ST.own_files()) >= 3


def test_mutation_check_writes_service_key_is_caught(tmp_path, monkeypatch, capsys):
    _tmp_env(tmp_path, monkeypatch)
    real = ST.check

    def writing_check(r):
        (IO.REP / "DX1_runtime_sec.txt").write_text("1.0")          # служебный ключ, как DX_runtime_sec в V4
        return real(r)

    monkeypatch.setattr(ST, "check", writing_check)
    assert run_v5_dx.main(["--stage", "DX1", "--check"]) == 1
    assert "изменил файлы" in capsys.readouterr().out


def test_check_catches_disk_not_equal_recompute(tmp_path, monkeypatch, capsys):
    _tmp_env(tmp_path, monkeypatch)
    p = IO.REP / "DX1_JUMPS.csv"
    p.write_text(p.read_text(encoding="utf-8").replace("freq", "frex", 1), encoding="utf-8")
    assert run_v5_dx.main(["--stage", "DX1", "--check"]) == 1
    assert "DX1_JUMPS.csv: диск ≠ пересчёт" in capsys.readouterr().out
