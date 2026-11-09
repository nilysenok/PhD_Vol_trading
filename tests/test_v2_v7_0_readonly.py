"""V7-0, G7: `--check` ничего не пишет (хэш выходов до = после; детектор ловит запись); модули листа 0 не импортируют обучатели (AST; детектор ловит подложенный импорт)."""
import ast
from pathlib import Path

import pytest

from models_v2.s14 import run_v7 as RUN, v7_0_io as IO

PKG = Path(RUN.__file__).parent
FILES = sorted(PKG.glob("v7_0_*.py")) + [PKG / "run_v7.py"]
FORBIDDEN = ("xgboost", "lightgbm", "sklearn", "torch", "optuna", "catboost", "models_v2.s5", "models_v2.s6", "models_v2.s7", "models_v2.s8", "models_v2.s9.hv2_fit", "models_v2.s10.hp_k23_calc", "models_v2.s11.v4_fit", "models_v2.s13.v6_1_fit", "models_v2.har")


def imports(src: str) -> list:
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out += [n.module] + ["%s.%s" % (n.module, a.name) for a in n.names]
    return out


def trainers(src: str) -> list:
    return [m for m in imports(src) if any(m == p or m.startswith(p + ".") for p in FORBIDDEN)]


def test_sheet_modules_import_no_trainers():
    assert len(FILES) >= 8
    for f in FILES:
        assert trainers(f.read_text(encoding="utf-8")) == [], f.name


@pytest.mark.parametrize("line", ["import xgboost", "from models_v2.s6.xgb import c_calc", "import lightgbm as lgb", "from models_v2.s6.lgbm import job", "from sklearn.linear_model import Ridge", "from models_v2.har import fit_window"])
def test_mutation_trainer_import_is_detected(line):
    assert trainers(line + "\n") != []


def test_allowed_imports_are_not_flagged():
    assert trainers("from models_v2 import metrics, daily as D\nimport numpy as np\nfrom models_v2.s14 import v7_0_io\n") == []


def test_check_changes_no_output_file():
    before = IO.outputs_digest()
    assert RUN.main(["--stage", "0", "--check"]) == 0
    assert IO.outputs_digest() == before


def test_digest_detects_a_write(tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "REP", tmp_path / "rep")
    monkeypatch.setattr(IO, "OUT", tmp_path / "out")
    monkeypatch.setattr(IO, "STATE", tmp_path / "state.md")
    (tmp_path / "rep").mkdir()
    (tmp_path / "rep" / "A.csv").write_text("a")
    d0 = IO.outputs_digest()
    (tmp_path / "rep" / "A.csv").write_text("b")
    assert IO.outputs_digest() != d0                                          # изменение файла
    d1 = IO.outputs_digest()
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "n.npz").write_text("x")
    assert IO.outputs_digest() != d1                                          # новый файл


def test_check_mode_never_calls_writers(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("запись в режиме --check")
    monkeypatch.setattr(IO, "write_csv", boom)
    monkeypatch.setattr(RUN, "write_stage", boom)
    assert RUN.main(["--stage", "0", "--check"]) == 0
