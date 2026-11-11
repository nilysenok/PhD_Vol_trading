"""W4-A, изоляция и «только читает»: `check` регистратора ничего не пишет; реальный реестр, CHECKPOINT и W4 не тронуты тестами; в тестах W3/W4-A нет присваиваний атрибутам модулей
(только monkeypatch, класс № 43); модули подмен восстановлены. Реальные файлы только читаются."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from models_v2.s16 import x1_w3_core as C
from models_v2.s16 import x1_w4_reg as W

ROOT = Path(W.__file__).resolve().parents[2]
TESTS = sorted(ROOT.glob("tests/test_v2_x1_w3_*.py")) + sorted(ROOT.glob("tests/test_v2_x1_w4_reg*.py"))
REAL = (W.TRIALS, W.CKPT)


def real_state() -> dict:
    s = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in REAL if p.exists()}
    s["W4"] = sorted(str(p) for p in W.W4.rglob("*")) if W.W4.exists() else []
    return s


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = real_state()
    yield
    assert real_state() == before                                                         # реальный ALL_TRIALS.csv, CHECKPOINT.json и каталог W4 прежние


def attr_assignments(src: str, mods: set) -> list:
    out = []
    for n in ast.walk(ast.parse(src)):
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else []):
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in mods:
                out.append(ast.unparse(n))
    return out


def test_no_module_attribute_assignment_in_w3_w4_tests_and_scanner_sees_one() -> None:
    assert len(TESTS) >= 5
    for p in TESTS:
        src = p.read_text(encoding="utf-8")
        mods = {n.asname or n.name for t in ast.walk(ast.parse(src)) if isinstance(t, ast.ImportFrom) and (t.module or "").startswith("models_v2") for n in t.names}
        assert attr_assignments(src, mods) == [], p.name
    assert attr_assignments("from models_v2.s16 import x1_w3_core as C\nC.closed_date = lambda t, d: d\n", {"C"})   # сканер видит подложенное присваивание


def test_swapped_functions_are_originals_after_other_tests() -> None:
    t = [1, 2]
    assert C.closed_date(t, 0) is t                                                       # подмена closed_date не осталась в модуле
    assert C.dftl.__code__.co_name == "dftl" and C.adahedge_fs.__code__.co_name == "adahedge_fs"
    assert W.register.__code__.co_name == "register" and W.require.__code__.co_name == "require" and W.check.__code__.co_name == "check"


def test_check_only_reads(tmp_path) -> None:
    t, w4 = tmp_path / "ALL_TRIALS.csv", tmp_path / "W4"
    W.register("11.11.2026 13:00:00", trials=_seed(t), w4=w4, ckpt=None)
    snap = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    assert W.check(trials=t, w4=w4) == []
    assert snap == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()} and not w4.exists()


def _seed(t: Path) -> Path:
    t.write_text("id,name,formula,params,status,registered_at\nY1,a,b,c,зарегистрирована,10.11.2026 04:21:03\n", encoding="utf-8")
    return t


def test_check_that_writes_is_caught_by_hash(tmp_path, monkeypatch) -> None:
    t, w4 = _seed(tmp_path / "ALL_TRIALS.csv"), tmp_path / "W4"
    W.register("11.11.2026 13:00:00", trials=t, w4=w4, ckpt=None)
    before = hashlib.sha256(t.read_bytes()).hexdigest()
    orig = W._parquet
    monkeypatch.setattr(W, "_parquet", lambda d: (t.write_text(t.read_text(encoding="utf-8") + "# изменено проверкой\n", encoding="utf-8"), orig(d))[1])      # мутация: check пишет
    W.check(trials=t, w4=w4)
    assert hashlib.sha256(t.read_bytes()).hexdigest() != before                           # внешний хэш выхода это видит


def write_calls(src: str) -> list:
    return [n.func.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("write_text", "write_bytes", "open", "to_csv", "dump")]


def func_src(src: str, name: str) -> str:
    return next(ast.unparse(f) for f in ast.parse(src).body if isinstance(f, ast.FunctionDef) and f.name == name)


def test_check_and_require_have_no_writes_and_scanner_sees_mutation() -> None:
    src = Path(W.__file__).read_text(encoding="utf-8")
    for f in ("check", "require", "_read", "_parquet"):
        assert write_calls(func_src(src, f)) == [], f
    assert write_calls(func_src(src, "register")) and write_calls(func_src(src, "merge_checkpoint"))        # писатели — только эти две
    assert write_calls(func_src(src.replace("    cur = _read(trials)\n    rows", "    trials.write_text('x')\n    rows", 1), "check"))
