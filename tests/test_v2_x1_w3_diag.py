"""W4-A, `x1_w3_diag --check`: строки данных W3_DIAG.csv совпадают с пересчётом побитно; проверка только читает (хэши файла и всех выходов до и после равны);
мутация «check пишет» ловится. Реальные выходы только читаются; подмены — monkeypatch (класс № 43)."""
from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from models_v2.s16 import x1_w3_diag as D

ROOT = Path(D.__file__).resolve().parents[2]
REAL = ROOT / "docs/otchety/harj_x1/W3_DIAG.csv"
WATCH = (ROOT / "docs/otchety/harj_x1", ROOT / "data/regen/harj_x1")


def snapshot(dirs=WATCH) -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for d in dirs if d.is_dir() for p in sorted(d.rglob("*")) if p.is_file()}


def run(flag: str):
    return subprocess.run([sys.executable, "-W", "ignore", "-m", "models_v2.s16.x1_w3_diag", flag], cwd=ROOT, capture_output=True, text=True, timeout=300)


def body(text: str) -> str:
    return text.split("\n", 1)[1]


def test_recomputation_equals_saved_data_rows_bitwise() -> None:
    assert D.data_text() == body(REAL.read_text(encoding="utf-8"))
    assert REAL.read_text(encoding="utf-8").startswith("# DEVELOPMENT / HISTORICAL WALK-FORWARD")


@pytest.mark.parametrize("flag,expect", [("--check", "check ok"), ("--check-pit", "pit ok"), ("--check-repro", "repro ok"), ("--check-known", "known ok")])
def test_checks_run_clean_and_change_no_output(flag, expect) -> None:
    before = snapshot()
    r = run(flag)
    after = snapshot()
    assert r.returncode == 0 and expect in r.stdout, r.stdout + r.stderr
    assert before == after and REAL.as_posix() in before                                  # хэши W3_DIAG.csv и всех выходов W3/W4/харнесса прежние


def fake_out(tmp_path, monkeypatch):
    saved = REAL.read_text(encoding="utf-8")
    out = tmp_path / "W3_DIAG.csv"
    out.write_text(saved, encoding="utf-8")
    monkeypatch.setattr(D, "OUT", out)
    return out, saved


def test_check_logic_honest_and_data_mismatch(tmp_path, monkeypatch, capsys) -> None:
    out, saved = fake_out(tmp_path, monkeypatch)
    monkeypatch.setattr(D, "data_text", lambda: body(saved))                              # дешёвый честный пересчёт: ровно строки файла
    assert D.check() is True and "check ok" in capsys.readouterr().out
    out.write_text("# другая шапка\n" + body(saved), encoding="utf-8")
    assert D.check() is True                                                              # шапка-комментарий в сверку не входит
    monkeypatch.setattr(D, "data_text", lambda: body(saved).replace("0.", "1.", 1))      # расхождение в одной цифре
    assert D.check() is False and "FAIL" in capsys.readouterr().out
    monkeypatch.setattr(D, "data_text", lambda: body(saved) + "extra\n")                  # лишняя строка
    assert D.check() is False


def test_mutation_check_that_writes_is_caught(tmp_path, monkeypatch) -> None:
    out, saved = fake_out(tmp_path, monkeypatch)

    def writing_recompute() -> str:
        out.write_text(saved + "# изменено проверкой\n", encoding="utf-8")                # мутация: пересчёт пишет в проверяемый файл
        return body(saved)

    monkeypatch.setattr(D, "data_text", writing_recompute)
    assert D.check() is False                                                             # sha до ≠ после — замечание
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    assert h != hashlib.sha256(saved.encode()).hexdigest()                                # и внешний хэш это видит


def write_calls(src: str) -> list:
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in ("write_text", "write_bytes", "to_parquet", "savez", "save", "dump") or (n.func.attr == "to_csv" and n.args) or (n.func.attr == "open" and "'w" in ast.dump(n)):
                out.append(n.func.attr)
    return out


def writers(src: str) -> list:
    return [f.name for f in ast.parse(src).body if isinstance(f, ast.FunctionDef) and write_calls(ast.unparse(f))]


def test_only_write_writes_and_scanner_sees_mutation() -> None:
    src = Path(D.__file__).read_text(encoding="utf-8")
    assert writers(src) == ["write"]                                                      # единственный пишущий режим — явный --write
    mutated = src.replace('    h1 = hashlib.sha256(OUT.read_bytes()).hexdigest()', '    OUT.write_text("x")\n    h1 = hashlib.sha256(OUT.read_bytes()).hexdigest()', 1)
    assert mutated != src and sorted(writers(mutated)) == ["check", "write"]


def test_default_flag_is_check_and_write_needs_explicit_flag() -> None:
    src = Path(D.__file__).read_text(encoding="utf-8")
    assert 'else "--check"' in src and '"--write": write' in src and "days.parquet" not in src.split("def write", 1)[1].split("def check", 1)[0]
