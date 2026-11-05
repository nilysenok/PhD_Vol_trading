"""Проверки охвата волны A: размеры файлов/функций, импорты, запрещённые строки."""
from __future__ import annotations

import ast
from pathlib import Path

from models_v2.s5 import common

ALLOWED = {"__future__", "ast", "json", "hashlib", "sys", "time", "argparse", "pathlib", "re", "itertools", "dataclasses", "numpy", "pandas", "scipy",
           "matplotlib", "xgboost", "models_v2", "pytest", "subprocess", "warnings"}
FORBIDDEN = ("test_locked", "_test.", ".env", "rvi_1850_test")


def files() -> list:
    return sorted((common.ROOT / "models_v2/s5").glob("*.py")) + sorted((common.ROOT / "tests").glob("test_v2_deep_*.py"))


def size_problems(max_file: int = 200, max_func: int = 40) -> list:
    bad = []
    for p in files():
        src = p.read_text()
        if len(src.splitlines()) > max_file:
            bad.append("%s: %d строк" % (p.name, len(src.splitlines())))
        lines = src.splitlines()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.end_lineno - n.lineno + 1 > max_func:
                if "xtask: длинная" not in "\n".join(lines[max(0, n.lineno - 3):n.lineno]):
                    bad.append("%s:%s %d строк" % (p.name, n.name, n.end_lineno - n.lineno + 1))
    return bad


def scope_problems() -> list:
    bad = []
    for p in files():
        if p.name in ("checks.py", "test_v2_deep_a_scope.py"):
            continue   # здесь запрещённые строки названы как данные проверки
        src = p.read_text()
        for n in ast.walk(ast.parse(src)):
            mods = [a.name.split(".")[0] for a in n.names] if isinstance(n, ast.Import) else [n.module.split(".")[0]] if isinstance(n, ast.ImportFrom) and n.module else []
            bad += ["%s: импорт %s" % (p.name, m) for m in mods if m not in ALLOWED]
        bad += ["%s: строка %s" % (p.name, s) for s in FORBIDDEN if s in src]
    return bad
