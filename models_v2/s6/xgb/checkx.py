"""B9 (GATES_B_xgb): гигиена — размеры файлов и функций, запрещённые обращения, прежние файлы не тронуты."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from models_v2.s6 import paths6

HERE = Path(__file__).resolve().parent
GATES = paths6.REP / "xgboost" / "GATES_B_xgb.md"
FORBID = re.compile(r"_test\.|test_locked|\.env\b")
MAX_LINES, MAX_FUNC = 200, 40


def size_violations() -> list:
    out = []
    for f in sorted(HERE.glob("*.py")):
        src = f.read_text(encoding="utf-8")
        lines = src.splitlines()
        if len(lines) > MAX_LINES:
            out.append("%s: %d строк" % (f.name, len(lines)))
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.end_lineno - n.lineno + 1 > MAX_FUNC:
                if "xtask: длинная" not in lines[n.lineno - 1] and "xtask: длинная" not in lines[n.lineno - 2]:
                    out.append("%s:%s %d строк" % (f.name, n.name, n.end_lineno - n.lineno + 1))
    return out


def forbidden_refs() -> list:
    """Обращения к запрещённым источникам в коде потока (сам этот файл — описание шаблона — пропускается)."""
    out = []
    for f in sorted(HERE.glob("*.py")):
        if f.name != "checkx.py":
            out += ["%s:%d" % (f.name, i + 1) for i, ln in enumerate(f.read_text(encoding="utf-8").splitlines()) if FORBID.search(ln)]
    return out


def newer_than_gates() -> list:
    """Файлы новее GATES_B_xgb.md вне разрешённых зон: models_v2 вне s6, прежние data/regen/v2_sber_h1* и docs/otchety/sber_h1_deep_v1."""
    t0, out = GATES.stat().st_mtime, []
    roots = [paths6.ROOT / "models_v2", *paths6.OLD_DATA, paths6.OLD_REP]
    for r in roots:
        for f in r.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts and f.stat().st_mtime > t0:
                if r.name == "models_v2" and "s6" in f.relative_to(r).parts[:1]:
                    continue
                out.append(str(f.relative_to(paths6.ROOT)))
    return out


def stage(rows: dict) -> int:
    bad = {"размеры": size_violations(), "запрещённые обращения": forbidden_refs(), "новее GATES вне s6": newer_than_gates()}
    n = sum(len(v) for v in bad.values())
    print("B9: нарушений %d %s" % (n, {k: v[:10] for k, v in bad.items() if v}))
    return 0 if n == 0 else 1
