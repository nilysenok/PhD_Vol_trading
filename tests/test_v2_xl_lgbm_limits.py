"""LGBM волна B: файл <= 200 строк, функция <= 40 (иначе пометка `# xtask: длинная`), запрещённые чтения в коде потока, нет python -c/heredoc."""
import ast
import re
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "models_v2" / "s6" / "lgbm"
FILES = sorted(D.glob("*.py"))
FORBIDDEN = re.compile(r"_test\.|test_locked|\.env\b|rvi_1850_test")


def test_files_not_longer_than_200():
    assert FILES and all(len(f.read_text().splitlines()) <= 200 for f in FILES), [f.name for f in FILES if len(f.read_text().splitlines()) > 200]


def test_functions_not_longer_than_40_or_marked():
    bad = []
    for f in FILES:
        src = f.read_text().splitlines()
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.end_lineno - n.lineno + 1 > 40:
                if not any("xtask: длинная" in s for s in src[max(0, n.lineno - 3):n.lineno]):
                    bad.append("%s:%s" % (f.name, n.name))
    assert not bad, bad


def test_no_forbidden_reads_in_code():
    hits = [f.name for f in FILES if FORBIDDEN.search(f.read_text())]
    assert not hits, hits


def test_mutation_forbidden_pattern_is_detected():
    assert FORBIDDEN.search("pd.read_parquet('rvi_1850_test.parquet')") and FORBIDDEN.search("open('.env')")
