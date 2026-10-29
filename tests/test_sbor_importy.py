"""Сбор всего `tests`: в каталоге нет __init__.py, соседей импортируют как `from test_x import`.
`from tests.test_x import` ломает сбор при запуске каталога целиком (в sys.path есть другой пакет `tests`,
moex_volatility/tests). Страж сам был красным на базе (его не гоняли), поэтому добавлен настоящий сбор."""
import re
import subprocess
import sys
from pathlib import Path

BAD = re.compile(r"^[ \t]*(from tests[. ]|import tests\b)", re.M)
HERE = Path(__file__).parent


def test_net_importov_cherez_paket_tests():
    assert not (HERE / "__init__.py").exists()
    bad = [p.name for p in HERE.glob("*.py") if p.name != Path(__file__).name
           and BAD.search(p.read_text(encoding="utf-8"))]
    assert bad == [], bad


def test_sbor_bez_oshibok():
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", str(HERE),
                        "--deselect", f"{Path(__file__).name}::test_sbor_bez_oshibok"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "error" not in r.stdout.lower().splitlines()[-1], r.stdout[-1500:]


def test_mutatsiya_import_cherez_paket_lovitsya():
    assert BAD.search("import numpy\nfrom tests.test_v5_filtr import GRID\n")
    assert BAD.search("def f():\n    from tests import test_eval_v5 as T\n")
    assert not BAD.search("from test_v5_filtr import GRID\n")
