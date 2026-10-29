"""Сбор всего `tests`: в каталоге нет __init__.py, соседей импортируют как `from test_x import`.
`from tests.test_x import` ломает сбор при запуске каталога целиком (ModuleNotFoundError, rassl-filtr-v6-sbor)."""
import re
from pathlib import Path

BAD = re.compile(r"^(from tests[. ]|import tests\b)", re.M)


def test_net_importov_cherez_paket_tests():
    here = Path(__file__).parent
    assert not (here / "__init__.py").exists()
    bad = [p.name for p in here.glob("test_*.py") if BAD.search(p.read_text(encoding="utf-8"))]
    assert bad == [], bad


def test_mutatsiya_import_cherez_paket_lovitsya():
    assert BAD.search("import numpy\nfrom tests.test_v5_filtr import GRID\n")
    assert not BAD.search("from test_v5_filtr import GRID\n")
