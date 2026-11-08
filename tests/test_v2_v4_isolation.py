"""G8 (в): oracle не импортируется путями прогноза. Мутация — код пути прогноза с импортом oracle (любой формы) ловится сканером."""
import pytest

from models_v2.s11 import v4_isolation as ISO


def test_real_tree_has_no_oracle_importers():
    assert ISO.violators() == []


@pytest.mark.parametrize("src", [
    "from models_v2.s11 import v4_oracle\n",
    "from models_v2.s11.v4_oracle import oracle\n",
    "import models_v2.s11.v4_oracle as o\n",
    "from . import v4_oracle\n",
    "def f():\n    from models_v2.s11 import v4_a, v4_oracle\n",
])
def test_mutation_oracle_import_is_caught(src):
    assert ISO.imports_oracle(src)


@pytest.mark.parametrize("src", ["from models_v2.s11 import v4_io\n", "import numpy as np\nx = 'v4_oracle'\n", "# from models_v2.s11 import v4_oracle\n"])
def test_clean_code_not_flagged(src):
    assert not ISO.imports_oracle(src)


def test_mutation_forecast_path_file_importing_oracle_is_listed(monkeypatch, tmp_path):
    """Подложенный файл пути прогноза в дереве s11 с импортом oracle попадает в нарушители."""
    f = ISO.ROOT / "models_v2" / "s11" / "v4_zz_mutant_forecast_path.py"
    f.write_text("from models_v2.s11 import v4_oracle\n", encoding="utf-8")
    try:
        assert "s11/v4_zz_mutant_forecast_path.py" in ISO.violators()
    finally:
        f.unlink()
    assert ISO.violators() == []
