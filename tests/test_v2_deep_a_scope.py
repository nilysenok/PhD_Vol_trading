"""A16/A17: охват волны A — импорты, запрещённые пути, размеры."""
from models_v2.s5 import checks


def test_importy_i_zapreshchennye_stroki():
    assert checks.scope_problems() == []


def test_razmery_faylov_i_funkciy():
    assert checks.size_problems() == []


def test_mut_proverka_lovit_chuzhoy_import(tmp_path):
    src = "import tensorflow\nfrom models import old\n"
    import ast
    mods = [n.module.split(".")[0] if hasattr(n, "module") else n.names[0].name for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert [m for m in mods if m not in checks.ALLOWED] == ["tensorflow", "models"]
