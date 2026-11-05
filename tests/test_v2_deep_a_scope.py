"""A16/A17: охват волны A — импорты, запрещённые пути, размеры.
D0 (Ж-204): optuna разрешена блоком (п. 25) только в `models_v2/s5/c/`, `models_v2/s5/d/` и их тестах `test_v2_deep_{c,d}_*`; math — стандартная библиотека."""
import ast

from models_v2.s5 import checks, common

STDLIB_OK = {"math"}
OPTUNA_OK_DIRS = ("c", "d")
OPTUNA_OK_TESTS = ("test_v2_deep_c_", "test_v2_deep_d_")


def imports_of(path):
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[0])
    return out


def optuna_users():
    root = common.ROOT
    found = [p for p in (root / "models_v2").rglob("*.py") if "optuna" in imports_of(p)]
    return found + [p for p in (root / "tests").glob("test_v2_*.py") if "optuna" in imports_of(p)]


def allowed_exception(msg):
    name, _, what = msg.partition(": импорт ")
    if what in STDLIB_OK:
        return True
    return what == "optuna" and name.startswith(OPTUNA_OK_TESTS)


def test_importy_i_zapreshchennye_stroki():
    assert [m for m in checks.scope_problems() if not allowed_exception(m)] == []


def test_optuna_tolko_v_s5_c_i_d():
    root = common.ROOT / "models_v2/s5"
    bad = []
    for p in optuna_users():
        in_s5 = root in p.parents and p.parent.name in OPTUNA_OK_DIRS and p.parent.parent == root
        in_tests = p.parent.name == "tests" and p.name.startswith(OPTUNA_OK_TESTS)
        if not (in_s5 or in_tests):
            bad.append(str(p.relative_to(common.ROOT)))
    assert bad == []


def test_optuna_v_s5_c_naydena():
    assert any(p.parent.name == "c" for p in optuna_users())


def test_mut_optuna_vne_s5_c_lovitsya():
    assert not allowed_exception("test_v2_deep_a_x.py: импорт optuna")
    assert not allowed_exception("test_v2_deep_b2_feat.py: импорт optuna")
    assert allowed_exception("test_v2_deep_c_tpe.py: импорт optuna")
    assert not allowed_exception("test_v2_deep_b2_feat.py: импорт tensorflow")


def test_razmery_faylov_i_funkciy():
    assert checks.size_problems() == []


def test_mut_proverka_lovit_chuzhoy_import(tmp_path):
    src = "import tensorflow\nfrom models import old\n"
    mods = [n.module.split(".")[0] if hasattr(n, "module") else n.names[0].name for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert [m for m in mods if m not in checks.ALLOWED] == ["tensorflow", "models"]
