"""W8-A, изоляция и «только читает» (№ 43, 123, 164): `check`/`require` регистратора и `check`/`verdict`/`render` диагностики ничего не пишут (AST + хэши); реальные ALL_TRIALS.csv, CHECKPOINT.json, `W8_*` в
`docs/otchety/harj_x1` и каталог `data/regen/harj_x1/W8` тестами не меняются (сравнение содержимого до и после, без ожидания пустоты); в тестах W3…W8 нет присваиваний атрибутам модулей (только
monkeypatch); реестр W8 на реальном файле — всё или ничего, по id. Реальные файлы только читаются."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from models_v2.s16 import x1_w8_diag as D
from models_v2.s16 import x1_w8_diag_v as V
from models_v2.s16 import x1_w8_reg as W

ROOT = Path(W.__file__).resolve().parents[2]
TESTS = sorted(ROOT.glob("tests/test_v2_x1_w[3-8]_*.py"))


def real_state() -> dict:
    s = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (W.TRIALS, W.CKPT) if p.exists()}
    s.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(W.REP.glob("W8_*")) if p.is_file()})
    s["W8"] = {str(p.relative_to(W.W8)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(W.W8.rglob("*")) if p.is_file()} if W.W8.exists() else {}
    return s


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = real_state()
    yield
    assert real_state() == before                                                         # реальные реестр, CHECKPOINT, W8_* и каталог W8 прежние


def attr_assignments(src: str, mods: set) -> list:
    out = []
    for n in ast.walk(ast.parse(src)):
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else []):
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in mods:
                out.append(ast.unparse(n))
    return out


def test_no_module_attribute_assignment_in_w3_to_w8_tests_and_scanner_sees_one() -> None:
    assert len(TESTS) >= 12 and any(p.name.startswith("test_v2_x1_w8_") for p in TESTS)
    for p in TESTS:
        src = p.read_text(encoding="utf-8")
        mods = {n.asname or n.name for t in ast.walk(ast.parse(src)) if isinstance(t, ast.ImportFrom) and (t.module or "").startswith("models_v2") for n in t.names}
        assert attr_assignments(src, mods) == [], p.name
    assert attr_assignments("from models_v2.s16 import x1_w8_reg as W\nW.register = lambda: 0\n", {"W"})      # сканер видит подложенное присваивание


def test_swapped_functions_are_originals_after_other_tests() -> None:
    assert all(getattr(W, f).__code__.co_name == f for f in ("register", "require", "check", "merge_checkpoint", "_sha"))       # подмены monkeypatch не остались в модуле
    assert all(getattr(D, f).__code__.co_name == f for f in ("render", "check", "block_a", "block_b", "block_f", "block_g", "cell"))
    assert V.verdict.__code__.co_name == "verdict" and D.K.terc_past.__code__.co_name == "terc_past" and D.FE.imoex_day.__code__.co_name == "imoex_day"


def seed(t: Path) -> Path:
    t.write_text("id,name,formula,params,status,registered_at\nY1,a,b,c,зарегистрирована,10.11.2026 04:21:03\n", encoding="utf-8")
    return t


def test_check_only_reads(tmp_path) -> None:
    t, w8 = seed(tmp_path / "ALL_TRIALS.csv"), tmp_path / "W8"
    W.register("11.11.2026 15:00:00", trials=t, w8=w8, ckpt=None)
    snap = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    assert W.check(trials=t, w8=w8) == [] and W.check(trials=t, w8=w8, after=True, ckpt=tmp_path / "none.json")
    assert snap == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()} and not w8.exists()


def test_check_that_writes_is_caught_by_hash(tmp_path, monkeypatch) -> None:
    t, w8 = seed(tmp_path / "ALL_TRIALS.csv"), tmp_path / "W8"
    W.register("11.11.2026 15:00:00", trials=t, w8=w8, ckpt=None)
    before = hashlib.sha256(t.read_bytes()).hexdigest()
    orig = W.W5R._parquet
    monkeypatch.setattr(W.W5R, "_parquet", lambda d: (t.write_text(t.read_text(encoding="utf-8") + "# изменено проверкой\n", encoding="utf-8"), orig(d))[1])      # мутация: check пишет
    W.check(trials=t, w8=w8)
    assert hashlib.sha256(t.read_bytes()).hexdigest() != before                           # внешний хэш выхода это видит


def write_calls(src: str) -> list:
    return [n.func.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("write_text", "write_bytes", "open", "to_csv", "dump", "to_parquet")]


def func_src(src: str, name: str) -> str:
    return next(ast.unparse(f) for f in ast.parse(src).body if isinstance(f, ast.FunctionDef) and f.name == name)


def test_readers_have_no_writes_and_scanner_sees_mutation() -> None:
    src = Path(W.__file__).read_text(encoding="utf-8")
    for f in ("check", "require", "_sha"):
        assert write_calls(func_src(src, f)) == [], f
    assert write_calls(func_src(src, "register")) and write_calls(func_src(src, "merge_checkpoint"))        # писатели — только эти две
    assert write_calls(func_src(src.replace("    cur = W5R._read(trials)\n    rows", "    trials.write_text('x')\n    rows", 1), "check"))
    d = Path(D.__file__).read_text(encoding="utf-8")
    for f in ("check", "cell", "block_a", "block_b", "block_f", "block_g"):
        assert [c for c in write_calls(func_src(d, f)) if c != "to_csv"] == [], f
    assert "write_text" in write_calls(func_src(d, "main")) and "write_text" not in write_calls(func_src(d, "check"))      # писатель диагностики — main с флагом --write
    for name in ("x1_w8_diag_v", "x1_w8_diag_core", "x1_w8_diag_feat", "x1_w8_diag_cde", "x1_w8_diag_null"):
        assert write_calls((Path(W.__file__).parent / (name + ".py")).read_text(encoding="utf-8")) == [], name
    assert write_calls(func_src(d.replace("    before = out.read_bytes()\n", "    out.write_text('x')\n    before = out.read_bytes()\n", 1), "check"))      # сканер видит запись в check диагностики


# ---------- реальный реестр: всё или ничего, по id ----------

def test_real_registry_is_all_or_nothing_and_by_id() -> None:
    cur = W.W5R._read(W.TRIALS)
    mine = cur[cur["id"].isin(W.IDS)]
    assert len(mine) in (0, 12)                                                           # частичной регистрации нет
    assert not cur[~cur["id"].isin(W.IDS)]["id"].str.match(r"^(W8D[0-9]+|W8-)").any()     # чужих W8-строк (вне разд. 8) нет
    if len(mine) == 12:
        assert mine["registered_at"].nunique() == 1
        pq = W.W5R._parquet(W.W8)
        if pq:                                                                            # режим «после прогноза»: check() законно отказывает
            assert any("parquet" in b for b in W.check())
            assert W.check(after=True) == []
        else:
            assert W.check() == []
        assert len(cur[cur["id"].str.match(r"^(W8D[0-9]+|W8-)")]) == 12                  # A.G3
        r = json.loads(W.CKPT.read_text(encoding="utf-8"))["W8.registered"]
        assert (r["n"], r["n_wide"], r["rows"], r["registered_at"]) == (814, 3333, 12, mine["registered_at"].iloc[0])
        assert r["spec_sha256"] == W._sha(W.REP / "W8_SPEC.md") and r["decomp_sha256"] == W._sha(W.REP / "W8_DECOMP.csv")        # вход зафиксирован при регистрации
    else:
        assert W.check() and not W.W5R._parquet(W.W8)                                     # не зарегистрировано, а прогнозов нет — check отказывает по строкам
        pd.testing.assert_frame_equal(mine, mine.iloc[0:0])
