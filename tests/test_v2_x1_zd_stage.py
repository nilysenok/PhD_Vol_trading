"""ZD, стадия и изоляция: модули ZD не пишут прогнозов, CHECKPOINT.json, ALL_TRIALS.csv (AST-проверка + мутация-детектор); `--check` только читает (мутация: проверка пишет в чужой выход — ловится хэшем); подмена числа в таблице/отчёте/манифесте ловится.
Запись и проверка — во ВРЕМЕННЫЙ каталог (пути стадии подменяются); реальные файлы не трогаются."""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_zd_stage as S

MODS = sorted(Path(S.__file__).parent.glob("x1_zd_*.py"))
FORBIDDEN = ("CHECKPOINT", "ALL_TRIALS", "to_parquet", "x1_reg")


def violations(src: str) -> list:
    """Строки-константы с запрещёнными именами и вызовы записи parquet в исходнике модуля."""
    tree, out = ast.parse(src), []
    docs = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr)}          # докстроки и голые строки — не код
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and any(k in n.value for k in FORBIDDEN[:2]):
            out.append(n.value[:40])
        if isinstance(n, ast.Attribute) and n.attr == "to_parquet":
            out.append("to_parquet")
        if isinstance(n, (ast.Import, ast.ImportFrom)) and "x1_reg" in ast.unparse(n):
            out.append("x1_reg")
    return out


def test_zd_modules_do_not_write_forecasts_checkpoint_or_registry() -> None:
    assert len(MODS) == 5
    for m in MODS:
        assert not violations(m.read_text(encoding="utf-8")), m.name


def test_detector_catches_write_to_checkpoint_registry_and_forecast() -> None:
    assert violations('P.write_text("x", encoding="u"); (REP / "CHECKPOINT.json").write_text("{}")')
    assert violations('df.to_csv(REP / "ALL_TRIALS.csv")')
    assert violations("f.to_parquet('y9.parquet')") and violations("from models_v2.s16 import x1_reg as R")


@pytest.fixture(scope="module")
def honest(tmp_path_factory):
    root = tmp_path_factory.mktemp("zd")
    mp = pytest.MonkeyPatch()
    mp.setattr(S, "REP", root), mp.setattr(S, "REPORT", root / "ZD_REPORT.md"), mp.setattr(S, "MANIFEST", root / "ZD_MANIFEST.json")
    mp.setattr(S, "WATCH", [root / "foreign"])
    (root / "foreign").mkdir()
    (root / "foreign/a.txt").write_text("чужой выход", encoding="utf-8")
    cache = S.compute()
    mp.setattr(S, "compute", lambda inp=None: cache)
    S.write()
    mp.undo()
    return root, cache


def env(tmp_path, honest, mp) -> Path:
    root = tmp_path / "w"
    shutil.copytree(honest[0], root)
    mp.setattr(S, "REP", root), mp.setattr(S, "REPORT", root / "ZD_REPORT.md"), mp.setattr(S, "MANIFEST", root / "ZD_MANIFEST.json"), mp.setattr(S, "WATCH", [root / "foreign"])
    mp.setattr(S, "compute", lambda inp=None: honest[1])
    return root


def test_clean_check_and_no_checkpoint_or_registry_written(honest, monkeypatch, tmp_path) -> None:
    root = env(tmp_path, honest, monkeypatch)
    assert S.check() == []
    names = {p.name for p in honest[0].iterdir()}
    assert names == {"ZD_STATES.csv", "ZD_PRED.csv", "ZD_CALIB.csv", "ZD_CAUSES.csv", "ZD_REPORT.md", "ZD_MANIFEST.json", "foreign"}
    assert not (root / "CHECKPOINT.json").exists() and not (root / "ALL_TRIALS.csv").exists()


def test_check_that_writes_is_caught(honest, monkeypatch, tmp_path) -> None:
    root = env(tmp_path, honest, monkeypatch)

    def writing(inp=None):
        (root / "foreign/a.txt").write_text("перезаписано проверкой", encoding="utf-8")
        return honest[1]
    monkeypatch.setattr(S, "compute", writing)
    assert any("изменил чужие выходы" in b for b in S.check())


def test_substituted_number_in_report_table_and_manifest_is_caught(honest, monkeypatch, tmp_path) -> None:
    root = env(tmp_path, honest, monkeypatch)
    rep = (root / "ZD_REPORT.md").read_text(encoding="utf-8")
    (root / "ZD_REPORT.md").write_text(rep.replace("| all | 1894 |", "| all | 1895 |", 1), encoding="utf-8")
    assert any("ZD_REPORT.md" in b for b in S.check())
    shutil.copy(honest[0] / "ZD_REPORT.md", root / "ZD_REPORT.md")
    tab = (root / "ZD_PRED.csv").read_text(encoding="utf-8").splitlines()
    tab[2] = tab[2].replace("Y1", "Y9", 1) if "Y1" in tab[2] else tab[2] + "0"
    (root / "ZD_PRED.csv").write_text("\n".join(tab) + "\n", encoding="utf-8")
    assert any("ZD_PRED.csv" in b for b in S.check())
    shutil.copy(honest[0] / "ZD_PRED.csv", root / "ZD_PRED.csv")
    (root / "ZD_MANIFEST.json").write_text((root / "ZD_MANIFEST.json").read_text(encoding="utf-8").replace('"configs_spent": 0', '"configs_spent": 1'), encoding="utf-8")
    assert any("MANIFEST" in b for b in S.check())


def test_report_first_line_zero_configs_and_identity() -> None:
    real = Path(S.__file__).parents[2] / "docs/otchety/harj_x1"
    assert (real / "ZD_REPORT.md").read_text(encoding="utf-8").splitlines()[0] == "диагностика, не отбор; историческая; без конфигураций"
    assert set(pd.read_csv(real / "ZD_PRED.csv", skiprows=1)["y"]) == {"Y1", "Y2", "Y3"}
    assert S.json.loads((real / "ZD_MANIFEST.json").read_text(encoding="utf-8"))["configs_spent"] == 0
