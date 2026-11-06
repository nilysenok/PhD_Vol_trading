"""A1 (Ж-206, волна A): состав INVENTORY.csv. Мутации: убран обязательный id; дубль id; убрана колонка; убраны лучшие 3 строки этапа 2; нет префикса fx_."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2.s6 import checks, paths6  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return paths6.read_csv(paths6.REP / "INVENTORY.csv")


def test_real_inventory_complete(df):
    assert checks.inventory_problems(df) == []
    assert df["id"].is_unique and len(df) >= 200
    assert (df["status"] == "DEVELOPMENT / HISTORICAL WALK-FORWARD").all()


@pytest.mark.parametrize("rid", checks.REQUIRED_ID)
def test_mutation_missing_required_id_caught(df, rid):
    assert checks.inventory_problems(df[df["id"] != rid]) == ["нет id " + rid]


def test_mutation_duplicate_id_caught(df):
    assert any("дубль" in p for p in checks.inventory_problems(pd.concat([df, df.iloc[[5]]], ignore_index=True)))


def test_mutation_missing_column_caught(df):
    assert checks.inventory_problems(df.drop(columns=["code_sha256"])) == ["нет колонки code_sha256"]


@pytest.mark.parametrize("lib", ["xgb", "lgbm"])
def test_mutation_stage2_top3_dropped_caught(df, lib):
    bad = checks.inventory_problems(df[~df["family"].astype(str).str.contains("этап2_top3_" + lib)])
    assert any("лучших 3 строки этапа 2 для " + lib in p for p in bad)


def test_mutation_no_fx_rows_caught(df):
    assert "нет строк с префиксом fx_" in checks.inventory_problems(df[~df["id"].astype(str).str.startswith("fx_")])


def test_audit_detects_long_function_and_file(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("def g():\n" + "    a = 1\n" * 45 + "\n" * 160, encoding="utf-8")
    big, longf = checks.audit_files([f])
    assert big == ["x.py"] and longf == ["x.py::g"]
