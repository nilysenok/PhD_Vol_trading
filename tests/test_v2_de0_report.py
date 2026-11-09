"""DE0, G1 и G5: реестр = таблицы программ; отчёт — генератор из таблиц (подмена числа ловится); `--check` ТОЛЬКО ЧИТАЕТ (запись в `--check` ловится); модули DE0 не импортируют обучатели."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s15 import de0_doc as DOC, de0_io as IO, de0_registry as R, de0_stage as S

TABLES, NUM = S.compute()
REG = TABLES["DE0_REGISTRY"]


def test_registry_covers_v1_v7_and_equals_program_tables():
    assert set(REG["program"]) >= {"V1", "V2", "V3", "V4", "V5", "V6", "V7", "контроль"}
    assert REG["verdict"].notna().all() and REG["direction"].notna().all()
    fin = pd.read_csv(IO.V7 / "FIN_ALL.csv", comment="#").set_index(["series", "window"])["qlike"]
    for _, r in REG[REG["program"] == "V7"].iterrows():
        assert abs(r["q_2018_2021"] - fin[(r["id"], "2018_2021")]) < 1e-12 and abs(r["q_all_2018_2026"] - fin[(r["id"], "all")]) < 1e-12
    fv6 = pd.read_csv(IO.V6 / "FIN_ALL.csv", comment="#").set_index(["series", "window"])["qlike"]
    for _, r in REG[(REG["program"] == "V6") & REG["id"].isin(R.V6_CFG)].iterrows():
        assert abs(r["q_2022_2025"] - fv6[(r["id"], "2022_2025")]) < 1e-12 and abs(r["q_all_table"] - fv6[(r["id"], "all")]) < 1e-12


def test_registry_checks_pass_and_substituted_number_is_caught():
    d = S.DATA.load()
    mx, rec = S._fin_all_vs_forecast(d)
    ck = R.checks(REG, {k: v for k, v in rec.items() if k.startswith("cfg")})
    assert mx < 1e-12 and ck["max_table_vs_forecast"] < 1e-12 and ck["max_window_vs_years"] < 1e-9
    bad = REG.copy()
    bad.loc[bad["id"] == "cfg8", "q_2022_2025"] += 1e-6                              # число реестра ≠ таблице программы
    assert R.checks(bad, {k: v for k, v in rec.items() if k.startswith("cfg")})["max_table_vs_forecast"] > 1e-7


def test_registry_window_vs_years_dropped_year_is_caught():
    yr = pd.DataFrame(dict(series="x", year=list(range(2017, 2027)), n=[250] * 10, qlike=np.linspace(.1, .4, 10)))
    full = R.pooled(yr, 2022, 2025)
    assert abs(full - yr[yr["year"].between(2022, 2025)]["qlike"].mean()) < 1e-15
    assert abs(R.pooled(yr[yr["year"] != 2024], 2022, 2025) - full) > 1e-3


def test_report_first_line_status_and_numbers_come_from_tables():
    txt = DOC.build(TABLES)
    first = txt.splitlines()[0]
    assert "диагностика, не отбор" in first and "adv" in first and "%" in first
    assert "Статус: диагностика, не отбор" in txt and "## Что не проверено" in txt
    c = TABLES["DE0_CAUSES"].iloc[0]
    assert "%+.4f" % c["d_adv"] in first and "%.0f %%" % (100 * c["share_base"]) in first


def test_mutation_substituted_number_in_table_changes_report():
    t2 = {k: v.copy() for k, v in TABLES.items()}
    t2["DE0_GAP_BASE"].loc[0, "q_hybrid"] += 0.001
    t2["DE0_CAUSES"].loc[2, "share_base"] += 0.2
    assert DOC.build(t2) != DOC.build(TABLES)


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(IO, "REP", tmp_path)
    monkeypatch.setattr(IO, "DOC", tmp_path / "DE0_REPORT.md")
    monkeypatch.setattr(IO, "CP_FILE", tmp_path / "CHECKPOINT.json")
    monkeypatch.setattr(S, "compute", lambda: ({k: v.copy() for k, v in TABLES.items()}, dict(NUM)))
    S.write()
    return tmp_path


def test_check_is_clean_after_write_and_substituted_number_in_report_is_caught(sandbox):
    assert S.check() == []
    p = sandbox / "DE0_REPORT.md"
    txt = p.read_text(encoding="utf-8")
    p.write_text(re.sub(r"(\d)\.(\d)", lambda m: m.group(1) + "." + str((int(m.group(2)) + 1) % 10), txt, count=1), encoding="utf-8")
    assert any("DE0_REPORT.md" in b for b in S.check())


def test_substituted_number_in_csv_is_caught(sandbox):
    p = sandbox / "DE0_GAP_BASE.csv"
    p.write_text(p.read_text(encoding="utf-8").replace("2018_2021", "2018_2022", 1), encoding="utf-8")
    assert any("DE0_GAP_BASE.csv" in b for b in S.check())


def test_check_writes_nothing_and_write_in_check_is_caught(sandbox, monkeypatch):
    h = IO.outputs_hash()
    assert S.check() == [] and IO.outputs_hash() == h

    def writing_compute():
        (IO.REP / "DE0_SERVICE.csv").write_text("служебный ключ", encoding="utf-8")   # мутация: проверка пишет файл
        return {k: v.copy() for k, v in TABLES.items()}, dict(NUM)
    monkeypatch.setattr(S, "compute", writing_compute)
    assert any("изменил выходы" in b for b in S.check())


def test_write_keeps_foreign_checkpoint_keys(sandbox):
    IO.merge_checkpoint("1", dict(x=1))
    S.write()
    assert IO.read_cp()["1"] == dict(x=1) and "0" in IO.read_cp()


FORBIDDEN = {"lightgbm", "xgboost", "sklearn", "catboost"}
S14_OK = {"v7_1m_data", "v7_4_evrun"}                                              # загрузчики сохранённых прогнозов, не обучатели


def imports_of(src: str) -> list:
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out += [(n.module or "") + "." + a.name for a in n.names] + [n.module or ""]
    return out


def violations(src: str, fname: str) -> list:
    bad = []
    for m in imports_of(src):
        root = m.split(".")[0]
        if root in FORBIDDEN or m.startswith("torch.") or m.startswith("models_v2.s13.v6_") and "fit" in m:
            bad.append(m)
        if m.startswith("models_v2.s14.") and m.split(".")[2] not in S14_OK:
            bad.append(m)
        if m == "torch" and fname != "__init__.py":
            bad.append(m)                                                           # torch — только порядок импорта в `__init__`
    return bad


def test_de0_modules_import_no_trainers():
    files = sorted(Path(IO.ROOT / "models_v2/s15").glob("*.py"))
    assert len(files) >= 8
    for f in files:
        assert violations(f.read_text(encoding="utf-8"), f.name) == [], f.name


def test_mutation_trainer_import_is_detected():
    for src in ("import lightgbm as lgb", "from xgboost import XGBRegressor", "import torch.optim", "from models_v2.s14 import v7_3_run", "import torch"):
        assert violations(src, "de0_x.py") != [], src
