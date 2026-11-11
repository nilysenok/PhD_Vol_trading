"""W1.7: отчёт — из таблиц. Подменённое число в отчёте или таблице ловится `--check`; число, вписанное в код генератора, ловится (AST); слова «лучше/хуже», «подтверждено», «превосходит» — по правилам. Реальные таблицы только читаются."""
from __future__ import annotations

import ast
import re

import pandas as pd
import pytest

from models_v2.s16 import x1_w1 as W, x1_w1_doc as DOC

ALLOWED_DECIMALS = {"0,05"}


def typed_numbers(src: str) -> list:
    """Числа, вписанные в код генератора: вещественные константы и десятичные числа в строковых литералах (кроме объявленного порога 0,05; форматные спецификации «%.2f» не считаются)."""
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Constant) and isinstance(n.value, float):
            out.append(n.value)
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            out += [m for m in re.findall(r"(?<![%\w.])\d+[.,]\d+", n.value) if m not in ALLOWED_DECIMALS]
    return out


@pytest.fixture(scope="module")
def real():
    return W.compute()


def test_report_equals_generator_and_disk(real) -> None:
    assert (W.REP / "W1_REPORT.md").read_text(encoding="utf-8") == real["md"] == DOC.build(real)
    assert real["md"].startswith("# W1_REPORT") and "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; диагностика после просмотра; Ж-250" in real["md"]
    assert W.files(real)["W1_REPORT.md"] == real["md"]


def test_substituted_number_in_report_or_table_is_caught_by_check(real, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(W, "compute", lambda: real)
    ck = tmp_path / "CHECKPOINT.json"
    ck.write_text('{\n "A": 1\n}', encoding="utf-8")
    W.write(tmp_path, ck)
    assert W.check(tmp_path) == []
    q = "%.6f" % float(real["table"]["qlike"].iloc[0])
    p = tmp_path / "W1_REPORT.md"
    txt = p.read_text(encoding="utf-8")
    assert q in txt
    p.write_text(txt.replace(q, "%.6f" % (float(q) + 1e-4), 1), encoding="utf-8")
    assert W.check(tmp_path) == ["W1_REPORT.md: текст ≠ пересчёту"]
    p.write_text(txt, encoding="utf-8")
    t = tmp_path / "W1_TABLE.csv"
    ct = t.read_text(encoding="utf-8")
    v = "%r" % float(real["table"]["qlike"].iloc[0])
    assert v in ct
    t.write_text(ct.replace(v, "%r" % (float(v) + 1e-6), 1), encoding="utf-8")
    assert W.check(tmp_path) == ["W1_TABLE.csv: текст ≠ пересчёту"]


def test_generator_has_no_typed_numbers_and_mutation_is_caught() -> None:
    src = open(DOC.__file__, encoding="utf-8").read()
    assert typed_numbers(src) == []
    assert typed_numbers(src + "\nX = 0.6180\n") == [0.618]                                     # число вписано в код
    assert typed_numbers(src.replace("## 7. Ответы (из таблиц)", "## 7. Ответы: Δ −1,09 % (из таблиц)")) == ["1,09"]       # число вписано в текст


def words_ok(md: str, stats: pd.DataFrame) -> bool:
    low = md.lower()
    banned = ("превосходит", "доказано", "выигрыша нет")
    holm_ok = bool((stats["p_holm_N"] < 0.05).any())
    return holm_ok or not any(b in low for b in banned)


def answer_signs_ok(md: str, stats: pd.DataFrame) -> bool:
    for b in ("K2", "K3"):
        r = stats[(stats["window"] == "M") & (stats["hybrid"] == "X1") & (stats["base"] == b)].iloc[0]
        line = [x for x in md.splitlines() if x.startswith("- **X1 против %s, окно M.**" % b)][0]
        if ("Численно: лучше" in line) != (r.delta_pct < 0) or ("Численно: не лучше" in line) != (r.delta_pct >= 0):
            return False
        if ("Статистически: подтверждено" in line) != (r.p_holm_N < 0.05 and r.delta_pct < 0):
            return False
    return True


def test_words_and_signs_follow_tables_and_mutations_are_caught(real) -> None:
    s, md = real["stats"], real["md"]
    assert words_ok(md, s) and answer_signs_ok(md, s)
    assert not words_ok(md + "\nX1 превосходит K2.", s)                                          # «превосходит» без p Холма < 0,05
    assert not words_ok(md + "\nВыигрыша нет.", s)                                               # «выигрыша нет» при Δ < 0 и p ≥ 0,05
    flipped = md.replace("Численно: лучше", "Численно: не лучше", 1)
    assert not answer_signs_ok(flipped, s)                                                       # слово перевёрнуто
    assert not answer_signs_ok(md.replace("Статистически: не подтверждено", "Статистически: подтверждено", 1), s)   # «подтверждено» без p Холма
    x = s[(s["window"] == "M") & (s["hybrid"] == "X1") & (s["base"] == "K2")]
    assert (x["delta_pct"] < 0).all() and (x["p_holm_N"] >= 0.05).all()                          # текущие данные: численно лучше, не подтверждено — и отчёт так и пишет
