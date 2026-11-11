"""W8-B, отчёт `W8_REPORT.md` (гейт B.G4; mutations.md № 125, 140, 158): числа отчёта = таблицам (подменённое число ловится), первая строка содержит три пометки разд. 6 дословно, «подтверждено» — по СВОЕЙ строке и только при
p Холма < 0,05, решение выведено кодом из `W8_GUARD.csv` (подмена `passed` ловится), «доказательство» рядом с У1–У6 только в отрицании, ни слова об «ожидаемом» проценте, порядок разделов — разд. 9; `--check`
только читает. Отчёт строится боевым кодом во временной папке; реальные файлы только читаются (autouse-хэши)."""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from models_v2.s16 import x1_w8_doc as D
from models_v2.s16 import x1_w8_eval as E

MARKS = ("признак IMOEX RV выбран из 10 после анализа истории окна M (W8 разд. 1.5); прохождение переноса на уровне семейства неотличимо от случайного (p перестановок 0,05, сдвигов 0,15)",
         "база W5-G: состояние gap выбрано из 24, оценщик β — после просмотра 24.02.2022", "исследовательский результат на раскрытой истории, не независимое подтверждение (Ж-250 п. 6)")        # разд. 6 спецификации, независимая копия
FIRST = "**" + "; ".join(MARKS) + "**"


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        s = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (E.REG.TRIALS, E.CKPT) if p.exists()}
        s.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.REP.glob("W8_*")) if p.is_file()})
        s["W8"] = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(E.OUT.rglob("*")) if p.is_file()} if E.OUT.exists() else {}
        return s
    before = st()
    yield
    assert st() == before


@pytest.fixture(scope="module")
def rep(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w8d")
    out = tmp / "W8"
    out.mkdir()
    for p in E.OUT.glob("*.parquet"):
        shutil.copy(p, out / p.name)
    shutil.copy(E.CKPT, tmp / "CHECKPOINT.json")
    E.write(tmp, tmp / "CHECKPOINT.json", out)
    (tmp / "W8_REPORT.md").write_text(D.build(tmp), encoding="utf-8")
    return tmp


def text(rep) -> str:
    return (rep / "W8_REPORT.md").read_text(encoding="utf-8")


def tamper(rep, name, old, new):
    p = rep / name
    keep = p.read_text(encoding="utf-8")
    assert old in keep, old
    p.write_text(keep.replace(old, new, 1), encoding="utf-8")
    return keep


def test_report_equals_tables_first_line_marks_and_section_order(rep) -> None:
    t = text(rep)
    assert D.problems(t, rep) == [] and t.splitlines()[0] == FIRST and D.FIRST == FIRST          # пометки разд. 6 — в первой строке дословно
    heads = [ln for ln in t.splitlines() if ln.startswith("## ")]
    keys = ["## 1. Общая таблица", "## 2. Δ W8-G", "## 3. Годы", "## 4. Основной критерий", "## 5. Чувствительность"]          # порядок отчёта разд. 9
    pos = [next(i for i, h in enumerate(heads) if h.startswith(k)) for k in keys]
    assert pos == sorted(pos) and len(heads) == 5 and "n = 2198" in heads[0] and "2018-01-04" in heads[0] and "2026-11-10" in heads[0]
    assert "N = 814" in t and "N_wide = 3333" in t and all(u in t for u in ("У1", "У2", "У3", "У4", "У5", "У6", "0.415", "0.5", "0.26", "0.1"))
    assert t.index("| W8-G |") > t.index("| K1 |") and t.index("| W8-Gb |") > t.index("| W8-G |")        # порядок строк таблицы 1: K1, K2, K3, X1, W5-G, W8-G; справочно W8-Gb


def test_substituted_number_marks_and_table_mutations_are_caught(rep) -> None:
    t, tbl = text(rep), pd.read_csv(rep / "W8_TABLE.csv", comment="#")
    q = "%.6f" % float(tbl[tbl["series"] == "X1"]["qlike"].iloc[0])
    assert t.count(q) >= 1
    assert any("≠ перерисовке" in b for b in D.problems(t.replace(q, q[:-1] + ("0" if q[-1] != "0" else "1"), 1), rep))          # число в тексте ≠ таблице
    line = next(ln for ln in (rep / "W8_TABLE.csv").read_text(encoding="utf-8").splitlines() if ln.startswith("main,X1,"))
    keep = tamper(rep, "W8_TABLE.csv", line, line.rsplit(",", 1)[0] + ",0.123456")                          # QLIKE X1 в таблице подменён
    try:
        assert any("≠ перерисовке" in b for b in D.problems(t, rep))                  # подмена числа в таблице при прежнем отчёте
    finally:
        (rep / "W8_TABLE.csv").write_text(keep, encoding="utf-8")
    for i, mark in enumerate(MARKS):                                                  # нет любой из трёх пометок в первой строке
        one = t.splitlines()
        one[0] = one[0].replace(mark, "результат")
        assert any("первая строка" in b for b in D.problems("\n".join(one), rep)), i
    assert any("первая строка" in b for b in D.problems("\n" + t, rep)) and D.problems(t, rep) == []        # пометки не в первой строке


def test_confirmed_only_for_own_row_with_holm_below_005(rep) -> None:
    s = pd.read_csv(rep / "W8_STATS.csv", comment="#")
    m = s[s["kind"] == "main"]
    assert (m["p_holm_N"] >= 0.05).all()                                              # предусловие: ни одна основная строка не значима (иначе тест пуст)
    assert not D.CONFIRMED.search(text(rep).replace("не подтверждено (p Холма", "")) and D._confirmed(text(rep), s) == []
    old = m.iloc[0]["verdict"]
    keep = tamper(rep, "W8_STATS.csv", old, "численно лучше базы; подтверждено (p Холма < 0,05)")           # мутация № 158: «подтверждено» при p Холма ≥ 0,05
    try:
        assert any("подтверждено" in b for b in D.problems(D.build(rep), rep))
    finally:
        (rep / "W8_STATS.csv").write_text(keep, encoding="utf-8")


def test_confirmed_wrongly_on_one_row_while_another_row_is_below_005_is_caught_per_row(rep) -> None:
    f = rep / "W8_STATS.csv"
    s = pd.read_csv(f, comment="#")
    m = s[s["kind"] == "main"]
    assert m.iloc[3]["base"] == "K3" and m.iloc[0]["base"] == "W5-G"
    k3 = next(ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.startswith("main,W8-G,K3,"))
    p = k3.split(",")
    keep = f.read_text(encoding="utf-8")
    new = k3.replace("," + p[13] + "," + p[14] + ",", ",0.001,0.001,", 1)                                 # строка K3: законно p Холма < 0,05
    assert new != k3
    try:
        f.write_text(keep.replace(k3, new, 1).replace(m.iloc[0]["verdict"], "численно лучше базы; подтверждено (p Холма < 0,05)", 1), encoding="utf-8")    # «подтверждено» — в строке W5-G (p ≥ 0,05)
        assert any("подтверждено" in b and "W5-G" in b for b in D.problems(D.build(rep), rep))              # по своей строке, а не по таблице в целом
    finally:
        f.write_text(keep, encoding="utf-8")
    assert D.problems(text(rep), rep) == []


def test_decision_is_derived_from_guard_and_substituted_passed_is_caught(rep) -> None:
    g = pd.read_csv(rep / "W8_GUARD.csv", comment="#")
    assert g.loc[g["id"] == "решение", "value"].iloc[0].startswith("W8-G не прошёл: У6") and f"**Решение (выведено кодом из таблицы):** {g.loc[g['id'] == 'решение', 'value'].iloc[0]}." in text(rep)
    line = next(ln for ln in (rep / "W8_GUARD.csv").read_text(encoding="utf-8").splitlines() if ln.startswith("У6,"))
    keep = tamper(rep, "W8_GUARD.csv", line, line.rsplit(",", 1)[0] + ",True")       # У6 в таблице объявлен выполненным, строка «решение» прежняя
    try:
        assert any("решение ≠ выведенному из W8_GUARD.csv" in b for b in D.problems(text(rep), rep))
    finally:
        (rep / "W8_GUARD.csv").write_text(keep, encoding="utf-8")
    dec = next(ln for ln in (rep / "W8_GUARD.csv").read_text(encoding="utf-8").splitlines() if ln.startswith("решение,"))
    keep = tamper(rep, "W8_GUARD.csv", dec, dec.replace("W8-G не прошёл: У6 (+0.1500)", "третья контрольная версия: W8-G"))      # подмена самой строки «решение»
    try:
        assert any("решение ≠ выведенному" in b for b in D.problems(D.build(rep), rep))
    finally:
        (rep / "W8_GUARD.csv").write_text(keep, encoding="utf-8")


def proof_claims(t: str) -> list:
    sec = t.split("## 4.")[1].split("## 5.")[0]
    return [m.group(0) for m in re.finditer(r"[^.]*доказательств[^.]*\.", sec) if D.NEGATION not in m.group(0)]


def test_word_proof_and_expected_percent_mutations_are_caught(rep, monkeypatch) -> None:
    t = text(rep)
    assert proof_claims(t) == [] and D.NEGATION in t and not re.search(r"ожидаем", t, re.I)
    orig = D.sec_guard
    monkeypatch.setattr(D, "sec_guard", lambda g: orig(g) + ["Пройденные пороги У1–У6 — доказательство улучшения.", ""])        # мутация: прохождение порогов названо доказательством
    mut = D.build(rep)
    assert proof_claims(mut) and D.problems(mut, rep)
    monkeypatch.setattr(D, "sec_guard", lambda g: orig(g) + ["Ожидаемое улучшение — 0,3 %.", ""])
    assert any("ожидаемый" in b for b in D.problems(D.build(rep), rep))


def test_cli_check_reads_only_and_real_report_equals_real_tables() -> None:
    p = D.REP / "W8_REPORT.md"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert D.problems(p.read_text(encoding="utf-8")) == [] and hashlib.sha256(p.read_bytes()).hexdigest() == h
    r = subprocess.run([sys.executable, "-W", "ignore", "-m", "models_v2.s16.x1_w8_doc", "--check"], cwd=D.REP.parents[2], capture_output=True, text=True)       # B.G4: код возврата 0
    assert r.returncode == 0 and "W8 doc --check: ok" in r.stdout and hashlib.sha256(p.read_bytes()).hexdigest() == h
    t = p.read_text(encoding="utf-8")
    assert t.splitlines()[0] == FIRST and proof_claims(t) == []
    assert D._confirmed(t, pd.read_csv(D.REP / "W8_STATS.csv", comment="#")) == []                 # «подтверждено» — только по своей строке с p Холма < 0,05
