"""W5-B, отчёт `W5_REPORT.md` (W5_SPEC 11.5–11.6; B.G4; mutations.md № 125, 140, 158): числа отчёта = таблицам (подменённое число ловится), первая строка содержит три пометки 11.5 дословно, статус W4 — формулировкой
шапки спецификации, «подтверждено» только при p Холма < 0,05, слово «доказательство» рядом с У1–У5 допустимо только в отрицании, ни слова об «ожидаемом» проценте, порядок разделов — по 11.6; `--check` только читает.
Отчёт строится боевым кодом во временной папке; реальные файлы только читаются (autouse-хэши)."""
from __future__ import annotations

import hashlib
import re
import shutil

import pandas as pd
import pytest

from models_v2.s16 import x1_w5_doc as D
from models_v2.s16 import x1_w5_eval as E
from models_v2.s16 import x1_w5_gap as G

FIRST = ("**состояние gap выбрано из 24 после анализа истории окна M; способ оценки β выбран после просмотра 24.02.2022 (Д-2); "
         "исследовательский результат на раскрытой истории, не новое независимое подтверждение (Ж-250 п. 6)**")
W4 = ("Предварительный вариант S1 улучшил QLIKE относительно M на 1,65 % на всём окне. Он не прошёл зарегистрированный порог ухудшения 2018–2021: +0,89 % при пороге +0,50 %. "
      "Ступень 1 не пройдена, полный гибрид W4 не рассчитывался. Порог W4 не меняется, W4 не перезапускается.")
REAL = [E.REP / n for n in E.OUTS] + [E.REP / "W5_REPORT.md", E.CKPT, G.REG.TRIALS, G.GAP_PIT]


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        s = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}
        s["W5"] = sorted((str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in G.OUT.rglob("*") if p.is_file()) if G.OUT.exists() else []
        return s
    before = st()
    yield
    assert st() == before


@pytest.fixture(scope="module")
def rep(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w5d")
    G.write(tmp / "W5", tmp / "W5_GAP_PIT.csv")
    shutil.copy(E.CKPT, tmp / "CHECKPOINT.json")
    E.write(tmp, tmp / "CHECKPOINT.json", tmp / "W5", tmp / "W5_GAP_PIT.csv")
    (tmp / "W5_REPORT.md").write_text(D.build(tmp), encoding="utf-8")
    return tmp


def text(rep) -> str:
    return (rep / "W5_REPORT.md").read_text(encoding="utf-8")


def tamper(rep, name, old, new):
    p = rep / name
    keep = p.read_text(encoding="utf-8")
    assert old in keep, old
    p.write_text(keep.replace(old, new, 1), encoding="utf-8")
    return keep


def test_report_equals_tables_first_line_marks_w4_status_and_order(rep) -> None:
    t = text(rep)
    assert D.problems(t, rep) == [] and t.splitlines()[0] == FIRST and ("**Статус W4.** " + W4) in t
    heads = [ln for ln in t.splitlines() if ln.startswith("## ")]
    keys = ["## 1. Общая таблица", "## 2. Δ W5-G", "## 3. Годы", "## 4. Дни gap = 1", "## 5. Защитные условия", "## 6. Чувствительность", "## 7. MCS"]
    assert [next(i for i, h in enumerate(heads) if h.startswith(k)) for k in keys] == sorted(next(i for i, h in enumerate(heads) if h.startswith(k)) for k in keys)    # порядок 11.6
    assert "n = 2198" in heads[0] and "2018-01-04" in heads[0] and "2026-11-10" in heads[0]
    assert "N = 798" in t and "N_wide = 3301" in t and "У1" in t and "0.415" in t and "0.26" in t


def test_substituted_number_and_marks_and_w4_status_mutations_are_caught(rep) -> None:
    t, tbl = text(rep), pd.read_csv(rep / "W5_TABLE.csv", comment="#")
    q = D.f6(float(tbl[tbl["series"] == "X1"]["qlike"].iloc[0]))
    assert t.count(q) >= 1
    assert any("≠ перерисовке" in b for b in D.problems(t.replace(q, q[:-1] + ("0" if q[-1] != "0" else "1"), 1), rep))          # число в тексте ≠ таблице
    line = next(ln for ln in (rep / "W5_TABLE.csv").read_text(encoding="utf-8").splitlines() if ln.startswith("main,X1,"))
    keep = tamper(rep, "W5_TABLE.csv", line, line.rsplit(",", 1)[0] + ",0.123456")                          # QLIKE X1 в таблице подменён
    try:
        assert any("≠ перерисовке" in b for b in D.problems(t, rep))                  # подмена числа в таблице при прежнем отчёте
    finally:
        (rep / "W5_TABLE.csv").write_text(keep, encoding="utf-8")
    one = t.splitlines()
    one[0] = one[0].replace("исследовательский результат на раскрытой истории, не новое независимое подтверждение (Ж-250 п. 6)", "результат")
    assert any("первая строка" in b for b in D.problems("\n".join(one), rep))        # нет одной из трёх пометок
    assert any("статус W4" in b for b in D.problems(t.replace("+0,89 % при пороге +0,50 %", "+0,29 % при пороге +0,50 %"), rep))      # W4 не формулировкой шапки
    assert D.problems(t, rep) == []


def test_confirmed_only_when_holm_below_005_mutation_in_table_is_caught(rep) -> None:
    s = pd.read_csv(rep / "W5_STATS.csv", comment="#")
    m = s[s["kind"] == "main"]
    assert (m["p_holm_N"] >= 0.05).all()                                              # предусловие: ни одна основная строка не значима (иначе тест пуст — менять данные теста)
    old = m.iloc[0]["verdict"]
    keep = tamper(rep, "W5_STATS.csv", old, "численно лучше базы; подтверждено (p Холма < 0,05)")           # мутация № 158: «подтверждено» при p Холма ≥ 0,05
    try:
        assert any("подтверждено" in b for b in D.problems(D.build(rep), rep))
    finally:
        (rep / "W5_STATS.csv").write_text(keep, encoding="utf-8")


def test_confirmed_wrongly_on_one_row_while_another_row_is_below_005_is_caught_per_row(rep) -> None:
    s = pd.read_csv(rep / "W5_STATS.csv", comment="#")
    m = s[s["kind"] == "main"]
    assert m.iloc[2]["base"] == "K3" and m.iloc[0]["base"] == "X1"
    old_x1 = m.iloc[0]["verdict"]
    k3 = next(ln for ln in (rep / "W5_STATS.csv").read_text(encoding="utf-8").splitlines() if ln.startswith("main,W5-G,K3,"))
    assert ",1.0,1.0," in k3
    keep = tamper(rep, "W5_STATS.csv", k3, k3.replace(",1.0,1.0,", ",0.001,0.001,", 1))                  # строка K3: законно p Холма < 0,05
    try:
        (rep / "W5_STATS.csv").write_text((rep / "W5_STATS.csv").read_text(encoding="utf-8").replace(old_x1, "численно лучше базы; подтверждено (p Холма < 0,05)", 1), encoding="utf-8")   # «подтверждено» — в первой строке (X1, p ≥ 0,05)
        assert any("подтверждено" in b for b in D.problems(D.build(rep), rep))        # по строке, а не по таблице
    finally:
        (rep / "W5_STATS.csv").write_text(keep, encoding="utf-8")


def proof_claims(t: str) -> list:
    """«Доказательство» в разделе У1–У5 допустимо только в отрицании «не является статистическим доказательством»."""
    sec = t.split("## 5.")[1].split("## 6.")[0]
    return [m.group(0) for m in re.finditer(r"[^.]*доказательств[^.]*\.", sec) if "не является статистическим доказательством" not in m.group(0)]


def test_word_proof_near_u1_u5_only_in_negation_and_mutation_is_caught(rep, monkeypatch) -> None:
    t = text(rep)
    assert proof_claims(t) == [] and "не является статистическим доказательством" in t and not re.search(r"ожидаем", t, re.I)
    orig = D.sec_guard
    monkeypatch.setattr(D, "sec_guard", lambda g: orig(g) + ["Пройденные пороги У1–У5 — доказательство улучшения.", ""])        # мутация: прохождение порогов названо доказательством
    mut = D.build(rep)
    assert proof_claims(mut)                                                          # детектор тестировщика видит подложенную фразу; видит ли её `problems` — тест ниже


def test_problems_flags_proof_claim_added_in_generator(rep, monkeypatch) -> None:
    orig = D.sec_guard
    monkeypatch.setattr(D, "sec_guard", lambda g: orig(g) + ["Пройденные пороги У1–У5 — доказательство улучшения.", ""])
    assert D.problems(D.build(rep), rep)


def test_check_only_reads_and_real_report_equals_real_tables() -> None:
    p = D.REP / "W5_REPORT.md"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    assert D.problems(p.read_text(encoding="utf-8")) == [] and hashlib.sha256(p.read_bytes()).hexdigest() == h
    t = p.read_text(encoding="utf-8")
    assert t.splitlines()[0] == FIRST and ("**Статус W4.** " + W4) in t and proof_claims(t) == []
