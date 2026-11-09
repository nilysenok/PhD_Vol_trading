"""V6-Z1 (Ж-230, G57–G58): A0g в итоговом документе — диагностический контроль, а не основная конфигурация. Метка в каждой таблице, лучшая конструкция только из 17 предрегистрированных,
подраздел «A0g: статус, причина, моменты» с временами из журнала. Реальные таблицы, журнал и документ — только чтение."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from models_v2.s13 import v6_fin_asm as A, v6_fin_data as D, v6_fin_doc as W, v6_fin_docheck as DC, v6_fin_z1 as Z1

T = W.load()
DOC = A.FINAL.read_text(encoding="utf-8")
JR = Z1.JOURNAL.read_text(encoding="utf-8")
ST = Z1.STATE.read_text(encoding="utf-8")
LBL = "диагностический контроль (Ж-230)"


def a0g_rows(doc: str) -> list:
    return [ln for ln in doc.splitlines() if ln.startswith("|") and re.search(r"\bA0g\b", ln)]


def test_every_table_row_with_a0g_has_the_label_and_document_equals_rebuild():
    rows = a0g_rows(DOC)
    assert len(rows) > 40 and all(LBL in ln for ln in rows)
    assert DC.check_doc(T) == [] and A.build(T) == DOC


def test_mutation_label_missing_in_one_row_is_caught():
    row = a0g_rows(DOC)[0]
    bad = DOC.replace(row, row.replace(Z1.W.LBL, "A0g"), 1)
    assert bad != DOC and any("без подписи" in x for x in DC.check_doc(T, bad))


def test_mutation_build_without_labelling_has_unlabelled_rows(monkeypatch):
    monkeypatch.setattr(Z1, "label_tables", lambda d: d)
    assert any(LBL not in ln for ln in a0g_rows(A.build(T)))


def test_prose_of_every_section_labels_first_a0g_mention():
    parts = [p for p in re.split(r"(?m)^(?=## )", DOC) if p.startswith("## ")]
    with_prose = [p for p in parts if any(re.search(r"\bA0g\b", ln) and not ln.startswith("|") for ln in p.split("\n"))]
    assert len(with_prose) >= 6 and all(Z1.first_prose(p) is None for p in parts)
    for p in with_prose:
        first = next(ln for ln in p.split("\n") if re.search(r"\bA0g\b", ln) and not ln.startswith("|"))
        i = re.search(r"\bA0g\b", first).start()
        assert "диагностический контроль" in first[i:i + 90]


def test_mutation_prose_label_removed_in_one_section_is_caught():
    parts = re.split(r"(?m)^(?=## )", DOC)
    p = next(x for x in parts if x.startswith("## 10."))
    assert Z1.W.LBL in p
    bad = DOC.replace(p, p.replace(Z1.W.LBL, "A0g", 1), 1)
    assert bad != DOC and any("проза раздела" in x for x in DC.check_doc(T, bad))


def test_best_construction_is_chosen_only_among_17_and_a0g_does_not_change_it():
    assert len(W.PRE) == 17 and "A0g" not in W.PRE and len(D.CFG) == 18
    for w in ("2017_2021", "all"):
        assert W.best(T, w) == W.best_with_diag(T, w) != "A0g"
    assert Z1.selection_fail(T) == [] and A.claims_fail(T) == []


def test_mutation_a0g_with_minimal_qlike_is_never_best_and_build_refuses():
    t = dict(T)
    d = T["all"].copy()
    m = (d["series"] == "A0g") & d["window"].isin(["2017_2021", "all"])
    d.loc[m, "qlike"] = 0.01
    t["all"] = d
    assert W.best_with_diag(t, "all") == "A0g" and W.best(t, "all") != "A0g" and W.best(t, "2017_2021") != "A0g"
    assert len(Z1.selection_fail(t)) == 2
    with pytest.raises(ValueError):
        A.build(t)


def test_subsection_states_status_reason_budget_and_direct_departure():
    sub = DOC.split("### 5.1. A0g: статус, причина, моменты (Ж-230)")[1].split("\n## 6.")[0]
    for s in ("дополнительный диагностический контроль (Ж-230), а не заранее утверждённая основная конфигурация",
              "17 предрегистрированных конфигураций + A0g, диагностический контроль на месте не посчитанной по правилу CB2; всего 18 из 18.",
              "обучатель не отделялся от признаков", "принято после просмотра результатов листа 1b — это отступление от предрегистрации, а не предрегистрация",
              "Выводы против A0 сохранены как были, против A0g — рядом", "Строка `Ж-227:A0g` есть в `ALL_TRIALS.csv`", "входит в N и N_wide, а не добавляется к ним"):
        assert s in sub, s
    heads = [ln for ln in DOC.splitlines() if ln.startswith("## ")]
    assert len(heads) == 13
    assert "- Ж-230: A0g утверждена как диагностический контроль; изменены только подписи и новый раздел; таблицы не менялись." in DOC.split("\n## 13.")[1]


def test_moments_come_from_journal_and_state():
    mo = Z1.moments(JR, ST)
    assert (mo["review_start"], mo["review_end"], mo["question"], mo["deadline"], mo["default"], mo["leaf6"], mo["owner"]) == ("02:55", "03:05", "03:06", "10:00", "10:01", "10:30", "13:47")
    sec = Z1.section(T)
    assert all(x in sec for x in ("03:06", "10:01", "10:30", "13:47")) and Z1.label_tables(sec) in DOC


def test_mutation_owner_time_is_taken_from_journal_not_from_generator():
    bad = JR.replace("### Ж-230 — 09.11.2026, 13:47 MSK", "### Ж-230 — 09.11.2026, 14:05 MSK")
    assert bad != JR
    sec = Z1.section(T, journal=bad)
    assert "14:05" in sec and "13:47" not in sec
    src = Path(Z1.__file__).read_text(encoding="utf-8")
    assert re.findall(r"(?<![\\\d])\d\d:\d\d", src) == [], "время вписано в текст генератора"


def test_mutation_times_out_of_order_or_missing_record_refuse():
    with pytest.raises(ValueError):
        Z1.moments(JR.replace("### Ж-230 — 09.11.2026, 13:47 MSK", "### Ж-230 — 09.11.2026, 09:47 MSK"), ST)
    with pytest.raises(ValueError):
        Z1.moments(JR.replace("вопрос владельцу 1 ниже", "вопрос ниже"), ST)
    with pytest.raises(ValueError):
        Z1.moments(JR, ST.replace("Лист 6 (абляционная база A0g", "Лист 6 (база"))
