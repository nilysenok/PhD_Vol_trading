"""S-C-EVAL, `cm_doc --check` (гейт E.G3; mutations.md № 125, 140, 158): `S_C_REPORT.md` = перестроению из таблиц `S_C_*.csv` и ключа `S.C.eval`; первая строка — пометка «исследовательский, не OOS»; таблица п. 16 первой и в порядке п. 16;
в каждом разделе с таблицами — строка D_M; «подтверждено» только на своей строке (Δ < 0, p Холма N < 0,05 и p SPA < 0,05); «лучше / хуже» по знаку Δ; решения NEWCTRL и STRUCT = строкам GUARD; `--check` только читает.
Мутации — копия текста отчёта во временном файле (`D.REPORT` подменён monkeypatch), реальный отчёт не меняется."""
from __future__ import annotations

import copy

import pytest

from models_v2.s18 import cm_doc as D


@pytest.fixture(scope="module")
def doc():
    T, ck = D.F.read_all(), D._key()
    return dict(T=T, ck=ck, text=D.REPORT.read_text(encoding="utf-8"), h0=D._tree())


def run_mut(monkeypatch, tmp_path, text):
    p = tmp_path / "S_C_REPORT.md"
    p.write_text(text, encoding="utf-8")
    monkeypatch.setattr(D, "REPORT", p)
    return D.check()


def test_check_ok_report_equals_rebuild_and_reads_only(doc):
    assert D.check() == [] and doc["text"] == D.build(doc["T"], doc["ck"]) and D._tree() == doc["h0"]
    assert doc["text"].split("\n", 1)[0] == D.MARK and doc["text"].split("\n## ")[1].startswith("1. Таблица")
    assert D._layout(doc["T"], doc["text"]) == [] and D._row_checks(doc["T"], doc["text"]) == []


def test_report_numbers_come_from_tables_and_first_table_is_p16(doc):
    T, text = doc["T"], doc["text"]
    sec1 = text.split("\n## ")[1]
    names = [ln.split(" | ")[0].lstrip("| ") for ln in sec1.splitlines() if ln.startswith("| ") and not ln.startswith("| Ряд")][:12]
    assert names == list(D.ORDER)
    assert "n = %d" % int(T["table"]["n"].iloc[0]) in sec1 and "2018-01-04…2026-11-10" in sec1
    assert "**" + T["guard"][T["guard"]["id"] == "решение"]["value"].iloc[0] + "**" in text and "Независимое подтверждение: нет" in text


def test_mutation_substituted_number_is_caught(doc, monkeypatch, tmp_path):
    s = doc["T"]["stats"]
    d = s[(s["kind"] == "main") & (s["hypothesis"] == "S-CK|W5-G")]["delta_pct"].iloc[0]
    from models_v2.s18.cm_doc_fmt import c_
    old = c_(d, 4, True)
    assert old in doc["text"]
    bad = run_mut(monkeypatch, tmp_path, doc["text"].replace(old, c_(d - 2.0, 4, True), 1))
    assert any("перестроению" in b for b in bad)


def test_mutation_flipped_better_worse_is_caught(doc, monkeypatch, tmp_path):
    lines = doc["text"].split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("| К1: ") and ln.endswith(" лучше |"))
    lines[i] = lines[i][:-len(" лучше |")] + " хуже |"
    text = "\n".join(lines)
    assert any("лучше / хуже" in b for b in D._row_checks(doc["T"], text))
    assert any("перестроению" in b for b in run_mut(monkeypatch, tmp_path, text))


def test_mutation_confirmed_on_foreign_row_is_caught(doc, monkeypatch, tmp_path):
    lines = doc["text"].split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("| S-CK к W5-G |") and "численно не лучше базы; не подтверждено" in ln)
    lines[i] = lines[i].replace("численно не лучше базы; не подтверждено", "численно лучше базы; подтверждено (p Холма N < 0,05 и p SPA < 0,05)")
    bad = run_mut(monkeypatch, tmp_path, "\n".join(lines))
    assert any("«подтверждено» в таблицах" in b for b in bad)


def test_confirmed_requires_both_p_by_own_row(doc):
    T = copy.deepcopy(doc["T"])
    st = T["stats"]
    i = st[(st["kind"] == "main") & (st["delta_pct"] < 0)].index[0]
    st.loc[i, ["p_holm_N", "p_spa_c"]] = (0.001, 0.001)
    st.loc[i, "verdict"] = "численно лучше базы; подтверждено (p Холма N < 0,05 и p SPA < 0,05)"
    assert D.confirmed(T) == 1 and D._row_checks(T, D.build(T, doc["ck"])) == []          # честная подтверждённая строка допустима
    st.loc[i, "p_spa_c"] = 0.20                                                           # p SPA не выполнено, а вывод остался «подтверждено»
    assert D.confirmed(T) == 0 and any("не на своей строке" in b for b in D._row_checks(T, D.build(T, doc["ck"])))
    st.loc[i, ["p_spa_c", "p_holm_N"]] = (0.001, 0.20)
    assert D.confirmed(T) == 0 and any("не на своей строке" in b for b in D._row_checks(T, D.build(T, doc["ck"])))


def test_mutation_decision_marker_order_and_dm_line_are_caught(doc, monkeypatch, tmp_path):
    t = doc["text"]
    bad = run_mut(monkeypatch, tmp_path, t.replace("**не прошёл", "**новая контрольная версия: S-CK (подмена) — не прошёл", 1))
    assert any("решение" in b for b in bad)
    bad = run_mut(monkeypatch, tmp_path, t.replace("эффект структуры: нет", "эффект структуры: да"))
    assert any("STRUCT" in b for b in bad) and any("перестроению" in b for b in bad)
    bad = run_mut(monkeypatch, tmp_path, t.replace(D.MARK, "результат подтверждён", 1))
    assert any("первая строка" in b for b in bad)
    lines = t.split("\n")
    a, b = (next(k for k, ln in enumerate(lines) if ln.startswith(p)) for p in ("| HAR-J (K1)", "| XGBoost (K2)"))
    lines[a], lines[b] = lines[b], lines[a]
    assert any("порядке п. 16" in x for x in D._layout(doc["T"], "\n".join(lines)))
    nodm = t.split("\n## 3. ")
    sec3 = nodm[1].replace(D.F.dm_line(doc["T"]), "", 1)
    assert any("без строки D_M" in x for x in D._layout(doc["T"], nodm[0] + "\n## 3. " + sec3))


def test_mutation_no_independent_confirmation_line_is_caught(doc, monkeypatch, tmp_path):
    bad = run_mut(monkeypatch, tmp_path, doc["text"].replace("Независимое подтверждение: нет", "Независимое подтверждение: есть", 1))
    assert any("независимое подтверждение" in b or "перестроению" in b for b in bad)
