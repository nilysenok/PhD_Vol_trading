"""S-R-EVAL, документ: отчёт = перестроению из таблиц; мутации «подменённое число», «лучше / хуже», «подтверждено по чужой строке», пометка, решения, `--check` пишет (E.G3). Отчёт мутируется в копии (tmp)."""
import pandas as pd
import pytest

from models_v2.s18 import r_doc as D
from models_v2.s18 import r_doc_fmt as F


@pytest.fixture()
def rep(tmp_path, monkeypatch):
    p = tmp_path / "S_R_REPORT.md"
    p.write_text(D.REPORT.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(D, "REPORT", p)
    return p


def _mut(p, old, new, count=1):
    t = p.read_text(encoding="utf-8")
    assert old in t, old
    p.write_text(t.replace(old, new, count), encoding="utf-8")


def test_real_report_check_passes():
    assert D.check() == []


def test_report_structure_first_line_first_table_and_independent_confirmation():
    t = D.REPORT.read_text(encoding="utf-8")
    assert t.split("\n", 1)[0] == D.MARK and t.split("\n## ")[1].startswith("1. Таблица") and "Независимое подтверждение: нет" in t
    assert t.count("\n## ") == 9


def test_mut_substituted_number_is_caught(rep):
    T = F.read_all()
    q = T["table"].set_index("series").at["S-RCK", "qlike"]
    _mut(rep, F.c_(q, 6), F.c_(q * 0.9, 6))
    assert any("перестроению" in b for b in D.check())


def test_mut_flipped_better_worse_word_is_caught(rep):
    t = rep.read_text(encoding="utf-8")
    line = next(ln for ln in t.splitlines() if ln.startswith("| П1 |"))
    new = line.replace(" | хуже |", " | лучше |") if " | хуже |" in line else line.replace(" | лучше |", " | хуже |")
    assert new != line
    _mut(rep, line, new)
    assert any("«лучше / хуже» не по знаку" in b or "перестроению" in b for b in D.check())


def test_mut_confirmed_on_foreign_row_is_caught(rep):
    t = rep.read_text(encoding="utf-8")
    line = next(ln for ln in t.splitlines() if ln.startswith("| П1 |") and "не подтверждено" in ln)
    _mut(rep, line, line.replace("не подтверждено", "подтверждено"))
    assert any("подтверждено" in b for b in D.check())


def test_mut_first_line_not_the_mark_is_caught(rep):
    _mut(rep, D.MARK, "результат подтверждён")
    assert any("пометке" in b for b in D.check())


def test_mut_decision_text_changed_is_caught(rep):
    g = F.read_all()["guard"]
    d = g[g["id"] == "решение"]["value"].iloc[0]
    _mut(rep, "**" + d + "**", "**новая контрольная версия: S-RA2**")
    assert any("решение" in b or "перестроению" in b for b in D.check())


def test_mut_table_order_of_series_changed_is_caught(rep):
    t = rep.read_text(encoding="utf-8")
    a = next(ln for ln in t.splitlines() if ln.startswith("| HAR-J (K1) |"))
    b = next(ln for ln in t.splitlines() if ln.startswith("| XGBoost (K2) |"))
    _mut(rep, a + "\n" + b, b + "\n" + a)
    assert any("не в порядке п. 16" in x for x in D.check())


def test_mut_check_that_writes_is_caught(rep, monkeypatch):
    orig = D._layout

    def writing(T, text):
        (D.REP / "S_R_REPORT.tmp_probe").write_text("x", encoding="utf-8")
        return orig(T, text)
    monkeypatch.setattr(D, "_layout", writing)
    try:
        assert any("изменил выходы" in b for b in D.check())
    finally:
        (D.REP / "S_R_REPORT.tmp_probe").unlink()


def test_confirmed_count_matches_rows_and_p_are_from_own_row():
    T = F.read_all()
    pr = T["proc"]
    own = ((pr["qlike_new"] < pr["qlike_old"]) & (pr["delta_trim_pct"] < 0) & (pr["p_holm_N"] < 0.05) & (pr["p_spa_c"] < 0.05)).tolist()
    assert own == [str(v) == "True" for v in pr["confirmed"]] and isinstance(F.confirmed(T), int)
    assert pd.notna(T["stats"]["p_holm_N"]).any()
