"""V4-FIN (G46–G47): итоговый документ. Главная строка и таблицы собираются из FIN_*.csv; подменённое число в документе (главная строка, таблица трёх лучших, ссылочные ряды, все 46 архитектур, окна) ловится пересборкой;
проверенных чисел не меньше 30; слова «лучше/хуже» следуют знаку Δ; разделы Ж-222 п. 14 и Ж-225 п. 9–10 на месте. Реальные таблицы и документ — только чтение."""
from __future__ import annotations

import pytest

from models_v2.s11 import v4_fin_check as CK, v4_fin_doc as DOC, v4_fin_write as W

T = DOC.tables_from_disk()
TXT = W.FINAL.read_text(encoding="utf-8")


def test_document_matches_tables_and_has_at_least_30_numbers():
    assert DOC.missing_claims(TXT, T) == []
    assert TXT.splitlines()[0] == DOC.first_line(T)
    assert sum(DOC.numbers_in(c) for c in DOC.claims(T)) >= 30
    assert DOC.numbers_in(DOC.first_line(T)) >= 15 and DOC.numbers_in(DOC.top3_block(T)) >= 30


@pytest.mark.parametrize("which", ["first_line", "top3", "ref", "all", "windows"])
def test_mutation_substituted_number_in_document_is_caught(which):
    """Мутация: в документе меняется одно число соответствующего фрагмента -> фрагмент не находится."""
    a, b = DOC.idx(T), T["top"][0]
    frag, num = {"first_line": (DOC.first_line(T), a.loc[(b, "all"), "crit_panel"]), "top3": (DOC.top3_block(T), a.loc[(b, "all"), "crit_panel"]), "ref": (DOC.ref_block(T), a.loc[("k1", "all"), "crit_panel"]),
                 "all": (DOC.all_block(T), a.loc[(b, "all"), "crit_panel"]), "windows": (DOC.windows_block(T), a.loc[(b, "2017"), "qlike"])}[which]
    old = "%.5f" % num
    assert old in frag
    bad = TXT.replace(frag, frag.replace(old, "%.5f" % (num + 0.00001), 1), 1)
    assert bad != TXT and DOC.missing_claims(bad, T) != []


def test_mutation_flipped_word_better_worse_is_caught():
    fl = DOC.first_line(T)
    assert "хуже" in fl
    bad = TXT.replace(fl, fl.replace("хуже", "лучше", 1), 1)
    assert DOC.missing_claims(bad, T) != [] and bad.splitlines()[0] != DOC.first_line(T)


def test_word_follows_sign_of_delta():
    assert DOC.word(-0.01) == "лучше" and DOC.word(0.01) == "хуже" and DOC.word(0.0) == "хуже"


def test_first_line_states_main_conclusion_with_both_measures_and_no_go():
    fl = TXT.splitlines()[0]
    assert fl.startswith("Устойчивого преимущества") or "прошли" in fl
    assert "crit_panel" in fl and "простое среднее QLIKE 2017–2021" in fl and "из 46" in fl
    assert not TXT.startswith("#")


def test_document_has_required_sections_and_statuses():
    for p in CK.DOC_PARTS:
        assert p in TXT, p
    assert "(в) Экономическая значимость — не оценивалась" in TXT and "НЕ независимый OOS" in TXT
    assert "ACCEPTED не присваивается" in TXT and "Ж-224 — это Ж-225" in TXT
    assert "bps" not in TXT.lower()                                      # владельцу — проценты, не bps


def test_reproduction_lists_every_stage():
    rep = W.REPRO.read_text(encoding="utf-8")
    for st in ("audit", "A", "B", "Ca", "Cb", "E", "DX", "CS", "NTP", "CLD", "RQC", "H", "I", "FIN"):
        assert "--stage %s --check" % st in rep and "--stage %s\n" % st in rep, st


def test_top3_are_lowest_crit_among_46_and_recorded_in_checkpoint():
    a = DOC.idx(T)
    from models_v2.s11 import v4_fin_data as FD
    order = sorted(FD.CFGS, key=lambda s: (a.loc[(s, "all"), "crit_panel"], s))
    assert T["top"] == order[:3] == T["cp"]["FIN_top3"]
