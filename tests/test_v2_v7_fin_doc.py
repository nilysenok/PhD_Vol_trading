"""V7-FIN (G50): итоговый документ. Каждое число собирается из таблиц на диске; подменённое число, перевёрнутое «лучше/хуже» и знак Δ ловятся пересборкой; словесные выводы стоят на проверяемых условиях (`claims_fail`);
числа вне таблиц — только определения (`STATIC_DECIMALS`). Реальные таблицы и документ — только чтение."""
from __future__ import annotations

import pandas as pd
import pytest

from models_v2.s14 import v7_fin_asm as A, v7_fin_data as D, v7_fin_doc as W, v7_fin_docheck as DC, v7_fin_text as TX

T = A.load()
DOC = A.FINAL.read_text(encoding="utf-8")


def test_document_equals_rebuild_from_tables_and_has_over_1000_numbers():
    assert DC.check_doc(T) == []
    n, bad = DC.compare(DOC, A.build(T))
    assert bad == 0 and n > 1000


def test_fifteen_sections_first_line_is_result_then_status_and_percent_not_bps():
    heads = [ln for ln in DOC.splitlines() if ln.startswith("## ")]
    assert len(heads) == 15 and [h.split(".")[0] for h in heads] == ["## %d" % i for i in range(1, 16)]
    lines = DOC.splitlines()
    assert lines[0].startswith("Нет: 0 из 9 посчитанных конфигураций V7") and "«go» 6.1 выполнено у 0 из 9" in lines[0]
    assert lines[2] == TX.STATUS and "первая цифра — потолок" in lines[2] and "дизайн K2/K3 выбирался на SBER 2017–2021" in lines[2] and "bps" not in DOC.lower()


def test_mutation_substituted_number_in_document_is_caught():
    tok = DC.tokens(DOC.split("\n## 2.")[0])[10]
    bad = DOC.replace(tok, "9," + tok.split(",")[1], 1)
    n, diff = DC.compare(bad, A.build(T))
    assert diff >= 1 and DC.check_doc(T, bad) != []


def test_mutation_substituted_number_in_table_changes_document_and_is_caught():
    """Число в FIN_ALL подменено (таблица): сборка даёт другой текст, документ на диске ≠ пересборке."""
    t = dict(T)
    a = T["FIN_ALL"].copy()
    m = (a["series"] == "cfg8") & (a["window"] == "2018_2021")
    a.loc[m, "qlike"] = a.loc[m, "qlike"] + 0.01
    t["FIN_ALL"] = a
    assert A.build(t) != DOC and DC.check_doc(t, DOC) != []


def test_mutation_flipped_word_and_sign_are_caught():
    assert "хуже" in DOC and "| −" in DOC and "| лучше |" in DOC
    assert DC.check_doc(T, DOC.replace("| хуже |", "| лучше |", 1)) != []
    assert DC.check_doc(T, DOC.replace("−", "+", 1)) != []
    assert "Δ к K3" in DOC and DC.check_doc(T, DOC.replace("Δ к K3", "Δ к K2", 1)) != []


def test_word_follows_sign_of_delta_and_formats():
    assert W.word(-0.01) == "лучше" and W.word(0.01) == "хуже" and W.word(0.0) == "хуже"
    assert W.s(-0.5) == "−0,50" and W.s(0.5) == "+0,50" and W.f(-0.5, 1) == "−0,5" and W.f(-0.00001, 2) == "0,00"


def test_table_sign_flip_of_delta_changes_document_and_is_caught():
    t = dict(T)
    d = T["FIN_TESTS"].copy()
    m = (d["config"] == "cfg8") & (d["base"] == "K2") & (d["window"] == "2018_2021")
    dl, lo, hi = d.loc[m, "delta_pct"].copy(), d.loc[m, "lo90"].copy(), d.loc[m, "hi90"].copy()
    d.loc[m, "delta_pct"], d.loc[m, "lo90"], d.loc[m, "hi90"] = -dl, -hi, -lo
    t["FIN_TESTS"] = d
    assert A.build(t) != DOC and DC.check_doc(t, DOC) != []


def test_first_line_flips_to_yes_when_an_interval_goes_below_zero_against_all_four():
    t = dict(T)
    d = T["FIN_TESTS"].copy()
    m = (d["config"] == "cfg8") & d["base"].isin(D.BASES) & (d["window"] == "2018_2021")
    d.loc[m, "hi90"] = -0.1
    t["FIN_TESTS"] = d
    assert W.first_line(t).startswith("Да: 1 из 9") and W.first_line(T).startswith("Нет:") and DC.check_doc(t, DOC) != []


def test_mutation_not_computed_config_shown_as_computed_is_refused():
    t = dict(T)
    r = T["FIN_REGISTRY"].copy()
    r.loc[r["place"] == 9, "status"] = "посчитана"
    t["FIN_REGISTRY"] = r
    assert A.claims_fail(T) == [] and any("реестр" in x for x in A.claims_fail(t))
    with pytest.raises(ValueError):
        A.build(t)


def test_mutation_cfg11_series_present_in_main_table_is_refused():
    t = dict(T)
    a = T["FIN_ALL"]
    t["FIN_ALL"] = pd.concat([a, a[a["series"] == "cfg8"].assign(series="cfg11")], ignore_index=True)
    assert any("FIN_ALL" in x for x in A.claims_fail(t))


def test_mutation_version_1_served_as_main_is_refused():
    t = dict(T)
    a, v1 = T["FIN_ALL"].copy(), T["FIN_V1"]
    m = (a["series"] == "cfg1") & (a["window"] == "2018_2021")
    a.loc[m, "qlike"] = float(v1[(v1["config"] == "cfg1") & (v1["window"] == "2018_2021")]["qlike_v1"].iloc[0])
    t["FIN_ALL"] = a
    assert any("cfg1" in x for x in A.claims_fail(t))
    with pytest.raises(ValueError):
        A.build(t)


def test_claims_fail_when_a2_story_about_2018_breaks():
    t = dict(T)
    a = T["FIN_ALL"].copy()
    m = (a["series"] == "V6:A2") & (a["window"] == "2019_2021")
    a.loc[m, "qlike"] = 0.01
    t["FIN_ALL"] = a
    assert any("2018" in x for x in A.claims_fail(t))


def test_registry_text_names_nine_computed_and_two_not_run_and_reserve_unused():
    sec = DOC.split("## 4.")[1].split("## 5.")[0]
    assert "посчитано 9 (1, 2, 3, 4, 5, 6, 7, 8, 10)" in sec and "не считалось по условию — 2 (9, 11)" in sec and "резерв (место 12): не использован" in sec
    assert "(`2.cond9`)" in sec and "(`4.cond11`)" in sec and sec.count("вычеркнуто") >= 5
