"""V5-FIN (G24): итоговый документ. Каждое число собирается из таблиц на диске; подменённое число, перевёрнутое «лучше/хуже» и знак Δ ловятся пересборкой; словесные выводы стоят на проверяемых условиях (`claims_fail`);
числа вне таблиц — только определения (`STATIC_DECIMALS`). Реальные таблицы и документ — только чтение."""
from __future__ import annotations

import pytest

from models_v2.s12 import v5_fin_asm as A, v5_fin_build as B, v5_fin_doc as W, v5_fin_docheck as DC, v5_fin_text as TX

T = W.load()
DOC = B.FINAL.read_text(encoding="utf-8")


def test_document_equals_rebuild_from_tables_and_has_over_1000_numbers():
    assert DC.check_doc(T) == []
    n, bad = DC.compare(DOC, A.build(T))
    assert bad == 0 and n > 1000


def test_first_line_is_result_then_status_and_percent_not_bps():
    lines = DOC.splitlines()
    assert lines[0].startswith("0 из 9 конфигураций прошли «go»") and "устойчивого преимущества над XGBoost (K2), LightGBM (K3) и над K8 нет" in lines[0]
    assert lines[1] == "" and lines[2].startswith("**Статус:** исследовательский, DEVELOPMENT / HISTORICAL WALK-FORWARD") and "первая цифра — потолок" in lines[2]
    assert "bps" not in DOC.lower() and "SBER в выбор и в M16 не входит" in DOC


def test_mutation_substituted_number_is_caught():
    tok = DC.tokens(DOC.split("\n## 2.")[0])[10]                  # число таблицы раздела 1
    bad = DOC.replace(tok, "9," + tok.split(",")[1], 1)
    n, diff = DC.compare(bad, A.build(T))
    assert diff >= 1 and DC.check_doc(T, bad) != []


def test_mutation_flipped_word_and_sign_are_caught():
    assert "лучше" in DOC and "| −" in DOC
    assert DC.check_doc(T, DOC.replace("лучше", "хуже", 1)) != []
    assert DC.check_doc(T, DOC.replace("| −", "| +", 1)) != []


def test_word_follows_sign_of_delta():
    assert W.word(-0.01) == "лучше" and W.word(0.01) == "хуже" and W.word(0.0) == "хуже"
    assert W.s(-0.5) == "−0,50" and W.s(0.5) == "+0,50" and W.f(-0.5, 1) == "−0,5"


def test_claims_fail_when_sp1_beats_k8_after_2018():
    assert B.claims_fail(T) == []
    t = dict(T)
    y = T["years"].copy()
    y.loc[(y["series"] == T["sp"]) & (y["year"] == 2019), "qlike_m16"] = 0.0
    t["years"] = y
    assert B.claims_fail(t) != []
    with pytest.raises(ValueError):
        A.build(t)


def test_claims_fail_when_go_is_true():
    t = dict(T)
    a = T["all"].copy()
    a.loc[(a["section"] == "go") & (a["series"] == "SP1"), "go"] = True
    t["all"] = a
    with pytest.raises(ValueError):
        A.build(t)


def test_numbers_outside_tables_are_only_definitions():
    assert DC.check_static(T) == []
    assert DC.static_decimals(T) <= TX.STATIC_DECIMALS


def test_required_sections_special_question_and_not_shown_part():
    for p in DC.PARTS:
        assert p in DOC, p
    sp = DOC.split("## 15.")[1].split("## 16.")[0]
    assert "действительно ли новые архитектуры извлекают информацию" in sp and "**Не показано.**" in sp and "Отсутствие значимости не доказывает отсутствия информации" in sp
    assert "Охват" in sp and "Совместные ошибки" in sp and "Абляция" in sp


def test_every_direction_has_rule_and_number():
    sec = DOC.split("## 14.")[1].split("## 15.")[0]
    for d in ("1 Graph-Score", "2 Spectral", "3 Moments", "4 Latent", "5 Jump", "6 Break", "7 сочетания"):
        assert d in sec
    assert "p Вальда" in sec and "QLIKE K8" in sec


def test_perturbed_tables_change_every_result_number():
    t = DC.perturb(T)
    n0 = DC.tokens(DOC)
    import models_v2.s12.v5_fin_build as BB
    orig = BB.claims_fail
    BB.claims_fail = lambda _t: []
    try:
        n1 = DC.tokens(A.build(t))
    finally:
        BB.claims_fail = orig
    same = sum(a == b for a, b in zip(n0, n1))
    assert len(n0) == len(n1) and same < 0.02 * len(n0)
