"""V6-FIN (G52): итоговый документ. Каждое число собирается из таблиц на диске; подменённое число, перевёрнутое «лучше/хуже» и знак Δ ловятся пересборкой; словесные выводы стоят на проверяемых условиях (`claims_fail`);
числа вне таблиц — только определения (`STATIC_DECIMALS`). Реальные таблицы и документ — только чтение."""
from __future__ import annotations

import pytest

from models_v2.s13 import v6_fin_asm as A, v6_fin_doc as W, v6_fin_docheck as DC, v6_fin_text as TX

T = W.load()
DOC = A.FINAL.read_text(encoding="utf-8")


def test_document_equals_rebuild_from_tables_and_has_over_1000_numbers():
    assert DC.check_doc(T) == []
    n, bad = DC.compare(DOC, A.build(T))
    assert bad == 0 and n > 1000


def test_thirteen_sections_and_first_line_is_result_then_status_and_percent_not_bps():
    heads = [ln for ln in DOC.splitlines() if ln.startswith("## ")]
    assert len(heads) == 13 and [h.split(".")[0] for h in heads] == ["## %d" % i for i in range(1, 14)]
    lines = DOC.splitlines()
    assert lines[0].startswith("Нет: ни у одной из 18 конфигураций V6") and "статистически неподтверждённое историческое улучшение" in lines[0]
    assert lines[2].startswith("DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок") and "bps" not in DOC.lower()


def test_mutation_substituted_number_is_caught():
    tok = DC.tokens(DOC.split("\n## 2.")[0])[10]
    bad = DOC.replace(tok, "9," + tok.split(",")[1], 1)
    n, diff = DC.compare(bad, A.build(T))
    assert diff >= 1 and DC.check_doc(T, bad) != []


def test_mutation_flipped_word_and_sign_are_caught():
    assert "хуже" in DOC and "| −" in DOC
    assert DC.check_doc(T, DOC.replace("хуже", "лучше", 1)) != []
    assert DC.check_doc(T, DOC.replace("| −", "| +", 1)) != []
    assert DC.check_doc(T, DOC.replace("Δ к K3", "Δ к K2", 1)) != []


def test_word_follows_sign_of_delta():
    assert W.word(-0.01) == "лучше" and W.word(0.01) == "хуже" and W.word(0.0) == "хуже"
    assert W.s(-0.5) == "−0,50" and W.s(0.5) == "+0,50" and W.f(-0.5, 1) == "−0,5"


def test_table_sign_flip_changes_document_and_is_caught():
    """Знак Δ в таблице тестов перевёрнут: сборка даёт другой текст, а документ на диске ≠ пересборке."""
    t = dict(T)
    d = T["tests"].copy()
    m = (d["config"] == "A2") & (d["base"] == "K1") & (d["window"] == "2017_2021") & (d["section"] == "delta_dm")
    dl, lo, hi = d.loc[m, "delta_pct"].copy(), d.loc[m, "lo90"].copy(), d.loc[m, "hi90"].copy()
    d.loc[m, "delta_pct"], d.loc[m, "lo90"], d.loc[m, "hi90"] = -dl, -hi, -lo
    t["tests"] = d
    assert A.build(t) != DOC and DC.check_doc(t, DOC) != []


def test_claims_fail_when_interval_goes_below_zero_vs_k3():
    """Если у какой-либо конфигурации интервал против K2 и K3 целиком ниже 0, первая строка «Нет» неверна: сборка отказывает."""
    assert A.claims_fail(T) == []
    t = dict(T)
    d = T["tests"].copy()
    m = (d["config"] == "A2") & (d["base"].isin(["K2", "K3"])) & (d["window"] == "2017_2021") & (d["section"] == "delta_dm")
    d.loc[m, "hi90"] = -0.1
    t["tests"] = d
    with pytest.raises(ValueError):
        A.build(t)


def test_claims_fail_when_a2_is_not_at_the_boundary_vs_a0g():
    t = dict(T)
    i = T["info"].copy()
    i.loc[(i["pair"] == "A2 vs A0g") & (i["window"] == "2017_2021"), "years_neg"] = 5
    t["info"] = i
    assert A.claims_fail(t) != []
    with pytest.raises(ValueError):
        A.build(t)


def test_claims_fail_when_a_config_passes_all_go_conditions():
    t = dict(T)
    g = T["go"].copy()
    g.loc[g["config"] == "A2", "passed"] = True
    t["go"] = g
    assert any("go" in k for k in A.claims_fail(t))


def test_numbers_outside_tables_are_only_declared_definitions():
    assert DC.check_static(T) == [] and TX.STATIC_DECIMALS >= {"0,05", "0,0005"}


def test_mutation_static_number_in_text_is_caught(monkeypatch):
    monkeypatch.setattr(TX, "REFINEMENTS", TX.REFINEMENTS + "\n- Лист 9: порог 0,77.")
    assert DC.check_static(T) != []


def test_perturbation_has_a_different_fixed_point_than_second_one():
    """Совпадение числа при искажении — не признак «числа вне таблиц»: два искажения с разными неподвижными точками."""
    assert abs(0.1371 / (1 - 1.31) - 0.0913 / (1 - 0.77)) > 0.01
