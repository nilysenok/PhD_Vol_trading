"""V5-FIN2: оговорки проверки в итоговом документе. О3 (охват SP1 отдельно к K8 и к M) стоит на условиях `claims_fail`;
раздел «Оговорки проверки перед слиянием» есть между «Отступлениями» и «Воспроизводимостью». Реальные таблицы и документ — только чтение."""
from __future__ import annotations

import copy

import pytest

from models_v2.s12 import v5_fin_asm as A, v5_fin_build as B, v5_fin_doc as W, v5_fin_docheck as DC

T = W.load()
DOC = B.FINAL.read_text(encoding="utf-8")
S1 = T["sp"]
CLAIMS = ("интервал охвата SP1 к M в 2017–2021 не включает 0", "интервалы охвата SP1 к M в 2022–2025 и вся включают 0", "интервалы охвата SP1 к K8 включают 0 во всех окнах")


def _mut(best, base, window, lo, hi):
    t = dict(T)
    d = copy.deepcopy(T["enc"])
    m = (d["best"] == best) & (d["base"] == base) & (d["window"] == window)
    assert m.sum() == 1
    d.loc[m, "w_lo90"], d.loc[m, "w_hi90"] = lo, hi
    t["enc"] = d
    return t


def test_real_tables_satisfy_o3_claims_and_sentence_has_table_numbers():
    assert not [c for c in B.claims_fail(T) if "SP1" in c]
    k8 = B.enc(T, S1, "K8", "2017_2021")
    m = B.enc(T, S1, "M", "2017_2021")
    assert k8.w_lo90 <= 0 <= k8.w_hi90 and m.w_lo90 > 0
    sent = B.sp1_claim(T, lambda r: "%.2f/%.2f" % (r.w_lo90, r.w_hi90))
    assert "к K8 включает 0 во всех окнах" in sent and "к M он не включает 0 в 2017–2021" in sent
    assert "%.2f/%.2f" % (m.w_lo90, m.w_hi90) in sent
    assert sent.split("(2017–2021, 2022–2025, вся история")[0].endswith("окнах ")
    assert "У SP1 интервал веса к K8 включает 0 во всех окнах" in DOC and "а не только" not in DOC


@pytest.mark.parametrize("best,base,window,lo,hi,claim", [(S1, "M", "2017_2021", -0.1, 1.0, CLAIMS[0]),       # О3 перевёрнут: к M в 2017–2021 нуль внутри
                                                          (S1, "M", "2022_2025", 0.2, 1.0, CLAIMS[1]),        # к M в 2022–2025 нуль снаружи
                                                          (S1, "K8", "all", 0.3, 0.9, CLAIMS[2])])           # к K8 нуль снаружи
def test_mutation_o3_reversed_makes_build_refuse(best, base, window, lo, hi, claim):
    t = _mut(best, base, window, lo, hi)
    assert claim in B.claims_fail(t)
    with pytest.raises(ValueError, match="утверждения документа не выполнены"):
        A.build(t)


def test_caveats_section_present_between_departures_and_reproducibility():
    i17, i18, i19 = (DOC.find("\n## %d. " % n) for n in (17, 18, 19))
    assert 0 < i17 < i18 < i19
    sec = DOC[i18:i19]
    assert "Оговорки проверки перед слиянием" in sec and "сливать с оговорками" in sec
    assert all("О%d." % k in sec for k in range(1, 6)) and "не подкладывал собственных мутаций" in sec
    assert DC.check_doc(T) == [] and DC.check_static(T) == []


def test_caveats_section_removed_makes_doc_differ_from_rebuild():
    i18, i19 = DOC.find("\n## 18. "), DOC.find("\n## 19. ")
    assert DC.check_doc(T, DOC[:i18] + DOC[i19:]) != []
