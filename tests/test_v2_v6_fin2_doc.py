"""V6-FIN2 (G56, замечание З-1 проверяющего): в разделе 12 итога нет слова «Подтверждено» для результата с p Холма (N = 722) >= 0,05;
p Холма в тексте берётся из `G_INFO.csv`. Реальные таблицы и документ — только чтение."""
from __future__ import annotations

from models_v2.s13 import v6_fin_asm as A, v6_fin_doc as W, v6_fin_doc3 as W3

T = W.load()
DOC = A.FINAL.read_text(encoding="utf-8")


def sec12(doc: str) -> str:
    return doc.split("## 12.")[1].split("\n## 13.")[0]


def bad_confirmed(doc: str, t: dict) -> list[str]:
    """Строки раздела 12 со словом «Подтверждено» (без «Не подтверждено»), пока у A0g против A0 p Холма >= 0,05."""
    p = float(W3.pair(t, "A0g vs A0").p_holm_N)
    if p < 0.05:
        return []
    return [ln for ln in sec12(doc).splitlines() if "Подтверждено" in ln]


def test_section12_has_no_confirmed_with_holm_p_at_least_005():
    assert float(W3.pair(T, "A0g vs A0").p_holm_N) >= 0.05
    assert bad_confirmed(DOC, T) == []
    assert bad_confirmed(A.build(T), T) == []


def test_section12_states_holm_p_from_table_and_section13_has_fin2_line():
    p = W.f(W3.pair(T, "A0g vs A0").p_holm_N, 3)
    line = [ln for ln in sec12(DOC).splitlines() if ln.startswith("- **Измерено (без поправки на множественность).**")]
    assert len(line) == 1 and "после поправки Холма (N = 722) не значимо: p Холма %s" % p in line[0]
    assert "- **Лист FIN2.** Формулировка раздела 12 по замечанию З-1 проверяющего (`PROVERKA_V6.md`)" in DOC.split("\n## 13.")[1]


def test_mutation_old_word_returned_is_caught():
    bad = DOC.replace("**Измерено (без поправки на множественность).**", "**Подтверждено (измерено).**", 1)
    assert bad != DOC and len(bad_confirmed(bad, T)) == 1
