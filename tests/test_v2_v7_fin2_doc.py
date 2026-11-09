"""V7-FIN2 (G53, замечания З-1, З-2, З-5 проверки): в итоге нет «Подтверждено» при p Холма (N = 733) >= 0,05, p берётся из таблицы; у причин (б), (г) — пометка правила DX; оговорка о z в разделах 12 и 14.
Реальные таблицы и документы — только чтение."""
from __future__ import annotations

from models_v2.s14 import v7_dx_rep as DXR, v7_fin_asm as A, v7_fin_doc as W, v7_fin_text as TX

T = A.load()
DOC = A.FINAL.read_text(encoding="utf-8")
DXDOC = (A.IO.REP / "DX_REPORT.md").read_text(encoding="utf-8")


def sec(doc: str, n: int) -> str:
    return doc.split("\n## %d." % n)[1].split("\n## %d." % (n + 1))[0] if n < 15 else doc.split("\n## 15.")[1]


def pmin(t: dict) -> float:
    p = t["FIN_PAIRS"]
    return float(p[(p["window"] == "2018_2021") & (p["rule62"] == "да")]["p_holm_N"].min())


def bad_confirmed(doc: str, t: dict) -> list:
    """Строки раздела 14 со словом «Подтверждено» (без «Не подтверждено»), пока p Холма пар 6.2 >= 0,05."""
    return [] if pmin(t) < 0.05 else [ln for ln in sec(doc, 14).splitlines() if "Подтверждено" in ln]


def test_section14_has_no_confirmed_with_holm_p_at_least_005_and_states_p_from_table():
    assert pmin(T) >= 0.05
    assert bad_confirmed(DOC, T) == [] and "Подтверждено" not in DOC
    head = "**Правило 6.2 выполнено (статистически не подтверждено: p Холма N = 733 — %s, порог 0,05)**" % W.f(pmin(T), 3)
    assert head in sec(DOC, 14) and head in A.build(T)


def test_mutation_confirmed_at_holm_p_ge_005_is_caught():
    bad = DOC.replace("**Правило 6.2 выполнено (статистически не подтверждено: p Холма N = 733 — 1,000, порог 0,05)**", "**Подтверждено**", 1)
    assert bad != DOC and len(bad_confirmed(bad, T)) == 1


def test_dx_causes_b_and_g_carry_the_note_in_final_and_in_dx_report():
    for line in (ln for ln in sec(DOC, 12).splitlines() if ln.startswith(("- **(б)", "- **(г)"))):
        assert "подтверждена" in line and TX.DX_NOTE in line
    assert DXR.NOTE == TX.DX_NOTE
    for h in ("## (б) Сигнал:", "## (г) Дублирование:"):
        ln = [x for x in DXDOC.splitlines() if x.startswith(h)][0]
        assert "подтверждена" in ln and DXR.NOTE in ln
    assert [x for x in DXDOC.splitlines() if x.startswith("Причина (по правилам")][0].endswith(DXR.NOTE + ".")
    assert "подтверждена" not in "".join(x for x in DXDOC.splitlines() if x.startswith(("## (а)", "## (в)")))


def test_mutation_note_removed_is_caught():
    bad = DOC.replace(TX.DX_NOTE, "")
    assert bad != DOC and not all(TX.DX_NOTE in ln for ln in sec(bad, 12).splitlines() if ln.startswith(("- **(б)", "- **(г)")))


def test_z_in_sample_caveat_in_sections_12_and_14():
    for n in (12, 14):
        assert TX.Z_IN_SAMPLE in sec(DOC, n)
    assert "не является утечкой" in TX.Z_IN_SAMPLE and "z внутри выборки" in TX.Z_IN_SAMPLE
