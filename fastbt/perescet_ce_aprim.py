"""Лист CE-9: пересчёт поверх A′ только замороженных точек C/D/E, без нового отбора (Ж-088 п. 2).

Контракт: docs/otchety/2026-10-28_plan_ce_train4.md (ветка arh-ce-prereg), раздел 4.
Берётся ровно список замороженных точек (docs/otbor/ce_tochki_A_strogaya.json), в том же порядке;
отбор (`otbor`) не вызывается никогда: параметр принят только затем, чтобы сигнатура совпала с A6
и чтобы тест мог подложить вызываемый, который падает при вызове.
Строки пересчёта помечаются istochnik = "perescet", baza = "A_prim:<sha>"; в N не входят (v_N = False).
"""
from copy import deepcopy

POLYA = ("family", "W", "label")


def baza_aprim(aprim):
    """aprim: None (исход (в): A′ = A_strogaya), dict с ключом 'sha' или объект с атрибутом `sha`."""
    if aprim is None:
        return "A_strogaya"
    sha = aprim.get("sha") if isinstance(aprim, dict) else getattr(aprim, "sha", None)
    if not sha:
        raise ValueError("aprim без sha: baza = 'A_prim:<sha>' не собрать")
    return f"A_prim:{sha}"


def perescet(zamorozhennye, aprim, otbor=None):
    """Точки поверх A′: копия замороженных (family, W, label сохранены, порядок тот же) + пометки пересчёта.

    `otbor` не вызывается. Пустой/неполный список — ошибка, а не молчаливый пустой результат.
    """
    if not zamorozhennye:
        raise ValueError("пустой список замороженных точек")
    baza = baza_aprim(aprim)
    out = []
    for t in zamorozhennye:
        lost = [k for k in POLYA if k not in t]
        if lost:
            raise ValueError(f"в замороженной точке нет полей {lost}: {t!r}")
        p = deepcopy(dict(t))
        p.update(istochnik="perescet", baza=baza, v_N=False)
        out.append(p)
    return out
