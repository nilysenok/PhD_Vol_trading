"""М-драйвер, часть 2: заморозка Морриса → точки полного прохода Соболя v6 (Ж-137 п. 2; инвентаризация §3.2). Данных не читает.
Замороженная новая ось ВЫНУТА из куба: у неё нет размерности Соболя, в каноне точки её нет («выкл.»). Новая ось, все родители которой
заморожены, замораживается тоже (в том числе неготовая и многородительская)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from fastbt import canon_v6


def zamknut(kub: dict, res: dict) -> dict:
    """Дополняет res {ось: (заморожена, причина)} дочерними осями замороженных родителей."""
    izm = True
    while izm:
        izm = False
        for n, v in canon_v6.novye(kub).items():
            rod = v.get("roditel")
            if rod and not res.get(n, (False,))[0] and all(res.get(p, (False,))[0] for p in rod):
                res[n], izm = (True, f"все родители {rod} заморожены"), True
    return res


def kub_bez(kub: dict, zam) -> dict:
    """Куб без замороженных новых осей и ссылок на них (roditel, snyato)."""
    zam, k = set(zam), copy.deepcopy(kub)
    for sec in canon_v6.NOVYE:
        for n in zam:
            k.get(sec, {}).pop(n, None)
        for v in k.get(sec, {}).values():
            if "roditel" in v:
                v["roditel"] = [p for p in v["roditel"] if p not in zam]
    k["snyato"] = {s: {a: b for a, b in d.items() if b not in zam} for s, d in kub.get("snyato", {}).items()}
    return k


def prochitat(path) -> set:
    """Замороженные оси из zamorozka.yaml (`osi`)."""
    return set(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["osi"])


def tochki_polnye(sid: str, zam, n0: int, n1: int, seed: str = "seed", kub: dict | None = None) -> tuple:
    """Точки полного прохода с заморозкой: ([(i, канон)], {vybrosheno}); замороженной оси в каноне нет."""
    return canon_v6.tochki_v6(kub_bez(kub or canon_v6.load_kub(sid), zam), n0, n1, seed)
