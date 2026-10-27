"""Развёртка сетки Э1-б (формат sweep-setka v1, fastbt/designs/sweep/S{k}_setka_e1b.yaml) в плоские base/grid для драйверов.

Ключ параметра: `имя` в безусловных блоках и `<блок>.<имя>` в блоках с `uslovie` (в S5/S6 max_hold и cooldown_bars
есть в нескольких RM-блоках). OAT по параметру условного блока идёт с подставленным условием (`oat_base`).
В `vne_v1` попадает всё, что нужно править в fastbt: dostup=pravka, значения из pravka_znach (ослабление зашитых порогов),
RM в √RV (sqrt_rv), trail_type=off. Они в base/grid не попадают; N в отчёте — верхняя граница.
"""
from __future__ import annotations

from pathlib import Path

DROP_BLOCKS = ("rm_sqrt_rv",)
DROP_PARAMS = ("rm_edinitsa",)
DROP_VALUES = {"rm_tip": ["sqrt_rv"], "trail_type": ["off"]}


def expand(setka) -> dict:
    """-> dict(base, grid, uslovie {ключ: {параметр: значение}}, vne_v1 [строки])."""
    import yaml
    s = setka if isinstance(setka, dict) else yaml.safe_load(Path(setka).read_text("utf-8"))
    base, grid, usl, vne = {}, {}, {}, []
    for blk, b in s["bloki"].items():
        for p in b.get("parametry", []):
            name = p["imya"]
            key = f"{blk}.{name}" if "uslovie" in b else name
            if blk in DROP_BLOCKS or name in DROP_PARAMS or p.get("dostup") == "pravka":
                vne.append(f"{key}: целиком вне v1 ({blk if blk in DROP_BLOCKS else p.get('dostup', 'RM в √RV')})")
                continue
            bad = list(p.get("pravka_znach") or []) + DROP_VALUES.get(name, [])
            vals = [v for v in p["setka"] if v not in bad]
            if len(vals) != len(p["setka"]):
                vne.append(f"{key}: значения {[v for v in p['setka'] if v in bad]} вне v1")
            base[key], grid[key] = p["ishodnoe"], vals
            if "uslovie" in b:
                usl[key] = dict(b["uslovie"])
    return dict(base=base, grid=grid, uslovie=usl, vne_v1=vne)


def oat_base(exp: dict, key: str) -> dict:
    """База OAT для параметра key: исходные значения + условие его блока."""
    return {**exp["base"], **exp["uslovie"].get(key, {})}
