"""C-L0 (контроль, не trial): наследованные HP И2 + F_top3 / F_F4 против собственных HP LB0 + те же наборы (из B). База развёрток = min crit_pen из четырёх."""
from __future__ import annotations

import json

import numpy as np

from models_v2.s6.lgbm import c_hp, c_world, grid, job
from models_v2.s6.lgbm import st_cfgs as sc
from models_v2.s6.lgbm.world import DATA

BASE_FILE = c_world.DATA_C / "C_BASE.json"
SETS = {"top3": "F_top3", "F4": "F_F4"}          # набор -> имя конфигурации B (собственные HP LB0)


def groups_of(set_name: str) -> tuple:
    return tuple(json.loads((DATA / "feat_registry.json").read_text())[SETS[set_name]]["groups"])


def hp_of(W: dict, kind: str) -> dict:
    return c_hp.lift_y(sc.hp_own() if kind == "own" else {Y: grid.from_i2(W["i2"][Y]) for Y in job.YEARS})


def spec_of(W: dict, name: str, set_name: str):
    return sc.feat_spec(W, name, sc.group_cols(W, groups_of(set_name)))


def items(W: dict) -> list:
    return [c_world.CItem("C0_inh_" + s, spec_of(W, "C0_inh_" + s, s), hp_of(W, "inh"), "control", "c_l0",
                          "C-L0: наследованные HP И2 + %s (%s)" % (SETS[s], "+".join(groups_of(s)))) for s in SETS]


def choose(crits: dict) -> str:
    """Минимум crit_pen; ничья |Δ| ≤ 1e-12 — собственные HP раньше наследованных, top3 раньше F4 (порядок словаря)."""
    m = min(crits.values())
    return next(k for k, v in crits.items() if v - m <= 1e-12)


def base_cfg(W: dict) -> tuple:
    """-> (имя базы, Spec, hp_y) из сохранённого C_BASE.json."""
    b = json.loads(BASE_FILE.read_text())
    return b["name"], spec_of(W, "base", b["set"]), hp_of(W, b["hp"])


def stage(W: dict) -> None:
    its = items(W)
    c_world.run_batch(W, its)
    names = {"own_top3": "F_top3", "own_F4": "F_F4", "inh_top3": "C0_inh_top3", "inh_F4": "C0_inh_F4"}
    crits = {k: c_world.get(n)[1]["crit"]["crit_pen"] for k, n in names.items()}
    pick = choose(crits)
    hp, s = pick.split("_")
    BASE_FILE.write_text(json.dumps({"name": names[pick], "hp": hp, "set": s, "crit_pen": crits[pick], "all": crits}, ensure_ascii=False, indent=1))
    df = c_world.rows(W, ["lb0_inh", "lb0_own"] + list(names.values()), names[pick])
    c_world.write(df, c_world.tab.REP / "C_L0.csv", "C-L0 контроль (trials 0): HP И2 / LB0 own × F_top3 / F_F4; база сравнения — %s" % names[pick])
    print("C-L0 crit_pen:", {k: round(v, 6) for k, v in crits.items()}, "-> база", names[pick], flush=True)
    assert np.isfinite(list(crits.values())).all()
