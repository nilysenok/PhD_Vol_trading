"""Стадии потока LGBM после LB0: loss (L1–L5), feat1 (F1–F6 по одной), feat2 (top2/top3, F-all), abl (обратные абляции)."""
from __future__ import annotations

import json

from models_v2.s5.b2 import engine as e2
from models_v2.s6.lgbm import cfg, tab
from models_v2.s6.lgbm import st_cfgs as sc
from models_v2.s6.lgbm.st_lb0 import OWN
from models_v2.s6.lgbm.world import DATA

REG = DATA / "feat_registry.json"
NOTE = "потолок: выбор и счёт на одних данных; trials obj_feat бюджета LGBM; Δ — к LB0 own"


def _reg() -> dict:
    return json.loads(REG.read_text()) if REG.exists() else {}


def _save_reg(upd: dict) -> None:
    r = _reg()
    r.update(upd)
    sc.dump_registry(REG, r)


def _done(names: list, path, note: str, W: dict) -> None:
    sc.table_rows(W, [OWN] + names, path, note)
    print("новых trials:", cfg.register(names), "CHECKPOINT:", tab.checkpoint()["used"])


def loss(W: dict) -> None:
    items = sc.objective_items(W)
    cfg.run_batch(W, items)
    _done([i.name for i in items], tab.REP / "OBJECTIVE_RESULTS.csv", "L1–L5 на LB0 own; L2 (ln-цель) = LB0 own; " + NOTE, W)


def feat1(W: dict) -> None:
    items = sc.single_items(W)
    cfg.run_batch(W, items)
    _save_reg({i.name: {"groups": [i.name[2:]], "drop": "", "role": "single"} for i in items})
    _done([i.name for i in items], tab.REP / "FEATURE_SETS_singles.csv", "F1–F6 по одной группе на LB0 own; " + NOTE, W)


def feat2(W: dict) -> None:
    singles = [n for n, v in _reg().items() if v["role"] == "single"]
    g2, g3, ga = sc.top_groups(W, singles, 2), sc.top_groups(W, singles, 3), tuple(sorted(W["acc"]))
    spec = {"F_top2": g2, "F_top3": g3, "F_all": ga}
    items = [sc.feat_item(W, n, g, desc="OWN + группы %s (%d столбцов)" % ("+".join(g), len(sc.group_cols(W, g)))) for n, g in spec.items()]
    cfg.run_batch(W, items)
    _save_reg({n: {"groups": list(g), "drop": "", "role": "combo" if n != "F_all" else "f_all"} for n, g in spec.items()})
    _done([i.name for i in items], tab.REP / "FEATURE_SETS_combos.csv", "top2/top3 по crit_pen одиночных + F-all; " + NOTE, W)


def abl(W: dict) -> None:
    reg = _reg()
    sets = [n for n, v in reg.items() if v["role"] in ("single", "combo", "f_all")]
    bf = cfg.best(sets)
    gr = tuple(reg[bf]["groups"])
    spec = {"A_%s_%s" % (k, bf): (gr, k, "обратная абляция: %s из %s" % (k, bf)) for k in e2.DROP}
    if len(gr) > 1:
        spec.update({"A_minus_%s_%s" % (g, bf): (tuple(x for x in gr if x != g), "", "%s без группы %s" % (bf, g)) for g in gr})
    items = [sc.feat_item(W, n, g, d, desc) for n, (g, d, desc) in spec.items()]
    cfg.run_batch(W, items)
    _save_reg({n: {"groups": list(g), "drop": d, "role": "ablation", "of": bf} for n, (g, d, _) in spec.items()})
    _done([i.name for i in items], tab.REP / "FEATURE_ABLATIONS.csv", "обратные абляции лучшего набора %s по crit_pen; %s" % (bf, NOTE), W)
    print("лучший набор для абляции:", bf)
