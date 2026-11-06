"""Волна D, поток LGBM: стадии win (D1), pick (окно по прошлому), freq (D2). Правила — GATES_D_lgbm.md (RD0–RD2), объявлены до счёта."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s4 import runner, select
from models_v2.s6 import paths6
from models_v2.s6.lgbm import c_l0, d_comp, d_world, grid, job
from models_v2.s6.lgbm.d_win import Tm

WINS = {"D_w252": Tm(win=252), "D_w504": Tm(win=504), "D_w756": Tm(win=756), "D_w1008": Tm(win=1008), "D_w1260": Tm(win=1260),
        "D_dec756": Tm(hl=756), "D_dec1260": Tm(hl=1260)}
CAND = ("D_w0",) + tuple(WINS)
FREQ = {"wk": "w", "mo": "m", "adapt": "adapt"}
THR = 0.0005
REP = d_world.tab.REP


def base_cfg(W: dict) -> tuple:
    name, spec, hp = c_l0.base_cfg(W)
    if name != d_world.BASE:
        raise ValueError("база C изменилась: %s" % name)
    return spec, hp


def win_items(W: dict) -> list:
    spec, hp = base_cfg(W)
    its = [d_world.DItem("D_w0", spec, hp, Tm(), "control", "контроль D: расширяющееся окно, ежедневно (пересчёт базы C0_inh_F4 кодом D)")]
    for n, tm in WINS.items():
        d = "rolling %d последних пар" % tm.win if tm.win else "экспоненциальные веса, полураспад %d строк" % tm.hl
        its.append(d_world.DItem(n, spec, hp, tm, "temporal", "D1: %s; база C0_inh_F4 (HP И2 + Б-2 + F4)" % d))
    return its


def parity(names=("D_w0",)) -> None:
    """D_w0 (код D) обязана совпасть с C0_inh_F4 (код C): иначе фолды/прогнозы D несопоставимы с базой."""
    fb, sb = d_world.get(d_world.BASE)
    for n in names:
        f0, s0 = d_world.get(n)
        rel = float(np.max(np.abs(f0["sigma2"].to_numpy() / fb["sigma2"].to_numpy() - 1)))
        dc = abs(s0["crit"]["crit_pen"] - sb["crit"]["crit_pen"])
        print("паритет %s: max отн. разность σ̂² %.3g, |Δcrit_pen| %.3g" % (n, rel, dc), flush=True)
        if rel > 1e-9 or dc > 1e-9:
            raise ValueError("паритет базы нарушен: %s" % n)


def pairs_table(W: dict, spec) -> pd.DataFrame:
    F, rec = W["F"], []
    cfgs = [("expanding", Tm())] + list(WINS.items())
    for Y in paths6.YEARS:
        T = select.cut_row(F, Y)
        idx = job.pairs(F, spec, T)
        lens = [len(runner.fold_split(F, spec, T, y)[0]) for y in range(select.FIRST_Y, Y)]
        for n, tm in cfgs:
            eff = len(idx) if not (tm.win or tm.hl) else (min(tm.win, len(idx)) if tm.win else float((0.5 ** ((T - idx) / tm.hl)).sum()))
            rec.append({"Y": Y, "config": n, "pairs_at_cut": len(idx), "pairs_used_or_effective": eff,
                        "share_folds_window_binding": float(np.mean([ln > tm.win for ln in lens])) if tm.win else 0.0, "n_folds": len(lens)})
    return pd.DataFrame(rec)


def stage_win(W: dict) -> None:
    its = win_items(W)
    d_world.run_batch(W, its)
    parity()
    added = d_world.register([i.name for i in its if i.direction == "temporal"])
    c = d_world.checkpoint()
    d_world.write(pairs_table(W, its[0].spec), REP / "D_PAIRS_BY_YEAR.csv", "доступность пар к 01.01.Y и доля фолдов, где окно ограничивает обучение")
    print("trials добавлено:", added, c["used_stream_d"], flush=True)
    for n in CAND:
        print(n, "crit_pen %.6f" % d_world.get(n)[1]["crit"]["crit_pen"], flush=True)


def stage_pick(W: dict) -> None:
    fcs = {n: d_world.get(n)[0] for n in CAND}
    pens = pd.DataFrame({n: {int(Y): v for Y, v in d_world.get(n)[1]["crit"]["by_year"].items()} for n in CAND}).sort_index()
    pick = d_comp.pick_by_year(pens)
    fc = d_comp.compose_by_year(fcs, pick)
    chosen = float(np.mean([pens.loc[Y, n] for Y, n in pick.items()]))
    st0 = d_world.get("D_w0")[1]
    tid = grid.trial_id("derived", {"rule": "pick_window"}, [], "", {"cands": list(CAND)})
    d_world.save("D_pick_window", fc, {k: v for k, v in st0.items() if k not in ("key", "crit")},
                 dict(key="derived", trial_id=tid, crit={"crit_pen": chosen, "crit_mean": float("nan"), "worst_fold": float("nan"), "by_year": {}},
                      hp_by_year={str(Y): n for Y, n in pick.items()}, tm={"pick": "по году"}, direction="temporal", secs_total=0.0,
                      desc="D1 окно по прошлому: для года Y окно из 8 с min pen_Y (цели < 01.01.Y); потолок: min по кандидатам, выбор и оценка на тех же фолдах"))
    print("выбор окна по годам:", pick, "crit_pen %.6f" % chosen, "added", d_world.register(["D_pick_window"]), d_world.checkpoint()["used_stream_d"], flush=True)


def best_window() -> str:
    cp = {n: d_world.get(n)[1]["crit"]["crit_pen"] for n in CAND}
    return min(cp, key=lambda k: (cp[k], CAND.index(k)))


def freq_items(W: dict) -> list:
    spec, hp = base_cfg(W)
    best = best_window()
    tm = Tm() if best == "D_w0" else WINS[best]
    return [d_world.DItem("%s_%s" % (best, s), spec, hp, Tm(tm.win, tm.hl, f), "temporal", "D2 частота %s на окне %s; фолды LOYO скопированы с ежедневной" % (f, best),
                          crit_from=best) for s, f in FREQ.items()]


def stage_freq(W: dict) -> None:
    its = freq_items(W)
    best = best_window()
    d_world.run_batch(W, its)
    added = d_world.register([i.name for i in its])
    names = [best] + [i.name for i in its]
    df = d_world.rows(W, names)
    cw = dict(zip(("d", "w", "m", "adapt"), df["crit_wf"]))
    df["secs_train"], df["n_trainings"] = [d_world.get(n)[1]["secs"] for n in names], [d_world.get(n)[1]["n_fits"] for n in names]
    df["verdict_rule_RD2"] = d_comp.frequency_verdict(cw["d"], cw["w"], cw["m"], THR)
    d_world.write(df[["model", "crit_pen", "crit_wf", "d_crit_wf_vs_base", "qlike_all", "qlike_p2017_2021", "qlike_y2022", "qlike_p2023_2025", "qlike_y2026",
                      "qlike_calm", "qlike_stress", "dloss_harj_no_20220224", "secs_train", "n_trainings", "verdict_rule_RD2", "status"]],
                  REP / "D_FREQUENCY.csv", "D2 частота на окне %s; порог RD2 %.4f по crit_wf; фолдовый crit_pen частоте нечувствителен (скопирован)" % (best, THR))
    print("best window", best, "crit_wf", {k: round(v, 6) for k, v in cw.items()}, "вердикт", df["verdict_rule_RD2"].iloc[0], "added", added,
          d_world.checkpoint()["used_stream_d"], flush=True)
