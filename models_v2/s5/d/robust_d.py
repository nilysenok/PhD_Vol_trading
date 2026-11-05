"""D: устойчивость финалистов — 5 seeds (42–46), соседи (бустеры C, top2, правило ансамбля), чувствительность к окну 1008. Не trials бюджета."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5 import common
from models_v2.s5.b1 import evalrows
from models_v2.s5.c import engine_c, mix, robust, tab_c
from models_v2.s5.c import stages as cstages
from models_v2.s5.d import ctx_d, engine_d, ens, stages_d

SEEDS, SEED_OK_MIN, PLATEAU_OK = robust.SEEDS, robust.SEED_OK_MIN, robust.PLATEAU_OK
FIN_C = ("c2_h3_roll_f4", "c1_opt", "c2_h3_roll_opt")
REF_ROW = "c3_f3calm_g4"          # справочная строка (не финалист)
ROLL = {"f4": "Xr", "opt": "opt_r", "top2": "top2_r", "L": "L_r", "Xh": "Xr"}


def finalists(best: str) -> list:
    return list(FIN_C) + [best, "b2_f4", "c3_f3calm_rv"]


def needs(rid: str, best: str) -> tuple:
    if rid == best:
        return tuple(dict.fromkeys({"X": "f4"}.get(m, m) for m in ens.SPEC[rid][0] if m != "H"))
    return {"b2_f4": ("f4",), "c1_opt": ("opt",)}.get(rid) or mix.NEEDS[rid]


def sigma(rid: str, best: str, ctx: dict, B: dict) -> np.ndarray:
    if rid == "b2_f4":
        return B["f4"]
    if rid == "c1_opt":
        return B["opt"]
    if rid == best:
        return ens.build(rid, ctx, {"X": B.get("f4"), "L": B.get("L"), "Xr": B.get("Xr"), "Xh": B.get("Xh")})
    return mix.build(rid, ctx, B)


def bundle(S, keys, seed: int = 42, roll: bool = False) -> dict:
    return {k: S.sig(ROLL.get(k, k) if roll else k, seed) for k in keys}


def seeds_table(S, ctx, ctrl, best, rids) -> pd.DataFrame:
    rv, h = ctx["rv"], ctx["h"]
    nf = ctx["td"].astype("datetime64[D]") != np.datetime64("2022-02-24")
    q2, qh = float(metrics.qlike(rv, ctrl["s2_b2"].to_numpy(float)).mean()), metrics.qlike(rv, h)
    rec = []
    for r in rids:
        for s in SEEDS:
            q = metrics.qlike(rv, sigma(r, best, ctx, bundle(S, needs(r, best), s)))
            rec.append({"row_id": r, "seed": s, "qlike_all": float(q.mean()), "d_harj_no_240222": float((q - qh)[nf].mean())})
    df = pd.DataFrame(rec)
    df["ok"] = (df["qlike_all"] < q2) & (df["d_harj_no_240222"] <= 0)
    agg = df.groupby("row_id").agg(mean=("qlike_all", "mean"), std=("qlike_all", "std"), min=("qlike_all", "min"), max=("qlike_all", "max"), n_ok=("ok", "sum")).reset_index()
    agg["seed_ok"] = agg["n_ok"] >= SEED_OK_MIN
    return df.merge(agg, on="row_id")


def window_table(S, ctx, ctrl, best, rids) -> pd.DataFrame:
    """Диагностика (не trials): то же правило на бустерах rolling 1008 вместо expanding."""
    rv, y = ctx["rv"], pd.Series(ctrl["target_date"]).dt.year.to_numpy()
    q2 = float(metrics.qlike(rv, ctrl["s2_b2"].to_numpy(float)).mean())
    rec = []
    for r in rids:
        qe, qr = (metrics.qlike(rv, sigma(r, best, ctx, bundle(S, needs(r, best), 42, roll))) for roll in (False, True))
        rec.append({"row_id": r, "qlike_expanding": float(qe.mean()), "qlike_rolling_1008": float(qr.mean()), "d_roll_minus_exp": float((qr - qe).mean()),
                    "d_roll_minus_exp_2017_2021": float((qr - qe)[y <= 2021].mean()), "d_roll_minus_exp_2022_2026": float((qr - qe)[y >= 2022].mean()),
                    "rolling_better_b2": bool(qr.mean() < q2)})
    return pd.DataFrame(rec)


def ens_neighbors(best: str, ctx: dict, M: dict, q_b2: float) -> pd.DataFrame:
    """Соседи правила ансамбля: окно 126/504 (roll, inv) и шаг сетки 0,05 (кроме inv); Q_wf соседа против Q_wf Б-2."""
    rule = ens.SPEC[best][1]
    nb = ([("win", w, {"win": w}) for w in (126, 504)] if rule in ("roll", "inv") else []) + ([("step", 0.05, {"step": 0.05})] if rule != "inv" else [])
    rec = []
    for ax, v, kw in nb:
        s2 = ens.build(best, ctx, M, **kw)
        qw = evalrows.q_wf(metrics.qlike(ctx["rv"], s2), pd.Series(ctx["td"]))
        rec.append({"unit": best, "kind": "ensemble_rule", "axis": ax, "dir": v, "crit": qw, "d_b2": qw - q_b2, "better_b2": bool(qw < q_b2)})
    return pd.DataFrame(rec)


def plateau_flags(pl: pd.DataFrame, rid: str, best: str, share_ens: float) -> bool:
    """plateau-ok строки: все её XGB-бустеры (f4/opt/top2) plateau-ok по C/D; для ансамбля — ещё доля соседей правила лучше Б-2 >= 0,75."""
    ok = pl[pl["kind"].str.startswith("booster")].groupby("unit")["plateau_ok"].first().to_dict()
    if rid == best:
        need = ["f4"] if any(m in ("X", "Xr", "Xh") for m in ens.SPEC[rid][0]) else []   # Xr/Xh/L наследуют HP XGB-И2 (↺ в GATES_D)
        if not (share_ens >= PLATEAU_OK):
            return False
    else:
        need = [k for k in needs(rid, best) if k in ("f4", "opt", "top2")]
    return bool(all(ok.get(b, False) for b in need))


def plateau_table(C, best, ctx, M) -> tuple:
    cp = pd.read_csv(common.REP / "C_PLATEAU.csv", comment="#")
    cp = cp.assign(unit=cp["booster"], kind="booster_C")[["unit", "kind", "axis", "dir", "crit", "d_b2", "better_b2", "plateau_ok"]]
    f = ctx_d.DATA / "crit_top2.json"
    with engine_c.pool(C["w"]) as pc:
        c0 = ctx_d.load_json(f.name, None) or engine_c.crit(engine_c.scores(pc, "top2", C["i2"]))
        ctx_d.save_json(f.name, c0)
        t2 = robust.plateau_table(pc, C["w"], ["top2"], C["i2"], cstages.chosen_hp(), {"top2": c0}, cstages.crit_b2())
    t2 = t2.assign(unit=t2["booster"], kind="booster_D")[["unit", "kind", "axis", "dir", "crit", "d_b2", "better_b2", "plateau_ok"]]
    en = ens_neighbors(best, ctx, M, tab_c.q_int_b2("wf", C["ctrl"], 0.0))
    share = float(en["better_b2"].mean()) if len(en) else float("nan")
    en = en.assign(share_better_b2=share, plateau_ok=bool(share >= PLATEAU_OK))
    return pd.concat([cp, t2, en], ignore_index=True), share


def run(C) -> None:
    ctrl, best = C["ctrl"], ctx_d.load_json("best.json", {})["best"]
    ctx = mix.context(C["w"].F, C["w"].grid, ctrl)
    rids = finalists(best) + [REF_ROW]
    with engine_d.pool(C["w"]) as pd_:
        S = stages_d.Src(C, pd_)
        sd, wn = seeds_table(S, ctx, ctrl, best, rids), window_table(S, ctx, ctrl, best, rids)
    M = stages_d.Src(C).members()
    pl, share = plateau_table(C, best, ctx, M)
    cs = pd.read_csv(common.REP / "C_SEEDS.csv", comment="#")
    mg = sd.merge(cs, on=["row_id", "seed"], suffixes=("", "_c"))
    dev = float((mg["qlike_all"] - mg["qlike_all_c"]).abs().max()) if len(mg) else float("nan")
    ctx_d.write(sd, common.REP / "D_SEEDS.csv", "5 seeds 42–46 финалистов и справочной c3_f3calm_g4; seed-ok: >= 4 из 5")
    ctx_d.write(pl, common.REP / "D_PLATEAU.csv", "соседи: бустеры C (C_PLATEAU), top2 (D), правило ансамбля (D)")
    ctx_d.write(wn, common.REP / "D_WINDOW.csv", "чувствительность к окну: expanding против rolling 1008 (диагностика, не trials)")
    fl = {r: {"seed_ok": bool(sd[sd.row_id == r]["seed_ok"].iloc[0]), "n_ok": int(sd[sd.row_id == r]["n_ok"].iloc[0]), "plateau_ok": plateau_flags(pl, r, best, share)}
          for r in rids}
    ctx_d.save_json("flags.json", {"flags": fl, "c_seeds_max_dev": dev, "ens_share": share})
    print("G8: флаги", fl, "| отклонение от C_SEEDS (общие строки) %.3e" % dev)
