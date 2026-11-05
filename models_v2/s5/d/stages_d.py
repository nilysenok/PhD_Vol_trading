"""D: источники σ² (члены ансамблей и виды окон), стадии control / comp / d1. Ничего не считается без команды run_d.py."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5 import common
from models_v2.s5.b1 import evalrows
from models_v2.s5.b2 import world as b2w
from models_v2.s5.c import mix, space, stages, tab_c
from models_v2.s5.d import ctx_d, ens, select_d, tab_d

# ключ -> (дизайн, HP, lib, окно, полураспад)
SPEC_M = {"f4": ("f4", "i2", "xgb", 0, 0), "opt": ("f4", "opt", "xgb", 0, 0), "top2": ("top2", "i2", "xgb", 0, 0),
          "L": ("f4", "i2", "lgbm", 0, 0), "Xr": ("f4", "i2", "xgb", 1008, 0), "Xh": ("f4", "i2", "xgb", 0, 504),
          "opt_r": ("f4", "opt", "xgb", 1008, 0), "top2_r": ("top2", "i2", "xgb", 1008, 0), "L_r": ("f4", "i2", "lgbm", 1008, 0)}
TAG = {"L": "lgbm_f4", "Xr": "xgb_f4_r1008", "Xh": "xgb_f4_hl504"}
COMP_ID = {"L": tab_d.COMPONENTS[0], "Xr": tab_d.COMPONENTS[1], "Xh": tab_d.COMPONENTS[2]}


class Src:
    """σ² источника по ключу SPEC_M и seed. Базовые (f4/opt/top2, seed 42) — из кэша C; остальное — engine_d с кэшем."""

    def __init__(self, C: dict, p=None):
        self.C, self.p = C, p

    def frame(self, key: str, seed: int = 42) -> tuple:
        nm, hk, lib, win, hl = SPEC_M[key]
        hp = self.C["i2"] if hk == "i2" else stages.chosen_hp()
        if seed == 42 and lib == "xgb" and not win and not hl:
            return stages.daily(self.C, None, nm, hp, 42)
        return ctx_d.cached(self.p, self.C, TAG.get(key, key), nm, hp, seed, lib, win, float(hl))

    def sig(self, key: str, seed: int = 42) -> np.ndarray:
        return self.frame(key, seed)[0]["sigma2"].to_numpy(float)

    def members(self, seed: int = 42, roll: bool = False) -> dict:
        r = {"f4": "Xr", "opt": "opt_r", "top2": "top2_r", "L": "L_r", "Xh": "Xr"} if roll else {}
        return {k: self.sig(r.get(k, k), seed) for k in ("L", "Xr", "Xh")} | {"X": self.sig(r.get("f4", "f4"), seed)}


def stage_control(C, p=None):
    ctrl = C["ctrl"]
    qb, qh = (float(metrics.qlike(ctrl["rv"], ctrl[c]).mean()) for c in ("s2_b2", "s2_harj"))
    assert (ctrl["rv"] > 0).all() and len(ctrl) == common.N_GRID
    res = {n: bool(np.array_equal(stages.daily(C, None, n, C["i2"])[0]["sigma2"].to_numpy(), pd.read_parquet(b2w.OUT / ("fc_b2_%s.parquet" % n))["sigma2"].to_numpy()))
           for n in ("f4", "top2")}
    ok = abs(qb - 0.220486502764) < 1e-12 and abs(qh - 0.233662115694) < 1e-12 and all(res.values())
    call = pd.read_csv(common.REP / "C_ALL.csv", comment="#").set_index("row_id")
    ctx = mix.context(C["w"].F, C["w"].grid, ctrl)
    b = stages.boosters(C, None, ("f4", "opt", "top2"))
    dev = {r: float(abs(metrics.qlike(ctrl["rv"].to_numpy(float), mix.build(r, ctx, b)).mean() - call.loc[r, "qlike_all"])) for r in
           ("c2_h3_roll_f4", "c2_h3_roll_opt", "c3_f3calm_rv", "c3_f3calm_g4")}
    dev["c1_opt"] = float(abs(metrics.qlike(ctrl["rv"].to_numpy(float), b["opt"]).mean() - call.loc["c1_opt", "qlike_all"]))
    print("G1: Б-2 %.12f HAR-J(б) %.12f; побитно к кэшу B2: %s; пересборка C к C_ALL: %s" % (qb, qh, res, dev))
    ctx_d.save_json("control.json", {"b2": qb, "harj": qh, "bitwise": res, "c_rebuild_dev": dev})
    if not ok or max(dev.values()) > 1e-12:
        sys.exit("G1 КРАСНЫЙ: контроль или пересборка C не воспроизведены — стоп и доклад")


def stage_comp(C, p):
    S, res = Src(C, p), {}
    for k in ("L", "Xr", "Xh"):
        fc, st = S.frame(k)
        q = float(metrics.qlike(fc["rv"].to_numpy(float), fc["sigma2"].to_numpy(float)).mean())
        assert len(fc) == common.N_GRID and (fc["sigma2"] > 0).all() and np.isfinite(fc["sigma2"]).all()
        res[COMP_ID[k]] = {"qlike": q, "secs": st["secs"], "trees": st["trees"]}
        print("G6", COMP_ID[k], "QLIKE %.6f, деревьев %.1f" % (q, st["trees"]), flush=True)
    ctx_d.save_json("comp.json", res)


def make_row(C, rid, fc, cfg, extra, trial=True, secs=0.0):
    q, td = tab_c.qlike_of(fc), fc["target_date"]
    qw, dev = evalrows.q_wf(q, td), select_d.q_wf_dev(q, td)
    tid = space.trial_id(cfg) if trial else ""
    f4 = stages.daily(C, None, "f4", C["i2"])[0]
    r = tab_c.build_row(rid, fc, C["ctrl"], f4, "wf", qw, stages.crit_b2(), False, tab_d.DIRECTION, tid, {**extra, "q_wf_dev": dev})
    desc = "qlike_all=%.6f;d_harj_no_240222=%.6f;d_b2=%.6f" % (r["qlike_all"], r["d_harj_no_240222"], r["d_b2"])
    ctx_d.put_row(rid, r, tab_d.trial_row(tid, cfg, "q_wf=%.6f;q_wf_dev=%.6f" % (qw, dev), desc) if trial else None)
    return r


def pick_best(df: pd.DataFrame) -> tuple:
    """(лучший по внутреннему критерию q_int, лучший по q_wf_dev); не по описанию qlike_all (MD11)."""
    return select_d.best_ensemble(df, "q_int"), select_d.best_ensemble(df, "q_wf_dev")


def stage_d1(C, p):
    S, ctrl = Src(C, p), C["ctrl"]
    ctx = mix.context(C["w"].F, C["w"].grid, ctrl)
    M = S.members()
    base = {"d": ctrl["d"], "target_date": ctrl["target_date"], "rv": ctrl["rv"]}
    for k in ("L", "Xr", "Xh"):
        fc, st = S.frame(k)
        make_row(C, COMP_ID[k], fc, {"kind": COMP_ID[k], "lib": SPEC_M[k][2], "win": SPEC_M[k][3], "hl": SPEC_M[k][4], "seed": 42},
                 {"loss_train": st["mse_train"], "loss_es": st["mse_block"], "trees_mean": st["trees"]}, secs=st["secs"])
    for rid, (mem, rule) in ens.SPEC.items():
        fc = pd.DataFrame({**base, "sigma2": ens.build(rid, ctx, M)})
        make_row(C, rid, fc, {"kind": rid, "members": mem, "rule": rule, "seed": 42, "step": 0.1, "win": ens.WIN}, {"members": "+".join(mem), "rule": rule},
                 trial=rid in ens.TRIALS)
    rows = ctx_d.load_json("rows.json", {})
    df = pd.DataFrame(rows.values())
    best, best_dev = pick_best(df)
    ctx_d.save_json("best.json", {"best": best, "best_dev": best_dev, "differs": best != best_dev})
    print("G7: лучший по Q_wf: %s; по Q_wf_dev: %s" % (best, best_dev))
    write_ensemble_tables(df)


def write_ensemble_tables(df: pd.DataFrame) -> None:
    tr = list(ctx_d.load_json("trials_rows.json", {}).values())
    tab_d.check_budget(tr, 0)
    pd.DataFrame(tr).to_csv(tab_d.TRIALS_D, index=False)
    d = df.assign(source="D")
    d = tab_c.holm_cols(tab_c.holm_cols(d, tab_d.FAMILY), tab_d.FAMILY_REF, "_ref116")
    c = pd.read_csv(common.REP / "C_ALL.csv", comment="#")
    ref = c[c["row_id"].isin(("c2_h3_roll_f4", "c2_h3_exp_f4"))].assign(source="C (E2, ссылка; не trials D)")
    ctx_d.write(pd.concat([d, ref], ignore_index=True), common.REP / "ENSEMBLE_RESULTS.csv",
                "Холм: N = %d (все trials программы), справочно N = %d (+10 контролей); строки E2 — из C_ALL" % (tab_d.FAMILY, tab_d.FAMILY_REF))
