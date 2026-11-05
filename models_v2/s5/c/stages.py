"""C: стадии c1 / c2 / c3 (строки, trials, таблицы). Ничего не считается без команды run_c.py."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s5 import common
from models_v2.s5.b1 import evalrows
from models_v2.s5.b2 import tab as b2tab, world as b2w
from models_v2.s5.c import ctx_c, engine_c, h2l4, mix, space, tab_c, tpe

FEAT, HAR, REG = "feature_engineering", "har_boosting", "regime_aware"


def chosen_hp() -> dict:
    return {int(Y): tpe.hp_of(v["hp"]) for Y, v in ctx_c.load_json("chosen.json", {}).items()}


def crit_b2() -> float:
    return float(json.loads((b2w.OUT / "st_BASE.json").read_text())["crit_loyo"])


def daily(C, p, name, hp_y, seed=42):
    return ctx_c.cached_daily(p, C["w"], name, hp_y, seed, C["L"]["key"])


def make_row(C, p, rid, fc, kind, q, opt, dirn, cfg, inner, desc, extra=None, secs=0.0, trial=True):
    tid = space.trial_id(cfg)
    f4 = daily(C, p, "f4", C["i2"])[0]
    r = tab_c.build_row(rid, fc, C["ctrl"], f4, kind, q, crit_b2(), opt, dirn, tid, extra)
    desc = desc or "qlike_all=%.6f;d_harj_no_240222=%.6f;d_b2=%.6f" % (r["qlike_all"], r["d_harj_no_240222"], r["d_b2"])
    ctx_c.put_row(rid, r, tab_c.trial_row(tid, dirn, cfg, inner, desc, secs) if trial else None)   # c1_opt — строка-процедура, не trial
    return r


def ext(st) -> dict:
    return {"loss_train": st["mse_train"], "loss_es": st["mse_block"], "loss_ratio": st["mse_block"] / st["mse_train"], "trees_mean": st["trees"]}


def stage_c1(C, p) -> None:
    w, i2, chosen = C["w"], C["i2"], chosen_hp()
    ch = ctx_c.load_json("chosen.json", {})
    feat_rows = []
    specs = [("c1_opt", "f4", chosen, float(np.mean([v["score"] for v in ch.values()])), True, {"kind": "c1_opt", "seed": 42, "hp": "TPE22+I2 по годам"}),
             ("c1_f4_range", "f4_range", i2, None, False, {"kind": "c1_f4_range", "add": C["cols"]["f4_range"], "hp": "I2", "seed": 42}),
             ("c1_f4_park", "f4_park", i2, None, False, {"kind": "c1_f4_park", "add": C["cols"]["f4_park"], "hp": "I2", "seed": 42}),
             ("c1_f4_f1", "f4_f1", i2, None, False, {"kind": "c1_f4_f1", "add": C["cols"]["f4_f1"], "hp": "I2", "seed": 42})]
    for rid, name, hp_y, q, opt, cfg in specs:
        fc, st = daily(C, p, name, hp_y)
        if q is None:
            q = engine_c.crit(engine_c.scores(p, name, hp_y))
        make_row(C, p, rid, fc, "loyo", q, opt, FEAT, cfg, "crit_loyo=%.6f" % q, "", ext(st), st["secs"], trial=rid != "c1_opt")
        r = ctx_c.load_json("rows.json", {})[rid]
        har, b2 = b2w.controls_frames()
        meta = {"trial_id": r["trial_id"], "direction": FEAT, "crit_loyo": q, "crit_loyo_b2": crit_b2(), "crit_wf_b2": b2tab.crit_wf(b2w.get(C["L"], b2w.BASE)[0])}
        feat_rows.append(b2tab.row(rid, fc, har, b2, st, meta))
    tab_c.append_table(common.REP / "FEATURE_ABLATIONS.csv", pd.DataFrame(feat_rows).rename(columns={}), "model")
    tpe_trials(C, p)


def tpe_trials(C, p) -> None:
    """22 trials Optuna в реестр: внутренние метрики; описание на 2271 датах — только для 3 лучших по J (θ постоянен по годам)."""
    done = ctx_c.load_json("tpe_cache.json", [])
    best = {d["k"] for d in sorted(done, key=lambda d: d["J"])[:3]}
    desc_rows = []
    for d in done:
        cfg = {"kind": "tpe", "k": d["k"], "params": d["params"], "features": "b2_f4", "study_seed": space.STUDY_SEED, "n_startup": space.N_STARTUP}
        tid, ds = space.trial_id(cfg), "описание: не считалось (стоимость; GATES_C §2)"
        if d["k"] in best:
            fc, _ = daily(C, p, "f4", {Y: tpe.hp_of(d["params"]) for Y in space.ALL_YEARS})
            q = tab_c.qlike_of(fc)
            y = fc["target_date"].dt.year.to_numpy()
            qh = tab_c.qlike_of(pd.DataFrame({"rv": C["ctrl"]["rv"], "sigma2": C["ctrl"]["s2_harj"]}))
            nf = (fc["target_date"] != pd.Timestamp("2022-02-24")).to_numpy()
            desc_rows.append({"k": d["k"], "J": d["J"], "qlike_all": q.mean(), "qlike_2017_2021": q[y <= 2021].mean(), "qlike_2022_2026": q[y >= 2022].mean(),
                              "d_harj_no_240222": (q - qh)[nf].mean()})
            ds = ";".join("%s=%.6f" % (k, v) for k, v in desc_rows[-1].items() if k != "k")
        inner = "J_2017_2021=%.6f;crit_all_years=%.6f" % (d["J"], np.mean(list(d["scores"].values())))
        tr = ctx_c.load_json("trials_rows.json", {})
        tr[tid] = tab_c.trial_row(tid, FEAT, cfg, inner, ds, 0.0)
        ctx_c.save_json("trials_rows.json", tr)
    pd.DataFrame(desc_rows).to_csv(common.REP / "C_TPE_DESC.csv", index=False)


def boosters(C, p, need) -> dict:
    out = {}
    for b in need:
        hp = chosen_hp() if b == "opt" else C["i2"]
        out[b] = daily(C, p, "f4" if b == "opt" else b, hp)[0]["sigma2"].to_numpy(float)
    return out


def stage_mix(C, p, rows_ids, direction: str) -> None:
    ctx = mix.context(C["w"].F, C["w"].grid, C["ctrl"])
    ctrl = C["ctrl"]
    for rid in rows_ids:
        b = boosters(C, p, mix.NEEDS[rid])
        fc = pd.DataFrame({"d": ctrl["d"], "target_date": ctrl["target_date"], "rv": ctrl["rv"], "sigma2": mix.build(rid, ctx, b)})
        q = evalrows.q_wf(tab_c.qlike_of(fc), fc["target_date"])
        make_row(C, p, rid, fc, "wf", q, rid.endswith("_opt"), direction, {"kind": rid, "needs": mix.NEEDS[rid], "seed": 42}, "q_wf=%.6f" % q, "")


def stage_c2(C, p) -> None:
    stage_mix(C, p, mix.C2_H3, HAR)


def stage_h2l4(C) -> None:
    """Вне пула C (процессов <= 6): b1.jobs открывает свой пул; кэш f4 уже посчитан в c1 (p не нужен)."""
    fc, ql, st = h2l4.run()
    make_row(C, None, h2l4.NAME + "_c05", fc, "loyo", ql, False, HAR, {"kind": "c2_h2l4_f4_c05", "c": 0.5, "seed": 42}, "crit_loyo=%.6f" % ql, "", st)


def stage_c3(C, p) -> None:
    stage_mix(C, p, mix.C3, REG)
