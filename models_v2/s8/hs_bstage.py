"""Ж-212 (HS5, HS11, G7, G8): стадии `base` (воспроизведение XJ/LF новым конвейером) и `b` (HAR как признаки: минимальный набор Bf, Bc × XGB, LGBM = 4 trials;
затем правило HS11 «продолжать»: Br, Bg, Be и Btop2 ещё 8). Таблица HAR_FEATURES_B.csv; тест обрезки (G14) на каждой строке — 3 даты × ряд, переобучение."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from models_v2.s6 import paths6
from models_v2.s8 import hs_bfit, hs_ctx, hs_eval, hs_leak, hs_ledger as L, hs_paths as P, hs_store

DIR = "B"
MIN_SET = (("XGB", ("Bf",)), ("XGB", ("Bc",)), ("LGBM", ("Bf",)), ("LGBM", ("Bc",)))
REST = ("Br", "Bg", "Be")
CTRL_COL = {"XGB": "XJ", "LGBM": "LF"}


def cname(fam: str, blocks: tuple) -> str:
    return "B_%s_%s" % (fam.lower(), "+".join(blocks))


def leak_rows(ctx: dict) -> list:
    F = ctx["w"]["F"]
    return [int(np.searchsorted(F.dates.astype("datetime64[ns]"), np.datetime64(d, "ns"), side="right") - 1) for d in paths6.CONTROL_DATES]


def stage_base(_a=None) -> int:
    ctx = hs_ctx.build()
    w, df = ctx["w"], ctx["df"]
    rows = []
    xg = hs_bfit.xgb_run(w, None)
    lf, _, _ = hs_bfit.lgbm_run(w, None, own=False)
    for nm, s2, ref in (("XJ", xg, df["XJ"].to_numpy(float)), ("LF", lf, df["LF"].to_numpy(float))):
        rows.append({"model": nm, "max_rel_diff": float(np.max(np.abs(s2 / ref - 1))), "n": len(s2), "ok_1e9": bool(np.max(np.abs(s2 / ref - 1)) <= 1e-9)})
    lo, sel, _ = hs_bfit.lgbm_run(w, None, own=True)
    hs_store.put(DIR, "LF_own", lo)
    L.control("LF_own", "LGBM на базе Б-2+F4 с собственным годовым LOYO-выбором HP (6 точек): контроль эффекта выбора HP, не trial")
    t = pd.DataFrame(rows)
    t["lgbm_own_sel"] = str(sel)
    P.write_csv(t, "BASE_REPRO.csv", "G7: повтор XJ и LF конвейером Ж-212; max |σ̂²/σ̂²_замороженная − 1|; lgbm_own_sel — {год: (num_leaves, lr)} собственного выбора")
    print(t.to_string(index=False), flush=True)
    return 0 if t["ok_1e9"].all() else 1


def run_cfg(ctx: dict, fam: str, blocks: tuple, direction_key: str = "har_features") -> dict:
    """Одна конфигурация B: реестр -> счёт (или кэш) -> строка оценки + обрезка."""
    name, w, df = cname(fam, blocks), ctx["w"], ctx["df"]
    L.register(direction_key, {"model": fam, "blocks": list(blocks), "base": "Б-2+F4 (33)", "control": CTRL_COL[fam],
                               "hp": "hp_by_year X2" if fam == "XGB" else "LOYO 6 точек nl×lr, остальное C0_inh_F4"})
    extra, names = hs_ctx.extra_for(w["F"], ctx["lb"], ctx["beta"], ctx["ml"], fam, blocks)
    t0, hp_y, sel = time.time(), None, None
    s2 = hs_store.get(DIR, name)
    if s2 is None:
        if fam == "XGB":
            s2 = hs_bfit.xgb_run(w, extra)
        else:
            s2, sel, hp_y = hs_bfit.lgbm_run(w, extra, own=True)
        hs_store.put(DIR, name, s2)
    elif fam == "LGBM":
        sel = None
    own = hs_store.get(DIR, "LF_own")
    r = hs_eval.row(df, name, s2, df[CTRL_COL[fam]].to_numpy(float),
                    {"family": fam, "blocks": "+".join(blocks), "n_cols": len(names), "robust_gain": None, "secs": round(time.time() - t0, 1)})
    r["robust_gain"] = hs_eval.robust_gain(r)
    if fam == "LGBM" and own is not None:
        r["d_crit_vs_lf_own"] = r["crit_wf"] - hs_eval.crit_e.cw(np.asarray(hs_eval.metrics.qlike(df["rv"], own), float), df["target_date"])
    r["lgbm_sel"] = str(sel) if sel else ""
    r["trunc_max_diff"] = max(hs_leak.trunc_diff(ctx, fam, blocks, d, hp_y) for d in leak_rows(ctx))
    return r


def top2(rows: list, fam: str) -> tuple:
    """Btop2: два блока с лучшим crit_wf среди пяти (механический выбор, потолок)."""
    r = sorted((x for x in rows if x["family"] == fam and "+" not in x["blocks"]), key=lambda x: x["crit_wf"])[:2]
    return tuple(x["blocks"] for x in r)


def stage_b(_a=None) -> int:
    ctx = hs_ctx.build()
    rows = [run_cfg(ctx, f, b) for f, b in MIN_SET]
    dec = hs_eval.hs11(rows)
    L.decide("har_features_min", dec)
    print("HS11 B (минимальный набор):", dec, flush=True)
    if dec["continue"]:
        for f in ("XGB", "LGBM"):
            rows += [run_cfg(ctx, f, (b,)) for b in REST]
        rows += [run_cfg(ctx, f, top2(rows, f)) for f in ("XGB", "LGBM")]
        L.decide("har_features", {**dec, "stop": False, "unused_trials": L.LIMITS["har_features"] - len(rows)})
    else:
        L.decide("har_features", {**dec, "stop": True, "unused_trials": L.LIMITS["har_features"] - len(rows), "reason": "HS11: правило продолжения не выполнено"})
    t = pd.DataFrame(rows)
    t["ceiling"] = t["blocks"].str.contains(r"\+")
    P.write_csv(t, "HAR_FEATURES_B.csv", "B: блок(и) поверх Б-2+F4 (33); Δ к замороженному XJ (XGB) / LF (LGBM), минус — лучше; Btop2 — потолок (выбор по crit_wf на всей истории)")
    print(t[["model", "qlike_all", "crit_wf", "d_crit", "d_qlike", "years_better", "robust_gain", "trunc_max_diff"]].round(5).to_string(index=False), flush=True)
    return 0 if (t["trunc_max_diff"] == 0).all() else 1
