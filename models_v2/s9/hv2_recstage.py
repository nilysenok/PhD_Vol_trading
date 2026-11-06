"""Ж-214, лист 3, стадия `rec` (V11, ступень Л9): R2S, R2J, R2T — только для схем V7, прошедших V13 (CHECKPOINT.decisions.V7.passed_V13); иначе «не запускалось» с причиной (0 trials).
σ̂²_Σ схемы — лучшая по crit_wf КОМПОНЕНТНАЯ конфигурация схемы (S1; J1; T1/T3), прошедшая V13 к A0 индивидуально (У17). V13 для R2: лучше КАЖДОГО из A0 и SUM (простая сумма).
Схема, не превосходящая и сумму, и прямой прогноз, отклоняется (R3x не запускается)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s9 import hv2_a0, hv2_dec as DC, hv2_eval as E, hv2_ledger as L, hv2_paths as P, hv2_rec as R, hv2_world as W

DIR, SUB = "REC", "reconciliation"
MODS = (R, DC, W, hv2_a0)
COMP_CFG = {"S": ("S1",), "J": ("J1",), "T": ("T1", "T3")}
CTRL = ("A0", "SUM", "XJ", "LF")


def compute(name: str, cfg: str, w: DC.DWorld) -> dict:
    f, wd, s2, a0, rec, flag = R.run_r2(w, cfg)
    return {name: f, name + "__w": wd, name + "__sum": s2, name + "__flag": flag.astype(float), **{"%s__k%d" % (name, k): rec[:, k] for k in range(rec.shape[1])}}


def pick(ctx: dict, x: str, dec: dict) -> str | None:
    """Компонентная конфигурация схемы x, прошедшая V13 к A0 индивидуально, с лучшей crit_wf."""
    ser = pd.read_parquet(P.OUT / "DEC_series.parquet")
    ok = []
    for c in COMP_CFG[x]:
        r = E.row(ctx, c, ser[c].to_numpy(float), {}, ("A0",))
        if E.v13([r], ("A0",), 1, 1)["continue"]:
            ok.append((r["crit_wf"], c))
    return min(ok)[1] if ok else None


def weights_table(name: str, wd: np.ndarray, w: DC.DWorld) -> pd.DataFrame:
    yr = pd.DatetimeIndex(w.d).year.to_numpy()
    return pd.DataFrame([{"model": name, "year": int(Y), "mean_w": float(wd[yr == Y].mean()), "share_w0": float((wd[yr == Y] == 0).mean()),
                          "share_w1": float((wd[yr == Y] == 1).mean()), "n": int((yr == Y).sum())} for Y in sorted(set(yr))])


def stage_rec(_a=None) -> int:
    ctx = E.context()
    dec = json.loads(L.CHECK.read_text())["decisions"]["V7"]
    w = DC.build(ctx["w"], pd.read_parquet(P.M10))
    rows, wt, ident, res, skipped = [], [], [], {}, {}
    for x in "SJT":
        c = pick(ctx, x, dec) if x in dec["passed_V13"] else None
        if c is None:
            skipped[x] = "схема не прошла V13 (V7)" if x not in dec["passed_V13"] else "ни одна компонентная конфигурация не прошла V13 к A0 индивидуально"
            continue
        name = "R2" + x
        key = L.register("reconciliation", {"name": name, "kind": "geo_blend", "scheme": x, "base_cfg": c, "spec": "f_д^(1−w)·f_Σ^w, w по закрытым (>=252), компоненты пропорционально",
                                            "controls": ["A0", "SUM"]})
        ser = E.cached(DIR, name, MODS, lambda c=c, name=name: compute(name, c, w), ctx)
        f, wd, s2 = ser[name], ser[name + "__w"], ser[name + "__sum"]
        rec = np.column_stack([ser["%s__k%d" % (name, k)] for k in range(len(DC.COMPS[c]))])
        ctx["refs"]["SUM"] = s2
        tr = max(DC.trunc_diff(lambda y, c=c: R.sigma(y, c), w, int(w.el[k])) for k in E.TRUNC_AT)
        rows.append(E.row(ctx, name, f, {"kind": "geo_blend", "scheme": x, "base_cfg": c, "key": key, "trunc_max_diff": tr}, CTRL))
        v = E.v13([rows[-1]], ("A0", "SUM"), 1, L.LIMITS["reconciliation"])
        res[x] = {"cfg": c, "v13": v["controls"], "continue": v["continue"], "rejected": not v["continue"]}
        wt.append(weights_table(name, wd, w))
        tot = rec.sum(axis=1)
        ident.append({"model": name, "min_component": float(rec.min()), "max_rel_sum_err": float(np.max(np.abs(tot / f - 1))), "min_sigma2": float(f.min()),
                      "n_equal_split_or_fallback": int((ser[name + "__flag"] > 0).sum())})
    refs = [E.row(ctx, k, v, {"kind": "контроль"}, ("A0", "XJ", "LF")) for k, v in ctx["refs"].items() if k in ("A0", "XJ", "LF")]
    P.write_csv(pd.DataFrame(refs + rows), "RECON.csv", "V11, R2x; Δ — строка минус контроль (A0 прямой; SUM — простая сумма компонент схемы), минус — строка лучше; потолок: w по закрытым, схема выбрана по V13 на всей истории", SUB)
    if wt:
        P.write_csv(pd.concat(wt), "RECON_WEIGHTS.csv", "доля прогноза схемы w в геометрическом согласовании по годам (w выбирается по закрытым прогнозам)", SUB)
        P.write_csv(pd.DataFrame(ident), "RECON_IDENTITIES.csv", "положительность и сумма согласованных компонент (Σ = согласованная сумма)", SUB)
    E.decide("V11", {"by_scheme": res, "skipped": skipped, "trials_used": len(rows), "R3x": "не запускалось (резерв, только если R2x лучше суммы и прямого)", "rule": "отклонить схему, не превосходящую сумму и прямой прогноз"})
    for r in rows:
        print({k: r[k] for k in ("model", "qlike_all", "crit_wf", "d_crit_A0", "d_crit_SUM", "years_better_A0", "years_better_SUM", "trunc_max_diff")})
    print("skipped:", skipped, "| V13:", {x: (v["continue"], v["v13"]) for x, v in res.items()}, flush=True)
    return 0 if all(r["trunc_max_diff"] == 0 for r in rows) else 1
