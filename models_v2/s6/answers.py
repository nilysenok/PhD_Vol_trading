"""Ответ (в): пересчёт сырых p и p Холма для c2_h3_roll_f4 из сохранённых прогнозов (волна C, N=90; волна D, N=106). Таблица — ANSWERS_V.csv."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s3 import dm as dmod
from models_v2.s5.b1 import evalrows
from models_v2.s5.c import ctx_c, mix
from models_v2.s5.c import stages as cst
from models_v2.s5.d import d2_d, ctx_d, robust_d, stats_d, tab_d
from models_v2.s6 import paths6

TARGET = "c2_h3_roll_f4"


def wave_c(C) -> dict:
    ctx = mix.context(C["w"].F, C["w"].grid, C["ctrl"])
    b = cst.boosters(C, None, ("f4", "opt", "top2"))
    call = paths6.read_csv(paths6.OLD_REP / "C_ALL.csv").set_index("row_id")
    rv, s2b = ctx["rv"], C["ctrl"]["s2_b2"].to_numpy(float)
    mk = evalrows.masks(C["ctrl"]["target_date"])["all"]
    p, src = {}, {}
    for rid in call.index:
        if rid.startswith("c1_") or rid == "c2_h2l4_f4_c05":
            sg = {"c1_opt": b["opt"]}.get(rid)
            if sg is None:
                p[rid], src[rid] = float(call.loc[rid, "p_b2"]), "таблица C_ALL (прогноз не пересобран)"
                continue
        else:
            sg = mix.build(rid, ctx, b)
        p[rid], src[rid] = evalrows.vs(rv, s2b, sg, mk)["p"], "пересчёт DM из прогнозов"
    ids = list(call.index)
    raw = np.array([p[i] for i in ids])
    holm = dmod.holm(np.concatenate([raw, np.ones(90 - len(raw))]))[:len(raw)]
    return {"ids": ids, "raw": raw, "holm": holm, "src": src, "table_raw": call["p_b2"].to_numpy(float), "table_holm": call["p_holm_b2"].to_numpy(float)}


def wave_d(C) -> dict:
    best = ctx_d.load_json("best.json", {})["best"]
    ctx, sig = d2_d.sigmas(C, best)
    rids = robust_d.finalists(best)
    rv, td = ctx["rv"], C["ctrl"]["target_date"]
    idx = stats_d.block_idx(len(rv))
    out = {"best": best, "rids": rids, "dm": [], "boot": []}
    for r in rids:
        out["dm"].append(stats_d.dm_periods(rv, sig["B2"], sig[r], td)[0]["p"])
        out["boot"].append(stats_d.boot_delta(common_q(rv, sig[r]) - common_q(rv, sig["B2"]), idx)["p_boot"])
    for k in ("dm", "boot"):
        out[k + "_holm"] = stats_d.holm_n(np.array(out[k]), tab_d.FAMILY)
    st = paths6.read_csv(paths6.OLD_REP / "STATISTICAL_TESTS.csv")
    for k, t in (("dm", "DM"), ("boot", "boot")):
        s = st[(st.test == t) & (st.base == "Б-2") & (st.period == "весь период")].set_index("row_id")
        out[k + "_table"] = [(float(s.loc[r, "p"]), float(s.loc[r, "p_holm_N106"])) for r in rids]
    return out


def common_q(rv, s2):
    from models_v2 import metrics
    return metrics.qlike(rv, s2)


def stage(a) -> int:
    C = ctx_c.load()
    c, d = wave_c(C), wave_d(C)
    i = c["ids"].index(TARGET)
    j = d["rids"].index(TARGET)
    rows = [{"wave": "C", "n_family": 90, "n_tests": len(c["ids"]), "padding_ones": 90 - len(c["ids"]), "test": "DM Ньюи–Уэст лаг 5", "base": "Б-2", "row": TARGET, "raw_p": c["raw"][i], "holm_p": c["holm"][i],
             "table_raw": c["table_raw"][i], "table_holm": c["table_holm"][i], "diff_holm": abs(c["holm"][i] - c["table_holm"][i])}]
    for k, lab in (("dm", "DM Ньюи–Уэст лаг 5"), ("boot", "блочный бутстрап (блок 20, 5000, seed 42)")):
        rows.append({"wave": "D", "n_family": tab_d.FAMILY, "n_tests": len(d["rids"]), "padding_ones": tab_d.FAMILY - len(d["rids"]), "test": lab, "base": "Б-2", "row": TARGET,
                     "raw_p": d[k][j], "holm_p": d[k + "_holm"][j], "table_raw": d[k + "_table"][j][0], "table_holm": d[k + "_table"][j][1], "diff_holm": abs(d[k + "_holm"][j] - d[k + "_table"][j][1])})
    df = pd.DataFrame(rows)
    paths6.write_csv(df, paths6.REP / "ANSWERS_V.csv", "пересчёт p Холма; семейство C: 19 строк C_ALL + 71 единица; D: финалисты %s + единицы до 106" % d["rids"])
    print(df.to_string())
    print("C: источники p:", {k: v for k, v in c["src"].items() if "таблица" in v})
    return int((df["diff_holm"] >= 1e-9).any())
