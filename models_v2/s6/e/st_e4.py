"""E4 (R8, R9, G10): ACCEPT/REJECT (раздельно XGB и LGBM; гибриды — против лучшего собственного компонента, включая HAR-J(б)) и выбор лучших по критериям A–F.
Правила объявлены в GATES_E.md до счёта. «Статистически подтверждено» — только при p скорр. (Холм, N) < 0,05."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.b1 import evalrows
from models_v2.s6.e import comp, crit_e, pe, rules, ser, st_a, st_b, st_c, st_e3, st_seeds

REP = pe.REP
PAR = {"fixed_loyo": 2, "inv_qlike_252": 1, "inv_rmse_252": 1, "roll_252": 1, "roll_126": 1, "roll_504": 1, "shrink_252": 2, "B": 1, "C2": 2, "C3": 2, "C4": 2, "C5": 2, "C6": 1}
PARAM_NOTE = "новые параметры: fixed_loyo K−1 весов + год; inv_*/roll_w: окно; shrink: окно + s; B: константа c0 (HP — прежнего выбора И2); C2/C3: порог группы + веса групп; C4: веса по режиму; C5: окно; C6: окно"


def pen(csv, key_col, key, col="crit_pen") -> float:
    t = pe.read(csv)
    return float(t.loc[t[key_col] == key, col].iloc[0])


def crit_pens() -> dict:
    x, l = REP / "xgboost", REP / "lightgbm"
    return {"B2X": pen(x / "OBJECTIVE_RESULTS.csv", "id", "ctrl_b2"), "XF": pen(x / "HP_RESULTS_C.csv", "id", "ctrl_f4"), "XJ": pen(x / "HP_RESULTS_C.csv", "id", "x2_best_j"),
            "B2L": pen(l / "C_L0.csv", "model", "lb0_inh"), "LF": pen(l / "C_L0.csv", "model", "C0_inh_F4")}


def seeds_of(sd: pd.DataFrame, n: str):
    r = sd[sd["id"] == n]
    return None if r.empty else [float(r["qlike_s%d" % s].iloc[0]) for s in comp.SEEDS]


def params_of(n: str, fin: dict) -> int:
    kind, a, b = fin[n]
    if kind == "A":
        return PAR[b] + (len(st_a.COMBOS[a]) - 2 if b == "fixed_loyo" else 0)
    return PAR["B"] if kind == "B" else PAR[a]


def reconstruct_ok(df, cdf, ctx, n, fin, W, ter, chg) -> tuple:
    """F: A — правило на сохранённых компонентах, C — правило на сохранённых рядах, B — повторное обучение (без кэша) и побитная сверка."""
    kind, a, b = fin[n]
    if kind == "A":
        s = rules.sigma2(b, comp.stack(cdf, st_a.COMBOS[a]), ctx)
    elif kind == "C":
        ad = pd.read_csv(pe.ENS / "ADAPTIVE_RESULTS.csv", comment="#")
        s = st_c.build(ctx, comp.col(cdf, "HAR"), comp.col(cdf, ad["base_ml"].iloc[0]), df[ad["base_b"].iloc[0]].to_numpy(float), ter, chg)[0][n]
    else:
        s = st_b.run_one(W, n, st_b.KIND[n], W["dz"].X, fresh=True)["sigma2"].to_numpy(float)
    return bool(s.tobytes() == df[n].to_numpy(float).tobytes()), float(np.max(np.abs(s / df[n].to_numpy(float) - 1)))


NOT_ADMITTED = "R8 пройдено; не допущен (R9, критерий C)"


def relabel_not_admitted(ar: pd.DataFrame, sel: pd.DataFrame) -> pd.DataFrame:
    """Гибрид с ACCEPT по R8, но admitted = False по R9, получает статус NOT_ADMITTED (колонки verdict, verdict_full); колонка accept (R8) не меняется."""
    adm = sel.set_index("id")["admitted"].to_dict()
    ar = ar.copy()
    m = (ar["role"] == "гибрид") & ar["accept"].astype(bool) & ar["id"].map(lambda i: i in adm and not adm[i])
    ar.loc[m, ["verdict", "verdict_full"]] = NOT_ADMITTED
    return ar


def _reuse_f(n: str):
    old = pe.read(pe.ENS / "HYBRID_SELECTION.csv").set_index("id")
    return bool(old.loc[n, "F_reconstruct_bitwise"]), float(old.loc[n, "F_max_rel"])


def stage(_a=None) -> int:
    from models_v2.s6.e import resid
    df, cdf = ser.load(), comp.load()
    ctx, sd, fin = comp.ctx_of(cdf), pe.read(pe.ENS / "SEEDS_E.csv"), st_seeds.finalists(ser.load())
    cands = st_e3.candidates(df)
    tests = pe.read(pe.ENS / "STATISTICAL_TESTS_all.csv")
    td, rv, pens = df["target_date"], df["rv"].to_numpy(float), crit_pens()
    mk, W = evalrows.masks(td), resid.build_world()
    ter, chg = st_c.groups(W)
    ref = {"XJ": "B2X", "XF": "B2X", "LF": "B2L"}
    rows = []
    for n, (fam, members) in cands.items():
        role = "самостоятельная" if n in ref else "гибрид"
        base = ref.get(n) or st_e3.best_component(df, members)
        q_n, q_b = ser.qloss(df, n), ser.qloss(df, base)
        d_n, d_b = crit_e.describe(n, df[n].to_numpy(float), df), crit_e.describe(base, df[base].to_numpy(float), df)
        cn, cb = (pens[n], pens[base]) if role == "самостоятельная" else (ser.crit(df, n), ser.crit(df, base))
        r = crit_e.accept(d_n, d_b, q_n, q_b, td, seeds_of(sd, n), seeds_of(sd, base), cn, cb)
        t = tests[(tests["candidate"] == n) & (tests["reference"] == base) & (tests["period"] == "вся история")]
        ph = float(t["p_holm"].iloc[0]) if len(t) else float("nan")
        post = mk["y2022"] | mk["p2023_2025"] | mk["y2026"]
        rows.append({"id": n, "role": role, "family": fam, "reference": base, "criterion": "crit_pen (LOYO)" if role == "самостоятельная" else "crit_wf", "crit_new": cn, "crit_ref": cb, **r,
                     "p_holm": ph, "statistically_confirmed": bool(ph < 0.05), "dq_2022_2026_vs_ref": float((q_n - q_b)[post].mean()),
                     "verdict_full": r["verdict"] + (" (потолок: x2_best_j видел 2017–2021; разность на 2022–2026 в колонке dq_2022_2026_vs_ref)" if n == "XJ" else "")})
    ar = pd.DataFrame(rows)
    ar["status"], ar["note"] = pe.STATUS, st_e3.NOTE
    sel = []
    for n in [x for x in cands if x in fin]:
        fam, members = cands[n]
        a = ar[ar["id"] == n].iloc[0]
        best_sa = min(("XJ", "LF"), key=lambda k: (ser.crit(df, k), k))
        q = ser.qloss(df, n)
        qb = ser.qloss(df, a["reference"])
        stab = {lab: float(q[mk[k]].mean() / qb[mk[k]].mean()) for k, lab in (("p2017_2021", "2017_2021"), ("y2022", "2022"), ("p2023_2025", "2023_2025"), ("y2026", "2026"))}
        nm = mk["no_240222"]
        dm = evalrows.vs(rv, df[a["reference"]].to_numpy(float), df[n].to_numpy(float), nm)
        F, maxrel = _reuse_f(n) if getattr(_a, "reuse_f", False) and fin[n][0] == "B" else reconstruct_ok(df, cdf, ctx, n, fin, W, ter, chg)
        simple = [c for c in df.columns if c.startswith("ctl_") and c.endswith("_mean_lvl")]
        sref = min(simple, key=lambda c: (ser.crit(df, c), c))
        ds = evalrows.vs(rv, df[sref].to_numpy(float), df[n].to_numpy(float), mk["all"])
        crit_n = ser.crit(df, n)
        row = {"id": n, "accept_R8": bool(a["accept"]), "A_dcrit_vs_best_standalone": float(ser.crit(df, best_sa) - crit_n), "B_max_ratio_vs_component": max(stab.values()), "B_ok": max(stab.values()) <= 1.02,
               "C_delta_no_240222": dm["d"], "C_p_raw": dm["p"], "C_ok": bool(dm["d"] < 0 and dm["p"] < 0.10), "D_p_holm": a["p_holm"], "E_new_params": params_of(n, fin), "F_reconstruct_bitwise": F, "F_max_rel": maxrel,
               "simple_ref": sref, "dcrit_vs_simple": float(ser.crit(df, sref) - crit_n), "p_vs_simple": ds["p"], "simple_suffices": bool(ser.crit(df, sref) - crit_n <= 0.0005 and ds["p"] > 0.10), "crit_wf": crit_n,
               **{"ratio_" + k: v for k, v in stab.items()}}
        row["admitted"] = bool(row["accept_R8"] and row["B_ok"] and row["C_ok"] and row["F_reconstruct_bitwise"])
        sel.append(row)
    s = pd.DataFrame(sel)
    ar = relabel_not_admitted(ar, s)
    pe.write(ar, pe.ENS / "ACCEPT_REJECT.csv", "R8: ACCEPT = пункты (1)–(4); (5) — отдельной колонкой; гибрид без допуска R9 — «%s»" % NOT_ADMITTED)
    best = s[s["admitted"]]["crit_wf"].min() if s["admitted"].any() else np.nan
    s["within_0.0005_of_best"] = s["admitted"] & (s["crit_wf"] <= best + 0.0005)
    s = s.sort_values(["admitted", "within_0.0005_of_best", "crit_wf"], ascending=[False, False, True])
    s["rank_note"] = PARAM_NOTE
    s["status"] = pe.STATUS
    pe.write(s, pe.ENS / "HYBRID_SELECTION.csv", "R9: критерии A–F; допуск = ACCEPT R8 + B + C + F; три лучших — только допущенные и в пределах 0,0005 от лучшего (при ничьей — меньше параметров)")
    print("G10 e4:\n", ar[["id", "role", "reference", "c1_crit", "c2_2017_2021", "c3_years", "c4_seeds", "verdict"]].to_string(index=False), "\n",
          s[["id", "admitted", "B_ok", "C_ok", "F_reconstruct_bitwise", "crit_wf", "within_0.0005_of_best", "simple_suffices"]].to_string(index=False), flush=True)
    return 0
