"""D3: таблицы 1–5 итогового отчёта (п. 30 блока) и сборка текстов. Все числа — из массивов σ² и csv этой папки; формулировки вывода — по правилу D3."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5 import common, stats
from models_v2.s5.b1 import evalrows
from models_v2.s5.d import ctx_d, doc_txt, robust_d

REP = common.REP
NAME = {"HARJ": "HAR-J(б)", "B2": "Б-2"}


f = doc_txt.f


def md(head, rows) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def years_win(rv, a, b, td) -> int:
    y = pd.Series(td).dt.year.to_numpy()
    qa, qb = metrics.qlike(rv, a), metrics.qlike(rv, b)
    return int(sum(qa[y == k].mean() < qb[y == k].mean() for k in sorted(set(y))))


def t1_t2(sig, rv, td, rk, st) -> tuple:
    mk = evalrows.masks(td)
    dmp = st[(st.test == "DM") & (st.base == "Б-2") & (st.period == "весь период")].set_index("row_id")
    stat = dict(zip(rk["row_id"], rk["status"])) | {"HARJ": "контроль (база)", "B2": "контроль (база, действующий)"}
    r1, r2 = [], []
    for k in ["B2", "HARJ"] + list(rk["row_id"]):
        m = stats.full_metrics(rv, sig[k])
        p = dmp["p"].get(k, np.nan) if k in dmp.index else np.nan
        r1.append((NAME.get(k, k), f(m["qlike"]), f(m["rmse_ln"]), f(m["mae_ln"]), f(m["r2_mz_ln"], 3), f(p, 4), stat[k]))
        q = metrics.qlike(rv, sig[k])
        r2.append((NAME.get(k, k),) + tuple(f(q[mk[x]].mean()) for x in ("p2017_2021", "y2022", "p2023_2025", "y2026", "all")))
    return md(("Модель", "QLIKE", "RMSE (ln)", "MAE (ln)", "R² MZ (ln)", "DM p к Б-2 (сырой)", "Статус"), r1), md(("Модель", "2017–2021", "2022", "2023–2025", "2026", "Вся история"), r2)


def t3(sig, rv, td, best, fl, mc) -> str:
    q = lambda k: float(metrics.qlike(rv, sig[k]).mean())   # noqa: E731
    ctl = ctx_d.load_json("rows.json", {})
    mean_best = min((r for r in ctl if r.startswith("mean_")), key=lambda r: ctl[r]["qlike_all"])
    tg = lambda k: "seed-ok %s, plateau-ok %s" % (fl[k]["seed_ok"], fl[k]["plateau_ok"])   # noqa: E731
    steps = [("HAR-J(б) → Б-2", "HARJ", "B2", ""), ("Б-2 → Б-2 + F4 (`b2_f4`)", "B2", "b2_f4", tg("b2_f4")),
             ("`b2_f4` → Optuna HP (`c1_opt`)", "b2_f4", "c1_opt", tg("c1_opt")),
             ("`b2_f4` → смесь с HAR-J(б) (`c2_h3_roll_f4`)", "b2_f4", "c2_h3_roll_f4", tg("c2_h3_roll_f4")),
             ("`b2_f4` → лучший ансамбль D1 (`%s`)" % best, "b2_f4", best, tg(best)),
             ("`b2_f4` → F3 в спокойном режиме (`c3_f3calm_rv`, потолок)", "b2_f4", "c3_f3calm_rv", tg("c3_f3calm_rv"))]
    rows = []
    for lab, a, b, tail in steps:
        rows.append((lab, f(q(a)), f(q(b)), f(q(a) - q(b)), "лет лучше: %d из 10; MCS p=%s; %s" % (years_win(rv, sig[b], sig[a], td), f(mc.get(b, np.nan), 2), tail)))
    rows.append(("простое среднее (`%s`) → лучший ансамбль" % mean_best, f(ctl[mean_best]["qlike_all"]), f(q(best)), f(ctl[mean_best]["qlike_all"] - q(best)), "контроль без параметров"))
    return md(("Изменение", "QLIKE до", "QLIKE после", "Выигрыш", "Устойчивость"), rows)


def t4() -> str:
    g = pd.read_csv(REP / "D_REGIMES.csv", comment="#")
    labs = list(dict.fromkeys(g["group"]))
    rows = [(NAME.get(m.upper(), m),) + tuple(f(float(g[(g.model == m) & (g.group == lb)]["qlike"].iloc[0])) for lb in labs) for m in dict.fromkeys(g["model"])]
    n = [int(g[g.group == lb]["n"].iloc[0]) for lb in labs]
    return md(("Модель",) + tuple("%s (n=%d)" % (lb, k) for lb, k in zip(labs, n)), rows)


def t5(sig, rv, td, rk, st, fl) -> str:
    y = pd.Series(td).dt.year.to_numpy()
    sd = pd.read_csv(REP / "D_SEEDS.csv", comment="#").drop_duplicates("row_id").set_index("row_id")
    wn = pd.read_csv(REP / "D_WINDOW.csv", comment="#").set_index("row_id")
    rows = []
    for k in rk["row_id"]:
        dq = {yy: float((metrics.qlike(rv, sig[k]) - metrics.qlike(rv, sig["B2"]))[y == yy].mean()) for yy in sorted(set(y))}
        wy = max(dq, key=dq.get)
        dmr = st[(st.test == "DM") & (st.row_id == k) & (st.base == "Б-2") & (st.period == "весь период")]
        dm_txt = "p=%s; Холм N=106: %s" % (f(float(dmr["p"].iloc[0]), 3), f(float(dmr["p_holm_N106"].iloc[0]), 3)) if len(dmr) else "не считался (справочная строка)"
        sens = "seeds: ст.откл. %s, %d/5 ok; соседи: %s; окно 1008: %s" % (f(sd.loc[k, "std"], 4), fl[k]["n_ok"], "ok" if fl[k]["plateau_ok"] else "нет", f(wn.loc[k, "d_roll_minus_exp"], sign=True))
        rows.append((k, "%d из 10 (к Б-2); %d из 10 (к HAR-J(б))" % (years_win(rv, sig[k], sig["B2"], td), years_win(rv, sig[k], sig["HARJ"], td)),
                     "%d (Δ к Б-2 %s)" % (wy, f(dq[wy], sign=True)), sens, dm_txt))
    return md(("Модель", "Годы с выигрышем", "Худший год", "Чувствительность к параметрам", "DM к Б-2"), rows)


def write_all(C, best, top, flags, rk, st, sig, ctx, counts) -> None:
    rv, td, fl = ctx["rv"], C["ctrl"]["target_date"], flags["flags"]
    mc = st[(st.test == "MCS") & (st.period == "весь период")].set_index("row_id")["p"].to_dict()
    tabs = {"t1": None}
    tabs["t1"], tabs["t2"] = t1_t2(sig, rv, td, rk, st)
    tabs.update(t3=t3(sig, rv, td, best, fl, mc), t4=t4(), t5=t5(sig, rv, td, rk, st, fl))
    doc_txt.write_final(C, best, top, flags, rk, st, tabs, counts)
    doc_txt.write_repro(best)
    doc_txt.write_readme(best, top, rk, counts)
    print("G10: FINAL.md, REPRODUCTION.md, README.md записаны; финалисты:", robust_d.finalists(best))
