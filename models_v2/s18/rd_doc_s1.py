"""S-RCD-EVAL: разделы 1–2 отчёта `S_RCD_REPORT.md` (первая таблица; разложение вклада разд. 6) — строки, собранные из таблиц `S_RCD_*.csv`; чисел, вписанных вручную, нет. В каждой таблице — границы дат и n.
Публичное API: `first_table(tb)`, `layer_table(tb)`, `contrast_table(t)`, `ident_table(t)`, `dm(t, section, item)`, `section2(t)`."""
from __future__ import annotations

import pandas as pd

from models_v2.s18 import rd_eval_cfg as C
from models_v2.s18.z_doc_fmt import ci, md, pick, pv, sgn

BASE_COLS = ("Δ к S-CDF, % [90 %]", "Δ к S-RCK, % [90 %]", "Δ к W5-G, % [90 %]", "Δ к X1, %", "Δ к K2, %", "Δ к K3, %")
COLS = ("ряд", "QLIKE", "первая t⁺", "последняя t⁺", "n") + BASE_COLS
CI_BASES = ("S-CDF", "S-RCK", "W5-G")


def first_table(tb: pd.DataFrame, series=C.SHOWN) -> str:
    rows = []
    for s in series:
        r = pick(tb, series=s)
        d = {"ряд": s, "QLIKE": "%.6f" % r["qlike"], "первая t⁺": r["first_tplus"], "последняя t⁺": r["last_tplus"], "n": str(int(r["n"]))}
        for lab, b in zip(BASE_COLS, C.BASES):
            k = "d_%s_pct" % b
            d[lab] = "—" if s == b else ci(r[k], r["lo90_" + b], r["hi90_" + b]) if b in CI_BASES else sgn(r[k])
        rows.append(d)
    return md(pd.DataFrame(rows), {c: "%s" for c in COLS})


def layer_table(tb: pd.DataFrame) -> str:
    t = tb[tb["role"] == "layer X1"]
    r = pd.DataFrame({"слой": t["series"], "QLIKE": t["qlike"], "первая t⁺": t["first_tplus"], "последняя t⁺": t["last_tplus"], "n": t["n"], "Δ к X1, %": t["d_X1_pct"], "90 % снизу": t["lo90_X1"], "90 % сверху": t["hi90_X1"]})
    return md(r, {"слой": "%s", "QLIKE": "%.6f", "первая t⁺": "%s", "последняя t⁺": "%s", "n": "%d", "Δ к X1, %": "%+.4f", "90 % снизу": "%+.4f", "90 % сверху": "%+.4f"})


def dm(t: dict, section: str, item: str, key=None) -> pd.Series:
    me = t["decomp"]
    m = (me["section"] == section) & (me["item"] == item)
    if key is not None:
        m &= me["key"] == key
    return me[m].iloc[0]


def contrast_table(t: dict) -> str:
    st, cn = t["stats"], t["conc"]
    rows = []
    for kind in ("main", "decomp", "reference", "active"):
        for _, r in st[st["kind"] == kind].iterrows():
            if kind == "main" and r["base"] not in ("S-CDF", "S-RCK", "W5-G"):
                continue
            tr = cn[(cn["series"] == r["series"]) & (cn["base"] == r["base"]) & (cn["item"] == "trim")]
            rows.append({"окно": r["window"] if kind == "active" else "D_M", "контраст (ряд к базе)": "%s к %s" % (r["series"], r["base"]), "первая t⁺": r["first"], "последняя t⁺": r["last"], "n": "%d" % r["n"],
                         "Q ряда": "%.6f" % r["qlike_h"], "Q базы": "%.6f" % r["qlike_b"], "Δ, % базы [90 %]": ci(r["delta_pct"], r["lo90"], r["hi90"]), "p DM NW-8": pv(r["p_dm"]),
                         "Δ_trim, %": sgn(tr["value"].iloc[0]) if len(tr) and kind != "active" else "—"})
    return md(pd.DataFrame(rows), {c: "%s" for c in rows[0]})


def ident_table(t: dict) -> str:
    rows = []
    for w in ("D_M", "active"):
        n = int(dm(t, "share_" + w, "n_dates")["value"])
        for sec, lab in (("e_", "e(X) — % QLIKE W5-G"), ("ident_", "слагаемые тождеств 1–3 (единицы e)")):
            me = t["decomp"][t["decomp"]["section"] == sec + w]
            rows += [{"окно": w, "n": "%d" % n, "группа": lab, "величина": r["item"], "значение": "%+.6f" % r["value"]} for _, r in me.iterrows()]
        s = dm(t, "share_" + w, "s")
        rows.append({"окно": w, "n": "%d" % n, "группа": "доля s", "величина": "s = e(S-RCD)/e(S-RCK)", "значение": "%s; %s" % (sgn(s["value"]), s["key"])})
    return md(pd.DataFrame(rows), {c: "%s" for c in rows[0]})


def section2(t: dict) -> list:
    tb = t["table"]
    cap = "t⁺ %s…%s, n = %d" % (tb["first_tplus"].iloc[0], tb["last_tplus"].iloc[0], tb["n"].iloc[0])
    fz = lambda tag: [dm(t, "decomp_" + tag, i)["value"] for i in ("total", "frozen_pi", "layer_reaction")]
    a, b = fz("vs_W5G"), fz("vs_CDF")
    return ["Квадрат {C-D, C-K} × {без R, R} на уровне F (разд. 6.1). Δ — в %% QLIKE базы, 90 %% интервал — стационарный бутстрап; p DM — односторонний, NW-8, сырой. D_M: %s; активное окно — d ≥ %s." % (cap, C.ACTIVE_FROM), "",
            contrast_table(t), "",
            "Аддитивное разложение (разд. 6.2): e(X) = 100·(Q(X) − Q(W5-G))/Q(W5-G) на датах окна; тождества 1–3 сходятся (невязки — строки `*_gap`); I < 0 означает, что R помогает C-K больше, чем C-D; s показана всегда, условием замены не является.", "",
            ident_table(t), "",
            "Разложение Δ(S-RCD) на «эффект при замороженном π» (F^fr) и «реакцию слоя» (%s): к W5-G — всего %s %% = заморожено %s %% + реакция слоя %s %%; к S-CDF — всего %s %% = заморожено %s %% + реакция слоя %s %%." % (cap, sgn(a[0]), sgn(a[1]), sgn(a[2]), sgn(b[0]), sgn(b[1]), sgn(b[2])), ""]
