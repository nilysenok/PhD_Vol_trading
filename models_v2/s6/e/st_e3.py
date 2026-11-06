"""E3 (R7, G9): статистика финалистов. DM (Ньюи–Уэст 5) по периодам, блочный бутстрап ΔQLIKE (блок 20, 5000, seed 42), Холм с N = pe.n_trials(), MCS 90 %, годы, режимы,
без 5 лучших и 5 худших дней, без топ-20 по |ΔQLIKE|, кумулятивные разности. Все p — после выбора на тех же данных (постселекционная оговорка в каждой строке)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.b1 import evalrows
from models_v2.s5.d import stats_d
from models_v2.s6.e import comp, crit_e, pe, resid, ser, st_a, st_c, st_seeds

NOTE = "p сырые и Холм (N = число уникальных trials Ж-206 на момент сравнения) посчитаны ПОСЛЕ выбора лучшего на тех же данных 2017–02.2026; не независимый OOS"
PER = (("all", "вся история"), ("p2017_2021", "2017–2021"), ("y2022", "2022"), ("p2023_2025", "2023–2025"), ("y2026", "2026"), ("no_240222", "без 24.02.2022"))
BASELINE = {"X": "B2X", "L": "B2L"}


def candidates(df) -> dict:
    """{id: (семейство X/L/XL/H, члены-компоненты)}: компоненты, лучшие A1…A4, B1, B2, C."""
    fin = st_seeds.finalists(df)
    ad = pd.read_csv(pe.ENS / "ADAPTIVE_RESULTS.csv", comment="#")
    ml, bb = ad["base_ml"].iloc[0], ad["base_b"].iloc[0]
    out = {"XJ": ("X", ()), "XF": ("X", ()), "LF": ("L", ())}
    for n, (kind, a, _) in fin.items():
        if kind == "A":
            out[n] = ({"A1": "X", "A2": "L", "A3": "XL", "A4": "XL"}[a], tuple(st_a.COMBOS[a]))
        elif kind == "B":
            out[n] = ("X" if a == "1" else "L", ("HAR", "XJ" if a == "1" else "LF"))
        else:
            out[n] = ("XL", ("HAR", bb if n == "C5" else ml))
    return out


def best_component(df, members) -> str:
    return min(members, key=lambda n: (ser.crit(df, n), n)) if members else ""


def compare(df, cand, ref, mk, idx) -> list:
    rv = df["rv"].to_numpy(float)
    s2c, s2r = df[cand].to_numpy(float), df[ref].to_numpy(float)
    dl = ser.qloss(df, cand) - ser.qloss(df, ref)
    rows = []
    for key, lab in PER:
        m = mk[key]
        r = evalrows.vs(rv, s2r, s2c, m)
        row = {"candidate": cand, "reference": ref, "period": lab, "n": int(m.sum()), "delta_qlike": float(dl[m].mean()), "dm": r["dm"], "p_raw": r["p"]}
        if key == "all":
            b = stats_d.boot_delta(dl, idx)
            s = np.sort(dl)
            row.update({"boot_lo95": b["lo95"], "boot_hi95": b["hi95"], "p_boot": b["p_boot"], "delta_no_best5_worst5": float(s[5:-5].mean()),
                        "delta_no_top20_abs": float(dl[np.argsort(-np.abs(dl))[20:]].mean())})
        rows.append(row)
    return rows


def holm_primary(t: pd.DataFrame) -> pd.DataFrame:
    """Холм по первичным сравнениям (период «вся история») с N = число уникальных trials Ж-206 (не меньше числа сравнений)."""
    prim = t[(t["period"] == "вся история")].index
    N = max(pe.n_trials(), len(prim))
    t = t.copy()
    t.loc[prim, "p_holm"] = stats_d.holm_n(t.loc[prim, "p_raw"].to_numpy(), N)
    t["n_hypotheses_holm"], t["status"], t["note"] = N, pe.STATUS, NOTE
    t["significant_holm_0.05"] = t["p_holm"] < 0.05
    return t


def stage(_a=None) -> int:
    df = ser.load()
    cands = candidates(df)
    mk = evalrows.masks(df["target_date"])
    idx = stats_d.block_idx(len(df))
    comps = {"HAR": ("H", ()), "B2X": ("X", ()), "B2L": ("L", ())}
    rows = []
    for c, (fam, members) in cands.items():
        refs = ["HAR", "B2X" if "X" in fam else "B2L"] + (["B2L"] if fam == "XL" else [])
        refs += [best_component(df, members)] if members else []
        refs += ["LF"] if c == "XJ" else []
        for r in dict.fromkeys(x for x in refs if x and x != c):
            for row in compare(df, c, r, mk, idx):
                rows.append({**row, "reference_role": "лучший собственный компонент" if r == best_component(df, members) and members else "baseline/HAR"})
    t = holm_primary(pd.DataFrame(rows))
    prim = t[(t["period"] == "вся история")].index
    N = int(t["n_hypotheses_holm"].iloc[0])
    ids = list(dict.fromkeys(list(cands) + ["B2X", "B2L", "HAR"]))
    L = np.column_stack([ser.qloss(df, k) for k in ids])
    mcs = stats_d.mcs(L, idx)
    t["mcs90_candidate"] = t["candidate"].map(dict(zip(ids, mcs["in"])))
    mcs_tab = pd.DataFrame({"id": ids, "mcs_p": mcs["p"], "in_mcs90": mcs["in"], "qlike_all": L.mean(axis=0)})
    W = resid.build_world()
    ter, _ = st_c.groups(W)
    reg = {"calm_g4": ~df["st_g4"].to_numpy(bool), "stress_g4": df["st_g4"].to_numpy(bool), "low_rv": ter == 0, "mid_rv": ter == 1, "high_rv": ter == 2}
    yr = pd.Series(df["target_date"]).dt.year.to_numpy()
    br = []
    for c in ids:
        q, rv = ser.qloss(df, c), df["rv"].to_numpy(float)
        d = crit_e.describe(c, df[c].to_numpy(float), df)
        br.append({**d, **{"qlike_y%d" % y: float(q[yr == y].mean()) for y in range(2017, 2027)}, **{"qlike_" + k: float(q[m].mean()) for k, m in reg.items()},
                   "qlike_no_top20": float(np.sort(q)[:-20].mean()), "kind": "candidate" if c in cands else "reference"})
    cum = pd.DataFrame({"d": df["d"]})
    for c, (fam, members) in cands.items():
        for r in dict.fromkeys(list(members) + ["HAR"]):
            if r != c and (members or c in ("XJ", "XF", "LF")):
                cum["%s_minus_%s" % (c, r)] = np.cumsum(ser.qloss(df, c) - ser.qloss(df, r))
    pe.write(t, pe.ENS / "STATISTICAL_TESTS_all.csv", "E3: DM (NW 5), бутстрап (блок 20, 5000, seed 42), Холм с N, периоды")
    pe.write(pd.DataFrame(br), pe.ENS / "HYBRID_BREAKDOWN.csv", "E3: QLIKE по годам, режимам, без топ-20; RMSE/MAE/R²/смещение")
    pe.write(mcs_tab, pe.ENS / "MCS90.csv", "E3: MCS 90 % на QLIKE (блоки 20, 5000)")
    cum.to_csv(pe.ENS / "CUMULATIVE_DIFF.csv", index=False, lineterminator="\n")
    _plot(cum)
    ok = bool((t["n_hypotheses_holm"] >= len(prim)).all()) and np.isfinite(t["p_raw"]).all()
    print("G9 E3: сравнений %d, первичных %d, N Холма %d, MCS90 в множестве: %s" % (len(t), len(prim), N, [i for i, m in zip(ids, mcs["in"]) if m]), flush=True)
    return 0 if ok else 1


def _plot(cum: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in cum.columns[1:]:
        if c.split("_minus_")[0].startswith(("A", "B", "C")):
            ax.plot(pd.to_datetime(cum["d"]), cum[c], lw=0.8, label=c)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Кумулятивная разность дневных QLIKE: гибрид − компонент (минус — гибрид лучше)")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(pe.ENS / "CUMULATIVE_DIFF.png", dpi=110)
    plt.close(fig)
