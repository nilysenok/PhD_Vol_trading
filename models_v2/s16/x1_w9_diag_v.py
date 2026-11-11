"""W9 `--check-verdict`: решение выводится из `W9_DIAG.csv` (блоки T, EP, A_sum, NULL), без пересчёта потерь и без записи.
П1, П2, П3, П5 — из `d_ctrl_pct` / `d_X1_pct` блока T; П4 — из EP (W9A: знак κ_G − κ_H; W9D*: порядок групп, ≥ 30 дней); П6 — из A_sum;
нуль: p_shift ≤ 0,10 (и p_perm ≤ 0,10 у W9D*). Сверяется с блоками V и DECISION; приоритет (а) перед (б), внутри — меньший max Δ к X1.
Запуск: `python3 -W ignore -m models_v2.s16.x1_w9_diag --check-verdict`."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "docs/otchety/harj_x1/W9_DIAG.csv"
FAM_A = ("W9A", "W9D1_agree", "W9D2_encshare", "W9D3_zinst")
FAM_B = ("W9B1_S", "W9B2_S_Sigma")


def read(path: Path = OUT) -> pd.DataFrame:
    text = "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")) + "\n"
    df = pd.read_csv(io.StringIO(text), dtype={"window": str, "group": str, "cand": str, "value": str}, keep_default_na=False)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def _same_order(df: pd.DataFrame, cand: str) -> bool:
    ep = df[(df["block"] == "EP") & (df["cand"] == cand)]
    e = {}
    for h in ("h1", "h2b22"):
        k = ep[(ep["window"] == h) & (ep["metric"] == "kappa")].set_index("group")["value"]
        n = ep[(ep["window"] == h) & (ep["metric"] == "n")].set_index("group")["value"]
        e[h] = {g: (k[g], n[g]) for g in k.index}
    keys = [k for k in e["h1"] if k in e["h2b22"] and e["h1"][k][1] >= 30 and e["h2b22"][k][1] >= 30]
    if len(keys) < 2:
        return False
    return all(np.sign(e["h1"][i][0] - e["h1"][j][0]) == np.sign(e["h2b22"][i][0] - e["h2b22"][j][0]) for i in keys for j in keys if int(i) < int(j))


def _p4_w9a(df: pd.DataFrame) -> bool:
    ep = df[(df["block"] == "EP") & (df["cand"] == "W9A")].set_index(["window", "group"])["value"]
    return bool(np.sign(ep[("h1", "kG")] - ep[("h1", "kH")]) == np.sign(ep[("h2b22", "kG")] - ep[("h2b22", "kH")]))


def derive(df: pd.DataFrame) -> dict:
    """-> {cand: dict(P=..., all=..., maxdx=..., passed=...)} из блоков T, EP, A_sum, NULL."""
    t = df[df["block"] == "T"]
    res = {}
    for cand in FAM_A + FAM_B:
        tc = t[t["cand"] == cand]
        d0 = {w: float(tc[(tc["window"] == w) & (tc["metric"] == "d_ctrl_pct")]["value"].iloc[0]) for w in ("T1", "T2", "T3", "T4")}
        dx = {w: float(tc[(tc["window"] == w) & (tc["metric"] == "d_X1_pct")]["value"].iloc[0]) for w in ("T1", "T2", "T3", "T4")}
        P = dict(P1=d0["T1"] < 0 and d0["T2"] < 0, P2=dx["T1"] < 0 and dx["T2"] < 0, P3=d0["T3"] < 0 and d0["T4"] < 0)
        if cand == "W9A":
            P["P4"] = _p4_w9a(df)
        elif cand in FAM_A:
            P["P4"] = _same_order(df, cand)
        P["P5"] = dx["T3"] <= 0 and dx["T4"] <= 0
        if cand == "W9A":
            P["P6"] = float(df[(df["block"] == "A_sum") & (df["window"] == "M_bez_2022")]["value"].iloc[0]) >= 0.10
        allp = all(P.values())
        nl = df[(df["block"] == "NULL") & (df["cand"] == cand)].set_index("metric")["value"]
        passed = allp
        if cand != "W9A":
            passed = passed and float(nl["p_shift"]) <= 0.10
        if cand in FAM_A[1:]:
            passed = passed and float(nl["p_perm"]) <= 0.10
        res[cand] = dict(P=P, all=allp, maxdx=max(dx.values()), passed=bool(passed))
    return res


def pick(res: dict) -> str:
    fa = [k for k in FAM_A if res[k]["passed"]]
    fb = [k for k in FAM_B if res[k]["passed"]]
    if fa:
        return min(fa, key=lambda k: res[k]["maxdx"])
    return min(fb, key=lambda k: res[k]["maxdx"]) if fb else "none"


def problems(df: pd.DataFrame) -> tuple[str, list]:
    """-> (решение, замечания): расхождения вывода с блоками V и DECISION файла."""
    res = derive(df)
    bad = []
    v = df[df["block"] == "V"]
    for cand, r in res.items():
        stored = {m: bool(val) for m, val in zip(v[v["cand"] == cand]["metric"], v[v["cand"] == cand]["value"])}
        for p, val in {**r["P"], "all": r["all"]}.items():
            if stored.get(p) != val:
                bad.append("%s %s: из таблицы %s, в блоке V %s" % (cand, p, val, stored.get(p)))
    dec = df[df["block"] == "DECISION"]
    for cand, val in zip(dec[dec["metric"] == "passed_with_null"]["cand"], dec[dec["metric"] == "passed_with_null"]["value"]):
        if bool(val) != res[cand]["passed"]:
            bad.append("DECISION %s: passed_with_null %s ≠ выведено %s" % (cand, bool(val), res[cand]["passed"]))
    chosen = list(dec[dec["metric"] == "chosen"]["cand"])
    got = pick(res)
    if chosen != [got]:
        bad.append("DECISION chosen %s ≠ выведено %s" % (chosen, got))
    return got, bad


def main() -> int:
    got, bad = problems(read())
    print("W9 --check-verdict: решение «%s»; кандидатов %d; %s" % (got, len(FAM_A + FAM_B), "расхождений с V и DECISION нет" if not bad else "РАСХОЖДЕНИЯ: " + "; ".join(bad)))
    return 1 if bad else 0
