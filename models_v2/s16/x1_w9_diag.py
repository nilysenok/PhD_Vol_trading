"""W9 build (перенос `w9/build.py`, лист кодера 11.11.2026; логика без изменений): `docs/otchety/harj_x1/W9_DIAG.csv`.
Запуск: `python3 -W ignore -m models_v2.s16.x1_w9_diag --check` — пересчёт и побайтное сравнение тела CSV, ничего не пишет (rc 1 при расхождении);
`--check-verdict` — решение выводится из CSV (`x1_w9_diag_v.py`); `--write` — пересчёт и запись (шапка-объявление сохраняется; в лист не запускалась)."""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w9_diag_a as A
from models_v2.s16 import x1_w9_diag_core as K
from models_v2.s16 import x1_w9_diag_ph as PH
from models_v2.s16 import x1_w9_diag_ph2 as PH2

OUT = K.ROOT / "docs/otchety/harj_x1/W9_DIAG.csv"
FEATS = ("W9D1_agree", "W9D2_encshare", "W9D3_zinst")
COLS = ["block", "cand", "window", "group", "metric", "value"]


def feat_raw(x: pd.DataFrame, name: str) -> np.ndarray:
    if name == "W9D1_agree":
        return (x["DG"].to_numpy(float) * x["DH"].to_numpy(float) > 0).astype(float)
    if name == "W9D2_encshare":
        a, b = np.abs(x["DG"].to_numpy(float)), np.abs(x["DH"].to_numpy(float))
        return a / (a + b)
    return x["zinst"].to_numpy(float)


def groups_of(name: str, v: np.ndarray) -> np.ndarray:
    if name == "W9D1_agree":
        return np.where(np.isfinite(v), v, -1).astype(int)
    return K.pit_terciles(v)


def rows_tr(cand, tr, ver, kind):
    out = []
    for t in tr:
        r = tr[t]
        out += [dict(block="T", cand=cand, window=t, group="", metric=m, value=v) for m, v in
                (("nB", r["nB"]), ("Q_cand", r["q_c"]), ("Q_ctrl", r["q_0"]), ("Q_X1", r["q_x1"]), ("d_ctrl_pct", ver["d0"][t]), ("d_X1_pct", ver["dx"][t]), ("k_ctrl", r["k0"]))]
        for gk, gv in r["kg"].items():
            out.append(dict(block="T", cand=cand, window=t, group=str(gk), metric="param", value=gv))
    for p, v in ver["P"].items():
        out.append(dict(block="V", cand=cand, window="", group="", metric=p, value=int(bool(v))))
    return out


def block_a(x) -> tuple[list, dict, dict]:
    """A_cell, A_sum (П6) и перенос W9A: -> (строки, passed, maxdx)."""
    rows = []
    c = A.cells(x)
    for _, r in c.iterrows():
        for m in c.columns[2:]:
            rows.append(dict(block="A_cell", cand="", window=str(r["period"]), group=r["state"], metric=m, value=r[m]))
    # П6: запас потолка вне 2022, по клеткам год × состояние (N, E)
    cc = c[(c["state"] != "all") & (c["period"] != 2022)]
    no22 = x["yr"].to_numpy() != 2022
    p6 = float((cc["n"] * (cc["c1"] - cc["c2"])).sum() / (no22.sum() * x["lX1"].to_numpy(float)[no22].mean()) * 100)
    p6_all = c[c["state"] != "all"]
    p6m = float((p6_all["n"] * (p6_all["c1"] - p6_all["c2"])).sum() / (len(x) * x["lX1"].mean()) * 100)
    p6x = float((p6_all["n"] * (p6_all["Q_X1"] - p6_all["c1"])).sum() / (len(x) * x["lX1"].mean()) * 100)
    rows += [dict(block="A_sum", cand="W9A", window="M_bez_2022", group="", metric="ceil_gain_c1_minus_c2_pct", value=p6),
             dict(block="A_sum", cand="W9A", window="M", group="", metric="ceil_gain_c1_minus_c2_pct", value=p6m),
             dict(block="A_sum", cand="W9A", window="M", group="", metric="X1_minus_c1_pct", value=p6x)]
    a = A.w9a(x)
    ver = K.verdict_group(a["tr"], {"h1": {}, "h2b22": {}})
    ver["P"]["P4"] = bool(np.sign(a["ep"]["h1"][1] - a["ep"]["h1"][0]) == np.sign(a["ep"]["h2b22"][1] - a["ep"]["h2b22"][0]))
    ver["P"]["P6"] = p6 >= 0.10
    ver["P"]["all"] = all(v for k, v in ver["P"].items() if k != "all")
    rows += rows_tr("W9A", a["tr"], ver, "A")
    for h, (kh, kg) in a["ep"].items():
        rows += [dict(block="EP", cand="W9A", window=h, group="kH", metric="kappa", value=kh), dict(block="EP", cand="W9A", window=h, group="kG", metric="kappa", value=kg)]
    return rows, {"W9A": ver["P"]["all"]}, {"W9A": max(ver["dx"].values())}


def block_d(x, lnM, piD, passed, maxdx) -> list:
    rows = []
    rng = np.random.default_rng(0)
    for name in FEATS:
        v = feat_raw(x, name)
        g = groups_of(name, v)
        tr, ep = K.group_transfer(x, g, lnM, piD), K.expost_groups(x, g, lnM, piD)
        ver = K.verdict_group(tr, ep)
        rows += rows_tr(name, tr, ver, "D")
        for h in ep:
            for gg, (kap, n) in ep[h].items():
                rows += [dict(block="EP", cand=name, window=h, group=str(gg), metric="kappa", value=kap), dict(block="EP", cand=name, window=h, group=str(gg), metric="n", value=n)]
        for gg in sorted(set(g)):
            rows.append(dict(block="G", cand=name, window="M", group=str(gg), metric="n", value=int((g == gg).sum())))
        ns = 0
        for s in K.SHIFTS:
            gs = groups_of(name, np.roll(v, s))
            vs = K.verdict_group(K.group_transfer(x, gs, lnM, piD), K.expost_groups(x, gs, lnM, piD))
            ns += int(vs["P"]["all"])
        npm = 0
        for _ in range(K.N_PERM):
            gp = rng.permutation(g)
            vp = K.verdict_group(K.group_transfer(x, gp, lnM, piD), K.expost_groups(x, gp, lnM, piD))
            npm += int(vp["P"]["all"])
        ps, pp = (1 + ns) / 20, (1 + npm) / (K.N_PERM + 1)
        rows += [dict(block="NULL", cand=name, window="", group="", metric="n_shift_pass", value=ns), dict(block="NULL", cand=name, window="", group="", metric="p_shift", value=ps),
                 dict(block="NULL", cand=name, window="", group="", metric="n_perm_pass", value=npm), dict(block="NULL", cand=name, window="", group="", metric="p_perm", value=pp)]
        passed[name] = ver["P"]["all"] and ps <= 0.10 and pp <= 0.10
        maxdx[name] = max(ver["dx"].values())
    return rows


def block_b(x, passed, maxdx) -> list:
    rows = []
    for name, cols in (("W9B1_S", K.S9), ("W9B2_S_Sigma", K.S9 + K.SIGMA12)):
        tr = A.ridge_transfer(x, cols)
        ver = A.verdict_ridge(tr)
        rows += rows_tr(name, tr, ver, "B")
        ns = sum(int(A.verdict_ridge(A.ridge_transfer(x, cols, s))["P"]["all"]) for s in K.SHIFTS)
        ps = (1 + ns) / 20
        rows += [dict(block="NULL", cand=name, window="", group="", metric="n_shift_pass", value=ns), dict(block="NULL", cand=name, window="", group="", metric="p_shift", value=ps)]
        passed[name] = ver["P"]["all"] and ps <= 0.10
        maxdx[name] = max(ver["dx"].values())
    return rows


def compute() -> tuple[pd.DataFrame, dict]:
    x, chk = K.load()
    lnM = np.log(x["M"].to_numpy(float))
    piD = (x["pi"] * x["D"]).to_numpy(float)
    rows, passed, maxdx = block_a(x)
    rows += block_d(x, lnM, piD, passed, maxdx)
    rows += block_b(x, passed, maxdx)
    fam_a = [k for k in ("W9A",) + FEATS if passed[k]]
    fam_b = [k for k in ("W9B1_S", "W9B2_S_Sigma") if passed[k]]
    pick = min(fam_a, key=lambda k: maxdx[k]) if fam_a else (min(fam_b, key=lambda k: maxdx[k]) if fam_b else "none")
    for k, v in passed.items():
        rows.append(dict(block="DECISION", cand=k, window="", group="", metric="passed_with_null", value=int(v)))
    rows.append(dict(block="DECISION", cand=pick, window="", group="", metric="chosen", value=1))
    # ПОСЛЕ ПРОСМОТРА (Д-2): блоки PH, PH2 — в вердикт не входят
    rows += PH.run(x)[0]
    rows += PH2.run(x)
    hon, cau = PH.check_pit(x)
    rows += [dict(block="CHECK", cand="PH-R", window="", group="", metric="pit_honest_unchanged", value=int(hon)),
             dict(block="CHECK", cand="PH-R", window="", group="", metric="mutation_closed_plus1_caught", value=int(cau))]
    return pd.DataFrame(rows, columns=COLS), chk


def fmt(df: pd.DataFrame) -> str:
    d = df.copy()
    d["value"] = d["value"].map(lambda v: "%.10g" % float(v))
    return d.to_csv(index=False, lineterminator="\n")


def body_of(text: str) -> str:
    """Тело CSV без строк-комментариев шапки."""
    return "\n".join(ln for ln in text.splitlines() if not ln.startswith("#")) + "\n"


def header(chk: dict) -> str:
    decl = [ln for ln in OUT.read_text(encoding="utf-8").splitlines() if ln.startswith("# ") and "данных ниже" not in ln and not ln.startswith("# sha256") and not ln.startswith("# сверки")]
    sh = ["%s %s" % (k, hashlib.sha256((K.ROOT / v).read_bytes()).hexdigest()[:16]) for k, v in {**K.W3.SRC, **K.SRC_EXTRA}.items()]
    return "\n".join(decl + ["# sha256 входов (16): " + "; ".join(sh), "# сверки: " + "; ".join("%s=%.3g" % kv for kv in chk.items())]) + "\n"


def check() -> tuple[bool, str]:
    """Пересчёт и сравнение тела CSV; ничего не пишет. -> (совпало, sha256[:16] файла)."""
    df, _ = compute()
    cur = OUT.read_text(encoding="utf-8")
    return body_of(cur) == fmt(df), hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]


def main():
    if "--check-verdict" in sys.argv:
        from models_v2.s16 import x1_w9_diag_v as V
        sys.exit(V.main())
    if "--check" in sys.argv:
        ok, sha = check()
        print("W9_DIAG --check:", "ok" if ok else "РАСХОЖДЕНИЕ", sha)
        sys.exit(0 if ok else 1)
    if "--write" in sys.argv:
        df, chk = compute()
        OUT.write_text(header(chk) + fmt(df), encoding="utf-8")
        print("wrote", OUT, len(df), "rows; sha256", hashlib.sha256(OUT.read_bytes()).hexdigest()[:16])
        return
    print(__doc__)


if __name__ == "__main__":
    main()
