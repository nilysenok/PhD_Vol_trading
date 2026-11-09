"""V6 лист Z2, G61 (правило 7): QLIKE каждого ряда, пересчитанный из СОХРАНЁННОГО прогноза, против числа из таблиц своей программы (допуск 1e-9). Источники чисел: V1 — `hybrid_final_v1/TABLE17_SBER_H1.csv`,
`FINAL_STRUCT_CONFIG.json`; V2, V4, V5 и прежние ряды — столбцы `reported_*` листа 0 (`PRIOR_SBER.csv`); V4, V5 — `FIN_SBER.csv`; V6 — `FIN_ALL.csv` (все окна). Совпадение K1/K2/K3 между программами — побайтно (sha256 массивов).
Ряд без числа в таблицах своей программы получает статус «нет числа в таблицах» (не расхождение). Только чтение."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_io as IO, v6_0_q as Q, v6_cmp_reg as R, v6_fin_data as FD

TOL = 1e-9
REP = IO.ROOT / R.REPORTS


def _csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(REP / rel, comment="#")


def claims() -> pd.DataFrame:
    rows = []
    add = lambda s, w, v, src, n=np.nan: rows.append(dict(series=s, window=w, source=src, source_value=float(v), n_source=n))
    pr = R.prior()
    for s, r in pr[pr["reported_qlike"].notna()].iterrows():
        add(s, "all", r["reported_qlike"], "PRIOR_SBER.csv (лист 0): " + str(r["reported_source"]))
    t17 = _csv("hybrid_final_v1/TABLE17_SBER_H1.csv").set_index("Модель")
    for name, s in (("HAR-J", "K1"), ("XGBoost", "K2"), ("LightGBM", "K3"), ("HAR + XGBoost [A1_roll_126]", "J210a:A1_roll_126")):
        add(s, "all", t17.loc[name, "QLIKE"], "hybrid_final_v1/TABLE17_SBER_H1.csv: " + name)
    cfg = json.loads((REP / "hybrid_struct_v1/FINAL_STRUCT_CONFIG.json").read_text(encoding="utf-8"))
    add("J212B:B_xgb_Bf", "all", cfg["specs"]["B_XGB"]["qlike_all"], "hybrid_struct_v1/FINAL_STRUCT_CONFIG.json specs.B_XGB.qlike_all")
    for prog, f, mp in (("V4", "harj_hybrid_v4/FIN_SBER.csv", {"k1": "K1", "k2": "K2", "k3": "K3", "d2": "PANlgbm:D2", "K8": "K8"}), ("V5", "harj_hybrid_v5/FIN_SBER.csv", {"k1": "K1", "k2": "K2", "k3": "K3", "M": "M", "d2": "PANlgbm:D2", "K8": "K8"})):
        for _, r in _csv(f).iterrows():
            add(mp.get(r["series"], "%s:%s" % (prog, r["series"])), r["window"], r["qlike"], f + ": " + r["series"], r["n"])
    mp = {"D2ctl": "PANlgbm:D2", **{k: k for k in ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V5:GS4", "V5:SP1")}, **{c: "V6:" + c for c in FD.CFG}}
    for _, r in _csv("harj_hybrid_v6/FIN_ALL.csv").iterrows():
        if r["series"] in mp:
            add(mp[r["series"]], str(r["window"]), r["qlike"], "harj_hybrid_v6/FIN_ALL.csv: " + r["series"], r["n"])
    return pd.DataFrame(rows)


def repro(u: dict, reg: pd.DataFrame) -> pd.DataFrame:
    c = claims()
    own, n_own, st = [], [], []
    for _, r in c.iterrows():
        if r["series"] not in u["f"]:
            own.append(np.nan), n_own.append(np.nan), st.append("нет ряда в реестре")
            continue
        m = Q.wmask(u["tp"], r["window"])
        v = float(Q.ql(u["rv"][m], u["f"][r["series"]][m]).mean())
        own.append(v), n_own.append(int(m.sum()))
        ok = abs(v - r["source_value"]) <= TOL and (np.isnan(r["n_source"]) or int(r["n_source"]) == int(m.sum()))
        st.append("совпало" if ok else "расхождение")
    c["own_value"], c["n_own"], c["status"] = own, n_own, st
    c["abs_diff"] = (c["own_value"] - c["source_value"]).abs()
    c.insert(0, "kind", "qlike")
    no = [dict(kind="qlike", series=s, window="all", source="в таблицах своей программы числа нет", own_value=float(Q.ql(u["rv"], u["f"][s]).mean()), n_own=len(u["rv"]), status="нет числа в таблицах")
          for s in dict.fromkeys(reg["series"]) if s not in set(c["series"])]
    return pd.concat([c, pd.DataFrame(no), identity(u)], ignore_index=True)[["kind", "series", "window", "source", "source_value", "own_value", "abs_diff", "n_source", "n_own", "status"]]


PAIRS = [("J210comp:HAR", "K1", "V1/V2: HAR-J из components.parquet"), ("J210comp:XJ", "K2", "V1/V2: XGBoost из components.parquet"), ("J210comp:LF", "K3", "V1/V2: LightGBM из components.parquet"),
         ("DEEP:s2_harj", "K1", "sber_h1_deep_v1: HAR-J"), ("PANlgbm:D2", "D2", "V3: финалист lgbm_D2 = контроль D2"), ("V4:K8", "K8", "V4: K8")]


def identity(u: dict) -> pd.DataFrame:
    """Совпадение контролей между программами: массивы побайтно (равенство и sha256)."""
    rows = []
    for a, b, why in PAIRS:
        eq = bool(np.array_equal(u["f"][a], u["f"][b]))
        rows.append(dict(kind="identity", series=a, window="все 2271 дат", source="%s = %s (sha256 %s | %s)" % (why, b, R.sha_arr(u["f"][a])[:16], R.sha_arr(u["f"][b])[:16]), status="совпало" if eq else "расхождение"))
    v6 = FD.load()["fs"]
    for k in ("K1", "K2", "K3", "M", "K8", "V4:NTP2"):
        eq = bool(np.array_equal(v6[k], u["f"][k]))
        rows.append(dict(kind="identity", series=k, window="все 2271 дат", source="V6 FIN (v6_fin_data) = лист 0 (sha256 %s | %s)" % (R.sha_arr(v6[k])[:16], R.sha_arr(u["f"][k])[:16]), status="совпало" if eq else "расхождение"))
    return pd.DataFrame(rows)


def unexplained(rp: pd.DataFrame) -> pd.DataFrame:
    """Необъяснённые расхождения (статус «нет числа в таблицах» — не расхождение, он перечислен отдельно)."""
    return rp[~rp["status"].isin(["совпало", "нет числа в таблицах"])]
