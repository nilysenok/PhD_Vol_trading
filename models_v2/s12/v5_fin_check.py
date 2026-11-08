"""V5 лист FIN, G23–G26: сверка (ТОЛЬКО ЧТЕНИЕ). (1) числа FIN_* против таблиц листов 0, 1a, 2 (допуск 1e-12) по общим ключам; (2) FIN_* на диске = пересчёт в памяти (1e-9; секунды замера FIN_COST не сравниваются);
(3) реестр и CHECKPOINT[FIN]. Итоговый документ сверяет `v5_fin_docheck`."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s11 import v4_fin_check as FC
from models_v2.s12 import v5_0_io as IO, v5_fin_cost as CO, v5_fin_data as D, v5_fin_stage as ST

TOL = 1e-12
MET = ["qlike", "rmse", "mae", "r2_rv", "r2_ln"]
TST = ["n_dates", "delta_mean", "dm_stat", "dm_p", "boot_lo", "boot_hi", "boot_p"]
CTRL = ["k1", "k2", "k3", "d2", "K8"]


def num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.lower().replace({"true": "1", "false": "0", "nan": np.nan, "": np.nan}), errors="coerce")


def cmp(label: str, fin: pd.DataFrame, ref: pd.DataFrame, keys: list, cols: list) -> list:
    """Макс. |ref − fin| по cols на ключах ref; ключ ref, которого нет в fin (или NaN против числа), — ошибка (inf)."""
    f, r = fin.set_index(keys)[cols].apply(num), ref.set_index(keys)[cols].apply(num)
    j = r.join(f, how="left", rsuffix="_f")
    worst = 0.0
    for c in cols:
        a, b = j[c].to_numpy(float), j[c + "_f"].to_numpy(float)
        d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
        worst = max(worst, np.inf if np.isnan(d).any() else float(d.max(initial=0.0)))
    return [] if len(j) and worst <= TOL else ["%s: макс. расхождение %.3g (допуск %.0e), ключей %d" % (label, worst, TOL, len(j))]


def one_sheet(t: dict, pre: str, series: list, best: str) -> list:
    rd, fa = IO.read_csv, t["FIN_ALL"]
    cmpd = rd(pre + "_COMPARISON.csv")
    m = cmpd[cmpd["section"] == "metrics"]
    bad = cmp(pre + " метрики", fa[fa["section"] == "metrics"], m, ["series", "window"], MET + ["bias_ln", "crit_panel", "EqY", "n_rows_m16"] + [c for c in m.columns if c.startswith("d_qlike_")])
    bad += cmp(pre + " go", fa[fa["section"] == "go"], cmpd[cmpd["section"] == "go"], ["series"], ["crit_panel", "crit_no2017", "crit_trim1", "thr1", "thr2", "thr3", "c1", "c2", "c3", "go", "EqY"])
    bad += cmp(pre + " годы", t["FIN_YEARS"], rd(pre + "_YEARS.csv"), ["series", "year"], ["qlike_m16"]) + cmp(pre + " SBER", t["FIN_SBER"], rd(pre + "_SBER.csv"), ["series", "window"], ["n"] + MET)
    p = rd(pre + "_PER_TICKER.csv")
    bad += cmp(pre + " бумаги", t["FIN_TICKERS"], p[p["ticker"] != "COUNT"], ["ticker", "window", "series"], ["n"] + MET + ["bias_ln", "d_qlike_k1_pct_of_k1"])
    bad += cmp(pre + " счётчики", t["FIN_TICKERS"], p[p["ticker"] == "COUNT"], ["ticker", "window", "series"], ["n_improve_vs_K8", "n_improve_vs_best_k2k3", "n_tickers"])
    ts, ft = rd(pre + "_TESTS.csv"), t["FIN_TESTS"]
    bad += cmp(pre + " тесты", ft[ft["config"] == best], ts[ts["config"] == best], ["config", "comparator", "window"], TST)
    e = rd(pre + "_ENCOMPASS.csv")
    bad += cmp(pre + " охват", t["FIN_ENCOMPASS"], e, ["best", "base", "window"], [c for c in e.columns if c not in ("best", "base", "window")])
    mc = rd(pre + "_MCS.csv")
    return bad + cmp(pre + " MCS (уровни)", t["FIN_MCS"], mc[mc["model"].isin(series)], ["window", "model"], ["mean_qlike", "n_dates"])


def ablation_sheet(t: dict) -> list:
    sa, fa = IO.read_csv("SP_ABLATION.csv"), t["FIN_ABLATION"]
    pr = fa[fa["section"] == "pair"]
    pairs = set(zip(pr["a"], pr["b"]))
    inc = lambda df: df[[(a, b) in pairs for a, b in zip(df["a"], df["b"])]]
    bad = cmp("SP абляция QLIKE", fa[fa["section"] == "qlike"], inc(sa[sa["section"] == "pair"]), ["window", "a", "b"], TST)
    bad += cmp("SP абляция Δ crit", pr, inc(sa[sa["section"] == "crit_diff"]), ["a", "b"], ["d_crit_panel", "d_crit_no2017", "d_crit_trim1", "d_EqY"])
    return bad + cmp("SP абляция crit", fa[fa["section"] == "crit"], sa[sa["section"] == "crit"], ["a"], ["crit_panel", "crit_no2017", "crit_trim1", "EqY"])


def stage0_sheet(t: dict, cp: dict) -> list:
    fa = t["FIN_ALL"]
    a = fa[(fa["section"] == "metrics") & (fa["window"] == "all")]
    ce = IO.read_csv("CRIT_EQY.csv")
    bad = cmp("лист 0 CRIT_EQY", a, ce[ce["series"].isin(CTRL + ["M"])], ["series"], ["crit_panel", "EqY"])
    got = a.set_index("series")["crit_panel"]
    return bad + ["CHECKPOINT controls %s: %.3g" % (s, abs(got[s] - v)) for s, v in cp["controls_crit_panel"].items() if abs(got[s] - v) > TOL]


def cross(r: dict) -> list:
    t, cp, d = r["t"], r["d"]["cp"], r["d"]
    return one_sheet(t, "GS", CTRL + D.GS, d["best_gs"]) + one_sheet(t, "SP", CTRL + D.SP, d["best_sp"]) + ablation_sheet(t) + stage0_sheet(t, cp)


def cost_ok(disk: pd.DataFrame, mem: pd.DataFrame) -> bool:
    keep = ["section", "stage", "command", "rc", "n_params", "note"]
    p = lambda x: x[x["section"] == "params"][keep].reset_index(drop=True)
    tm = disk[disk["section"] == "time"]
    return FC.frames_equal(p(disk), p(mem), 0.0) and list(tm["stage"].astype(str)) == [s for s, _ in CO.STAGES] and bool((tm["rc"] == 0).all()) and bool((tm["seconds"] > 0).all())


def disk_vs_memory(r: dict) -> list:
    bad = []
    for n in D.FIN_TABLES:
        if not (IO.REP / (n + ".csv")).exists():
            bad.append("%s: нет файла" % n)
            continue
        disk, mem = IO.read_csv(n + ".csv"), r["t"][n]
        if not (cost_ok(disk, mem) if n == "FIN_COST" else FC.frames_equal(disk, mem, 1e-9)):
            bad.append("%s: диск ≠ пересчёт" % n)
    return bad


def registry(r: dict) -> list:
    cp, tr = D.checkpoint(), pd.read_csv(IO.REP / "ALL_TRIALS.csv")
    cfgs = sorted(json.loads(c)["cfg"] for c in tr["config"])
    bad = []
    if cfgs != sorted(D.GS + D.SP) or tr["key"].duplicated().any() or not tr["key"].str.startswith("Ж-226:").all():
        bad.append("G25: ALL_TRIALS: %d строк, конфигурации %s" % (len(tr), cfgs))
    f = cp.get("FIN", {})
    if f.get("used_total") != 9 or f.get("budget") != 24 or (f.get("N_declared_before_stats"), f.get("N_wide")) != (704, 3104) or f.get("configs_spent_by_FIN") != 0:
        bad.append("G25: CHECKPOINT[FIN]: расход/N: %s" % {k: f.get(k) for k in ("used_total", "budget", "N_declared_before_stats", "N_wide", "configs_spent_by_FIN")})
    if set(f.get("closed_directions", {})) != set(ST.CLOSED) or f.get("other_keys_sha") != ST.other_keys_sha(cp):
        bad.append("G25: CHECKPOINT[FIN]: закрытые направления или прежние ключи изменены")
    if f.get("tables_sha") != ST.tables_sha() or f.get("cache_keys") != r["d"]["keys"]:
        bad.append("G25: CHECKPOINT[FIN]: хэш таблиц FIN или ключи кэша не сходятся")
    return bad
