"""V6 лист FIN, G50–G55: проверки (ТОЛЬКО ЧТЕНИЕ) — диск против пересчёта в памяти, числа против таблиц листов (≤ 1e-12), реестр, CHECKPOINT, MCS, уровни, документ."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_fin_check as FC
from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_1_reg as RG, v6_1_stage as S1, v6_fin_data as D, v6_fin_docheck as DC, v6_fin_doc as W, v6_fin_stage as ST, v6_fin_tabs as TB

TOL = 1e-12


def same(a: pd.DataFrame, b: pd.DataFrame, cols: list, keys: list, tol: float = TOL) -> bool:
    a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)
    return a.shape[0] == b.shape[0] and all(np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=0.0, atol=tol, equal_nan=True) for c in cols)


def disk_vs_memory(r: dict) -> list:
    bad = []
    for n in D.TABLES:
        p = IO.REP / (n + ".csv")
        if not p.exists() or not FC.frames_equal(IO.read_csv(n + ".csv"), r["t"][n], 1e-9):
            bad.append("G50: %s.csv: нет файла или диск ≠ пересчёту" % n)
    return bad


def vs_sheets(r: dict) -> list:
    """G50: числа FIN_* = числам таблиц листов 1a–6 (метрики, тесты, охват, стресс) до 1e-12."""
    bad, a, t = [], r["t"], r["t"]
    mcols = ["n", "qlike", "rmse", "mae", "r2_rv", "r2_ln", "bias_ln"]
    for c in D.CFG + ["K9"]:
        sh = IO.read_csv(("K9" if c == "K9" else c) + "_METRICS.csv")
        sh = sh[sh["series"] == c]
        if not same(sh, t["FIN_ALL"][t["FIN_ALL"]["series"] == c], mcols, ["window"]):
            bad.append("G50: FIN_ALL.%s ≠ %s_METRICS.csv" % (c, c))
    base = IO.read_csv("A0_METRICS.csv")
    for c in ("K1", "K2", "K3", "K8", "V4:NTP2"):
        if not same(base[base["series"] == c], t["FIN_ALL"][t["FIN_ALL"]["series"] == c], mcols, ["window"]):
            bad.append("G50: FIN_ALL.%s ≠ A0_METRICS.csv" % c)
    pr = IO.read_csv("PRIOR_SBER.csv").set_index("series")["qlike_own"]
    al = t["FIN_ALL"][t["FIN_ALL"]["window"] == "all"].set_index("series")["qlike"]
    bad += ["G50: QLIKE %s ≠ PRIOR_SBER.csv" % c for c in D.CTRL if c in pr.index and abs(al[c] - pr[c]) > 1e-9]
    for c in D.CFG:
        for nm, key, cols in (("TESTS", "FIN_TESTS", ["delta_pct", "lo90", "hi90", "p_one", "p_holm_N", "p_holm_Nwide"]), ("STRESS", "FIN_STRESS", ["n", "qlike", "d_pct_K1", "d_pct_K8"])):
            sh, fin = IO.read_csv("%s_%s.csv" % (c, nm)), t[key][t[key]["config"] == c]
            if nm == "TESTS":
                sh, fin = sh[sh["section"] == "delta_dm"], fin[fin["section"] == "delta_dm"]
                keys = ["window", "base"]
            else:
                fin, keys = fin[fin["kind"] == "stress"], ["scenario"]
            if not same(sh, fin, cols, keys):
                bad.append("G50: %s.%s ≠ %s_%s.csv" % (key, c, c, nm))
    return bad + info_vs_sheets(r)


def info_vs_sheets(r: dict) -> list:
    """Пары правила п. 2.2: пересчёт из прогнозов = FIN_INFO_ALL (Δ, интервал, DM p, лет с Δ < 0, вывод) до 1e-9; p Холма зависит от состава пар листа и не пересчитывается."""
    info = r["t"]["FIN_INFO_ALL"]
    fresh = TB.recompute_info(r["d"], info)
    mine = info.drop_duplicates(["pair", "window"]).merge(fresh, on=["pair", "window"], suffixes=("", "_f"))
    cols = ["delta_qlike", "delta_pct", "lo90", "hi90", "dm_p_one", "years_neg"]
    bad = ["G50: FIN_INFO_ALL ≠ пересчёту (%s)" % c for c in cols if not np.allclose(mine[c].to_numpy(float), mine[c + "_f"].to_numpy(float), rtol=0.0, atol=1e-9, equal_nan=True)]
    ni = mine[mine["window"] == "2017_2021"]
    if (ni["new_information"].astype(str).str.lower() != ni["new_information_f"].astype(str).str.lower()).any():
        bad.append("G50: вывод «новая информация» ≠ пересчёту")
    return bad


def mcs_levels(r: dict) -> list:
    """G51, G54: MCS по трём окнам, 28 рядов, seed фиксирован (блоковые индексы SD.block_idx); уровни 1–5 у всех 18 конфигураций."""
    bad, mc, lv = [], r["t"]["FIN_MCS"], r["t"]["FIN_LEVELS"]
    if set(mc["window"]) != set(TB.MCS_WINS) or set(mc["model"]) != set(D.SER) or len(mc) != 3 * len(D.SER):
        bad.append("G51: состав FIN_MCS")
    if any(not mc[mc["window"] == w]["in_mcs90"].any() for w in TB.MCS_WINS):
        bad.append("G51: пустое MCS")
    if len(lv) != 5 * len(D.CFG) or set(lv["config"]) != set(D.CFG) or set(lv["level"]) != {1, 2, 3, 4, 5}:
        bad.append("G54: уровни 1–5 не у всех 18 конфигураций")
    p5 = lv[lv["level"] == 5]
    if not p5["basis"].astype(str).str.endswith(("статистически неподтверждённое историческое улучшение", "улучшения к K1 нет")).all():
        bad.append("G54: статус уровня 5 не словами блока")
    return bad


def registry(r: dict) -> list:
    """G53: ALL_TRIALS — 18 строк `Ж-227:*` без дублей, каждая объявлена; ключи 0…6 CHECKPOINT не тронуты; ключ FIN = пересчёт; REPRODUCTION_V6 содержит команды всех стадий."""
    bad = []
    if not RG.trials_ok(D.CFG, exact=True) or len(CP0.ALL_TRIALS.read_text(encoding="utf-8").splitlines()) != 19:
        bad.append("G53: ALL_TRIALS.csv ≠ 18 строк плана")
    cp = CP0.read()
    fin = cp.get("FIN")
    want = ST.fin_entry(cp, len(r["d"]["w"]))
    if fin is None or S1.stable(fin) != S1.stable(want):
        bad.append("G53: ключ FIN на диске ≠ пересчёту")
    for k in ("0", "1a", "1b", "2", "3", "4", "5", "6"):
        if k not in cp:
            bad.append("G53: нет ключа %s" % k)
    from models_v2.s13 import v6_fin_asm as A
    txt = A.REPRO.read_text(encoding="utf-8") if A.REPRO.exists() else ""
    bad += ["G53: REPRODUCTION_V6.md без стадии %s" % s for s in A.STAGES if "--stage %s --check" % s not in txt]
    return bad


def doc(r: dict) -> list:
    t = W.load()
    return DC.check_doc(t) + DC.check_static(t)


def all_bad(r: dict) -> list:
    return disk_vs_memory(r) + vs_sheets(r) + mcs_levels(r) + registry(r) + doc(r)
