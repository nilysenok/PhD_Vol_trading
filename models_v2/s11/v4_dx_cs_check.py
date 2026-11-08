"""V4-DX-CS: проверки гейтов G27–G28 для `run_v4 --stage CS --check`. Каждая функция возвращает список нарушений (пусто — гейт закрыт)."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_b_decl as DECL, v4_c_cache as CC, v4_cb_run as CB, v4_cb_years as Y, v4_cs_fit as FS, v4_cs_run as R, v4_cs_stage as ST, v4_dx_cs_report as RP, v4_dx_iso as DXI, v4_io as IO,
                           v4_isolation as ISO, v4_z_build as ZB, v4_z_feat as ZF)

DAY = "2018-06-01"          # обрезка боевого пути: первый торговый день месяца
CUT = "2018-09-01"


def _declared() -> float:
    return dt.datetime.strptime(json.loads(DECL.CP.read_text())["CS_declared_at"], "%Y-%m-%d %H:%M:%S.%f").timestamp()


def g27_files(r: dict) -> list:
    bad, P = [], r["P"]
    for cfg in ST.CFGS:
        f, w = pd.read_parquet(CC.OUT / (cfg + ".parquet")), pd.read_parquet(CC.OUT / (cfg + "_weights.parquet"))
        if len(f) != len(P) or f["ticker"].nunique() != 17 or not (np.isfinite(f["F"]) & (f["F"] > 0)).all() or not ((f["w"] > 0) & (f["w"] < 1)).all():
            bad.append("G27: %s — состав/нефинитные F/вес вне (0,1)" % cfg)
        if w[w["ticker"] == "PANEL"]["converged"].mean() < 0.95:
            bad.append("G27: %s — сходимость < 0,95" % cfg)
        if (CC.OUT / (cfg + ".parquet")).stat().st_mtime <= _declared():
            bad.append("G27: %s создан не позже объявления листа" % cfg)
        c8 = pd.read_parquet(CC.OUT / "C8.parquet")
        a, b = (x[x["date"] < "2017-02-01"].sort_values(["date", "ticker"])["F"].to_numpy() for x in (f, c8))
        if len(a) == 0 or not np.array_equal(a, b):
            bad.append("G27: %s — приорные месяцы (до 60 пар) не равны C8 побитно" % cfg)
    d = IO.read_csv("CS_DIAG.csv")
    need = {"k1", "k2", "k3", "d2", "C8", "CS1", "CS2", "a_C8_params_end2017", "b_best_const_weight_2017", "loss_cold_start_a", "loss_cold_start_b", "prior_before_2017"}
    if not need <= set(d["item"]) or set(d["group"]) != {"M16", "SBER"} or "нет данных" not in str(d[d["item"] == "prior_before_2017"]["note"].iloc[0]):
        bad.append("G27: CS_DIAG неполон (нужны (а), (б), потери, строка «приор до 2017 — нет данных»)")
    return bad


def g27_pit(r: dict) -> list:
    bad, inp = [], ZB.real_inputs()
    P = r["P"][r["P"]["date"] < CUT]
    for cfg in ST.CFGS:
        if R.pit_violation(cfg, P, inp, DAY):
            bad.append("G27: %s — обрезка: данные после d меняют прогноз дня d" % cfg)
    P0 = r["P"][r["P"]["date"] < "2017-07-01"]
    zl, orig = ZF.build_panel_z(P0, inp), FS.sched
    FS.sched = lambda n: np.full(np.shape(n), 250.0) if np.ndim(n) else 250.0
    try:
        flat = {c: R.forecast(c, P0, zl)[0]["F"].to_numpy() for c in ST.CFGS}
    finally:
        FS.sched = orig
    ref = CB.forecast("C8", P0, zl)[0]["F"].to_numpy()
    bad += ["G27: %s при расписании ≡ 250 (n → 0) не воспроизводит C8 (1e-8)" % c for c, v in flat.items() if not np.allclose(v, ref, rtol=1e-8, atol=0)]
    return bad + ["G26/G27: диагностика импортирована путём прогноза: %s" % x for x in DXI.violators() + ISO.violators()]


def g28(r: dict) -> list:
    bad, cols = [], r["cols"]
    for name, rows in (("CS_COMPARISON.csv", len(cols) * 4), ("CS_TESTS.csv", len(ST.CFGS) * 7 * 4), ("CS_SBER.csv", len(cols) * 4), ("CS_YEARS.csv", len(cols) * 10)):
        if len(IO.read_csv(name)) != rows:
            bad.append("G28: %s — строк %d, ожидалось %d" % (name, len(IO.read_csv(name)), rows))
    crit = S.crit_panel(r["k"], cols).set_index("series")["crit_panel"]
    comp = IO.read_csv("CS_COMPARISON.csv")
    if any(abs(float(comp[comp["series"] == s]["crit_panel"].iloc[0]) - v) > 1e-12 for s, v in crit.items()):
        bad.append("G28: crit_panel в CS_COMPARISON не равен hp_stats_sel.crit_panel")
    old = IO.read_csv("C_COMPARISON.csv")
    if any(abs(float(old[old["series"] == s]["crit_panel"].iloc[0]) - crit[s]) > 1e-12 for s in ("k1", "k2", "k3", "d2", "C1", "C4", "C8")):
        bad.append("G28: crit_panel контролей/C1/C4/C8 изменился против C_COMPARISON.csv")
    yc = Y.crit_from_years(IO.read_csv("CS_YEARS.csv"))
    if any(abs(yc[s] - crit[s]) > 1e-12 for s in cols):
        bad.append("G28: сумма crit_part по годам не равна crit_panel")
    t = IO.read_csv("CS_TESTS.csv")
    m = t[t["kind"] == "main"].dropna(subset=["p_holm"])
    if m.empty or (m["p_holm_wide"] < m["p_holm"] - 1e-15).any() or set(t["comparator"]) != {"k1", "k2", "k3", "d2", "C1", "C4", "C8"} or set(t["config"]) != set(ST.CFGS):
        bad.append("G28: CS_TESTS — Холм / состав")
    if IO.read_csv("CS_PER_TICKER.csv")["ticker"].nunique() != 17 or IO.read_csv("CS_MCS.csv").empty:
        bad.append("G28: CS_PER_TICKER/CS_MCS неполны")
    s = IO.read_csv("CS_SBER.csv")
    sa = s[(s["window"] == "all") & (s["series"] == "k1")]
    if sa.empty or int(sa["n"].iloc[0]) != 2271 or abs(float(sa["ref_a1_roll_126"].iloc[0]) - 0.20814) > 1e-4:
        bad.append("G28: CS_SBER — не 2271 даты или нет ориентира")
    lt = IO.read_csv("CS_LOTO.csv")
    L = lt[lt["held_out"].isin(IO.M16 + ["M16"])]
    if set(lt["config"]) != {r["best"]} or set(L["held_out"]) != set(IO.M16 + ["M16"]) or not L["delta_zero_check"].astype(str).eq("True").all():
        bad.append("G28: CS_LOTO неполон или параметры исключённой ≠ панельным")
    keys = [k for k in pd.read_csv(IO.REP / "ALL_TRIALS.csv")["key"] if k.startswith("Ж-222:") and not k.startswith(("Ж-222:NTP", "Ж-222:CLD", "Ж-222:K8", "Ж-222:RQC"))]       # отступление (листы NTP, CLD, K8-RQC): строки NTP1–NTP4, CLD1–CLD4, K8, RQC1–RQC3 других стадий не считаются
    pref = sorted(k.split(":")[1] for k in keys)
    if sorted(x for x in pref if x.startswith("CS")) != ST.CFGS or len(keys) != 23 or len(set(pref)) != 23:
        bad.append("G28: ALL_TRIALS — нет строк Ж-222:CS1, CS2 (по одной) или потеряны строки других стадий (всего ожидалось 23)")
    return bad


def g28_report(r: dict, diag: pd.DataFrame) -> list:
    txt = (IO.REP / "CS_REPORT.md").read_text()
    cp = json.loads(DECL.CP.read_text())
    bad = [] if txt.splitlines()[0] == RP.first_line(r, diag) else ["G28: первая строка CS_REPORT не соответствует пересчёту go/stop"]
    if len(txt.split()) > 1000:
        bad.append("G28: CS_REPORT %d слов > 1000" % len(txt.split()))
    if not all(x in txt for x in ("2017", "2018", "SBER", "LOTO", "Отступления", "механическим", "нет данных")):
        bad.append("G28: в CS_REPORT нет обязательных разделов")
    if cp.get("CS_stop_go") != ("go" if r["go"]["cfg"][r["best"]]["go"] else "stop") or cp.get("used_after_CS") != cp["used_after_E"] + 2 or not txt.startswith(cp["CS_stop_go"]):
        bad.append("G28: CHECKPOINT — CS_stop_go / расход")
    return bad


def all_checks(r: dict, diag: pd.DataFrame) -> list:
    return g27_files(r) + g27_pit(r) + g28(r) + g28_report(r, diag)
