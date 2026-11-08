"""V4-C: проверки гейтов G15–G18 для `run_v4 --stage Ca --check`. Каждая функция возвращает список нарушений (пусто — гейт закрыт)."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_b_data as D, v4_b_decl as DECL, v4_b_ops as OP, v4_c_cache as CC, v4_c_fit as CF, v4_c_run as CR, v4_c_stage as ST, v4_io as IO, v4_z_build as ZB, v4_z_feat as ZF

CSVS = ["Z_REPORT.csv", "C_COMPARISON.csv", "C_PER_TICKER.csv", "C_TESTS.csv", "C_MCS.csv", "C_WEIGHTS.csv", "C_SBER.csv", "C_LOTO.csv", "C_REPORT.md"]


def _declared() -> float:
    return dt.datetime.strptime(json.loads(DECL.CP.read_text())["C_declared_at"], "%Y-%m-%d %H:%M:%S.%f").timestamp()


def g15(o: dict) -> list:
    bad, Zl, P = [], o["Zl"], o["P"]
    if sorted(Zl["ticker"].unique()) != sorted(IO.T17) or len(Zl) != len(P) or not set(ZF.ALL) <= set(Zl.columns):
        bad.append("G15: Z — состав бумаг/дат/колонок не равен панели")
    for t in IO.T17:
        if not (CC.OUT / ("Z_%s.parquet" % t)).exists():
            bad.append("G15: нет Z_%s.parquet" % t)
    ref = ZF.cross(Zl, IO.T17).reindex(pd.DatetimeIndex(Zl["date"]))
    if not (np.allclose(ref["med_lrv"].to_numpy(), Zl["med_lrv"].to_numpy(), atol=0, equal_nan=True) and np.allclose(ref["sur_mkt"].to_numpy(), Zl["sur_mkt"].to_numpy(), atol=0, equal_nan=True)):
        bad.append("G15: поперечные признаки не равны медиане по 17 бумагам")
    try:
        ZF.cross(Zl, IO.T17 + ["GCHE"])
        bad.append("G15: бумага E11 допущена в поперечные признаки")
    except ValueError:
        pass
    z = o["zrep"]
    if set(z["feature"]) != set(ZF.ALL) or z["leak_suspect"].any() or not z["truncation_ok"].all() or z["coverage"].min() < 0.999:
        bad.append("G15: Z_REPORT — признак с утечкой по сдвигу, обрезка не прошла или покрытие < 99,9 %%: %s" % z[["feature", "leak_suspect", "coverage"]].to_dict("records"))
    inp = ZB.real_inputs()
    for cfg in ("C1", "C4"):
        if CR.pit_violation(cfg, P[P["date"] < "2018-09-01"], inp, "2018-06-01"):
            bad.append("G15: %s — обрезка: данные после d меняют прогноз дня d" % cfg)
    files = [IO.REP / f for f in CSVS if (IO.REP / f).exists()] + sorted(CC.OUT.glob("Z_*.parquet")) + sorted(CC.OUT.glob("C*.parquet"))
    return bad + ["G15: файл создан не позже объявления листа: " + str(f) for f in files if f.stat().st_mtime <= _declared()]


def g16(o: dict) -> list:
    bad, P = [], o["P"]
    for cfg in ST.BASE:
        f, w = pd.read_parquet(CC.OUT / (cfg + ".parquet")), pd.read_parquet(CC.OUT / (cfg + "_weights.parquet"))
        if len(f) != len(P) or f["ticker"].nunique() != 17 or not (np.isfinite(f["F"]) & (f["F"] > 0)).all() or not ((f["w"] > 0) & (f["w"] < 1)).all():
            bad.append("G16: %s — состав/нефинитные F/вес вне (0,1)" % cfg)
        pn = w[w["ticker"] == "PANEL"]
        if pn["converged"].mean() < 0.95 or w["alpha"].isna().any():
            bad.append("G16: %s — сходимость %.3f < 0,95 или пустые коэффициенты" % (cfg, pn["converged"].mean()))
    H, M = P["h"].to_numpy(float), OP.mean_m(P["k2"], P["k3"])
    wh = np.random.default_rng(3).uniform(0.05, 0.95, len(H))
    F, _ = CF.forward(np.log(H), np.log(M / H), CF.logit_w_of_b2(wh))
    if not np.allclose(F, OP.forward("B2", H, M, wh)[0], rtol=1e-8, atol=0):
        bad.append("G16: β = 0, δ = 0, ℓ-константа не воспроизводит B2 (1e-8) на реальных H, M")
    w = pd.read_csv(IO.REP / "C_WEIGHTS.csv", comment="#")
    if not {"mean_w", "b_ell", "b_herf", "alpha", "converged"} <= set(w.columns) or w["month"].nunique() < 100:
        bad.append("G16: C_WEIGHTS неполон")
    return bad


def g17(o: dict) -> list:
    bad, n = [], len(o["cols"])
    comp = IO.read_csv("C_COMPARISON.csv")
    for name, rows in (("C_COMPARISON.csv", n * 4), ("C_TESTS.csv", len(ST.BASE) * 5 * 4), ("C_SBER.csv", len(o["cols_main"]) * 4)):
        if len(IO.read_csv(name)) != rows:
            bad.append("G17: %s — строк %d, ожидалось %d" % (name, len(IO.read_csv(name)), rows))
    k = D_eval(o)
    crit = S.crit_panel(k, o["cols"]).set_index("series")["crit_panel"]
    if any(abs(float(comp[comp["series"] == s]["crit_panel"].iloc[0]) - v) > 1e-12 for s, v in crit.items()):
        bad.append("G17: crit_panel в C_COMPARISON не равен hp_stats_sel.crit_panel")
    t = IO.read_csv("C_TESTS.csv")
    m = t[t["kind"] == "main"].dropna(subset=["p_holm"])
    if m.empty or (m["p_holm_wide"] < m["p_holm"] - 1e-15).any() or set(t["comparator"]) != {"k1", "k2", "k3", "d2", "B6"}:
        bad.append("G17: C_TESTS — Холм / компараторы")
    s = IO.read_csv("C_SBER.csv")
    sa = s[(s["window"] == "all") & (s["series"] == "k1")]
    if sa.empty or int(sa["n"].iloc[0]) != 2271 or abs(float(sa["ref_a1_roll_126"].iloc[0]) - 0.20814) > 1e-4:
        bad.append("G17: C_SBER — не 2271 даты или нет ориентира")
    if IO.read_csv("C_PER_TICKER.csv")["ticker"].nunique() != 17 or IO.read_csv("C_MCS.csv").empty:
        bad.append("G17: C_PER_TICKER/C_MCS неполны")
    return bad + g17_loto_trials_b(o)


def D_eval(o: dict) -> pd.DataFrame:
    return D.eval_frame(o["P"], {"B6": pd.read_parquet(CC.OUT / "B_B6.parquet"), **{c: o["fc"][c] for c in ST.BASE},
                                 **{"C4_diag_" + a: o["fc"]["C4"][["ticker", "date", "F_abl_" + a]].rename(columns={"F_abl_" + a: "F"}) for a in ST.ABL}})


def g17_loto_trials_b(o: dict) -> list:
    bad, lt = [], IO.read_csv("C_LOTO.csv")
    L = lt[lt["held_out"].isin(IO.M16 + ["M16"])]
    if set(L["held_out"]) != set(IO.M16 + ["M16"]) or not L["delta_zero_check"].astype(str).eq("True").all() or set(lt["config"]) != {o["best"]}:
        bad.append("G17: C_LOTO неполон или δ ≠ 0 у исключённой")
    tr = pd.read_csv(IO.REP / "ALL_TRIALS.csv")
    if sorted(k.split(":")[1] for k in tr["key"] if k.startswith("Ж-222:C") and k.split(":")[1] in ST.BASE) != ST.BASE:       # строки C5–C10 других стадий сохраняются (G22)
        bad.append("G17: ALL_TRIALS — нет строк Ж-222:C1…C4")
    bc = IO.read_csv("B_COMPARISON.csv")
    if not {"rmse", "mae", "r2_rv", "r2_ln"} <= set(bc.columns) or {"rmse_ln", "r2_lvl"} & set(bc.columns) or {"rmse", "r2_rv"} - set(IO.read_csv("B_PER_TICKER.csv").columns):
        bad.append("G17: колонки B не пересчитаны по hp_hyb_out")
    h = IO.read_csv("archive_pre_metrics/HASHES.csv")
    arch = IO.REP / "archive_pre_metrics"
    if len(h) != 4 or any(IO.sha256(arch / x) != y for x, y in zip(["B_COMPARISON.csv", "B_PER_TICKER.csv"], h[h["role"].str.startswith("до")]["sha256"])):
        bad.append("G17: archive_pre_metrics/HASHES.csv не сходится с архивом")
    return bad


def g18(o: dict) -> list:
    txt = (IO.REP / "C_REPORT.md").read_text()
    first = txt.splitlines()[0]
    crit = S.crit_panel(D_eval(o), o["cols"]).set_index("series")["crit_panel"]
    best = min(ST.BASE, key=lambda c: crit[c])
    bad = [] if best == o["best"] and best in first and ("%.5f" % crit[best]) in first and ("%.5f" % crit["B6"]) in first else ["G18: первая строка C_REPORT не соответствует лучшей C-a"]
    if len(txt.split()) > 1200 or not all(x in txt for x in ("C1", "C4", "HAR-surprise", "Intraday Shape", "SBER", "LOTO", "по годам")):
        bad.append("G18: C_REPORT > 1200 слов или нет обязательных разделов")
    return bad


def all_checks(o: dict) -> list:
    return g15(o) + g16(o) + g17(o) + g18(o)
