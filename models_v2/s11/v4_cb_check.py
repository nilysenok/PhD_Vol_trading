"""V4-C, лист C-b: проверки гейтов G19–G21 для `run_v4 --stage Cb --check`. Каждая функция возвращает список нарушений (пусто — гейт закрыт)."""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_b_decl as DECL, v4_c_cache as CC, v4_c_run as CR, v4_cb_report as RP, v4_cb_run as R, v4_cb_stage as ST, v4_cb_years as Y, v4_io as IO, v4_z_build as ZB, v4_z_feat as ZF

DAY = "2018-06-01"          # обрезка боевого пути: все периоды (месяц/год) — «закрыто + 1» меняет прогноз даты <= DAY


def _declared() -> float:
    return dt.datetime.strptime(json.loads(DECL.CP.read_text())["C_b_declared_at"], "%Y-%m-%d %H:%M:%S.%f").timestamp()


def g19_files(r: dict) -> list:
    bad, P = [], r["P"]
    for cfg in ST.CFGS:
        f, w = pd.read_parquet(CC.OUT / (cfg + ".parquet")), pd.read_parquet(CC.OUT / (cfg + "_weights.parquet"))
        if len(f) != len(P) or f["ticker"].nunique() != 17 or not (np.isfinite(f["F"]) & (f["F"] > 0)).all() or not ((f["w"] > 0) & (f["w"] < 1)).all():
            bad.append("G19: %s — состав/нефинитные F/вес вне (0,1)" % cfg)
        pn = w[w["ticker"] == "PANEL"]
        if cfg != "C7" and pn["converged"].mean() < 0.95:
            bad.append("G19: %s — сходимость %.3f < 0,95" % (cfg, pn["converged"].mean()))
        if (CC.OUT / (cfg + ".parquet")).stat().st_mtime <= _declared():
            bad.append("G19: %s создан не позже объявления листа" % cfg)
    c9 = pd.read_parquet(CC.OUT / "C9.parquet")
    s = c9[["wH", "wK2", "wK3"]]
    if (s.sum(1) - 1.0).abs().max() > 1e-12 or (s < 0).any().any() or not np.allclose(c9["w"], c9["wK2"] + c9["wK3"], atol=1e-15):
        bad.append("G19: C9 — веса вне симплекса")
    w6 = pd.read_parquet(CC.OUT / "C6_weights.parquet")
    g = w6[(w6["ticker"] == "PANEL")]["b_ewqd"].dropna()
    if g.empty or g.min() < -1e-12:
        bad.append("G19: C6 — γ < 0 (min %.3g)" % (g.min() if len(g) else np.nan))
    return bad


def g19_pit(r: dict) -> list:
    bad, inp = [], ZB.real_inputs()
    P = r["P"][r["P"]["date"] < "2018-09-01"]
    for cfg in ST.CFGS:
        if R.pit_violation(cfg, P, inp, DAY):
            bad.append("G19: %s — обрезка: данные после d меняют прогноз дня d" % cfg)
    zl = ZF.build_panel_z(P, inp)
    a, b = CR.forecast("C1", P, zl)[0]["F"].to_numpy(), R.forecast("C1x", P, zl)[0]["F"].to_numpy()
    if not np.allclose(a, b, rtol=1e-8, atol=0):
        bad.append("G19: линейный контур не воспроизводит C1 (1e-8)")
    return bad


def g20(r: dict) -> list:
    bad, cols, ca = [], r["cols"], r["ca"]
    for name, rows in (("C_COMPARISON.csv", len(cols) * 4), ("C_TESTS.csv", len(ST.ALLC) * 5 * 4), ("C_SBER.csv", len(r["cols_main"]) * 4), ("C_YEARS.csv", len(r["cols_main"]) * 10)):
        if len(IO.read_csv(name)) != rows:
            bad.append("G20: %s — строк %d, ожидалось %d" % (name, len(IO.read_csv(name)), rows))
    comp, crit = IO.read_csv("C_COMPARISON.csv"), S.crit_panel(r["k"], cols).set_index("series")["crit_panel"]
    if any(abs(float(comp[comp["series"] == s]["crit_panel"].iloc[0]) - v) > 1e-12 for s, v in crit.items()):
        bad.append("G20: crit_panel в C_COMPARISON не равен hp_stats_sel.crit_panel")
    old = ca["crit"].set_index("series")["crit_panel"]
    if any(crit[s] != old[s] for s in old.index):
        bad.append("G20: crit_panel рядов листа C-a / контролей изменился после добавления C5–C10")
    keep = old.index.tolist()
    if not r["comp"][r["comp"]["series"].isin(keep)].reset_index(drop=True).equals(ca["comp"][ca["comp"]["series"].isin(keep)].reset_index(drop=True)):
        bad.append("G20: строки C1–C4 / контролей в C_COMPARISON не совпали с листом C-a побитно")
    yt = IO.read_csv("C_YEARS.csv")
    yc = Y.crit_from_years(yt)
    if any(abs(yc[s] - crit[s]) > 1e-12 for s in r["cols_main"]):
        bad.append("G20: сумма crit_part по годам не равна crit_panel")
    t = IO.read_csv("C_TESTS.csv")
    m = t[t["kind"] == "main"].dropna(subset=["p_holm"])
    if m.empty or (m["p_holm_wide"] < m["p_holm"] - 1e-15).any() or set(t["config"]) != set(ST.ALLC) or set(t["comparator"]) != {"k1", "k2", "k3", "d2", "B6"}:
        bad.append("G20: C_TESTS — Холм / состав")
    if IO.read_csv("C_PER_TICKER.csv")["ticker"].nunique() != 17 or IO.read_csv("C_MCS.csv").empty:
        bad.append("G20: C_PER_TICKER/C_MCS неполны")
    s = IO.read_csv("C_SBER.csv")
    sa = s[(s["window"] == "all") & (s["series"] == "k1")]
    if sa.empty or int(sa["n"].iloc[0]) != 2271 or abs(float(sa["ref_a1_roll_126"].iloc[0]) - 0.20814) > 1e-4:
        bad.append("G20: C_SBER — не 2271 даты или нет ориентира")
    return bad + g20_loto_trials(r)


def g20_loto_trials(r: dict) -> list:
    bad, lt = [], IO.read_csv("C_LOTO.csv")
    need = {"C1", r["best"]}
    L = lt[lt["held_out"].isin(IO.M16 + ["M16"])]
    if set(lt["config"]) != need or not L["delta_zero_check"].astype(str).eq("True").all() or any(set(L[L["config"] == c]["held_out"]) != set(IO.M16 + ["M16"]) for c in need):
        bad.append("G20: C_LOTO неполон или параметры исключённой ≠ панельным")
    keys = [k for k in pd.read_csv(IO.REP / "ALL_TRIALS.csv")["key"] if k.startswith("Ж-222:C") and not k.startswith("Ж-222:CS")]       # отступление (лист DX-CS): строки CS1, CS2 другой стадии сохраняются
    if sorted(k.split(":")[1] for k in keys) != sorted(ST.ALLC):
        bad.append("G20: ALL_TRIALS — нет строк Ж-222:C1…C10 (ровно по одной)")
    return bad


def g21(r: dict) -> list:
    txt = (IO.REP / "C_REPORT.md").read_text()
    crit = S.crit_panel(r["k"], r["cols"]).set_index("series")["crit_panel"]
    best = min(ST.ALLC, key=lambda c: crit[c])
    thr = json.loads(DECL.CP.read_text())["go_threshold_B"]
    want = RP.first_line(crit, r["comp"], best, thr)
    bad = [] if txt.splitlines()[0] == want and best == r["best"] else ["G21: первая строка C_REPORT не соответствует лучшей из C1–C10 / go-stop"]
    if not all(x in txt for x in ("C5", "C10", "по годам", "2017", "SBER", "LOTO", "Отступления")):
        bad.append("G21: в C_REPORT нет обязательных разделов")
    cp = json.loads(DECL.CP.read_text())
    if cp.get("C_stop_go") != ("go" if crit[best] <= thr else "stop") or cp.get("used_after_C_b") != cp["used_after_C_a"] + 6:
        bad.append("G21: CHECKPOINT — C_stop_go / расход")
    return bad


def all_checks(r: dict) -> list:
    return g19_files(r) + g19_pit(r) + g20(r) + g21(r)
