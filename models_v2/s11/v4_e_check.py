"""V4-E: проверки гейтов G22–G25 после записи стадии E (`run_v4 --stage E --check`). Возвращают список нарушений (пусто = OK)."""
from __future__ import annotations

import datetime as dt
import json
from unittest import mock

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_b_data as D, v4_cb_years as Y, v4_e_calib as CAL, v4_e_report as RP, v4_e_run as R, v4_e_stage as ST, v4_io as IO

DAY = pd.Timestamp("2019-06-03")
CSV = ["E_COMPARISON.csv", "E_PER_TICKER.csv", "E_TESTS.csv", "E_MCS.csv", "E_YEARS.csv", "E_SBER.csv", "E_LOTO.csv", "E_BIAS.csv"]


def g23_forecasts(o: dict) -> list:
    bad, P = [], o["P"]
    n = len(P)
    for cfg in R.CFGS:
        f = o["fc"][cfg]
        if len(f) != n or f["ticker"].nunique() != 17 or not (np.isfinite(f["F"]) & (f["F"] > 0)).all():
            bad.append("G23: %s — прогнозы неполны / не конечны" % cfg)
        if not (f["b"].abs() <= CAL.B_CLIP + 1e-12).all():
            bad.append("G23: %s — |b| > %.1f" % (cfg, CAL.B_CLIP))
        if (f.loc[f["date"] < "2017-03-20", "b"] != 0).any():
            bad.append("G23: %s — b ≠ 0 до первых 60 закрытых пар" % cfg)
    return bad


def g23_pit(o: dict) -> list:
    """Обрезка (подмена RV после d) на реальных данных; мутация «закрыто + 1» ловится боевым путём (E1, E3, E5, E6)."""
    bad, P, G = [], o["P"], pd.read_parquet(ST.EC.OUT / "B_B2.parquet")
    for cfg in R.CFGS:
        if R.pit_violation(cfg, P, G, DAY):
            bad.append("G23: %s — подмена RV после d меняет F дня ≤ d" % cfg)
    with mock.patch.object(CAL, "closed_date", lambda tp, d: d):
        for cfg in ("E1", "E3", "E5", "E6"):
            if not R.pit_violation(cfg, P, G, DAY):
                bad.append("G23: мутация «закрыто + 1» не поймана на %s" % cfg)
    return bad


def g23_identities(o: dict) -> list:
    bad, P, G = [], o["P"], pd.read_parquet(ST.EC.OUT / "B_B2.parquet")
    e1, _ = R.forecast("E1", P, G, zero_b=True)
    e2, _ = R.forecast("E2", P, G, zero_b=True)
    Pm = P.merge(G[["ticker", "date", "F"]], on=["ticker", "date"]).sort_values(["date", "ticker"])
    if not np.array_equal(e1["F"].to_numpy(), Pm["h"].to_numpy()):
        bad.append("G23: E1 при b ≡ 0 не равен H побитно")
    if not np.array_equal(e2["F"].to_numpy(), Pm["F"].to_numpy()):
        bad.append("G23: E2 при b ≡ 0 не равен G побитно")
    return bad


def g23_loto(o: dict) -> list:
    """Исключённая бумага не входит в панельную ĉ: подмена её RV не меняет ĉ_panel ни в одном дне."""
    P, G, bad = o["P"], pd.read_parquet(ST.EC.OUT / "B_B2.parquet"), []
    for t in ("LKOH", "AFLT"):
        Q = P.copy()
        Q.loc[Q["ticker"] == t, "rv"] *= 5.0
        a, b = R.forecast("E3", P, G, exclude=t)[0], R.forecast("E3", Q, G, exclude=t)[0]
        if not np.array_equal(a["c_panel"].to_numpy(), b["c_panel"].to_numpy()):
            bad.append("G23: LOTO — исключённая %s влияет на панельную ĉ" % t)
        c, d = R.forecast("E3", P, G)[0], R.forecast("E3", Q, G)[0]
        if np.array_equal(c["c_panel"].to_numpy(), d["c_panel"].to_numpy()):
            bad.append("G23: контроль LOTO — подмена RV бумаги не меняет ĉ_panel без исключения (тест не чувствителен)")
    return bad


def g24_tables(o: dict) -> list:
    bad = []
    for f in CSV:
        if not (IO.REP / f).exists():
            bad.append("G24: нет " + f)
    if bad:
        return bad
    comp, tst = IO.read_csv("E_COMPARISON.csv"), IO.read_csv("E_TESTS.csv")
    if len(comp) != (4 + 2 + 6) * 4 or len(IO.read_csv("E_PER_TICKER.csv")["ticker"].unique()) != 17:
        bad.append("G24: E_COMPARISON / E_PER_TICKER неполны")
    cs = S.crit_panel(o["k"], o["cols_main"]).set_index("series")["crit_panel"]
    cc = comp[comp["window"] == "all"].set_index("series")["crit_panel"]
    if max(abs(cs[s] - cc[s]) for s in cs.index) > 1e-9:
        bad.append("G24: crit_panel в E_COMPARISON ≠ hp_stats_sel.crit_panel")
    if len(tst) != 6 * 6 * 4 or not (tst["p_holm_wide"].dropna() >= tst["p_holm"].dropna() - 1e-15).all():
        bad.append("G24: E_TESTS — размер или порядок p_holm ≤ p_holm_wide")
    yr = IO.read_csv("E_YEARS.csv")
    if max(abs(Y.crit_from_years(yr)[s] - cs[s]) for s in cs.index) > 1e-12:
        bad.append("G24: сумма crit_part по годам ≠ crit_panel")
    tr = pd.read_csv(IO.REP / "ALL_TRIALS.csv")
    j = tr[tr["key"].str.startswith("Ж-222:") & ~tr["key"].str.startswith(("Ж-222:CS", "Ж-222:NTP"))]["key"].str.split(":").str[1]       # отступление (лист DX-CS): строки CS1, CS2 другой стадии не считаются
    if sorted(j.tolist()) != sorted(["B1", "B2", "B3", "B4", "B6"] + ["C%d" % i for i in range(1, 11)] + R.CFGS) or len(j) != 21:
        bad.append("G24: ALL_TRIALS — строки Ж-222 не B1–B4,B6 + C1–C10 + E1–E6 по одной (%d)" % len(j))
    return bad


def g25_report(o: dict) -> list:
    txt = (IO.REP / "E_REPORT.md").read_text()
    bad = []
    if txt.splitlines()[0] != RP.first_line(o):
        bad.append("G25: первая строка E_REPORT не равна пересчитанной по crit_panel и порогу")
    if len(txt.split()) > 1200:
        bad.append("G25: E_REPORT длиннее 1200 слов")
    cp = json.loads((IO.REP / "CHECKPOINT.json").read_text())
    cs = o["crit"].set_index("series")["crit_panel"]
    g = o["go"]["cfg"][o["best"]]
    if g["go"] != (g["c1"] and g["c2"] and g["c3"]) or g["c1"] != bool(cs[o["best"]] <= cp["go_threshold_B"]):
        bad.append("G25: «go» не равно конъюнкции условий 1–3 / условие 1 не по crit_panel")
    if not txt.splitlines()[0].startswith("go" if g["go"] else "stop") or "(2)" not in txt.splitlines()[0] or "(3)" not in txt.splitlines()[0]:
        bad.append("G25: первая строка не содержит итог по условиям 1–3")
    if cp.get("E_stop_go") != ("go" if g["go"] else "stop") or cp.get("used_after_E") != 21:
        bad.append("G25: CHECKPOINT — E_stop_go / used_after_E")
    t = dt.datetime.strptime(cp["E_declared_at"], "%Y-%m-%d %H:%M:%S.%f").timestamp()       # местное время, как mtime файлов
    early = [f for f in ST.EC.OUT.glob("E*.parquet") if f.stat().st_mtime <= t]
    if early:
        bad.append("G25: файлы E созданы не позже объявления: %s" % [f.name for f in early])
    return bad


def all_checks(o: dict) -> list:
    return g23_forecasts(o) + g23_pit(o) + g23_identities(o) + g23_loto(o) + g24_tables(o) + g25_report(o)
