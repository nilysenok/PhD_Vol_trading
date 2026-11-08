"""V4-NTP-диагностика: проверки гейтов G29–G31 для `run_v4 --stage NTP --check`. Каждая функция возвращает список нарушений (пусто — гейт закрыт)."""
from __future__ import annotations

import ast
import datetime as dt
import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_b_decl as DECL, v4_c_cache as CC, v4_cb_years as Y, v4_io as IO, v4_isolation as ISO, v4_ntp_stage as ST, v4_ntp_run as R, v4_ntp_state as SS,
                           v4_ntpd_report as RP, v4_ntpd_states as STS, v4_z_build as ZB, v4_z_feat as ZF)

DAY, CUT = "2018-06-01", "2018-09-01"
PRIOR = ("B", "C", "E", "Z", "CS")


def _declared() -> float:
    return dt.datetime.strptime(json.loads(DECL.CP.read_text())["NTP_declared_at"], "%Y-%m-%d %H:%M:%S.%f").timestamp()


def g29_files(r: dict) -> list:
    bad, P = [], r["P"]
    for cfg in ST.CFGS:
        f = pd.read_parquet(CC.OUT / (cfg + ".parquet"))
        if len(f) != len(P) or f["ticker"].nunique() != 17 or not (np.isfinite(f["F"]) & (f["F"] > 0)).all() or not ((f["pi"] >= 0) & (f["pi"] <= 1)).all():
            bad.append("G29: %s — состав/нефинитные F/π вне [0,1]" % cfg)
        if (CC.OUT / (cfg + ".parquet")).stat().st_mtime <= _declared():
            bad.append("G29: %s создан не позже объявления листа" % cfg)
    n4 = r["fc"]["NTP4"].sort_values(["date", "ticker"])
    if SS.max_step(n4["pi"].to_numpy(), n4["ticker"].astype("category").cat.codes.to_numpy(), 17) > SS.STEP + 1e-12:
        bad.append("G29: NTP4 — шаг π больше %.2f" % SS.STEP)
    jump = np.where(np.arange(400) % 2 == 0, 0.9, 0.1)
    old, SS.STEP = SS.STEP, 0.05
    try:
        caught = SS.max_step(SS.smooth(jump, np.zeros(400, int), 1), np.zeros(400, int), 1) > 0.02 + 1e-9
    finally:
        SS.STEP = old
    return bad + ([] if caught else ["G29: мутация «шаг 0,05» не поймана проверкой шага"])


def g29_pit(r: dict) -> list:
    bad, inp = [], ZB.real_inputs()
    P = r["P"][r["P"]["date"] < CUT]
    for cfg in ST.CFGS:
        if R.pit_violation(cfg, P, r["G"], inp, DAY):
            bad.append("G29: %s — обрезка: данные после d меняют прогноз дня d" % cfg)
    orig = SS.closed_date
    SS.closed_date = lambda t, d: d
    try:
        for cfg in ST.CFGS:
            if not R.pit_violation(cfg, P, r["G"], inp, DAY):
                bad.append("G29: %s — мутация «закрыто + 1» не поймана боевым путём" % cfg)
    finally:
        SS.closed_date = orig
    return bad


def g29_identities(r: dict) -> list:
    """π ≡ 1 → C8 побитно, π ≡ 0 → M побитно (боевой путь на реальных данных; NTP1–NTP4)."""
    bad, P, G, zl = [], r["P"], r["G"], r["Zl"]
    Pm = P.merge(G.rename(columns={"F": "c8"}), on=["ticker", "date"]).sort_values(["date", "ticker"])
    m = np.sqrt(Pm["k2"].to_numpy() * Pm["k3"].to_numpy())
    for cfg in ST.CFGS:
        one, zero = (R.forecast(cfg, P, G, zl, force_pi=v)[0]["F"].to_numpy() for v in (1.0, 0.0))
        if not (np.array_equal(one, Pm["c8"].to_numpy()) and np.array_equal(zero, m)):
            bad.append("G29: %s — тождество π ≡ 1 → C8 / π ≡ 0 → M нарушено" % cfg)
    return bad


def g29_loto(r: dict) -> list:
    """Подмена RV исключённой бумаги не меняет панельную величину (NTP1/2/4) и параметры панели (NTP3); без исключения — меняет."""
    bad, P, G, zl = [], r["P"], r["G"], r["Zl"]
    Q = P.copy()
    Q.loc[Q["ticker"] == "LKOH", "rv"] *= 5.0
    z16 = ZF.recross(zl, [x for x in IO.T17 if x != "LKOH"])
    for cfg in ("NTP1", "NTP2", "NTP4"):
        a, b, c = (R.forecast(cfg, X, G, z16, exclude=e)[0]["pan"].to_numpy() for X, e in ((P, "LKOH"), (Q, "LKOH"), (Q, None)))
        if not (np.array_equal(a, b, equal_nan=True) and not np.array_equal(a, c, equal_nan=True)):
            bad.append("G29: %s — исключённая бумага в панельной величине (LOTO)" % cfg)
    (_, wa), (_, wb) = (R.forecast("NTP3", X, G, z16, exclude="LKOH") for X in (P, Q))
    pa, pb = (w[w["ticker"] == "PANEL"][["alpha", "b_zmq", "b_abs_ell", "b_med_lrv"]].to_numpy() for w in (wa, wb))
    return bad + ([] if np.array_equal(pa, pb) else ["G29: NTP3 — параметры панели видят исключённую бумагу (LOTO)"])


def imports_ntpd(source: str) -> bool:
    """True, если код импортирует модуль `v4_ntpd_*` любым способом."""
    for n in ast.walk(ast.parse(source)):
        names = [a.name for a in n.names] + [getattr(n, "module", None) or ""] if isinstance(n, (ast.Import, ast.ImportFrom)) else []
        if any(p.startswith("v4_ntpd_") for x in names for p in x.split(".")):
            return True
    return False


def g29_isolation() -> list:
    """Модули `v4_ntpd_*` (читают RV задним числом) не импортируются путями прогноза: все файлы `s10`/`s11` вне `v4_ntpd_*` и `run_v4`."""
    bad = []
    for d in ("s10", "s11"):
        for p in sorted((ISO.ROOT / "models_v2" / d).glob("*.py")):
            if not (p.name.startswith("v4_ntpd_") or p.name == "run_v4.py") and imports_ntpd(p.read_text(encoding="utf-8")):
                bad.append("G29: %s/%s импортирует диагностику NTP" % (d, p.name))
    return bad + ["G29: oracle импортирован путём прогноза: %s" % x for x in ISO.violators()]


def g30(r: dict) -> list:
    bad, cols = [], r["cols"]
    for name, rows in (("NTP_COMPARISON.csv", len(cols) * 4), ("NTP_TESTS.csv", len(ST.CFGS) * 7 * 4), ("NTP_SBER.csv", len(cols) * 4), ("NTP_YEARS.csv", len(cols) * 10), ("NTP_STATES.csv", 3 * 3 * 2)):
        if len(IO.read_csv(name)) != rows:
            bad.append("G30: %s — строк %d, ожидалось %d" % (name, len(IO.read_csv(name)), rows))
    crit = S.crit_panel(r["k"], cols).set_index("series")["crit_panel"]
    comp = IO.read_csv("NTP_COMPARISON.csv")
    if any(abs(float(comp[comp["series"] == s]["crit_panel"].iloc[0]) - v) > 1e-12 for s, v in crit.items()):
        bad.append("G30: crit_panel в NTP_COMPARISON не равен hp_stats_sel.crit_panel")
    old = IO.read_csv("C_COMPARISON.csv")
    if any(abs(float(old[old["series"] == s]["crit_panel"].iloc[0]) - crit[s]) > 1e-12 for s in ("k1", "k2", "k3", "d2", "C1", "C4", "C8")):
        bad.append("G30: crit_panel контролей/C1/C4/C8 изменился против C_COMPARISON.csv")
    yc = Y.crit_from_years(IO.read_csv("NTP_YEARS.csv"))
    if any(abs(yc[s] - crit[s]) > 1e-12 for s in cols):
        bad.append("G30: сумма crit_part по годам не равна crit_panel")
    t = IO.read_csv("NTP_TESTS.csv")
    m = t[t["kind"] == "main"].dropna(subset=["p_holm"])
    if m.empty or (m["p_holm_wide"] < m["p_holm"] - 1e-15).any() or set(t["comparator"]) != {"k1", "k2", "k3", "d2", "C1", "C4", "C8"} or set(t["config"]) != set(ST.CFGS):
        bad.append("G30: NTP_TESTS — Холм / состав")
    if IO.read_csv("NTP_PER_TICKER.csv")["ticker"].nunique() != 17 or set(IO.read_csv("NTP_WEIGHTS.csv")["config"]) != set(ST.CFGS):
        bad.append("G30: NTP_PER_TICKER/NTP_WEIGHTS неполны")
    s = IO.read_csv("NTP_SBER.csv")
    sa = s[(s["window"] == "all") & (s["series"] == "k1")]
    if sa.empty or int(sa["n"].iloc[0]) != 2271 or abs(float(sa["ref_a1_roll_126"].iloc[0]) - 0.20814) > 1e-4:
        bad.append("G30: NTP_SBER — не 2271 даты или нет ориентира")
    lt = IO.read_csv("NTP_LOTO.csv")
    L = lt[lt["held_out"].isin(IO.M16 + ["M16"])]
    if set(lt["config"]) != {r["best"]} or set(L["held_out"]) != set(IO.M16 + ["M16"]) or not L["delta_zero_check"].astype(str).eq("True").all():
        bad.append("G30: NTP_LOTO неполон или исключённая бумага в панельной величине")
    keys = [k for k in pd.read_csv(IO.REP / "ALL_TRIALS.csv")["key"] if k.startswith("Ж-222:") and not k.startswith(("Ж-222:CLD", "Ж-222:K8", "Ж-222:RQC", "Ж-222:H", "Ж-222:I"))]       # отступление (листы CLD, K8-RQC, H): строки CLD1–CLD4, K8, RQC1–RQC3 других стадий не считаются
    pref = sorted(k.split(":")[1] for k in keys)
    if sorted(x for x in pref if x.startswith("NTP")) != ST.CFGS or len(keys) != 27 or len(set(pref)) != 27:
        bad.append("G30: ALL_TRIALS — нет строк Ж-222:NTP1…NTP4 (по одной) или потеряны строки других стадий (всего ожидалось 27)")
    return bad


def g30_states(r: dict, st: pd.DataFrame) -> list:
    """Группы по π NTP3 на дату прогноза: π строки не зависит от её цели (подмена RV цели после d не меняет группу) и обе группы непусты в окне выбора."""
    f = STS.frame(r)
    w = f[(f["t_plus"].dt.year <= 2021)]
    bad = [] if {"pi_gt_half", "pi_le_half"} <= set(st[st["window"] == "2017_2021"]["group"]) and (w["pi"] > 0.5).any() and (w["pi"] <= 0.5).any() else ["G30: NTP_STATES — группа пуста"]
    pi = r["fc"]["NTP3"]
    return bad + ([] if len(pi) == len(r["P"]) and pi["pi"].notna().all() else ["G30: у NTP3 нет π на части дат"])


def g31(r: dict, st: pd.DataFrame, pb: pd.DataFrame) -> list:
    txt = (IO.REP / "NTP_REPORT.md").read_text()
    cp = json.loads(DECL.CP.read_text())
    bad = [] if txt.splitlines()[0] == RP.first_line(r) else ["G31: первая строка NTP_REPORT не соответствует пересчёту go/stop"]
    if len(txt.split()) > 1000:
        bad.append("G31: NTP_REPORT %d слов > 1000" % len(txt.split()))
    if not all(x in txt for x in ("2017", "2018", "2022", "SBER", "LOTO", "Отступления", "по состояниям", STS.answer(st))):
        bad.append("G31: в NTP_REPORT нет обязательных разделов")
    if cp.get("NTP_stop_go") != ("go" if r["go"]["cfg"][r["best"]]["go"] else "stop") or cp.get("used_after_NTP") != cp["used_after_CS"] + 4 or not txt.startswith(cp["NTP_stop_go"]):
        bad.append("G31: CHECKPOINT — NTP_stop_go / расход")
    return bad


def all_checks(r: dict, st: pd.DataFrame, pb: pd.DataFrame) -> list:
    return g29_files(r) + g29_pit(r) + g29_identities(r) + g29_loto(r) + g29_isolation() + g30(r) + g30_states(r, st) + g31(r, st, pb)
