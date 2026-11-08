"""V4-CLD-диагностика: проверки гейтов G32–G34 для `run_v4 --stage CLD --check`. Каждая функция возвращает список нарушений (пусто — гейт закрыт)."""
from __future__ import annotations

import ast
import datetime as dt
import json

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import (v4_b_decl as DECL, v4_b_run as BR, v4_c_cache as CC, v4_cb_years as Y, v4_cld_brk as BK, v4_cld_run as R, v4_cld_stage as ST, v4_cldd_pred as PR, v4_cldd_report as RP,
                           v4_io as IO, v4_isolation as ISO, v4_z_build as ZB)

DAY, CUT = "2018-06-01", "2018-09-01"


def _declared() -> float:
    return dt.datetime.strptime(json.loads(DECL.CP.read_text())["CLD_declared_at"], "%Y-%m-%d %H:%M:%S.%f").timestamp()


def g32_files(r: dict) -> list:
    bad, P = [], r["P"]
    for cfg in ST.CFGS:
        f = pd.read_parquet(CC.OUT / (cfg + ".parquet"))
        if len(f) != len(P) or f["ticker"].nunique() != 17 or not (np.isfinite(f["F"]) & (f["F"] > 0)).all() or not ((f["w"] >= 0) & (f["w"] <= 1)).all():
            bad.append("G32: %s — состав/нефинитные F/w вне [0,1]" % cfg)
        if (CC.OUT / (cfg + ".parquet")).stat().st_mtime <= _declared():
            bad.append("G32: %s создан не позже объявления листа" % cfg)
    b = pd.read_parquet(CC.OUT / "BRK.parquet")
    if list(b.columns) != ["date", "med_lrv", "brk", "fired"] or not b["fired"].any() or not (b["brk"] >= 0).all():
        bad.append("G32: BRK.parquet — состав / ни одного срабатывания")
    return bad


def g32_brk(r: dict) -> list:
    """Известный ответ CUSUM на синтетике со сдвигом уровня; PIT на реальной медиане; мутация «данные d+1» ловится."""
    bad, n, t0 = [], 520, 400
    x = pd.Series(np.random.default_rng(3).normal(0, 1, n), index=pd.bdate_range("2017-01-02", periods=n))
    x.iloc[t0:] += 10.0
    fired = BK.cusum(x)[1].to_numpy()
    if not (fired[t0] and not fired[:t0].any()):
        bad.append("G32: CUSUM — на синтетике со сдвигом срабатывание не в день сдвига")
    m = BK.median_series(r["Zl"])
    day = m.index[m.index <= DAY][-1]
    m2 = m.copy()
    m2[m2.index > day] += 8.0
    eq = lambda lead: np.array_equal(BK.cusum(m, lead)[0][:day].to_numpy(), BK.cusum(m2, lead)[0][:day].to_numpy())
    if not eq(0):
        bad.append("G32: brk реального ряда зависит от данных после d")
    if eq(1):
        bad.append("G32: мутация «срабатывание с данными d+1» не поймана на реальном ряду")
    return bad


def g32_pit(r: dict) -> list:
    bad, inp = [], ZB.real_inputs()
    P = r["P"][r["P"]["date"] < CUT]
    for cfg in ST.CFGS:
        if R.pit_violation(cfg, P, inp, DAY):
            bad.append("G32: %s — обрезка: данные после d меняют прогноз дня d" % cfg)
    orig = BR.closed_mask
    BR.closed_mask = lambda t_plus, d0: np.asarray(t_plus) <= np.datetime64(d0) + np.timedelta64(3, "D")
    try:
        for cfg in ST.CFGS:
            if not R.pit_violation(cfg, P, inp, DAY):
                bad.append("G32: %s — мутация «закрыто + 1» в регрессии D не поймана боевым путём" % cfg)
    finally:
        BR.closed_mask = orig
    for cfg in ST.CFGS:
        if not R.pit_violation(cfg, P, inp, DAY, leak=True):
            bad.append("G32: %s — мутация «D цели в признаках» не поймана боевым путём" % cfg)
    return bad


def g32_identities(r: dict) -> list:
    """D̂ ≡ 0 → w = 0,5 побитно (боевой путь на реальных данных; CLD1–CLD4)."""
    bad = []
    for cfg in ST.CFGS:
        if not (R.forecast(cfg, r["P"], r["Zl"], force_dhat=0.0)[0]["w"] == 0.5).all():
            bad.append("G32: %s — тождество D̂ ≡ 0 → w = 0,5 нарушено" % cfg)
    return bad


def g32_loto(r: dict) -> list:
    """Исключённая бумага не в оценке (подмена её RV не меняет прогнозы остальных); при поперечных z по 17 — меняет (мутация ловится)."""
    bad, P, Zl, t = [], r["P"], r["Zl"], "LKOH"
    Z16 = ST.ZF.recross(Zl, [x for x in IO.T17 if x != t])
    out = R.forecast(r["best"], P, Z16, exclude=t)[0]
    if not ST.zero_ok(r["best"], P, Z16, t, out):
        bad.append("G32: LOTO — исключённая бумага влияет на оценку")
    Q = P.copy()
    Q.loc[Q["ticker"] == t, "rv"] *= 5.0
    a, b = (x[x["ticker"] != t].sort_values(["date", "ticker"])["F"].to_numpy() for x in (R.forecast(r["best"], P, Zl)[0], R.forecast(r["best"], Q, Zl)[0]))
    return bad + ([] if not np.array_equal(a, b) else ["G32: LOTO — мутация «бумага внутри оценки» не поймана"])


def _imports(source: str, prefix: str) -> bool:
    for n in ast.walk(ast.parse(source)):
        names = [a.name for a in n.names] + [getattr(n, "module", None) or ""] if isinstance(n, (ast.Import, ast.ImportFrom)) else []
        if any(p.startswith(prefix) for x in names for p in x.split(".")):
            return True
    return False


def g32_isolation() -> list:
    """Модули `v4_cldd_*` (читают D задним числом) не импортируются путями прогноза: все файлы `s10`/`s11` вне `v4_cldd_*` и `run_v4`."""
    bad = []
    for d in ("s10", "s11"):
        for p in sorted((ISO.ROOT / "models_v2" / d).glob("*.py")):
            if not (p.name.startswith("v4_cldd_") or p.name == "run_v4.py") and _imports(p.read_text(encoding="utf-8"), "v4_cldd_"):
                bad.append("G32: %s/%s импортирует диагностику CLD" % (d, p.name))
    return bad + ["G32: oracle импортирован путём прогноза: %s" % x for x in ISO.violators()]


def g33(r: dict, pred: pd.DataFrame) -> list:
    bad, cols = [], r["cols"]
    for name, rows in (("CLD_COMPARISON.csv", len(cols) * 4), ("CLD_TESTS.csv", len(ST.CFGS) * 8 * 4), ("CLD_SBER.csv", len(cols) * 4), ("CLD_YEARS.csv", len(cols) * 10)):
        if len(IO.read_csv(name)) != rows:
            bad.append("G33: %s — строк %d, ожидалось %d" % (name, len(IO.read_csv(name)), rows))
    crit = S.crit_panel(r["k"], cols).set_index("series")["crit_panel"]
    comp = IO.read_csv("CLD_COMPARISON.csv")
    if any(abs(float(comp[comp["series"] == s]["crit_panel"].iloc[0]) - v) > 1e-12 for s, v in crit.items()):
        bad.append("G33: crit_panel в CLD_COMPARISON не равен hp_stats_sel.crit_panel")
    old = IO.read_csv("NTP_COMPARISON.csv")
    if any(abs(float(old[old["series"] == s]["crit_panel"].iloc[0]) - crit[s]) > 1e-12 for s in ("k1", "k2", "k3", "d2", "C1", "C4", "C8", "NTP1", "NTP3")):
        bad.append("G33: crit_panel контролей/C1/C4/C8/NTP1/NTP3 изменился против NTP_COMPARISON.csv")
    yc = Y.crit_from_years(IO.read_csv("CLD_YEARS.csv"))
    if any(abs(yc[s] - crit[s]) > 1e-12 for s in cols):
        bad.append("G33: сумма crit_part по годам не равна crit_panel")
    t = IO.read_csv("CLD_TESTS.csv")
    m = t[t["kind"] == "main"].dropna(subset=["p_holm"])
    if m.empty or (m["p_holm_wide"] < m["p_holm"] - 1e-15).any() or set(t["comparator"]) != {"k1", "k2", "k3", "d2", "C1", "C4", "C8", "NTP1"} or set(t["config"]) != set(ST.CFGS):
        bad.append("G33: CLD_TESTS — Холм / состав")
    if IO.read_csv("CLD_PER_TICKER.csv")["ticker"].nunique() != 17 or set(IO.read_csv("CLD_WEIGHTS.csv")["config"]) != set(ST.CFGS):
        bad.append("G33: CLD_PER_TICKER/CLD_WEIGHTS неполны")
    cf = IO.read_csv("CLD_COEF.csv")
    if set(cf["config"]) != set(ST.CFGS) or set(cf["feature"]) != set(R.FEATS) or cf["month"].nunique() < 100:
        bad.append("G33: CLD_COEF неполон")
    p = IO.read_csv("CLD_PRED.csv")
    if set(p["config"]) != set(ST.CFGS) or not {"2017_2021", "2022_2025", "all"} <= set(p[p["kind"] == "window"]["period"]) or len(p) != len(pred) or (p[p["kind"] == "month"]["n"] <= 0).any():
        bad.append("G33: CLD_PRED — состав окон/месяцев")
    s = IO.read_csv("CLD_SBER.csv")
    sa = s[(s["window"] == "all") & (s["series"] == "k1")]
    if sa.empty or int(sa["n"].iloc[0]) != 2271 or abs(float(sa["ref_a1_roll_126"].iloc[0]) - 0.20814) > 1e-4:
        bad.append("G33: CLD_SBER — не 2271 даты или нет ориентира")
    lt = IO.read_csv("CLD_LOTO.csv")
    L = lt[lt["held_out"].isin(IO.M16 + ["M16"])]
    if set(lt["config"]) != {r["best"]} or set(L["held_out"]) != set(IO.M16 + ["M16"]) or not L["delta_zero_check"].astype(str).eq("True").all():
        bad.append("G33: CLD_LOTO неполон или исключённая бумага в оценке")
    keys = [k for k in pd.read_csv(IO.REP / "ALL_TRIALS.csv")["key"] if k.startswith("Ж-222:") and not k.startswith(("Ж-222:K8", "Ж-222:RQC"))]       # отступление (лист K8-RQC): строки K8, RQC1–RQC3 не считаются
    pref = sorted(k.split(":")[1] for k in keys)
    if sorted(x for x in pref if x.startswith("CLD")) != ST.CFGS or len(keys) != 31 or len(set(pref)) != 31:
        bad.append("G33: ALL_TRIALS — нет строк Ж-222:CLD1…CLD4 (по одной) или потеряны строки других стадий (всего ожидалось 31)")
    return bad


def g34(r: dict, pred: pd.DataFrame) -> list:
    txt = (IO.REP / "CLD_REPORT.md").read_text()
    cp = json.loads(DECL.CP.read_text())
    bad = [] if txt.splitlines()[0] == RP.first_line(r) else ["G34: первая строка CLD_REPORT не соответствует пересчёту go/stop"]
    if len(txt.split()) > 1000:
        bad.append("G34: CLD_REPORT %d слов > 1000" % len(txt.split()))
    if not all(x in txt for x in ("2017", "2018", "2022", "SBER", "LOTO", "Отступления", "C4", "NTP3", "среднее QLIKE 2017–2021", RP.verdict(r, pred))):
        bad.append("G34: в CLD_REPORT нет обязательных разделов")
    if cp.get("CLD_stop_go") != ("go" if r["go"]["cfg"][r["best"]]["go"] else "stop") or cp.get("used_after_CLD") != cp["used_after_NTP"] + 4 or not txt.startswith(cp["CLD_stop_go"]):
        bad.append("G34: CHECKPOINT — CLD_stop_go / расход")
    return bad


def all_checks(r: dict, pred: pd.DataFrame) -> list:
    return g32_files(r) + g32_brk(r) + g32_pit(r) + g32_identities(r) + g32_loto(r) + g32_isolation() + g33(r, pred) + g34(r, pred)
