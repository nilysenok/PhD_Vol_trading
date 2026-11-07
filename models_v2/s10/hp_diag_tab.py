"""P3c, G13: сравнительные таблицы итога (`diagnostics/FINAL_TABLES.md`) и `FINAL_PANEL_CONFIG.json` из готовых `statistics_p17/` и входов 17. Пересчётов статистики нет; только чтение и форматирование.
17 (M16 — вывод) и E11 (перенос) — раздельные таблицы. Статус: исследовательский, HISTORICAL WALK-FORWARD, потолок."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd

from models_v2.s10 import hp_diag as DG, hp_engine as E, hp_hyb_core as HC, hp_paths as P

S17 = P.REP / "statistics_p17"
HYB = [m + "_" + n for m in ("xgb", "lgbm") for n in ("A1", "A2", "A3", "A4", "B1", "B2", "C1", "C2", "D1", "D2")]
SERIES = ["k1", "k4", "k5", "k6", "k7"] + HYB
WIN = {"all": "all", "conf": "confirm_2022_2026"}
POP = {"17": ("M16", "17"), "E11": ("E11", "E11")}          # (scope в таблицах, population)


def _rd(name: str) -> pd.DataFrame:
    return pd.read_csv(S17 / name, comment="#")


def f(x, nd=4) -> str:
    return "—" if x is None or not np.isfinite(x) else ("%.*f" % (nd, x)).replace(".", ",")


def md(df: pd.DataFrame) -> str:
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def _mcs(mc: pd.DataFrame, scope: str, name: str, w: str) -> str:
    x = mc[(mc["scope"] == scope) & (mc["window"] == WIN[w]) & (mc["model"] == name)]
    return "—" if x.empty else ("да" if bool(x["in_mcs90"].iloc[0]) else "нет")


def table_levels(pop: str, cmp_: pd.DataFrame, sel: pd.DataFrame, mc: pd.DataFrame) -> pd.DataFrame:
    scope, pp = POP[pop]
    rows = []
    for s in SERIES:
        q = {w: cmp_[(cmp_["scope"] == scope) & (cmp_["population"] == pp) & (cmp_["period"] == WIN[w]) & (cmp_["model"] == s)] for w in WIN}
        cp = sel[sel["series"] == s]["crit_panel"]
        rows.append({"ряд": s, "crit_panel (M16 2017–2021)": f(cp.iloc[0], 6) if pop == "17" and len(cp) else "—", "QLIKE вся": f(q["all"]["qlike_mean"].iloc[0]) if len(q["all"]) else "—",
                     "QLIKE 2022–02.2026": f(q["conf"]["qlike_mean"].iloc[0]) if len(q["conf"]) else "—", "RMSE вся (среднее по бумагам)": f(q["all"]["rmse_mean"].iloc[0], 2) if len(q["all"]) else "—",
                     "MCS 90 % вся / подтв.": "%s / %s" % (_mcs(mc, scope, s, "all"), _mcs(mc, scope, s, "conf"))})
    return pd.DataFrame(rows)


def _cell(te: pd.DataFrame, pop: str, s: str, comp: str, w: str, k23: bool) -> str:
    scope, pp = POP[pop]
    x = te[(te["population"] == pp) & (te["scope"] == scope) & (te["config"] == s) & (te["comparator"] == comp) & (te["window"] == ("k23_" if k23 else "") + WIN[w]) & te["kind"].isin(["main", "transfer", "control"])]
    if x.empty:
        return "—"
    r = x.iloc[0]
    tail = "; n бумаг: %s" % ("—" if not np.isfinite(r["n_k23_tickers"]) else int(r["n_k23_tickers"])) if k23 else ""
    return "%s (p %s; Холм %s%s)" % (f(r["delta_mean"], 5), f(r["dm_p"], 3), f(r["p_holm"], 2), tail)


def table_vs(pop: str, te: pd.DataFrame, bases: tuple, series: list) -> pd.DataFrame:
    """bases: «k1», «own» (K4 для XGB, K5 для LGBM), «k2», «k3»."""
    rows = []
    for s in series:
        r = {"ряд": s}
        for b in bases:
            comp = ("k4" if s.startswith("xgb") else "k5") if b == "own" else b
            for w in WIN:
                r["Δ к %s, %s" % (b, "вся" if w == "all" else "подтв.")] = _cell(te, pop, s, comp, w, b in ("k2", "k3")) if not (b == "own" and s not in HYB) else "—"
        rows.append(r)
    return pd.DataFrame(rows)


def build() -> str:
    cmp_, sel, mc, te = _rd("COMPARISON.csv"), _rd("SELECTION.csv"), _rd("MCS.csv"), _rd("TESTS.csv")
    out = ["# Таблицы итога (генератор `hp_diag_tab.py`; источники — `statistics_p17/*.csv`, не пересчитывались)", "Δ = QLIKE(ряд) − QLIKE(база); минус — ряд лучше. DM — Ньюи–Уэст 5 по датам; p Холма — N = 631. «вся» = цели ≥ 2017-01-03, «подтв.» = 2022 — 02.2026.", ""]
    for pop, title in (("17", "Обучение на 17 (вывод — M16; SBER в обучении)"), ("E11", "E11 — перенос зафиксированной модели (не полностью независимая выборка, Ж-220 п. 9)")):
        out += ["## %s" % title, "### Уровни (QLIKE среднее по бумагам, общие даты ряда)", md(table_levels(pop, cmp_, sel, mc)), "### Против K1 и своего самостоятельного бустинга (K4 для XGB, K5 для LGBM)",
                md(table_vs(pop, te, ("k1", "own"), ["k4", "k5", "k6", "k7"] + HYB)), "### Против K2 и K3 (замороженные самостоятельные бустинги бумаги; только бумаги, где они есть)", md(table_vs(pop, te, ("k2", "k3"), HYB)), ""]
    return "\n".join(out)


def config() -> dict:
    sel = json.loads((S17 / "SELECTION.json").read_text())
    arch = {"lgbm_D2": "Confidence-Gated D2: λ_{i,t} = σ(w·z), z = (Ḡ 63 дня, разброс g по 5 бэггинг-копиям, ln RV_d/RV_m IMOEX, ln RV_d/RV_w, доля знака g за 22 дня), w — ридж-логистика по закрытым; f = HAR-J·exp(λ·clip(g, ±c)), g — LGBM QLIKE на 28 признаках z_*",
            "xgb_D2": "то же, модуль XGBoost (второй финалист; порог R9 к K1 не прошёл: запас −0,000031)"}
    code = {p.name: P.sha16(p) for p in sorted((P.ROOT / "models_v2/s10").glob("hp_*.py")) + [P.ROOT / "models_v2/s10/run_hp.py"]}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=P.ROOT, capture_output=True, text=True).stdout.strip()
    return {"status": "исследовательский, HISTORICAL WALK-FORWARD, не независимый OOS; не ACCEPTED; цифры — потолок", "horizon": "h = 1", "training_universe": {"tickers": P.MAIN17, "n": 17, "note": "Ж-220: E11 в обучении, выборе и поперечных признаках нет"},
            "finalists": {"primary": "lgbm_D2", "second": "xgb_D2"}, "architecture": arch, "features": HC.Z28, "grid": {"depth": list(E.DEPTHS), "c": list(E.CS), "lambda": list(E.LAMS), "hp": E.HP, "default_first_year": list(E.DEFAULT)},
            "yearly_selection_rule": "глубина, c, λ (κ, a, b, w) на год Y — по закрытым прогнозам года Y−1 по 17 бумагам (средний по бумагам QLIKE); первый год — default", "selection": {"crit_panel_window": "M16 × цели 2017–2021", "margin_vs_K1": 0.0005, "N_holm": sel["N_declared"],
            "triple": sel["triple_cfgs"], "u16": sel["u16"]}, "input_sha256_16": {**sel["hashes"], "panel_base.parquet": P.sha16(P.OUT / "panel_base.parquet"), "loto_p17_xgb.parquet": P.sha16(P.OUT / "loto_p17_xgb.parquet"),
            "loto_p17_lgbm.parquet": P.sha16(P.OUT / "loto_p17_lgbm.parquet"), "k45_p17.parquet": P.sha16(P.OUT / "k45_p17.parquet")}, "code": {"git_head": head, "note": "models_v2/ в этой ветке не отслеживается git (так было до работы); ниже — SHA-256[:16] файлов кода", "files_sha256_16": code}}


def run() -> None:
    DG.OUT.mkdir(parents=True, exist_ok=True)
    tables = build()
    (DG.OUT / "FINAL_TABLES.md").write_text(tables, encoding="utf-8")
    tpl = DG.OUT / "FINAL_TEXT.md"       # текст итога (пишется вручную; числа — из таблиц); маркер {{TABLES}} заменяется таблицами
    if tpl.exists():
        body = ("\n" + tables.split("\n", 3)[3]).replace("\n## ", "\n### ").replace("\n### Уровни", "\n#### Уровни").replace("\n### Против", "\n#### Против")
        (P.ROOT / "docs/otchety/HARJ_PANEL_HYBRID_FINAL.md").write_text(tpl.read_text(encoding="utf-8").replace("{{TABLES}}", body), encoding="utf-8")
    (P.REP / "FINAL_PANEL_CONFIG.json").write_text(json.dumps(config(), ensure_ascii=False, indent=1), encoding="utf-8")
