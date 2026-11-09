"""DE0, G1: сводный реестр V1–V7 (SBER, h = 1). Числа берутся из ТАБЛИЦ ПРОГРАММ и сверяются независимо:
V1–V5 и контроли — `cmp/CMP_SERIES|WINDOWS|YEARS|TESTS`; V6 — `FIN_ALL` (годы, окна), `ALL_TRIALS`, `*_GO`, `FIN_INFO_ALL`, `FIN_LEVELS`; V7 — `FIN_ALL`, `ALL_TRIALS`, `FIN_GO`.
QLIKE: 2018_2021, 2022_2025, `all_2018_2026` (общая история V7 — для V1–V6 пересчитана из годовых строк весами n) и `all_table` (окно `all` самой таблицы: у V1–V6 — 2271 дата 2017–2026).
Проверка равенства (`checks`): окна таблицы = взвешенные годовые QLIKE (V1–V6), qlike из FIN_ALL V7 = qlike из прогнозов (задаёт стадия)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s15 import de0_io as IO

V6_CFG = ["A0", "A1", "A2", "A3", "C1", "C2", "C3", "C4", "E1", "E2", "E3", "D1", "D2", "D3", "CB1", "F1", "F2", "A0g"]
V7_CFG = [1, 2, 3, 4, 5, 6, 7, 8, 10]
TOL = 1e-9


def _rd(path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def pooled(years: pd.DataFrame, a: int, b: int) -> float:
    """Σ n·q / Σ n по годовым строкам (series, year, n, qlike) за годы a…b."""
    g = years[(years["year"].astype(int) >= a) & (years["year"].astype(int) <= b)]
    return float((g["n"] * g["qlike"]).sum() / g["n"].sum())


def _qrow(years: pd.DataFrame, wins: pd.DataFrame) -> dict:
    """years: n, qlike, year (одна серия); wins: window → qlike (таблица программы). -> столбцы QLIKE и невязки проверки."""
    w = {r["window"]: float(r["qlike"]) for _, r in wins.iterrows()}
    out = dict(q_2018_2021=pooled(years, 2018, 2021), q_2022_2025=w["2022_2025"], q_all_2018_2026=pooled(years, 2018, 2026), q_all_table=w["all"])
    out["check_2022_2025"] = abs(pooled(years, 2022, 2025) - w["2022_2025"])
    out["check_2017_2021"] = abs(pooled(years, 2017, 2021) - w["2017_2021"])
    out["check_all"] = abs(pooled(years, 2017, 2026) - w["all"])
    return out


def cmp_rows() -> list:
    ser, yr, win, tst = (_rd(IO.CMP / ("CMP_%s.csv" % n)) for n in ("SERIES", "YEARS", "WINDOWS", "TESTS"))
    rows = []
    for _, s in ser.iterrows():
        k = s["series"]
        t = tst[(tst["series"] == k) & (tst["base"] == "K2") & (tst["role"] == s["role"])] if k in set(tst["series"]) else tst.iloc[0:0]
        late = t[t["window"] == "2022_2025"]
        ver = "Δ к K2 2022–2025: %.2f %% (p_holm_Nprog %.3f)" % (late.iloc[0]["delta_pct"], late.iloc[0]["p_holm_Nprog"]) if len(late) else "теста к K2 в CMP_TESTS нет (ряд вне множества тестов или сам K2/контроль)"
        rows.append(dict(program=str(s["program"]).split(" ")[0] if s["program"] != "общие" else "контроль", id=k, direction="%s [%s]" % (str(s["declared_by"])[:110], s["row"]), cls=s["cls"],
                         verdict=ver + "; роль: " + str(s["role"]), source="CMP_*.csv", **_qrow(yr[yr["series"] == k], win[win["series"] == k])))
    return rows


def v6_rows() -> list:
    fin = _rd(IO.V6 / "FIN_ALL.csv")
    trials = _rd(IO.V6 / "ALL_TRIALS.csv").set_index("key")
    info = _rd(IO.V6 / "FIN_INFO_ALL.csv")
    lev = _rd(IO.V6 / "FIN_LEVELS.csv")
    rows = []
    for c in V6_CFG:
        f = fin[fin["series"] == c]
        years = f[f["window"].str.fullmatch(r"\d{4}")].rename(columns={"window": "year", "qlike": "qlike"})
        years = years.assign(year=years["year"].astype(int))
        wins = f[~f["window"].str.fullmatch(r"\d{4}")]
        go = _rd(IO.V6 / ("%s_GO.csv" % c))
        ni = info[(info["x"] == c) & (info["window"] == "2017_2021") & info["role"].isna()]
        ni_txt = "; ".join("новая информация vs %s: %s" % (r["y"], r["new_information"]) for _, r in ni.iterrows()) or "правило 2.2: пар нет"
        lv = lev[(lev["config"] == c) & (lev["passed"] == "да")]["level"].astype(int).tolist()
        ver = "go: %s (%d из %d условий); %s; уровни пройдены: %s" % ("да" if go["passed"].astype(bool).all() else "нет", int(go["passed"].astype(bool).sum()), len(go), ni_txt, lv or "—")
        rows.append(dict(program="V6", id=c, direction=str(trials.loc["Ж-227:" + c, "direction"]), cls="PIT-сопоставим", verdict=ver, source="FIN_ALL.csv, ALL_TRIALS.csv, %s_GO.csv, FIN_INFO_ALL.csv, FIN_LEVELS.csv" % c,
                         **_qrow(years.rename(columns={"year": "year"}), wins)))
    return rows


def v7_rows() -> list:
    fin = _rd(IO.V7 / "FIN_ALL.csv")
    trials = _rd(IO.V7 / "ALL_TRIALS.csv").set_index("key")
    go = _rd(IO.V7 / "FIN_GO.csv")
    rows = []
    for k in V7_CFG:
        c = "cfg%d" % k
        f = fin[fin["series"] == c].set_index("window")["qlike"]
        g = go[go["config"] == c]
        name = json.loads(trials.loc["Ж-233:%d" % k, "config"])["name"]
        passed = g["passed"].astype(bool)
        rows.append(dict(program="V7", id=c, direction="%s — %s" % (trials.loc["Ж-233:%d" % k, "direction"], name), cls="PIT-сопоставим",
                         verdict="go: %s (%d из %d условий)" % ("да" if passed.all() else "нет", int(passed.sum()), len(g)), source="FIN_ALL.csv, ALL_TRIALS.csv, FIN_GO.csv",
                         q_2018_2021=float(f["2018_2021"]), q_2022_2025=float(f["2022_2025"]), q_all_2018_2026=float(f["all"]), q_all_table=float(f["all"]),
                         check_2022_2025=0.0, check_2017_2021=0.0, check_all=0.0))
    return rows


def build() -> pd.DataFrame:
    return pd.DataFrame(cmp_rows() + v6_rows() + v7_rows())


def checks(reg: pd.DataFrame, recomputed: dict) -> dict:
    """recomputed: {id: {'q_2018_2021', 'q_2022_2025', 'q_all_2018_2026'}} из СОХРАНЁННЫХ прогнозов (V7-ряды). Возвращает макс. невязки: окна/годы V1–V6 и таблица ↔ прогнозы V7."""
    cols = ["check_2022_2025", "check_2017_2021", "check_all"]
    out = dict(max_window_vs_years=float(np.nanmax(reg[cols].to_numpy(float))), max_table_vs_forecast=0.0)
    for k, v in recomputed.items():
        r = reg[reg["id"] == k]
        for c in ("q_2018_2021", "q_2022_2025", "q_all_2018_2026"):
            out["max_table_vs_forecast"] = max(out["max_table_vs_forecast"], float(np.abs(r[c].to_numpy(float) - v[c]).max()))
    return out
