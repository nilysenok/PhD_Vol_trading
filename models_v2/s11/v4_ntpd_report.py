"""V4-NTP-диагностика, G31: NTP_REPORT.md (≤ 1000 слов). Первая строка — «go»/«stop» по трём условиям (Ж-225 п. 10; `v4_e_go.rules`), лучшая NTP, её crit_panel против порога и отрыв от лучшего из K2, K3, D2 в % уровня K1;
все числа берутся из таблиц стадии. Модуль — вне путей прогноза (использует диагностику состояний)."""
from __future__ import annotations

import pandas as pd

from models_v2.s11 import v4_b_report as BR, v4_ntpd_states as STS

DEVIATIONS = [
    "M = √(K2·K3), C8 — сохранённый прогноз (`C8.parquet`) без переобучения; D′ = QLIKE(C8) − QLIKE(M). До 60 закрытых пар бумаги π = 0,5 (приор, как в B/C/E); NTP4 стартует с π = 0,5.",
    "Панельные EWMA (m, s панели): затухание на каждый торговый день панели, полураспад 60; бумажные — по парам бумаги (полураспад 60). s_t — бумажное, без стягивания; стягивается только m (κ = 250, n — эффективное число пар).",
    "NTP2: состояние по бумаге (логит веса C8), η = 1/медиана QLIKE обеих моделей по закрытым парам панели на день d; π = 0,5 до 60 пар, хотя вес накапливается с первой пары.",
    "NTP3: ℓ_d = ln(M/K1) (как в C), z_mkt_level = `med_lrv`, m/s — признак NTP1; оценка и штрафы — как C-a (`v4_c_fit.fit` с базой ln M и «ℓ» = ln(C8/M)), месячная.",
    "LOTO: NTP1/2/4 без параметров — исключённая бумага не входит в панельные величины (её собственное состояние — по её закрытым парам); NTP3 — панель на 16, δ = 0.",
    "NTP_STATES: ряд для DM — среднее по бумагам группы на дату (даты без бумаг группы не входят); группы — по π NTP3 на дату прогноза (PIT).",
    "`run_v4.py` был ровно 200 строк: цепочку `if a.stage == …` заменил словарём (минимальная правка прежнего модуля; поведение прежних стадий не менялось).",
]


def _gap(crit: pd.Series, best: str) -> tuple:
    ref = min(("k2", "k3", "d2"), key=lambda s: crit[s])
    return ref, float(crit[best] - crit[ref]), 100.0 * float(crit[best] - crit[ref]) / float(crit["k1"])


def first_line(r: dict) -> str:
    best, crit, go, cp = r["best"], r["crit"].set_index("series")["crit_panel"], r["go"], r["cp"]
    g, (ref, d, dp) = go["cfg"][best], _gap(crit, best)
    return ("%s — NTP: лучшая %s, (1) crit_panel %.5f ≤ порога %.5f: %s; (2) без 2017 %.5f: %s; (3) без 1 %% дат %.5f: %s; отрыв от лучшего из K2, K3, D2 (%s %.5f): %+.5f = %+.2f %% уровня K1 %.5f (нужно ≤ −0,0005)" % (
        "go" if g["go"] else "stop", best, crit[best], cp["go_threshold_B"], "да" if g["c1"] else "нет", g["crit_no2017"], "да" if g["c2"] else "нет", g["crit_trim1"], "да" if g["c3"] else "нет", ref, crit[ref], d, dp, crit["k1"]))


def deltas(r: dict, best: str) -> list:
    """Δ QLIKE лучшей NTP к K1, K2, K3, D2, C8 в % уровня K1: вся история, 2017, 2018, 2022–2025 (M16)."""
    comp = r["comp"].set_index(["series", "window"])["qlike"]
    yt = r["years"].pivot_table(index="year", columns="series", values="ann_qlike")
    L = ["| период | K1 | K2 | K3 | D2 | C8 |", "|---|---|---|---|---|---|"]
    for name, get in (("вся", lambda s: comp[(s, "all")]), ("2017", lambda s: yt.loc[2017, s]), ("2018", lambda s: yt.loc[2018, s]), ("2022–2025", lambda s: comp[(s, "2022_2025")])):
        L.append("| %s | %s |" % (name, " | ".join("%+.2f" % (100.0 * (get(best) - get(s)) / get("k1")) for s in ("k1", "k2", "k3", "d2", "C8"))))
    return L


def _crit_lines(crit: pd.Series, go: dict) -> list:
    L = ["| ряд | crit_panel | без 2017 | без 1 % дат |", "|---|---|---|---|"]
    for s in ("k1", "k2", "k3", "d2", "C1", "C4", "C8", "NTP1", "NTP2", "NTP3", "NTP4"):
        g = go["ref"].get(s) or go["cfg"][s]
        L.append("| %s | %.5f | %.5f | %.5f |" % (s, crit[s], g["crit_no2017"], g["crit_trim1"]))
    return L


def _pi_lines(pb: pd.DataFrame) -> list:
    p = pb.pivot_table(index="config", columns="period", values="mean_pi")
    cols = ["2017", "2018", "2022", "shock_2018-04-06_pm14d", "shock_2022-02-24_p28d"]
    return ["| конфигурация | 2017 | 2018 | 2022 | шок 06.04.2018 | шок 24.02.2022 |", "|---|---|---|---|---|---|"] + ["| %s | %s |" % (i, " | ".join("%.3f" % p.loc[i, c] for c in cols)) for i in p.index]


def _state_lines(st: pd.DataFrame) -> list:
    s = st[st["group"].isin(["pi_gt_half", "pi_le_half"]) & (st["window"] != "all")]
    return ["- %s, %s, %s: доля строк %.2f, Δ %.4f, DM p %.3f, ДИ [%.4f; %.4f]" % (x.window, x.group, x.comparison, x.share_rows, x.delta_mean, x.dm_p, x.boot_lo, x.boot_hi) for x in s.itertuples()]


def build(r: dict, st: pd.DataFrame, pb: pd.DataFrame) -> str:
    cp, crit, best, go = r["cp"], r["crit"].set_index("series")["crit_panel"], r["best"], r["go"]
    L = [first_line(r), "", "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок; выбор — только crit_panel (2017–2021); окна 2022+ — описательно (Ж-225).", "",
         "## Отступления и толкования (до выводов)"] + ["- " + x for x in DEVIATIONS]
    L += ["", "## crit_panel и три условия «go» (M16 × цели 2017–2021)", ""] + _crit_lines(crit, go)
    L += ["", "Порог go (1): %.5f; (2), (3): min(K2, K3, D2) − 0,0005 по тому же критерию." % cp["go_threshold_B"]]
    L += ["", "## Δ QLIKE лучшей (%s) к K1, K2, K3, D2, C8 (%% уровня K1, M16; минус — лучше)" % best, ""] + deltas(r, best)
    L += ["", "## Добавляет ли HAR-J информацию по состояниям (NTP_STATES.csv; группы по π NTP3 на дату прогноза)", "", STS.answer(st), ""] + _state_lines(st)
    L += ["", "## Поведение π (средний по M16)", ""] + _pi_lines(pb)
    L += ["", "## Холм (лучшая; семейство M16 × {2022_2025, all} × NTP1–NTP4, N=%d; N_wide=%d; K1, D2, C8)" % (cp["N_declared_before_stats"], cp["N_wide"]), ""] + BR.holm_lines({**r, "tests": r["tests"][r["tests"]["comparator"].isin(["k1", "d2", "C8"])]})
    L += ["", "## SBER (отдельно, в выбор не входит; 2271 дата; ориентир 0,2081 — потолок)", ""] + BR.sber_lines(r)
    lt = r["loto"][r["loto"]["held_out"] == "M16"].set_index("window")
    L += ["", "## LOTO лучшей (%s; M16: LOTO / пул / K1)" % best, ""] + ["- %s: %.4f / %.4f / %.4f" % (w, lt.loc[w, "q_loto"], lt.loc[w, "q_pool"], lt.loc[w, "q_k1"]) for w in ("2017_2021", "2022_2025", "all")]
    L += ["", "Израсходовано конфигураций листа: 4 (NTP1–NTP4) из 60; после листа %d из 60 (CHECKPOINT.json, `used_after_NTP`)." % (int(cp["used_after_CS"]) + 4)]
    return "\n".join(L) + "\n"
