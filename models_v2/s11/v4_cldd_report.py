"""V4-CLD-диагностика, G34: CLD_REPORT.md (≤ 1000 слов). Первая строка — «go»/«stop» по трём условиям (Ж-225 п. 10; `v4_e_go.rules`), лучшая CLD, её crit_panel против порога и простое среднее QLIKE 2017–2021 против контролей;
все числа — из таблиц стадии. Модуль вне путей прогноза (читает CLD_PRED)."""
from __future__ import annotations

import pandas as pd

from models_v2.s11 import v4_b_report as BR, v4_io as IO

DEVIATIONS = [
    "Признаки — ровно объявленные 11 (ewqd и доля скачков из Z этапа C в списке Ж-225 п. 5 нет — не входят); без отбора. Регрессия панельная, без эффектов бумаг; λ = 250 — штраф λ‖β‖² при стандартизованных столбцах, свободный член не штрафуется.",
    "CLD3: ŝ для δ — с. о. остатка гребневой подгонки того же окна (не MAD), гребень λ = 250 сохранён. CLD4: 1 внутренний узел в медиане окна, вне границ — значение на границе; в CLD_COEF — с. о. вклада признака.",
    "CUSUM: окно стандартизации 252 торговых дня строго до d (≥ 60); до первого срабатывания — дни с начала ряда; в LOTO — по 16 бумагам. R² — к историческому среднему D окна оценки; по M16, строки панели не независимы.",
    "`run_v4.py` (197 строк): `main_ntp` заменён таблицей `LATE` (NTP, CLD) — поведение прежних стадий не менялось.",
]


def _ref(crit: pd.Series, comp: pd.DataFrame) -> tuple:
    q = comp.set_index(["series", "window"])["qlike"]
    return {s: float(q[(s, "2017_2021")]) for s in ("k1", "k2", "k3", "d2")}


def first_line(r: dict) -> str:
    best, crit, go, cp = r["best"], r["crit"].set_index("series")["crit_panel"], r["go"], r["cp"]
    g, sm = go["cfg"][best], _ref(None, r["comp"])
    q = r["comp"].set_index(["series", "window"])["qlike"]
    return ("%s — CLD: лучшая %s, (1) crit_panel %.5f ≤ порога %.5f: %s; (2) без 2017 %.5f: %s; (3) без 1 %% дат %.5f: %s; простое среднее QLIKE 2017–2021 %.5f против K1 %.5f, K2 %.5f, K3 %.5f, D2 %.5f" % (
        "go" if g["go"] else "stop", best, crit[best], cp["go_threshold_B"], "да" if g["c1"] else "нет", g["crit_no2017"], "да" if g["c2"] else "нет", g["crit_trim1"], "да" if g["c3"] else "нет",
        q[(best, "2017_2021")], sm["k1"], sm["k2"], sm["k3"], sm["d2"]))


def deltas(r: dict, best: str) -> list:
    """Δ QLIKE лучшей CLD к K1, K2, K3, D2, C8, NTP1 в % уровня K1: вся история, 2017, 2018, 2022–2025 (M16)."""
    comp = r["comp"].set_index(["series", "window"])["qlike"]
    yt = r["years"].pivot_table(index="year", columns="series", values="ann_qlike")
    S = ("k1", "k2", "k3", "d2", "C8", "NTP1")
    L = ["| период | K1 | K2 | K3 | D2 | C8 | NTP1 |", "|---|---|---|---|---|---|---|"]
    for name, get in (("вся", lambda s: comp[(s, "all")]), ("2017", lambda s: yt.loc[2017, s]), ("2018", lambda s: yt.loc[2018, s]), ("2022–2025", lambda s: comp[(s, "2022_2025")])):
        L.append("| %s | %s |" % (name, " | ".join("%+.2f" % (100.0 * (get(best) - get(s)) / get("k1")) for s in S)))
    return L


def _crit_lines(r: dict) -> list:
    crit, go, q = r["crit"].set_index("series")["crit_panel"], r["go"], r["comp"].set_index(["series", "window"])["qlike"]
    L = ["| ряд | crit_panel | без 2017 | без 1 % дат | среднее QLIKE 2017–2021 |", "|---|---|---|---|---|"]
    for s in ("k1", "k2", "k3", "d2", "C1", "C4", "C8", "NTP1", "NTP3", "CLD1", "CLD2", "CLD3", "CLD4"):
        g = go["ref"].get(s) or go["cfg"][s]
        L.append("| %s | %.5f | %.5f | %.5f | %.5f |" % (s, crit[s], g["crit_no2017"], g["crit_trim1"], q[(s, "2017_2021")]))
    return L


def _pred_lines(pred: pd.DataFrame) -> list:
    w = pred[pred["kind"] == "window"]
    L = ["| CLD | окно | n | R² OOS | знак | доля D>0 | корр. |", "|---|---|---|---|---|---|---|"]
    return L + ["| %s | %s | %d | %.4f | %.3f | %.3f | %.3f |" % (x.config, x.period, x.n, x.r2_oos, x.sign_share, x.share_pos, x.corr) for x in w.itertuples() if x.period != "2026" and x.config != "CLD2"] + ["", "CLD2 — та же регрессия, что CLD1 (D̂ те же, отличается только отображение в вес), строки в CLD_PRED.csv."]


def verdict(r: dict, pred: pd.DataFrame) -> str:
    """Можно ли превратить прогноз D в устойчивые веса: по числам CLD_PRED (R², знак против доли D>0, корреляция) и crit_panel против прямого обучения (C4, NTP3)."""
    best, crit = r["best"], r["crit"].set_index("series")["crit_panel"]
    p = pred[(pred["config"] == best) & (pred["kind"] == "window")].set_index("period")
    pr = bool(p.loc["2017_2021", "r2_oos"] > 0 and p.loc["2017_2021", "corr"] > 0)
    beat = bool(crit[best] < min(crit["C4"], crit["NTP3"]))
    w = r["fc"][best]
    w = w[w["ticker"].isin(IO.M16) & w["dbar"].notna() & (w["t_plus"].dt.year <= 2021)]["w"]
    concl = ("прогноз дифференциала превращается в веса лучше прямого обучения" if (pr and beat) else
             "информативные веса из D̂ получить нельзя: перед прямым обучением преимущество есть, но D̂ ничего не предсказывает, вес почти постоянный (w = %.3f ± %.3f в 2017–2021) — выигрыш даёт сжатие к 0,5, не условность" % (w.mean(), w.std()) if beat else
             "ни предсказуемого D, ни выигрыша у прямого обучения")
    return ("D %s предсказуем вне выборки (%s: R² %.4f, корр. %.3f, знак %.3f при доле D>0 %.3f в 2017–2021; 2022–2025: R² %.4f, корр. %.3f); %s лучше прямого обучения по QLIKE: %.5f против C4 %.5f, NTP3 %.5f. Вывод: %s." % (
        "слабо" if pr else "НЕ", best, p.loc["2017_2021", "r2_oos"], p.loc["2017_2021", "corr"], p.loc["2017_2021", "sign_share"], p.loc["2017_2021", "share_pos"], p.loc["2022_2025", "r2_oos"], p.loc["2022_2025", "corr"],
        best if beat else "не", crit[best], crit["C4"], crit["NTP3"], concl))


def build(r: dict, pred: pd.DataFrame) -> str:
    cp, best = r["cp"], r["best"]
    L = [first_line(r), "", "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок; выбор — только crit_panel (2017–2021); окна 2022+ — описательно (Ж-225).", "",
         "## Отступления и толкования (до выводов)"] + ["- " + x for x in DEVIATIONS]
    L += ["", "## crit_panel, три условия «go» и простое среднее QLIKE 2017–2021 (M16)", ""] + _crit_lines(r)
    L += ["", "Порог go (1): %.5f; (2), (3): min(K2, K3, D2) − 0,0005 по тому же критерию." % cp["go_threshold_B"]]
    L += ["", "## Предсказуемость D вне выборки (CLD_PRED.csv; M16, по году цели)", ""] + _pred_lines(pred)
    L += ["", "## Прогноз дифференциала против прямого обучения весов по QLIKE (C4, NTP3)", "", verdict(r, pred)]
    L += ["", "## Δ QLIKE лучшей (%s) к K1, K2, K3, D2, C8, NTP1 (%% уровня K1, M16; минус — лучше)" % best, ""] + deltas(r, best)
    L += ["", "## Холм (лучшая; семейство M16 × {2022_2025, all} × CLD1–CLD4, N=%d; N_wide=%d; K1, D2, C8)" % (cp["N_declared_before_stats"], cp["N_wide"]), ""] + BR.holm_lines({**r, "tests": r["tests"][r["tests"]["comparator"].isin(["k1", "d2", "C8"])]})
    L += ["", "## SBER (отдельно, в выбор не входит; 2271 дата; ориентир 0,2081 — потолок)", ""] + BR.sber_lines(r)
    lt = r["loto"][r["loto"]["held_out"] == "M16"].set_index("window")
    L += ["", "## LOTO лучшей (%s; M16: LOTO / пул / K1)" % best, ""] + ["- %s: %.4f / %.4f / %.4f" % (w, lt.loc[w, "q_loto"], lt.loc[w, "q_pool"], lt.loc[w, "q_k1"]) for w in ("2017_2021", "2022_2025", "all")]
    L += ["", "Израсходовано конфигураций листа: 4 (CLD1–CLD4) из 60; после листа %d из 60 (CHECKPOINT.json, `used_after_CLD`)." % (int(cp["used_after_NTP"]) + 4)]
    return "\n".join(L) + "\n"
