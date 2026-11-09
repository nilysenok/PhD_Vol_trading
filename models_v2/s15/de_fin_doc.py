"""FIN: итоговый документ — генератор из таблиц (числа руками не вписываются). Часть 1: помощники, правила слов, первая строка, разделы 1, 2, 5. Часть 2 — `de_fin_doc2`.
Каждое десятичное число в тексте записывается `fx` (или `txt` для текста из файлов); `check` сверяет множество десятичных чисел документа с записанными — число, набранное руками, не пройдёт."""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from models_v2.s15.de_e1_doc import WINS, YEARS, f, md
from models_v2.s15.de_fin_tabs import ALL

ALPHA = 0.05                      # «подтверждено» — только при правиле 6.2 «да» И p Холма (N = 738) < ALPHA
FOUR = (("V4:NTP2", "NTP2"), ("V6:A2", "A2"), ("K2", "XGBoost (K2)"), ("K3", "LightGBM (K3)"))
X = ("X1", "X2", "X3", "X4")
NUMS: list = []
DATE = re.compile(r"\d{2}\.\d{2}\.\d{4}")
DEC = re.compile(r"\d+\.\d+")
STATUS = "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок; дизайн K2/K3 выбирался на SBER 2017–2021. Только SBER, h = 1; расходы пилота — 4 конфигурации из 5 (X5 не считается). Окна 2022+ — только оценка, выбора по ним нет; весь период многократно просмотрен."


def fx(x, nd: int = 4) -> str:
    s = f(x, nd)
    if s != "—":
        NUMS.append(s.lstrip("-"))
    return s


def txt(s: str) -> str:
    """Текст из файла (реестр, DE0): его десятичные числа записываются как «из файла»."""
    NUMS.extend(DEC.findall(DATE.sub("", s)))
    return s


STATIC_OK = {"6.1", "6.2"}        # номера правил, не результаты


def doc_decimals(doc: str) -> Counter:
    return Counter(x for x in DEC.findall(DATE.sub("", doc)) if x not in STATIC_OK)


def recorded() -> Counter:
    return Counter(x for x in NUMS if x not in STATIC_OK)


def better(dq: float) -> str:
    return "лучше" if dq < 0 else "хуже"


def pair(t: dict, x: str, y: str, w: str):
    r = t["PAIRS"]
    return r[(r["x"] == x) & (r["y"] == y) & (r["window"] == w)].iloc[0]


def verdict(r) -> str:
    if str(r["rule62"]) == "—":
        return "диагностика (окно не окно выбора)"
    return "подтверждено" if (str(r["rule62"]) == "да" and float(r["p_holm_N"]) < ALPHA) else "не подтверждено"


def qlike(t: dict, s: str, w: str) -> float:
    m = t["METRICS"]
    return float(m.loc[(m["series"] == s) & (m["window"] == w), "qlike"].iloc[0])


def best_x(t: dict, w: str) -> str:
    return min(X, key=lambda x: (qlike(t, x, w), x))


def pair_txt(r, name: str) -> str:
    return "против %s: Δ = %s %% уровня K1 [%s; %s], DM p = %s, p Холма (N = 738) = %s, правило 6.2: %s, %s (QLIKE %s)" % (name, fx(r["delta_pct"], 2), fx(r["lo90"], 2), fx(r["hi90"], 2), fx(r["p_one"], 3), fx(r["p_holm_N"], 3), "не применимо" if r["rule62"] == "—" else r["rule62"], verdict(r), better(float(r["delta_q"])))


def has_value(t: dict) -> bool:
    return any(all(verdict(pair(t, x, y, "2018_2021")) == "подтверждено" for y, _ in FOUR) for x in X)


def first_line(t: dict, num: dict) -> str:
    b, c = best_x(t, "2018_2021"), best_x(t, "2022_2025")
    k = sum(all(verdict(pair(t, x, y, "2018_2021")) == "подтверждено" for y, _ in FOUR) for x in X)
    s = ["**Дополнительная прогностическая ценность над XGBoost (K2), LightGBM (K3) и лучшим прежним гибридом (NTP2, A2) на одинаковых датах %s … %s (окно разработки 2018–2021, правило 6.2, Холм N = 738): %s.** Подтверждено у %d из 4 конфигураций пилота" % (
        num["first"], num["last"], "есть" if has_value(t) else "нет", k), "Лучшая по QLIKE 2018–2021 — %s (%s)" % (b, fx(qlike(t, b, "2018_2021")))]
    s += [pair_txt(pair(t, b, y, "2018_2021"), n) for y, n in FOUR]
    s.append("на 2022–2025 (диагностика, не отбор) лучшая по QLIKE — %s (%s)" % (c, fx(qlike(t, c, "2022_2025"))))
    s += [pair_txt(pair(t, c, y, "2022_2025"), n) for y, n in FOUR]
    return "; ".join(s) + "."


def sec1(t: dict) -> list:
    rows = []
    for w in ("2018_2021", "2022_2025"):
        for x in X:
            for y, n in FOUR:
                r = pair(t, x, y, w)
                rows.append([w, x, n, fx(r["delta_pct"], 2), fx(r["lo90"], 2), fx(r["hi90"], 2), fx(r["p_one"], 3), fx(r["p_holm_N"], 3), r["rule62"], verdict(r)])
    return ["## 1. Результат: X1–X4 против XGBoost, LightGBM, NTP2, A2", "", "Δ < 0 — конфигурация лучше базы (знак как в таблицах листов); интервал 90 %% блоковым бутстрапом (20 · 2000 · seed 0); DM лаг 5; Холм по семейству пар окна, N = 738. «Подтверждено» — правило 6.2 «да» И p Холма < %s; окно 2022–2025 — диагностика." % fx(ALPHA, 2), "",
            md(rows, ["окно", "конфигурация", "база", "Δ %", "lo90", "hi90", "DM p", "p Холм N", "6.2", "вердикт"]), ""]


def sec2(t: dict, causes: pd.DataFrame) -> list:
    adv, st = t["ADV"].set_index("series"), t["STATE"].set_index(["series", "window", "state"])["qlike"]
    out = ["## 2. Главный вопрос Ж-239 п. 2: причина проигрыша 2022–2025 и что устранено", "",
           "Причина (DE0, `DE0_CAUSES.csv`): разрыв HAR-J к K2 вырос после 2021 года, и гибрид закрывает его лишь частично; смена преимущества над K2 разложена на три части — K2 оторвался от HAR-J (`part_base`), выигрыш гибрида над HAR-J на недооценке u > 0 (`part_gu`) и на переоценке u ≤ 0 (`part_go`).", "",
           md([[r.hybrid, fx(r.adv_2018_2021), fx(r.adv_2022_2025), fx(r.d_adv), fx(r.part_base), fx(r.part_gu), fx(r.part_go)] for r in causes.itertuples()], ["гибрид (DE0)", "преимущество над K2 2018–2021", "то же 2022–2025", "смена", "K2 оторвался", "недооценка", "переоценка"]), "",
           "Что устранено — преимущество над K2 (QLIKE(K2) − QLIKE(ряд); плюс — ряд лучше K2) и его смена, те же формулы для X1–X4 (`FIN_ADV.csv`):", "",
           md([[n, fx(adv.loc[n, "adv_2018_2021"]), fx(adv.loc[n, "adv_2022_2025"]), fx(adv.loc[n, "d_adv"])] for n in list(X) + ["ENS", "V4:NTP2", "V6:A2", "V6:A0g", "cfg8", "cfg4"]], ["ряд", "2018–2021", "2022–2025", "смена"]), "",
           "QLIKE по состоянию «экстрем / норма» (экстрем — экстремальный M10-бар дня d; состояние — s_d):", "",
           md([[s, w, fx(st[(s, w, "экстрем")]), fx(st[(s, w, "норма")])] for s in ("X1", "V4:NTP2", "cfg8", "K2", "V6:A0g") for w in ("2018_2021", "2022_2025")], ["ряд", "окно", "экстрем", "норма"]), ""]
    for y, n in (("K2", "K2"), ("V4:NTP2", "NTP2")):
        r = pair(t, "X1", y, "2022_2025")
        out.append("- X1 на 2022–2025 (диагностика) " + pair_txt(r, n))
    return out + [""]


def sec5(t: dict) -> list:
    m = t["METRICS"].set_index(["series", "window"])["qlike"]
    cols = YEARS + WINS + ["all"]
    out = ["## 5. QLIKE по годам — все ряды; сравнение с лучшим прежним гибридом", "", md([[s] + [fx(m.get((s, w), float("nan"))) for w in cols] for s in sorted(set(t["METRICS"]["series"]), key=ALL.index)], ["ряд"] + cols), "",
           "Сравнение с лучшим прежним гибридом: NTP2 — по всей истории, A2 — по окну 2018–2021 (Δ в % уровня K1; минус — конфигурация лучше):", ""]
    tt, rows = t["TESTS"], []
    for x in X:
        for b, w in (("V4:NTP2", "2018_2021"), ("V4:NTP2", "2022_2025"), ("V4:NTP2", "all"), ("V6:A2", "2018_2021")):
            r = tt[(tt["config"] == x) & (tt["base"] == b) & (tt["window"] == w)].iloc[0]
            rows.append([x, b, w, fx(r["delta_pct"], 2), fx(r["lo90"], 2), fx(r["hi90"], 2), fx(r["p_one"], 3), fx(r["p_holm_N"], 3)])
    return out + [md(rows, ["конфигурация", "база", "окно", "Δ %", "lo90", "hi90", "DM p", "p Холм N"]), ""]
