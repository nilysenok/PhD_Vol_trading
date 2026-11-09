"""V7 лист FIN: помощники формата и разделы 1–7 итогового документа. Каждое число берётся из таблиц `t` (FIN_*, CHECKPOINT); «лучше / хуже» выводится из знака Δ (`word`), знак Δ печатается из таблицы."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_fin_data as D, v7_fin_tabs as TB, v7_fin_text as TX

WN = {"2018_2021": "2018–2021", "2019_2021": "2019–2021", "2022_2025": "2022–2025", "2026": "2026", "all": "вся история 2018–2026"}
SH = {"V4:NTP2": "NTP2", "V6:A2": "A2", "V6:A0": "A0", "V6:A0g": "A0g"}


def nm(x: str) -> str:
    return SH.get(x, x)


def f(x, n: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return ("−" if x < 0 and round(abs(x), n) > 0 else "") + ("%.*f" % (n, abs(x))).replace(".", ",")


def s(x, n: int = 2) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else ("−" if x < 0 else "+") + ("%.*f" % (n, abs(x))).replace(".", ",")


def word(delta: float) -> str:
    return "лучше" if delta < 0 else "хуже"


def tbl(head: list, rows: list) -> str:
    return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def q(t: dict, series: str, win: str) -> float:
    a = t["FIN_ALL"]
    return float(a[(a["series"] == series) & (a["window"] == win)]["qlike"].iloc[0])


def tt(t: dict, cfg: str, base: str, win: str):
    x = t["FIN_TESTS"]
    return x[(x["config"] == cfg) & (x["base"] == base) & (x["window"] == win)].iloc[0]


def iv(r) -> str:
    return "%s [%s; %s]" % (s(r["delta_pct"]), s(r["lo90"]), s(r["hi90"]))


def go_count(t: dict) -> int:
    g = t["FIN_GO"]
    return int(g[g["cond"] == 0]["passed"].astype(str).str.lower().eq("true").sum())


def first_line(t: dict) -> str:
    adv, b18, b22 = TB.advantage(t), TB.best(t, "2018_2021"), TB.best(t, "2022_2025")
    ph = min(float(tt(t, b18, b, "2018_2021")["p_holm_N"]) for b in D.BASES)
    qs = lambda c, w: "%s; K3 %s, NTP2 %s, A2 %s" % (f(q(t, c, w), 4), f(q(t, "K3", w)), f(q(t, "V4:NTP2", w)), f(q(t, "V6:A2", w)))
    return ("%s: %d из %d посчитанных конфигураций V7 имеют на 2018–2021 90 %% интервал Δ целиком ниже нуля против K2 (XGBoost), K3 (LightGBM), NTP2 и A2 одновременно; «go» 6.1 выполнено у %d из %d; "
            "лучшая по QLIKE 2018–2021 — %s (%s), лучшая по 2022–2025 — %s (%s); наименьший p Холма (N = 733) у лучшей 2018–2021 против этих четырёх — %s; статус — %s."
            % ("Да" if adv else "Нет", len(adv), len(D.CFG), go_count(t), len(D.CFG), b18, qs(b18, "2018_2021"), b22, qs(b22, "2022_2025"), f(ph, 3),
               "подтверждено" if adv and go_count(t) and ph < 0.05 else "статистически неподтверждённое историческое улучшение" if go_count(t) else "улучшения не подтверждено"))


def best_block(t: dict, cfg: str, win: str) -> str:
    rows = []
    for b in D.BASES + ["K1"]:
        r = tt(t, cfg, b, win)
        rows.append([nm(b), f(q(t, b, win)), iv(r), word(r["delta_pct"]), f(r["p_one"], 3), f(r["p_holm_N"], 3), f(r["p_holm_Nwide"], 3)])
    return tbl(["база", "QLIKE базы", "Δ, % K1 [90 %]", "конфигурация", "DM p", "Холм N", "Холм N_wide"], rows)


def mcs_block(t: dict) -> str:
    m, rows = t["FIN_MCS"], []
    for w in ("2018_2021", "2022_2025", "all"):
        k = m[(m["window"] == w) & m["in_mcs"].astype(str).str.lower().eq("true")]
        rows.append([WN[w], len(k), ", ".join(nm(x) for x in k["series"])])
    return tbl(["окно", "в MCS 90 %", "ряды"], rows)


def sec1(t: dict) -> str:
    b18, b22 = TB.best(t, "2018_2021"), TB.best(t, "2022_2025")
    out = "Лучшая по QLIKE 2018–2021: **%s** (%s); лучшая по 2022–2025: **%s** (%s). Δ = QLIKE(конфигурация) − QLIKE(база) в %% среднего QLIKE K1 окна: минус — конфигурация лучше. Ряды — на одинаковых датах, `FIN_TESTS.csv`.\n\n" % (b18, f(q(t, b18, "2018_2021")), b22, f(q(t, b22, "2022_2025")))
    out += "**%s, окно 2018–2021:**\n\n%s\n**%s, окно 2022–2025:**\n\n%s\n" % (b18, best_block(t, b18, "2018_2021"), b22, best_block(t, b22, "2022_2025"))
    if b22 != b18:
        out += "**%s, окно 2022–2025:**\n\n%s\n" % (b18, best_block(t, b18, "2022_2025"))
    return out + "MCS 90 %% (`FIN_MCS.csv`):\n\n%s" % mcs_block(t)


def sec2(t: dict, cp: dict, enc: dict) -> str:
    p, c = cp["1a"]["params"], cp["1b"]["choice3"]["choice"]
    e = enc[c.lower()]
    rows = [["1", "TCN-w5 + g(z)", enc["tcn"], 5, enc["tcn"] + 5], ["2", "CNN-w5 + g(z)", enc["cnn"], 5, enc["cnn"] + 5], ["3", "%s-w1 + g(z)" % c, e, 5, e + 5],
            ["4", "плоский корректор g(z, Z)", e, p["flat"], e + p["flat"]], ["5", "MoE, маршрутизатор A", e, p["router"]["A"] + p["experts"], e + p["router"]["A"] + p["experts"]],
            ["6", "MoE, маршрутизатор B (основная)", e, p["moe_b"], e + p["moe_b"]], ["7", "MoE, маршрутизатор C", e, p["router"]["C"] + p["experts"], e + p["router"]["C"] + p["experts"]],
            ["8", "MoE-B + якорь w·ℓ", e, p["anchor_b"], e + p["anchor_b"]], ["10", "совместное обучение (Stage B) для 6", e, p["moe_b"], e + p["moe_b"]]]
    return "%s\n\n%s\n**Число обучаемых параметров** (энкодер — по предобучению; в Stage B конфигурации 10 энкодер дообучается вместе с головой; `1a.params`, `v7_1e_net.PARAMS`):\n\n%s" % (
        TX.ARCH, TX.RULES, tbl(["№", "конфигурация", "энкодер", "голова", "всего"], rows))


def sec3(t: dict) -> str:
    d, ch = t["FIN_DATA"], t["FIN_CHANNELS"]
    rows = [[int(r.year), int(r.n_days), int(r.n_bars), int(r.bars_mode), int(r.days_not_mode), f(r.zero_return_share), r.date if isinstance(r.date, str) else "—", f(r.n_a, 0), f(r.n_b, 0)] for r in d.itertuples()]
    c = [[r.channel, "да" if pd.isna(r.available) else r.available, "" if pd.isna(r.note) else r.note] for r in ch.itertuples()]
    return ("Бары M10 основной сессии SBER (10:00–18:40, 53 слота), Σr² = RV; паспорт — `M10_PASSPORT.csv`, число закрытых пар на первую дату года — `SAMPLES.csv` (n_a — для мета-слоя с K2/K3, n_b — для корректора H):\n\n%s\n"
            "**Каналы входа энкодера (11):**\n\n%s\nПризнаки маршрутизатора Z — раздел 2." % (tbl(["год", "дней", "баров", "мода баров", "дней не по моде", "доля нулевых r", "первая дата года", "пар n_a", "пар n_b"], rows), tbl(["канал", "надёжен", "примечание"], c)))


def sec4(t: dict) -> str:
    r = t["FIN_REGISTRY"]
    rows = [[int(x.place), x.name, x.status, "да" if x.v1_archived else "—", f(q(t, "cfg%d" % x.place, "2018_2021")) if "cfg%d" % x.place in D.CFG else "—", f(q(t, "cfg%d" % x.place, "2022_2025")) if "cfg%d" % x.place in D.CFG else "—"] for x in r.itertuples()]
    done = [x.place for x in r.itertuples() if x.status == "посчитана"]
    skip = [x.place for x in r.itertuples() if str(x.status).startswith("не считается")]
    out = "Места 1–11 реестра (`ALL_TRIALS.csv`, Ж-233); посчитано %d (%s), не считалось по условию — %d (%s); резерв (место 12): не использован, в реестре %d мест.\n\n%s\n" % (len(done), ", ".join(map(str, done)), len(skip), ", ".join(map(str, skip)), len(r), tbl(["место", "конструкция", "статус", "v1 в архиве", "QLIKE 2018–2021", "QLIKE 2022–2025"], rows))
    out += "**Почему не считались:**\n\n" + "".join("- место %d — %s.\n" % (n, TX.NOT_RUN[n]) for n in skip) + "\n**Вычеркнуты как дубли конструкции (план, раздел 2):**\n\n" + TX.CROSSED
    return out


def sec5(t: dict) -> str:
    out = "Окно 2017–2021 справочно — не применимо: энкодеры и мета-слой начинают оценку с 2018-01-03 (2017 — только обучение). Ряды: контроли K1, K2, K3, M, K8, NTP2, A2, A0, A0g и конфигурации (`FIN_ALL.csv`).\n\n"
    for w in ("2018_2021", "2019_2021", "2022_2025", "2026"):
        a = t["FIN_ALL"]
        a = a[a["window"] == w].set_index("series").loc[D.SER]
        out += "**%s** (n = %d):\n\n%s\n" % (WN[w], int(a["n"].iloc[0]), tbl(["ряд", "QLIKE", "RMSE", "MAE", "R² (RV)", "R² (ln)", "bias (ln)"], [[nm(k), f(r.qlike), f(r.rmse), f(r.mae), f(r.r2_rv, 3), f(r.r2_ln, 3), f(r.bias_ln, 3)] for k, r in a.iterrows()]))
    return out


def sec6(t: dict) -> str:
    y = t["FIN_YEARS"]
    return tbl(["год"] + [nm(c) for c in D.SER], [[int(r["year"])] + [f(r[c]) for c in D.SER] for _, r in y.iterrows()])


def sec7(t: dict) -> str:
    out = ""
    for w in ("2018_2021", "2022_2025"):
        d = [[c] + [iv(tt(t, c, b, w)) for b in D.BASES] for c in D.CFG]
        p = [[c] + ["%s / %s / %s" % (f(tt(t, c, b, w)["p_one"], 3), f(tt(t, c, b, w)["p_holm_N"], 3), f(tt(t, c, b, w)["p_holm_Nwide"], 3)) for b in D.BASES] for c in D.CFG]
        out += "**Δ, %% уровня K1 [90 %%], %s:**\n\n%s\n**DM p (односторонний «конфигурация лучше») / Холм N = 733 / Холм N_wide = 3166, %s:**\n\n%s\n" % (WN[w], tbl(["конфигурация"] + [nm(b) for b in D.BASES], d), WN[w], tbl(["конфигурация"] + [nm(b) for b in D.BASES], p))
    return out + "Базы: K2 — XGBoost, K3 — LightGBM, NTP2 — прежний гибрид V4, A2 — прежний гибрид V6.\n"
