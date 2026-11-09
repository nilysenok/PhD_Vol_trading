"""V6 лист Z3: документ причин `HARJ_SBER_V6_PRICHINY.md` — ЦЕЛИКОМ из таблиц `docs/otchety/harj_hybrid_v6/why/WHY_*.csv` (разделы 1–3; 4–7, вывод и сборка — `v6_why_doc2`). Ни одно результатное число не вписано руками:
десятичные числа текста идут только из функций ниже (сверка `v6_why_docheck`). Формат: десятичная запятая, минус «−»; Δ — разность средних QLIKE (гибрид − база) в % уровня QLIKE K1 (минус = гибрид лучше)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_why_data as DT

KIND_RU = {"low": "низкий терциль K1", "mid": "средний терциль K1", "high": "высокий терциль K1", "no_group": "первые 250 дат (без группы)", "up": "RV(t⁺) > F_K1 (по исходу)", "down": "RV(t⁺) ≤ F_K1 (по исходу)", "total": "всего"}


def load_tables() -> dict:
    return {n: DT.read_csv(n) for n in DT.TABLES}


def f(x, nd: int = 2) -> str:
    return "—" if pd.isna(x) else ("%.*f" % (nd, x)).replace(".", ",").replace("-", "−")


def s(x, nd: int = 2) -> str:
    return "—" if pd.isna(x) else ("%+.*f" % (nd, x)).replace(".", ",").replace("-", "−")


def md(head: list, rows: list) -> list:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def one(df: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    if m.sum() != 1:
        raise KeyError("строк %d для %s" % (m.sum(), kw))
    return df[m].iloc[0]


def pr(h: str, b: str) -> str:
    return "%s к %s" % (h, b)


def year_stats(t: dict, h: str, b: str) -> dict:
    y = t["WHY_YEARS"]
    y = y[(y.hybrid == h) & (y.base == b) & (y.year != "total")]
    return dict(neg=int((y.delta < 0).sum()), n=len(y), y21=float(one(y, year="2021").delta_pct), y22=float(one(y, year="2022").delta_pct), total=float(one(t["WHY_YEARS"], hybrid=h, base=b, year="total").delta_pct))


def sec_years(t: dict) -> list:
    out = ["## 1. Годы", "", "Δ_y — разность средних QLIKE (гибрид − база) по году цели t⁺, в % уровня QLIKE K1 того же года; вклад года = доля дат года × Δ_y, в % уровня K1 всей истории; сумма вкладов равна общей Δ (тождество, допуск 1e-12, проверено при сборке).", ""]
    y = t["WHY_YEARS"]
    for h, b in DT.MAIN:
        z = y[(y.hybrid == h) & (y.base == b)]
        out += ["**%s**" % pr(h, b), ""] + md(["год", "дат", "доля дат", "Δ_y, % K1", "вклад, % K1", "знак"], [[r.year, r.n, f(r.weight, 4), s(r.delta_pct), s(r.contrib_pct), "−" if r.sign < 0 else "+"] for r in z.itertuples()])
        k = year_stats(t, h, b)
        out += ["%s: лет с Δ_y < 0 — %d из %d; 2021 → 2022: %s → %s %% уровня K1 (%s)." % (pr(h, b), k["neg"], k["n"], s(k["y21"]), s(k["y22"]), "знак меняется" if k["y21"] * k["y22"] < 0 else "знак не меняется"), ""]
    rows = []
    for h, b in DT.PAIRS:
        k = year_stats(t, h, b)
        rows.append([pr(h, b), s(k["total"]), "%d из %d" % (k["neg"], k["n"]), s(k["y21"]), s(k["y22"]), "да" if k["y21"] * k["y22"] < 0 else "нет"])
    return out + ["Все пары (годовые строки всех пар — `WHY_YEARS.csv`):", ""] + md(["пара", "Δ вся история, % K1", "лет с Δ < 0", "Δ 2021, % K1", "Δ 2022, % K1", "смена знака 2021 → 2022"], rows)


def sec_tails(t: dict) -> list:
    out = ["## 2. Крайние дни", "", "Дни с наибольшим |l_гибрид − l_база| (l — дневная QLIKE; доли 1 % и 5 % дат, округление до целого числа дней). Вклад топа = сумма разностей топ-дней / число всех дат, в % уровня K1 всей истории; Δ без топа — среднее по остальным дням, в % уровня K1 тех же дней; топ + остальные = общей Δ (тождество). Доля топа > 100 % означает, что остальные дни имеют Δ противоположного знака.", ""]
    z = t["WHY_TAILS"]
    rows = [[pr(r.hybrid, r.base), "%d %%" % round(100 * r.frac), r.n_top, r.n_top_worse, s(r.delta_pct), s(r.contrib_top_pct), f(r.share_top_pct, 1), s(r.delta_without_pct)] for r in z.itertuples()]
    return out + md(["пара", "доля дат", "дней в топе", "из них гибрид хуже", "Δ общая, % K1", "вклад топа, % K1", "доля топа в Δ, %", "Δ без топа, % K1"], rows)


def sec_regime(t: dict) -> list:
    out = ["## 3. Режимы (PIT)", "", "Терцили уровня прогноза K1 на дату d: пороги — только по значениям K1 на датах ≤ d (расширяющееся окно); первые 250 дат — без группы. Группы «по исходу» (RV(t⁺) против F_K1) используют будущее и только описывают. Δ группы — в % уровня K1 тех же дат; вклад = сумма разностей группы / число всех дат, в % уровня K1 всей истории; сумма вкладов групп равна общей Δ (тождество).", ""]
    g = t["WHY_REGIME"]
    for h, b in DT.MAIN:
        z = g[(g.hybrid == h) & (g.base == b)]
        out += ["**%s**" % pr(h, b), ""] + md(["разбиение", "группа", "дат", "Δ, % K1", "вклад, % K1"], [[("терцили PIT" if r.kind == "tercile" else "по исходу"), KIND_RU[r.group], r.n, s(r.delta_pct), s(r.contrib_pct)] for r in z.itertuples()])
    rows = []
    for h, b in DT.PAIRS:
        z = g[(g.hybrid == h) & (g.base == b)]
        rows.append([pr(h, b)] + [s(one(z, group=k).delta_pct) for k in ("low", "mid", "high", "up", "down")])
    return out + ["Все пары, Δ в % уровня K1 группы:", ""] + md(["пара", "низкий терциль", "средний", "высокий", "RV > F_K1 (по исходу)", "RV ≤ F_K1 (по исходу)"], rows)
