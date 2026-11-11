"""S-T: `S_T_REPORT.md` — текст целиком из таблиц `build()` (чисел, вписанных руками, нет). Порядок: полная таблица (Ж-256 п. 1), годы, срезы (диагностика), совпадение исключённых дат."""
from __future__ import annotations

import pandas as pd

MARK = "исследовательский результат, не независимый OOS; DEVELOPMENT / HISTORICAL WALK-FORWARD"
BASES = ("K1", "K2", "K3", "X1", "W5-G")


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def _f(v, p=4, sign=False) -> str:
    return "—" if pd.isna(v) else (("%+." + str(p) + "f") if sign else ("%." + str(p) + "f")) % v


def _full(tb: pd.DataFrame) -> str:
    rows = [[r.series, r.role, r.status, "" if pd.isna(r.n) else int(r.n), _f(r.qlike, 6)] + [_f(getattr(r, "d_%s_pct" % b.replace("-", "_")), 3, True) for b in BASES] for r in tb.rename(columns=lambda c: c.replace("-", "_")).itertuples()]
    return _md(pd.DataFrame(rows, columns=["ряд", "роль", "статус", "n", "QLIKE"] + ["Δ к %s, %%" % b for b in BASES]))


def _years(y: pd.DataFrame) -> str:
    per = y[(y["base"] == "-")]
    order = list(dict.fromkeys(per["period"]))
    q = per.pivot(index="series", columns="period", values="q_series").reindex(list(dict.fromkeys(per["series"])))[order]
    n = per[per["series"] == per["series"].iloc[0]].set_index("period")["n"]
    t1 = pd.DataFrame([[s] + [_f(v, 4) for v in q.loc[s]] for s in q.index], columns=["ряд"] + ["%s (n=%d)" % (p, n[p]) for p in order])
    out = ["### QLIKE по периодам t⁺ и половинам (h1 = 2018–2021, h2 = 2022–2026)", _md(t1)]
    for b in ("W5-G", "X1"):
        d = y[y["base"] == b]
        o = list(dict.fromkeys(d["period"]))
        pv = d.pivot(index="series", columns="period", values="delta_pct").reindex(list(dict.fromkeys(d["series"])))[o]
        tot = d.groupby("series", sort=False)["delta_total_pct"].first()
        t2 = pd.DataFrame([[s] + [_f(v, 3, True) for v in pv.loc[s]] + [_f(tot[s], 3, True)] for s in pv.index], columns=["ряд"] + o + ["все 2198"])
        out += ["### Δ к %s по периодам, %% QLIKE %s в периоде (минус — лучше)" % (b, b), _md(t2)]
    return "\n\n".join(out)


def _slices(s: pd.DataFrame) -> str:
    rows = [[r.candidate, r.base, r.slice, r.kind, r.n, "%s: %s" % (r.denominator.split(" на ")[0], _f(r.q_base, 6)), _f(r.delta_pct, 3, True)] for r in s.itertuples()]
    return _md(pd.DataFrame(rows, columns=["кандидат", "база (контроль)", "срез", "статус", "n", "знаменатель Δ", "Δ, %"]))


def _dates(ts: pd.DataFrame) -> str:
    pr = ts[ts["kind"] == "пара сравнений"]
    key = pr[(pr["a"] == "S-RCK | W5-G") & (pr["b"] == "S-RCK | X1")].iloc[0]
    line = "Исключённые при обрезке 22 + 22 даты **не одинаковы** для разных сравнений: общих дат из 44 — от %d до %d (медиана %.1f), Жаккар от %.3f до %.3f. S-RCK к W5-G и к X1: общих %d из 44, Жаккар %.3f." % (
        pr["n_common"].min(), pr["n_common"].max(), pr["n_common"].median(), pr["jaccard"].min(), pr["jaccard"].max(), key["n_common"], key["jaccard"])
    if pr["n_common"].min() == pr["n_each"].iloc[0] and pr["n_common"].max() == pr["n_each"].iloc[0]:
        line = line.replace("**не одинаковы**", "**одинаковы**")
    lst = ["- %s, %s: %s" % (r.a, r.side, r.dates) for r in ts[ts["kind"] == "перечень дат"].itertuples()]
    return "\n\n".join([line] + lst)


def report(t: dict, tp) -> str:
    tp = pd.DatetimeIndex(tp)
    head = "# S-T: сводная таблица и разбор сохранённых прогнозов (Ж-256 пп. 1, 4)\n\n> %s. Обучения нет: читаются сохранённые прогнозы. D_M: t⁺ %s…%s, n = %d. Основной результат — обычная QLIKE (среднее дневных потерь) на всех общих датах; Δ = 100·(QLIKE ряда − QLIKE базы)/QLIKE базы, минус — лучше. Кандидат S-RCK берётся как ряд F (не слой X1_R), поэтому Δ к X1 здесь — по F." % (
        MARK, tp.min().date(), tp.max().date(), len(tp))
    s2 = "## 1. Полная таблица на одинаковых датах\n\n" + _full(t["table"]) + "\n\nСтроки «не рассчитывался» — варианты, которых нет среди сохранённых прогнозов."
    s3 = "## 2. По годам\n\n" + _years(t["years"])
    s4 = "## 3. Срезы — диагностика (не основной результат)\n\nДиагностика: контрольная модель и знаменатель названы в каждой строке; знаменатель — QLIKE базы на тех же оставшихся датах. Строка «основной» — Δ на всех датах без обрезки.\n\n" + _slices(t["slices"])
    s5 = "## 4. Совпадение исключённых дат (диагностика)\n\n" + _dates(t["trimsets"])
    return "\n\n".join([head, s2, s3, s4, s5, "_" + MARK + "._"]) + "\n"
