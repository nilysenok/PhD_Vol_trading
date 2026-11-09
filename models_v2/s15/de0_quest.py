"""DE0, G3/G4: ответы на вопросы 1–8 п. 3 блока Ж-239 — по одной строке с числами, ТОЛЬКО из таблиц DE0 (правила выбора объявлены здесь).
Клетка = (признак, группа) с n ≥ 50 в окне; строка `все` не участвует. Δ — гибрид − K2 (минус — гибрид лучше), в скобках — % QLIKE K2 в той же клетке."""
from __future__ import annotations

import pandas as pd

MIN_N = 50
WINS = ("2018_2021", "2022_2025")


def _cells(g: pd.DataFrame, w: str) -> pd.DataFrame:
    c = g[(g["feature"] != "все") & (g["window"] == w) & (g["n"] >= MIN_N)]
    return c.assign(d_H=c["q_H"] - c["q_K2"])


def _name(r) -> str:
    return "%s / %s (n=%d)" % (r["feature"], r["bin"], r["n"])


def _pct(r, col: str) -> str:
    return "%+.4f (%+.1f %%)" % (r[col], 100.0 * r[col] / r["q_K2"])


def _all(g: pd.DataFrame, w: str):
    return g[(g["feature"] == "все") & (g["window"] == w)].iloc[0]


def answers(g: pd.DataFrame, acf: pd.DataFrame, runs: pd.DataFrame, ridge: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in WINS:
        c, a = _cells(g, w), _all(g, w)
        hi, lo, k2, im, wo = c.loc[c["u_H"].idxmax()], c.loc[c["u_H"].idxmin()], c.loc[c["d_H"].idxmax()], c.loc[c["d_cfg8"].idxmin()], c.loc[c["d_cfg8"].idxmax()]
        rows += [(1, w, "HAR-J недооценивает (u > 0) там, где u(H) наибольший: %s, средний u(H) = %+.3f; клеток с u(H) > 0: %d из %d; окно в целом %+.3f" % (_name(hi), hi["u_H"], int((c["u_H"] > 0).sum()), len(c), a["u_H"])),
                 (2, w, "HAR-J переоценивает (u < 0) сильнее всего: %s, средний u(H) = %+.3f; клеток с u(H) < 0: %d из %d; окно в целом %+.3f" % (_name(lo), lo["u_H"], int((c["u_H"] < 0).sum()), len(c), a["u_H"])),
                 (3, w, "XGBoost лучше HAR-J сильнее всего: %s, QLIKE(H) − QLIKE(K2) = %+.4f (в окне в целом %+.4f)" % (_name(k2), k2["d_H"], a["q_H"] - a["q_K2"])),
                 (4, w, "гибрид cfg8 улучшает K2 сильнее всего: %s, Δ = %s; A2: лучшая клетка Δ = %+.4f (%s)" % (_name(im), _pct(im, "d_cfg8"), c["d_A2"].min(), _name(c.loc[c["d_A2"].idxmin()]))),
                 (5, w, "гибрид cfg8 ухудшает K2 сильнее всего: %s, Δ = %s; A2: худшая клетка Δ = %+.4f (%s); окно в целом Δ cfg8 %+.4f, A2 %+.4f" % (_name(wo), _pct(wo, "d_cfg8"), c["d_A2"].max(), _name(c.loc[c["d_A2"].idxmax()]), a["d_cfg8"], a["d_A2"]))]
        ac, rn = acf[(acf["window"] == w) & (acf["diff"] == "l_cfg8 − l_K2")].iloc[0], runs[(runs["window"] == w) & (runs["diff"] == "l_cfg8 − l_K2")].iloc[0]
        rows.append((6, w, "ошибки повторяются: ACF дифференциала cfg8−K2 лаг 1 = %+.3f (лаги 2–5: %s; 95 %%-полоса ±%.3f); серии d_t > 0: средняя длина %.2f при независимости %.2f, максимум %d" % (
            ac["lag1"], ", ".join("%+.3f" % ac["lag%d" % k] for k in (2, 3, 4, 5)), ac["band95"], rn["mean_len"], rn["exp_mean_len"], rn["max_len"])))
        e = g[(g["feature"] == "d_ekstrem") & (g["window"] == w)].set_index("bin")
        rows.append((7, w, "экстремальный M10-бар (да n=%d / нет n=%d): u(H) %+.3f / %+.3f; Δ cfg8 %+.4f / %+.4f; Δ A2 %+.4f / %+.4f; QLIKE(H) %.4f / %.4f" % (
            e.loc["да", "n"], e.loc["нет", "n"], e.loc["да", "u_H"], e.loc["нет", "u_H"], e.loc["да", "d_cfg8"], e.loc["нет", "d_cfg8"], e.loc["да", "d_A2"], e.loc["нет", "d_A2"], e.loc["да", "q_H"], e.loc["нет", "q_H"])))
    p = ridge[ridge["year"] == "пул"]
    parts = []
    for t, x in p.groupby("target", sort=False):
        b = x.loc[x["r2_oos"].idxmax()]
        parts.append("%s: лучший набор %s, R² вне выборки %+.4f (n=%d)" % (t, b["features"], b["r2_oos"], b["n"]))
    rows.append((8, "все годы вне выборки", "связь с прошлыми M10-признаками (PIT-гребень, S/Σ/z): " + "; ".join(parts)))
    return pd.DataFrame(rows, columns=["q", "window", "answer"])
