"""S-A-EVAL (Ж-254; `S_A_SPEC.md`, разд. 6.2, 6.4, 6.5): годы и вклады, концентрация, мощность, защита У1-S…У5-S и решение — чистые функции по таблицам и массивам; ничего не пишут.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Все пороги — константы ЭТОГО файла (из спецификации), не из результата.
Публичное API:
- `THR` = {У1-S: 0,415; У2-S: 0,50; У3-S: 1,00} (%), `ORDER` = (S-A2, S-A1) — порядок рассмотрения кандидатов (A0 и A2p кандидатами не являются), `COMPS` — 12 пар (ряд, база).
- `years(tp, ls)`: ряды и сравнения по периодам 2018…2025, 2026_до_27.02, 2026_F и половинам h1 (2018–2021, n = 1010) / h2 (2022–2026, n = 1188); вклад c_p = 100·Σ(l_h − l_b)/(n·Q_b), Σ c_p = Δ (1e-12); `identity_gap(yrs)`.
- `conc(tp, ls)`: summary / loyo периодов (как W1) + срезы Δ без 24.02.2022, без 2022, без самого влиятельного дня (max |l_h − l_b|), без самого влиятельного календарного года, доля дисперсии разности от крупнейшего дня и от 24.02.2022 — ТОЛЬКО диагностика.
- `power_table(tp, ls)`: σ_LR (NW-94), эффект, n для мощности 0,5 / 0,8 — для М1 и М2 по A0 / A1 / A2, на D_M и без 2022.
- `guard(tbl, yrs, cnc)`: по кандидату (A2, A1): М1, М2, У1-S…У3-S для обеих пар, У4-S, У5-S; `decide(g)`: выводит решение «новая контрольная версия» из строк, порядок A2 → A1 (не по минимуму QLIKE)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w1_stat as SS, x1_w5_eval_t as T5
from models_v2.s18 import a_eval_stat as S
from models_v2.s18.a_eval_tab import BASES, CANDS, hkey

THR = {"У1-S": 0.415, "У2-S": 0.50, "У3-S": 1.00}
ORDER = ("S-A2", "S-A1")
H1, H2 = T5.H1, T5.H2
COMPS = [(hkey(c, b), b) for c in CANDS for b in BASES]
PAIRS = (("М1", "X1"), ("М2", "W5-G"))
SKIP = ("кандидат", "решение")


def years(tp, ls: dict) -> pd.DataFrame:
    mk, rows = T5.masks(tp), []

    def kind(p):
        return "half" if p in (H1, H2) else "period"
    for k in ls:
        rows += [dict(series=k, base="-", kind=kind(p), period=p, n=int(m.sum()), q_series=float(ls[k][m].mean()), q_base=np.nan, delta_pct=np.nan, contrib_pct=np.nan, delta_total_pct=np.nan) for p, m in mk.items()]
    for h, b in COMPS:
        c, tot = SS.contrib(ls[h], ls[b], mk), 100.0 * float(ls[h].mean() - ls[b].mean()) / float(ls[b].mean())
        rows += [dict(series=h, base=b, kind=kind(p), period=p, n=int(m.sum()), q_series=float(ls[h][m].mean()), q_base=float(ls[b][m].mean()), delta_pct=100.0 * float(ls[h][m].mean() / ls[b][m].mean() - 1.0),
                      contrib_pct=c[p], delta_total_pct=tot) for p, m in mk.items()]
    return pd.DataFrame(rows)


def identity_gap(yrs: pd.DataFrame) -> float:
    p = yrs[yrs["base"] != "-"]
    return float(max(abs(g["contrib_pct"].sum() - g["delta_total_pct"].iloc[0]) for _, g in p.groupby(["series", "base", "kind"])))


def _dl(d, lb, keep) -> float:
    return 100.0 * float(d[keep].sum()) / float(lb[keep].sum())


def conc(tp, ls: dict) -> pd.DataFrame:
    tp, pm, out = pd.DatetimeIndex(tp), SS.period_masks(tp), []
    y, n = tp.year.to_numpy(), len(tp)
    for h, b in COMPS:
        d, lb = ls[h] - ls[b], ls[b]
        c, tot = SS.contrib(ls[h], lb, pm), 100.0 * float(ls[h].mean() - lb.mean()) / float(lb.mean())
        out += SS.conc_rows(h, b, ls, pm, c, tot)
        cy = {int(k): float(d[y == k].sum()) for k in np.unique(y)}
        top_y, top_d, e = max(cy, key=lambda k: abs(cy[k])), int(np.argmax(np.abs(d))), (d - d.mean()) ** 2
        day22 = tp == pd.Timestamp("2022-02-24")
        keep_d = np.ones(n, bool)
        keep_d[top_d] = False
        out += [dict(series=h, base=b, item=i, key=k, value=v) for i, k, v in (
            ("no_day", "2022-02-24", _dl(d, lb, ~day22)), ("no_year", "2022", _dl(d, lb, y != 2022)), ("no_top_day", str(tp[top_d].date()), _dl(d, lb, keep_d)),
            ("no_top_year", str(top_y), _dl(d, lb, y != top_y)), ("var_share_top_day", str(tp[int(np.argmax(e))].date()), float(e.max() / e.sum())), ("var_share_day", "2022-02-24", float(e[day22].sum() / e.sum())))]
    return pd.DataFrame(out)


def power_table(tp, ls: dict) -> pd.DataFrame:
    y, rows = pd.DatetimeIndex(tp).year.to_numpy(), []
    for c in CANDS:
        for pid, b in PAIRS:
            h = hkey(c, b)
            for w, m in (("D_M", np.ones(len(y), bool)), ("без 2022", y != 2022)):
                lag = S.nw_lag(int(m.sum()))
                rows.append(dict(candidate=c, pair=pid, series=h, base=b, window=w, n=int(m.sum()), lag=lag, **S.power(ls[h][m] - ls[b][m], float(ls[b][m].mean()), lag)))
    return pd.DataFrame(rows)


def _g(rows, cand, pair, i, cond, v, thr, f, ok) -> None:
    rows.append(dict(candidate=cand, pair=pair, id=i, condition=cond, value=v, threshold=thr, formula=f, passed=bool(ok)))


def _guard_c(rows, c, q, yrs, cnc) -> None:
    for pid, b in PAIRS:
        h = hkey(c, b)
        w = yrs[(yrs["series"] == h) & (yrs["base"] == b)].set_index("period")
        d = float(w["delta_total_pct"].iloc[0])
        _g(rows, c, pid, pid, "Δ(%s к %s) на D_M < 0, %%" % (h, b), d, 0.0, "строго меньше нуля", d < 0)
        for i, p in (("У1-S", H1), ("У2-S", H2), ("У3-S", "2018")):
            v = float(w.at[p, "delta_pct"])
            _g(rows, c, pid, i, "Δ(%s к %s) в %s ≤ порога, %%" % (h, b, p), v, THR[i], "константа спецификации S_A_SPEC.md разд. 6.2", v <= THR[i])
    ctr = ("S-A2p", "S-A0") if c == "S-A2" else ("S-A0",)
    _g(rows, c, "—", "У4-S", "QLIKE(F_%s) < QLIKE(F_%s) на D_M" % (c, ", F_".join(ctr)), "; ".join("%s=%.6f" % (k, q[k]) for k in (c,) + ctr), "—", "строгое неравенство с контролями (б) и (в)", all(q[c] < q[k] for k in ctr))
    w = yrs[(yrs["series"] == c) & (yrs["base"] == "W5-G")].set_index("period")
    best = w[w["kind"] == "period"]["contrib_pct"].idxmin()
    loyo = float(cnc[(cnc["series"] == c) & (cnc["base"] == "W5-G") & (cnc["item"] == "loyo") & (cnc["key"] == best)]["value"].iloc[0])
    _g(rows, c, "М2", "У5-S", "Δ(F_%s к W5-G) без лучшего для F_A периода (%s) ≤ 0, %%" % (c, best), loyo, 0.0, "Δ без периода с наименьшим вкладом", loyo <= 0)


def _b(v) -> bool:
    return v if isinstance(v, (bool, np.bool_)) else str(v) == "True"


def guard(tbl, yrs, cnc) -> pd.DataFrame:
    q, rows = tbl.set_index("series")["qlike"], []
    for c in ORDER:
        _guard_c(rows, c, q, yrs, cnc)
        sub = [r for r in rows if r["candidate"] == c]
        _g(rows, c, "—", "кандидат", "все условия М1, М2, У1-S…У5-S выполнены", "%d из %d" % (sum(r["passed"] for r in sub), len(sub)), "—", "выведено из строк выше", all(r["passed"] for r in sub))
    g = pd.DataFrame(rows)
    return pd.concat([g, pd.DataFrame([dict(candidate="—", pair="—", id="решение", condition="правило 6.5, порядок A2 → A1", value=decide(g), threshold="—", formula="выведено кодом из строк выше",
                                           passed=decide(g).startswith("новая"))])], ignore_index=True)


def _fmt(v) -> str:
    try:
        return "%+.4f" % float(v)
    except (TypeError, ValueError):
        return str(v)


def decide(g: pd.DataFrame) -> str:
    tail = "X1 и W5-G остаются контрольными версиями; W8-G — экспериментальная"
    bad_all = []
    for c in ORDER:
        sub = g[(g["candidate"] == c) & ~g["id"].isin(SKIP)]
        if len(sub) != 10:
            raise ValueError("%s: строк условий %d ≠ 10" % (c, len(sub)))
        bad = sub[~sub["passed"].map(_b)]
        if bad.empty:
            return "новая контрольная версия: %s (решение разработки, не «подтверждено»); %s" % (c, tail)
        bad_all.append("%s: %s" % (c, ", ".join("%s%s (%s)" % (r.id, "" if r.pair in ("—", r.id) else "|" + r.pair, _fmt(r.value)) for r in bad.itertuples())))
    return "не прошёл — " + "; ".join(bad_all) + "; " + tail
