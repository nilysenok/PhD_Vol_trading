"""S-RCD-EVAL (Ж-259; `S_RCD_SPEC.md`, разд. 6.1, 7.1–7.3, 7.6): годы и вклады, концентрация, защитные условия (два набора), H-RCD1 и решение RCD-DEC — чистые функции по таблицам и массивам; ничего не пишут.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Все пороги — константы `rd_eval_cfg` (из спецификации), не из результата; выбор по минимуму QLIKE и победы по годам не используются; S-RCK не заменяется без решения владельца (класс A).
Публичное API:
- `COMPS` — пары (ряд, база) таблиц YEARS и CONC; `years`, `conc` (прежние строки + trim / share_top1pct при K = 22 + n_slice), `identity_gap`.
- `guard(yrs, cnc) -> DataFrame`: по набору («к W5-G», «к S-RCK») пять условий У1-S, У2-S, У3-S, У5-S, У6-R, итог набора; У6-R — Δ_trim < 0 у F и у слоя X1_R.
- `e_of(tbl)`, `share_s(tbl)`: e(X) = 100·(Q(X) − Q(W5-G))/Q(W5-G); s = e(S-RCD)/e(S-RCK) при e(S-RCK) < 0.
- `dec_rows(tbl, g, st) -> DataFrame`, `decide(dec, g) -> (код 1…4, текст)`: выведено из строк; «превзошёл» (код 1) — только при H-RCD1 ∧ Q(S-RCD) < Q(S-RCK) ∧ оба набора ∧ подтверждение превосходства (p Холма N < 0,05 и p SPA (б) < 0,05)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import a_eval_yr as AY
from models_v2.s18 import r_eval_tab as RT
from models_v2.s18 import rd_eval_cfg as C
from models_v2.s18.rd_eval_tab import CDF, DEC4, FAM, R, REF, RL

SETS = (("к W5-G", "W5-G", "X1"), ("к S-RCK", "S-RCK", "X1_R(S-RCK)"))
COMPS = list(FAM[:4]) + [(RL, "X1_R(S-RCK)"), ("S-RCK", R)] + list(DEC4) + list(REF[:2]) + [(RL, "X1_V(S-CDF)")]
identity_gap = AY.identity_gap
assert C.KTRIM == RT.KTRIM


def years(tp, ls: dict) -> pd.DataFrame:
    return AY.years(tp, ls, COMPS)


def conc(tp, ls: dict) -> pd.DataFrame:
    base = AY.conc(tp, ls, COMPS)
    ex = [dict(series=h, base=b, item=i, key="K=%d" % C.KTRIM, value=f(ls[h], ls[b], C.KTRIM)) for h, b in COMPS for i, f in (("trim", RT.trim_delta), ("share_top1pct", RT.share_top))]
    return pd.concat([base, pd.DataFrame(ex + _n_slices(tp, base))], ignore_index=True)


def _n_slices(tp, base: pd.DataFrame) -> list:
    """Число дат каждого среза (для отчёта): no_day, no_year, no_top_day, no_top_year, trim."""
    tp, n = pd.DatetimeIndex(tp), len(tp)
    out = []
    for h, b in COMPS:
        top_y = int(base[(base["series"] == h) & (base["base"] == b) & (base["item"] == "no_top_year")]["key"].iloc[0])
        cnt = dict(no_day=int((tp != pd.Timestamp("2022-02-24")).sum()), no_year=int((tp.year != 2022).sum()), no_top_day=n - 1, no_top_year=int((tp.year != top_y).sum()), trim=n - 2 * C.KTRIM)
        out += [dict(series=h, base=b, item="n_slice", key=k, value=v) for k, v in cnt.items()]
    return out


def _cv(cnc, h, b, item, key=None) -> float:
    m = (cnc["series"] == h) & (cnc["base"] == b) & (cnc["item"] == item)
    if key is not None:
        m &= cnc["key"] == key
    return float(cnc[m]["value"].iloc[0])


def _row(rows, st, i, cond, v, thr, f, ok) -> None:
    AY._g(rows, R, st, i, cond, v, thr, f, ok)


def _set(rows, name, b, lay, yrs, cnc) -> None:
    w = yrs[(yrs["series"] == R) & (yrs["base"] == b)].set_index("period")
    for i, p in (("У1-S", AY.H1), ("У2-S", AY.H2), ("У3-S", "2018")):
        v = float(w.at[p, "delta_pct"])
        _row(rows, name, i, "Δ(F_RCD к %s) в %s ≤ порога, %%" % (b, p), v, C.THR[i], "константа S_RCD_SPEC.md разд. 7.3", v <= C.THR[i])
    best = w[w["kind"] == "period"]["contrib_pct"].idxmin()
    loyo = _cv(cnc, R, b, "loyo", best)
    _row(rows, name, "У5-S", "Δ(F_RCD к %s) без лучшего для кандидата периода (%s) ≤ 0, %%" % (b, best), loyo, 0.0, "Δ без периода с наименьшим вкладом", loyo <= 0)
    t1, t2 = _cv(cnc, R, b, "trim"), _cv(cnc, RL, lay, "trim")
    _row(rows, name, "У6-R", "Δ_trim (K = %d с каждой стороны) < 0 у F к %s и у слоя X1_R к %s, %%" % (C.KTRIM, b, lay), "F %+.4f; слой %+.4f" % (t1, t2), 0.0, "симметричная обрезка, база — оставшиеся даты", t1 < 0 and t2 < 0)


def guard(yrs: pd.DataFrame, cnc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, b, lay in SETS:
        n0 = len(rows)
        _set(rows, name, b, lay, yrs, cnc)
        sub = rows[n0:]
        _row(rows, name, "набор", "все условия У1-S…У3-S, У5-S, У6-R набора %s выполнены" % name, "%d из %d" % (sum(r["passed"] for r in sub), len(sub)), "—", "выведено из строк выше", all(r["passed"] for r in sub))
    return pd.DataFrame(rows).rename(columns={"pair": "set", "candidate": "series"})


def e_of(tbl: pd.DataFrame) -> dict:
    q = tbl.set_index("series")["qlike"]
    return {k: 100.0 * (q[k] - q["W5-G"]) / q["W5-G"] for k in q.index}


def share_s(tbl: pd.DataFrame) -> float:
    e = e_of(tbl)
    return float(e[R] / e["S-RCK"]) if e["S-RCK"] < 0 else float("nan")


def _fmt(v) -> str:
    return AY._fmt(v)


def _stat(st: pd.DataFrame, b: str) -> pd.Series:
    return st[(st["kind"] == "main") & (st["hypothesis"] == "%s|%s" % (R, b))].iloc[0]


def dec_rows(tbl: pd.DataFrame, g: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    q = tbl.set_index("series")["qlike"]
    h1, hk = _stat(st, CDF), _stat(st, "S-RCK")
    c1 = bool(h1["delta_pct"] < 0 and h1["p_holm_N"] < S.ALPHA and h1["p_spa_c"] < S.ALPHA)
    ck = bool(hk["delta_pct"] < 0 and hk["p_holm_N"] < S.ALPHA and hk["p_spa_c"] < S.ALPHA)
    rows = [dict(id="H-RCD1", condition="QLIKE(S-RCD) < QLIKE(S-CDF) на D_M, все даты; Δ в % QLIKE(S-CDF)", value=100.0 * (q[R] - q[CDF]) / q[CDF], threshold=0.0, passed=bool(q[R] < q[CDF])),
            dict(id="S-RCD против S-RCK", condition="QLIKE(S-RCD) < QLIKE(S-RCK) на D_M; Δ в % QLIKE(S-RCK)", value=100.0 * (q[R] - q["S-RCK"]) / q["S-RCK"], threshold=0.0, passed=bool(q[R] < q["S-RCK"]))]
    for name, *_ in SETS:
        s = g[(g["set"] == name) & (g["id"] == "набор")].iloc[0]
        rows.append(dict(id="У " + name, condition="все условия разд. 7.3 " + name, value=s["value"], threshold="—", passed=bool(s["passed"])))
    rows += [dict(id="подтверждение H-RCD1", condition="Δ < 0, p Холма (N = %d) < 0,05 и p SPA (а) < 0,05" % int(h1["n_holm"]), value="Δ %+.4f; p Холма %.4g; p SPA %.4g" % (h1["delta_pct"], h1["p_holm_N"], h1["p_spa_c"]), threshold=S.ALPHA, passed=c1),
             dict(id="подтверждение над S-RCK", condition="Δ < 0, p Холма (N = %d) < 0,05 и p SPA (б) < 0,05" % int(hk["n_holm"]), value="Δ %+.4f; p Холма %.4g; p SPA %.4g" % (hk["delta_pct"], hk["p_holm_N"], hk["p_spa_c"]), threshold=S.ALPHA, passed=ck),
             dict(id="доля s", condition="s = e(S-RCD)/e(S-RCK) при e(S-RCK) < 0; показ, не условие", value=share_s(tbl), threshold=C.SHARE_HALF, passed=bool(share_s(tbl) >= C.SHARE_HALF))]
    code, text = decide(pd.DataFrame(rows), g)
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([dict(id="RCD-DEC", condition="решение S-RCD (разд. 7.6), выведено кодом", value=code, threshold="—", passed=code == "1", text=text)])], ignore_index=True)


def decide(dec: pd.DataFrame, g: pd.DataFrame) -> tuple:
    p, d = dec.set_index("id")["passed"].map(bool), dec.set_index("id")
    own = "S-RCK не заменяется без решения владельца (класс A)."
    ha = bool(p["H-RCD1"] and p["У к W5-G"])
    if ha and p["S-RCD против S-RCK"] and p["У к S-RCK"] and p["подтверждение над S-RCK"]:
        return "1", "S-RCD превзошёл S-RCK (H-RCD1, оба набора, подтверждено над S-RCK). Предложение заменить кандидата идёт владельцу (класс A). До его решения S-RCK остаётся кандидатом. " + own
    if ha:
        conf = "H-RCD1 не подтверждено: p Холма N ≥ 0,05" if not p.get("подтверждение H-RCD1", False) else "H-RCD1 подтверждено"
        return "2", "R численно улучшает прямую коррекцию (%s); S-RCK остаётся кандидатом-контролем; S-RCD — сравнительная версия, как S-RL1 и S-RCKZ. %s" % (conf, own)
    if p["H-RCD1"]:
        bad = ["%s|%s (%s)" % (r.id, r.set, _fmt(r.value)) for r in g[(g["set"] == "к W5-G") & ~g["passed"].map(bool) & (g["id"] != "набор")].itertuples()]
        return "3", "R улучшает прямую коррекцию относительно S-CDF, но S-RCD не проходит защиту к W5-G (%s). %s" % (", ".join(bad), own)
    return "4", "R не улучшил прямую коррекцию на D_M (Δ к S-CDF = %s %%). %s" % (_fmt(d.loc["H-RCD1", "value"]), own)
