"""S-Z-EVAL (Ж-258; `S_Z_SPEC.md`, разд. 7.1–7.3, 7.7): годы и вклады, концентрация, защитные условия (два набора) и решение Z-DEC — чистые функции по таблицам и массивам; ничего не пишут.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Все пороги — константы `z_eval_cfg` (из спецификации), не из результата; кандидат один (S-RCKZ); выбор по минимуму QLIKE запрещён; замену кандидата решает только владелец (класс A).
Публичное API:
- `COMPS` — пары (ряд, база) таблиц YEARS и CONC; `years`, `conc` (прежние строки + trim / share_top1pct при K = 22), `identity_gap`.
- `guard(yrs, cnc) -> DataFrame`: по набору («к W5-G», «к S-RCK») пять условий У1-S, У2-S, У3-S, У5-S, У6-R, итог набора; У6-R — Δ_trim < 0 у F и у слоя X1_R (K = 22 с каждой стороны, база процентов — оставшиеся даты).
- `hz(tbl) -> dict`: H-Z1 (QLIKE(S-RCKZ) < QLIKE(S-RCK)) и H-Z2 (< QLIKE(S-RCKZs)), строго, на всех датах D_M.
- `dec_rows(tbl, g, st) -> DataFrame`, `decide(dec) -> (код 1/2/3, текст)`: выведено из строк; «превзошёл» (код 1) — только при H-Z1 ∧ H-Z2 ∧ оба набора ∧ подтверждении H-Z1 (p Холма N < 0,05 и p SPA (а) < 0,05)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import a_eval_yr as AY
from models_v2.s18 import r_eval_tab as RT
from models_v2.s18 import z_eval_cfg as C
from models_v2.s18.z_eval_tab import Z, ZL

SETS = (("к W5-G", "W5-G", "X1"), ("к S-RCK", "S-RCK", "X1_R(S-RCK)"))
COMPS = [(Z, "S-RCK"), (Z, "W5-G"), (ZL, "X1"), (ZL, "X1_R(S-RCK)"), (Z, "S-RCKZs"), ("S-RCKZs", "S-RCK"), ("S-RZ", "W5-G"), ("S-RZ", "S-RCK")]
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
    AY._g(rows, "S-RCKZ", st, i, cond, v, thr, f, ok)


def _set(rows, name, b, lay, yrs, cnc) -> None:
    w = yrs[(yrs["series"] == Z) & (yrs["base"] == b)].set_index("period")
    for i, p in (("У1-S", AY.H1), ("У2-S", AY.H2), ("У3-S", "2018")):
        v = float(w.at[p, "delta_pct"])
        _row(rows, name, i, "Δ(F_Z к %s) в %s ≤ порога, %%" % (b, p), v, C.THR[i], "константа S_Z_SPEC.md разд. 7.3", v <= C.THR[i])
    best = w[w["kind"] == "period"]["contrib_pct"].idxmin()
    loyo = _cv(cnc, Z, b, "loyo", best)
    _row(rows, name, "У5-S", "Δ(F_Z к %s) без лучшего для кандидата периода (%s) ≤ 0, %%" % (b, best), loyo, 0.0, "Δ без периода с наименьшим вкладом", loyo <= 0)
    t1, t2 = _cv(cnc, Z, b, "trim"), _cv(cnc, ZL, lay, "trim")
    _row(rows, name, "У6-R", "Δ_trim (K = %d с каждой стороны) < 0 у F к %s и у слоя X1_R к %s, %%" % (C.KTRIM, b, lay), "F %+.4f; слой %+.4f" % (t1, t2), 0.0, "симметричная обрезка, база — оставшиеся даты", t1 < 0 and t2 < 0)


def guard(yrs: pd.DataFrame, cnc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, b, lay in SETS:
        n0 = len(rows)
        _set(rows, name, b, lay, yrs, cnc)
        sub = rows[n0:]
        _row(rows, name, "набор", "все условия У1-S…У3-S, У5-S, У6-R набора %s выполнены" % name, "%d из %d" % (sum(r["passed"] for r in sub), len(sub)), "—", "выведено из строк выше", all(r["passed"] for r in sub))
    return pd.DataFrame(rows).rename(columns={"pair": "set", "candidate": "series"})


def hz(tbl: pd.DataFrame) -> dict:
    q = tbl.set_index("series")["qlike"]
    return {"H-Z1": bool(q[Z] < q["S-RCK"]), "H-Z2": bool(q[Z] < q["S-RCKZs"])}


def _fmt(v) -> str:
    return AY._fmt(v)


def dec_rows(tbl: pd.DataFrame, g: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    q, h = tbl.set_index("series")["qlike"], hz(tbl)
    r = st[(st["kind"] == "main") & (st["hypothesis"] == "%s|S-RCK" % Z)].iloc[0]
    conf = bool(r["delta_pct"] < 0 and r["p_holm_N"] < S.ALPHA and r["p_spa_c"] < S.ALPHA)
    rows = [dict(id="H-Z1", condition="QLIKE(S-RCKZ) < QLIKE(S-RCK) на D_M, все даты", value=100.0 * (q[Z] - q["S-RCK"]) / q["S-RCK"], threshold=0.0, passed=h["H-Z1"]),
            dict(id="H-Z2", condition="QLIKE(S-RCKZ) < QLIKE(S-RCKZs) на D_M, все даты", value=100.0 * (q[Z] - q["S-RCKZs"]) / q["S-RCKZs"], threshold=0.0, passed=h["H-Z2"])]
    for name, *_ in SETS:
        s = g[(g["set"] == name) & (g["id"] == "набор")].iloc[0]
        rows.append(dict(id="У " + name, condition="все условия разд. 7.3 " + name, value=s["value"], threshold="—", passed=bool(s["passed"])))
    rows.append(dict(id="подтверждение H-Z1", condition="H-Z1: Δ < 0, p Холма (N = %d) < 0,05 и p SPA (а) < 0,05" % int(r["n_holm"]), value="Δ %+.4f; p Холма %.4g; p SPA %.4g" % (r["delta_pct"], r["p_holm_N"], r["p_spa_c"]), threshold=S.ALPHA, passed=conf))
    code, text = decide(pd.DataFrame(rows), g)
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([dict(id="Z-DEC", condition="решение S-Z (разд. 7.7), выведено кодом", value=code, threshold="—", passed=code == "1", text=text)])], ignore_index=True)


def decide(dec: pd.DataFrame, g: pd.DataFrame) -> tuple:
    p = dec.set_index("id")["passed"].map(bool)
    ok = bool(p["H-Z1"] and p["H-Z2"] and p["У к W5-G"] and p["У к S-RCK"])
    own = "Замена кандидата — только решение владельца (класс A)."
    if ok and p["подтверждение H-Z1"]:
        return "1", "S-RCKZ превзошёл S-RCK (H-Z1, H-Z2, все условия разд. 7.3, подтверждено). Предложение заменить кандидата идёт владельцу. " + own
    if ok:
        return "2", "S-RCKZ численно лучше S-RCK; превосходство не установлено. S-RCK остаётся кандидатом, S-RCKZ — сравнительная версия, как S-RL1. " + own
    d = dec.set_index("id")
    bad = [("%s (%s)" % (i, _fmt(d.loc[i, "value"]))) for i in ("H-Z1", "H-Z2") if not p[i]]
    bad += ["%s|%s (%s)" % (r.id, r.set, _fmt(r.value)) for r in g[~g["passed"].map(bool) & (g["id"] != "набор")].itertuples()]
    return "3", "S-RCKZ не прошёл: " + ", ".join(bad) + ". S-RCK остаётся кандидатом. " + own
