"""S-R-EVAL (Ж-254; `S_R_SPEC.md`, разд. 4.2–4.4, 4.7): годы и вклады, концентрация (У6-R), мощность, защита (М1, М2, У1-S…У3-S, У5-S, У6-R), процедура (П1, П2) и решение NEWCTRL-R — чистые функции по таблицам и массивам; ничего не пишут.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Все пороги — константы ЭТОГО файла (из спецификации), не из результата; порядок кандидатов S-RA2 → S-RCK; абляции и прежние ряды кандидатами не являются; выбор по минимуму QLIKE запрещён.
Публичное API:
- `THR` = {У1-S: 0,415; У2-S: 0,50; У3-S: 1,00} (%), `ORDER` = (S-RA2, S-RCK), `COMPS` — пары (ряд, база) таблиц YEARS и CONC; `years`, `conc` (прежние строки + trim / share_top1pct), `identity_gap`.
- `power_table(tp, ls)`: σ_LR (NW-94), эффект, n для мощности 0,5 / 0,8 — (М1, М2, П) кандидатов, на D_M и без 2022.
- `guard(tbl, yrs, cnc)`: по кандидату 10 условий: М1, М2, У1-S…У3-S для обеих пар, У5-S, У6-R; итог кандидата; решение `decide(g)` (NEWCTRL-R).
- `proc_table(tbl, ls, st)`: П1 / П2 — QLIKE(F_R) < QLIKE(F_прежний) ∧ Δ_trim(F_R против F_прежний) < 0; p Холма N и p SPA из STATS; «подтверждено» = П ∧ p Холма N < 0,05 ∧ p SPA < 0,05; R < Rb и R < Rc («нужны оба элемента»); `decide_proc(pt)`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import a_eval_yr as Y
from models_v2.s18 import r_eval_tab as T

THR, ORDER, SKIP = Y.THR, T.CANDS, ("кандидат", "решение")
PAIRS, H1, H2 = Y.PAIRS, Y.H1, Y.H2
COMPS = [(T.hkey(c, b), b) for c in T.CANDS for b in T.BASES] + [(c, T.OLD[c]) for c in T.CANDS] + [(c, a) for c in T.CANDS for a in T.ABL[c]]
identity_gap = Y.identity_gap


def years(tp, ls: dict) -> pd.DataFrame:
    return Y.years(tp, ls, COMPS)


def conc(tp, ls: dict) -> pd.DataFrame:
    base = Y.conc(tp, ls, COMPS)
    ex = []
    for h, b in COMPS:
        ex += [dict(series=h, base=b, item="trim", key="K=%d" % T.KTRIM, value=T.trim_delta(ls[h], ls[b])), dict(series=h, base=b, item="share_top1pct", key="K=%d" % T.KTRIM, value=T.share_top(ls[h], ls[b]))]
    return pd.concat([base, pd.DataFrame(ex)], ignore_index=True)


def power_table(tp, ls: dict) -> pd.DataFrame:
    y, rows = pd.DatetimeIndex(tp).year.to_numpy(), []
    for c in T.CANDS:
        for pid, h, b in (("М1", T.hkey(c, "X1"), "X1"), ("М2", c, "W5-G"), ("П", c, T.OLD[c])):
            for w, m in (("D_M", np.ones(len(y), bool)), ("без 2022", y != 2022)):
                lag = S.nw_lag(int(m.sum()))
                rows.append(dict(candidate=c, pair=pid, series=h, base=b, window=w, n=int(m.sum()), lag=lag, **S.power(ls[h][m] - ls[b][m], float(ls[b][m].mean()), lag)))
    return pd.DataFrame(rows)


def _trim(cnc, h, b) -> float:
    return float(cnc[(cnc["series"] == h) & (cnc["base"] == b) & (cnc["item"] == "trim")]["value"].iloc[0])


def _guard_c(rows, c, yrs, cnc) -> None:
    for pid, b in PAIRS:
        h = T.hkey(c, b)
        w = yrs[(yrs["series"] == h) & (yrs["base"] == b)].set_index("period")
        d = float(w["delta_total_pct"].iloc[0])
        Y._g(rows, c, pid, pid, "Δ(%s к %s) на D_M < 0, %%" % (h, b), d, 0.0, "строго меньше нуля", d < 0)
        for i, p in (("У1-S", H1), ("У2-S", H2), ("У3-S", "2018")):
            v = float(w.at[p, "delta_pct"])
            Y._g(rows, c, pid, i, "Δ(%s к %s) в %s ≤ порога, %%" % (h, b, p), v, THR[i], "константа спецификации S_R_SPEC.md разд. 4.2", v <= THR[i])
    w = yrs[(yrs["series"] == c) & (yrs["base"] == "W5-G")].set_index("period")
    best = w[w["kind"] == "period"]["contrib_pct"].idxmin()
    loyo = float(cnc[(cnc["series"] == c) & (cnc["base"] == "W5-G") & (cnc["item"] == "loyo") & (cnc["key"] == best)]["value"].iloc[0])
    Y._g(rows, c, "М2", "У5-S", "Δ(F_%s к W5-G) без лучшего для F_R периода (%s) ≤ 0, %%" % (c, best), loyo, 0.0, "Δ без периода с наименьшим вкладом", loyo <= 0)
    t2, t1 = _trim(cnc, c, "W5-G"), _trim(cnc, T.hkey(c, "X1"), "X1")
    Y._g(rows, c, "М1, М2", "У6-R", "Δ_trim (без %d лучших и %d худших дат) < 0 для F против W5-G и X1_R против X1, %%" % (T.KTRIM, T.KTRIM), "М2 %+.4f; М1 %+.4f" % (t2, t1), 0.0, "симметричная обрезка 1 %% с обеих сторон, K = %d" % T.KTRIM, t2 < 0 and t1 < 0)


def guard(tbl, yrs, cnc) -> pd.DataFrame:
    rows = []
    for c in ORDER:
        _guard_c(rows, c, yrs, cnc)
        sub = [r for r in rows if r["candidate"] == c]
        Y._g(rows, c, "—", "кандидат", "все условия М1, М2, У1-S…У3-S, У5-S, У6-R выполнены", "%d из %d" % (sum(r["passed"] for r in sub), len(sub)), "—", "выведено из строк выше", all(r["passed"] for r in sub))
    g = pd.DataFrame(rows)
    d = decide(g)
    return pd.concat([g, pd.DataFrame([dict(candidate="—", pair="—", id="решение", condition="NEWCTRL-R, порядок S-RA2 → S-RCK", value=d, threshold="—", formula="выведено кодом из строк выше", passed=d.startswith("новая"))])], ignore_index=True)


def decide(g: pd.DataFrame) -> str:
    tail = "X1 и W5-G остаются контрольными версиями; W8-G — экспериментальная"
    bad_all = []
    for c in ORDER:
        sub = g[(g["candidate"] == c) & ~g["id"].isin(SKIP)]
        if len(sub) != 10:
            raise ValueError("%s: строк условий %d ≠ 10" % (c, len(sub)))
        bad = sub[~sub["passed"].map(Y._b)]
        if bad.empty:
            return "новая контрольная версия: %s (решение разработки, не «подтверждено»); %s" % (c, tail)
        bad_all.append("%s: %s" % (c, ", ".join("%s%s (%s)" % (r.id, "" if r.pair in ("—", r.id) else "|" + r.pair, Y._fmt(r.value)) for r in bad.itertuples())))
    return "не прошёл — " + "; ".join(bad_all) + "; " + tail


def proc_table(tbl, ls: dict, st: pd.DataFrame) -> pd.DataFrame:
    q, rows = tbl.set_index("series")["qlike"], []
    for i, c in enumerate(T.CANDS):
        o, (rb, rc), s = T.OLD[c], T.ABL[c], st[(st["kind"] == "proc") & (st["hypothesis"] == "П%d" % (i + 1))].iloc[0]
        tr = T.trim_delta(ls[c], ls[o])
        p = bool(q[c] < q[o] and tr < 0)
        conf = bool(p and s["p_holm_N"] < S.ALPHA and s["p_spa_c"] < S.ALPHA)
        rows.append(dict(pid="П%d" % (i + 1), candidate=c, old=o, qlike_new=q[c], qlike_old=q[o], delta_pct=100.0 * (q[c] - q[o]) / q[o], delta_trim_pct=tr, cond_qlike=bool(q[c] < q[o]), cond_trim=bool(tr < 0), passed=p,
                         p_holm_N=float(s["p_holm_N"]), p_spa_c=float(s["p_spa_c"]), confirmed=conf, qlike_rb=q[rb], qlike_rc=q[rc], r_lt_rb=bool(q[c] < q[rb]), r_lt_rc=bool(q[c] < q[rc]), both_needed=bool(q[c] < q[rb] and q[c] < q[rc])))
    return pd.DataFrame(rows)


def decide_proc(pt: pd.DataFrame) -> str:
    out = []
    for r in pt.itertuples():
        out.append("%s: %s" % (r.pid, ("да" + (", подтверждено" if r.confirmed else ", не подтверждено статистически")) if r.passed else "нет"))
    return "; ".join(out)
