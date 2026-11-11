"""S-C-EVAL (Ж-254; `S_C_SPEC.md`, разд. 4.2–4.6): годы и вклады, концентрация, мощность, защита (М1, М2, У1-S…У5-S, У4-C), STRUCT и решение NEWCTRL — чистые функции по таблицам и массивам; ничего не пишут.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Все пороги — константы ЭТОГО файла (из спецификации), не из результата; порядок кандидатов S-CK → S-CKA, S-CD кандидатом не является; выбор по минимуму QLIKE запрещён.
Публичное API:
- `THR` = {У1-S: 0,415; У2-S: 0,50; У3-S: 1,00} (%), `ORDER` = (S-CK, S-CKA), `CTRL` = (S-CKL, S-C7) — контроли У4-C; `COMPS` — пары (ряд, база) таблиц YEARS и CONC.
- `years(tp, ls)`, `conc(tp, ls)`: `a_eval_yr.years` / `conc` с парами C; Σ вкладов = Δ (1e-12): `identity_gap`.
- `power_table(tp, ls)`: σ_LR (NW-94), эффект, n для мощности 0,5 / 0,8 — М1, М2 (S-CK, S-CKA) и К1 (S-CK против S-CD), на D_M и без 2022.
- `guard(tbl, yrs, cnc, stats)`: по кандидату: М1, М2, У1-S…У3-S для обеих пар, У4-C, У5-S, итог кандидата; структура: К1–К3, STRUCT, «STRUCT подтверждён» (по p Холма N программы К1 и p SPA из `stats`), S-CD справочно; решения `decide(g)` (NEWCTRL) и `decide_struct(g)`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import a_eval_yr as Y
from models_v2.s18.cm_eval_tab import BASES, CANDS, STRUCT, hkey, verdict

THR = Y.THR
ORDER, CTRL, SKIP = CANDS, ("S-CKL", "S-C7"), ("кандидат", "решение")
COMPS = [(hkey(c, b), b) for c in CANDS for b in BASES] + [(c, b) for _, c, b in STRUCT] + [("S-CKA", "S-CK"), ("S-CD", "W5-G"), ("X1_C(S-CD)", "X1")]
PAIRS = Y.PAIRS
H1, H2 = Y.H1, Y.H2
SD = "структура"


def years(tp, ls: dict) -> pd.DataFrame:
    return Y.years(tp, ls, COMPS)


def conc(tp, ls: dict) -> pd.DataFrame:
    return Y.conc(tp, ls, COMPS)


identity_gap = Y.identity_gap


def power_table(tp, ls: dict) -> pd.DataFrame:
    y, rows = pd.DatetimeIndex(tp).year.to_numpy(), []
    sets = [(c, pid, hkey(c, b), b) for c in CANDS for pid, b in PAIRS] + [("S-CK", "К1", "S-CK", "S-CD")]
    for c, pid, h, b in sets:
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
            _g(rows, c, pid, i, "Δ(%s к %s) в %s ≤ порога, %%" % (h, b, p), v, THR[i], "константа спецификации S_C_SPEC.md разд. 4.3", v <= THR[i])
    _g(rows, c, "—", "У4-C", "QLIKE(F_%s) < QLIKE(F_%s) на D_M" % (c, ") и < QLIKE(F_".join(CTRL)), "; ".join("%s=%.6f" % (k, q[k]) for k in (c,) + CTRL), "—", "строгое неравенство с контролями (б) и K7-классом", all(q[c] < q[k] for k in CTRL))
    w = yrs[(yrs["series"] == c) & (yrs["base"] == "W5-G")].set_index("period")
    best = w[w["kind"] == "period"]["contrib_pct"].idxmin()
    loyo = float(cnc[(cnc["series"] == c) & (cnc["base"] == "W5-G") & (cnc["item"] == "loyo") & (cnc["key"] == best)]["value"].iloc[0])
    _g(rows, c, "М2", "У5-S", "Δ(F_%s к W5-G) без лучшего для F_V периода (%s) ≤ 0, %%" % (c, best), loyo, 0.0, "Δ без периода с наименьшим вкладом", loyo <= 0)


def _struct_rows(rows, q, st) -> None:
    ok = []
    for i, c, b in STRUCT:
        _g(rows, SD, i, i, "QLIKE(F_%s) < QLIKE(F_%s) на D_M" % (c, b), "%s=%.6f; %s=%.6f" % (c, q[c], b, q[b]), "—", "строгое неравенство на n = 2198", q[c] < q[b])
        ok.append(q[c] < q[b])
    _g(rows, SD, "—", "STRUCT", "К1 ∧ К2 ∧ К3 (численный эффект компонентной структуры)", "%d из 3" % sum(ok), "—", "выведено из строк выше", all(ok))
    r = st[(st["kind"] == "struct") & (st["hypothesis"] == "К1")].iloc[0]
    conf = verdict(r["delta_pct"], r["p_holm_N"], r["p_spa_c"]).startswith("численно лучше базы; подтверждено")
    _g(rows, SD, "—", "STRUCT подтверждён", "STRUCT ∧ p Холма N (К1) < 0,05 ∧ p SPA (К1) < 0,05", "p Холма N=%.4f; p SPA=%.4f" % (r["p_holm_N"], r["p_spa_c"]), "0,05", "по строке К1 таблицы STATS", all(ok) and conf)
    cd, w5 = q["S-CD"], q["W5-G"]
    note = ("прямая коррекция на рыночных признаках, справочно, не кандидат; структура отличия не доказала" if cd < w5 and not all(ok)
            else "условие п. 4.6 не применимо: S-CD численно не лучше W5-G" if not cd < w5 else "условие п. 4.6 не применимо: S-CD лучше W5-G, но STRUCT выполнен")
    _g(rows, SD, "—", "S-CD справочно", "S-CD численно лучше W5-G, а STRUCT не выполнен", note, "—", "п. 4.6; продвижение S-CD — только новой регистрацией", cd < w5 and not all(ok))


def guard(tbl, yrs, cnc, st) -> pd.DataFrame:
    q, rows = tbl.set_index("series")["qlike"], []
    for c in ORDER:
        _guard_c(rows, c, q, yrs, cnc)
        sub = [r for r in rows if r["candidate"] == c]
        _g(rows, c, "—", "кандидат", "все условия М1, М2, У1-S…У5-S, У4-C выполнены", "%d из %d" % (sum(r["passed"] for r in sub), len(sub)), "—", "выведено из строк выше", all(r["passed"] for r in sub))
    _struct_rows(rows, q, st)
    g = pd.DataFrame(rows)
    d, ds = decide(g), decide_struct(g)
    return pd.concat([g, pd.DataFrame([dict(candidate="—", pair="—", id="решение", condition="NEWCTRL, порядок S-CK → S-CKA", value=d, threshold="—", formula="выведено кодом из строк выше", passed=d.startswith("новая")),
                                       dict(candidate="—", pair="—", id="решение STRUCT", condition="STRUCT и подтверждение", value=ds, threshold="—", formula="выведено кодом из строк выше", passed=ds.startswith("эффект структуры: да"))])], ignore_index=True)


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
        bad = sub[~sub["passed"].map(Y._b)]
        if bad.empty:
            return "новая контрольная версия: %s (решение разработки, не «подтверждено»); %s" % (c, tail)
        bad_all.append("%s: %s" % (c, ", ".join("%s%s (%s)" % (r.id, "" if r.pair in ("—", r.id) else "|" + r.pair, _fmt(r.value)) for r in bad.itertuples())))
    return "не прошёл — " + "; ".join(bad_all) + "; " + tail


def decide_struct(g: pd.DataFrame) -> str:
    s = g[g["candidate"] == SD].set_index("id")["passed"].map(Y._b)
    k = [i for i in ("К1", "К2", "К3") if not s[i]]
    if k:
        return "эффект структуры: нет (не выполнено: %s); подтверждения нет" % ", ".join(k)
    return "эффект структуры: да (К1 ∧ К2 ∧ К3); " + ("подтверждён" if s["STRUCT подтверждён"] else "не подтверждён статистически (p Холма N (К1) ≥ 0,05 или p SPA ≥ 0,05)")
