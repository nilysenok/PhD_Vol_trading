"""S-D-EVAL (Ж-256; `S_D_SPEC.md`, разд. 5): таблицы абляций S-RCKm, S-RCKf — чистые функции по массивам потерь; ничего не пишут.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, диагностика вклада элементов R, исследовательский результат, не OOS. Основная оценка — обычная QLIKE (среднее дневных потерь, без обрезки) на всех 2198 датах D_M;
Δ = 100·(QLIKE(ряд) − QLIKE(база))/QLIKE(база), минус — ряд лучше; интервал — стационарный бутстрап (B = 10 000, блок 20, seed 0, `a_eval_stat`); p DM NW-8 — справочно; Холма и SPA нет.
Константы — из спецификации (не из результата): `THR_PM` = −0,0313, `THR_PF` = 0,0157, `KTRIM` = 22.
Публичное API: `losses(rv, F)`, `table(rv, tp, F)`, `deltas(ls, tp, pairs)`, `verdict(delta, lo, hi) -> код`, `VERDICT` (код → текст), `elements(dl)` (вклад (iii) и (i) по строкам Δ к S-RCK), `summary(codes)`, `years(tp, ls, pairs)`, `slices(tp, ls, pairs)`, `trim_keep(d)`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_stat as S

KTRIM, THR_PM, THR_PF, SHARE_SPEC = 22, -0.0313, 0.0157, 0.89
SERIES = ("S-RCK", "S-RCKm", "S-RCKf", "S-RCKb", "W5-G", "X1", "K2", "K3")
ELEMS = (("(iii) поправка уровня m", "S-RCKm", "S-RCK"), ("(i) потеря на уровне F вместо C", "S-RCKf", "S-RCK"), ("(ii) обрезка без m (S-RCKm против S-RCKb)", "S-RCKm", "S-RCKb"))
VERDICT = {"needed": "элемент необходим для выигрыша S-RCK; без него хуже на %+.4f %%", "worse_noise": "без элемента хуже на %+.4f %%; вклад от шума не отделён",
           "not_found": "вклад элемента не обнаружен (Δ = %+.4f %%)", "harms": "элемент вредит: без него лучше на %+.4f %%. Ряд кандидатом не становится, только новой регистрацией"}
SUMMARY = {"both_needed": "выигрыш S-RCK = обрезка с поправкой m + связь уровня C со слоем доверия через потерю уровня F",
           "i_not_found": "выигрыш достигается на уровне C; связь с π не нужна", "iii_harms": "m лишняя; обрезка без m лучше (без смены кандидата)"}


def ql(rv, F) -> np.ndarray:
    return S.Q.ql(np.asarray(rv, float), np.asarray(F, float))


def losses(rv, F: dict) -> dict:
    return {k: ql(rv, v) for k, v in F.items()}


def table(rv, tp, F: dict) -> pd.DataFrame:
    tp, rows = pd.DatetimeIndex(tp), []
    q = {k: float(ql(rv, F[k]).mean()) for k in F}
    for k in SERIES:
        rows.append(dict(series=k, n=len(tp), first_tplus=str(tp.min().date()), last_tplus=str(tp.max().date()), qlike=q[k], **{"d_vs_%s_pct" % b: 100.0 * (q[k] - q[b]) / q[b] for b in ("S-RCK", "W5-G", "X1", "K2", "K3")}))
    return pd.DataFrame(rows)


def deltas(ls: dict, tp, pairs) -> pd.DataFrame:
    idx, rows, n = S.idx_for(len(tp)), [], len(tp)
    for h, b in pairs:
        ci, dm = S.boot_ci(ls[h], ls[b], idx), S.dm_nw(ls[b], ls[h], S.nw_lag(n))
        rows.append(dict(series=h, base=b, n=n, first_tplus=str(pd.DatetimeIndex(tp).min().date()), last_tplus=str(pd.DatetimeIndex(tp).max().date()), q_series=float(ls[h].mean()), q_base=float(ls[b].mean()),
                         delta_pct=ci["delta_pct"], lo90=ci["lo90"], hi90=ci["hi90"], p_boot=ci["p_boot"], p_dm_nw8=dm["p_one"], boot_B=S.B, boot_block=S.MEAN_BLOCK, boot_seed=S.SEED, nw_lag=dm["lag"]))
    return pd.DataFrame(rows)


def verdict(delta: float, lo: float, hi: float) -> str:
    """Разд. 5: x — ряд без элемента, Δ_x к базе (S-RCK / S-RCKb); плюс — без элемента хуже."""
    if delta > 0:
        return "needed" if lo > 0 else "worse_noise"
    return "harms" if (delta < 0 and hi < 0) else "not_found"


def elements(dl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, x, b in ELEMS:
        r = dl[(dl["series"] == x) & (dl["base"] == b)].iloc[0]
        code = verdict(float(r["delta_pct"]), float(r["lo90"]), float(r["hi90"]))
        rows.append(dict(element=name, series=x, base=b, delta_pct=r["delta_pct"], lo90=r["lo90"], hi90=r["hi90"], p_dm_nw8=r["p_dm_nw8"], code=code, verdict=VERDICT[code] % r["delta_pct"], main="да" if b == "S-RCK" else "нет (без вердикта разд. 5)"))
    return pd.DataFrame(rows)


def summary(codes: dict) -> str:
    """codes: {"iii": код, "i": код}. Применимые правила разд. 5, в порядке перечня; иначе — прямое указание на сочетание."""
    out = ([SUMMARY["both_needed"]] if codes["i"] == codes["iii"] == "needed" else []) + ([SUMMARY["i_not_found"]] if codes["i"] == "not_found" else []) + ([SUMMARY["iii_harms"]] if codes["iii"] == "harms" else [])
    return "; ".join(out) if out else "сочетание не предусмотрено спецификацией: (i) = %s, (iii) = %s" % (codes["i"], codes["iii"])


def years(tp, ls: dict, pairs) -> pd.DataFrame:
    y, rows = pd.DatetimeIndex(tp).year.to_numpy(), []
    for h, b in pairs:
        d, tot = ls[h] - ls[b], float(ls[b].sum())
        for yr in sorted(set(y)):
            m = y == yr
            rows.append(dict(series=h, base=b, period=str(yr), n=int(m.sum()), q_series=float(ls[h][m].mean()), q_base=float(ls[b][m].mean()), contrib_pct=100.0 * float(d[m].sum()) / tot, delta_total_pct=100.0 * float(d.sum()) / tot))
    return pd.DataFrame(rows)


def trim_keep(d, k: int = KTRIM) -> np.ndarray:
    o, keep = np.argsort(np.asarray(d, float), kind="stable"), np.ones(len(d), bool)
    keep[o[:k]], keep[o[len(d) - k:]] = False, False
    return keep


SLICES = ("все даты (основной)", "без t⁺ = 2022-02-24", "без 2022 года", "симметричная обрезка %d + %d дат (даты выбраны по разности потерь пары)" % (KTRIM, KTRIM))


def slices(tp, ls: dict, pairs) -> pd.DataFrame:
    tp, rows = pd.DatetimeIndex(tp), []
    for h, b in pairs:
        d = ls[h] - ls[b]
        for i, (nm, m) in enumerate(zip(SLICES, (np.ones(len(tp), bool), tp != pd.Timestamp("2022-02-24"), tp.year.to_numpy() != 2022, trim_keep(d)))):
            rows.append(dict(series=h, base=b, slice=nm, kind="основной" if i == 0 else "диагностика", n=int(m.sum()), n_removed=int((~m).sum()), control_and_denominator="контроль %s; знаменатель — QLIKE %s на оставшихся %d датах" % (b, b, int(m.sum())),
                             q_base=float(ls[b][m].mean()), q_series=float(ls[h][m].mean()), delta_pct=100.0 * float(d[m].sum()) / float(ls[b][m].sum())))
    return pd.DataFrame(rows)


def trim_overlap(tp, ls: dict, p1, p2) -> dict:
    """Общие исключённые даты обрезки двух пар: даты выбраны по d_t своей пары, наборы разные."""
    ex = [set(np.flatnonzero(~trim_keep(ls[h] - ls[b]))) for h, b in (p1, p2)]
    return dict(n_each=len(ex[0]), n_common=len(ex[0] & ex[1]), jaccard=len(ex[0] & ex[1]) / len(ex[0] | ex[1]))
