"""FIN: сводные таблицы `FIN_*.csv` пилота Deep Extension — чистые функции от массивов (файлов нет). Оценка — функции V7 (импорт через `de_e1_eval`), Холм N = 738 / N_wide = 3181 (как листы E1–E4).
Ряды на ОДИНАКОВЫХ датах 2018-01-03 … 2026-02-26: X1–X4, ENS (ансамбль без затвора, справочный), контроли K1, K2, K3, M, K8, NTP2, A2, A0, A0g, cfg4, cfg8, cfg10 (только чтение).
Разделы Ж-239 п. 10 по годам 2022…2026 и окну 2022–2025: `stab_table` (поправка к HAR-J = ln(F/K1)); QLIKE по состояниям «экстрем / норма» — `state_table`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s15 import de_e1_eval as E1V

EV, ST, E2, Q = E1V.EV, E1V.ST, E1V.E2, E1V.Q
X = ("X1", "X2", "X3", "X4")
CTRL = ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "V6:A0", "V6:A0g", "cfg4", "cfg8", "cfg10")
ALL = X + ("ENS",) + CTRL
BOOST = ("K2", "K3")
PERIODS = ("2022", "2023", "2024", "2025", "2026", "2022_2025")
SPECIAL = {("X1", "V4:NTP2"): "X1 против NTP2 (6.2, польза состояния)", ("X2", "V6:A0g"): "X2 против A0g (6.2)", ("X3", "V6:A0g"): "X3 против A0g (6.2)", ("X4", "ENS"): "X4 против ансамбля без затвора (6.2, польза затвора)",
           ("X4", "V6:A0g"): "X4 против A0g (6.2)", ("X2", "cfg4"): "X2 против cfg4"}
BASES = ("V4:NTP2", "V6:A2", "V6:A0g", "cfg8", "K2", "K3")
PAIRS = tuple((SPECIAL.get((x, y), "%s против %s" % (x, y)), x, y) for x in X for y in BASES) + tuple((SPECIAL.get(k, "%s против %s" % k), *k) for k in (("X2", "cfg4"), ("X4", "ENS"), ("X2", "X1"), ("X3", "X1"), ("X4", "X1")))
STATE_SERIES = X + ("ENS", "V6:A0g", "V4:NTP2", "V6:A2", "cfg4", "cfg8", "M", "K2", "K3")
LEAF = {"X1": "E1", "X2": "E2", "X3": "E3", "X4": "E4"}


def assemble(ser: dict, x: dict) -> dict:
    """Словарь рядов в порядке ALL; X1–X4 и ENS — прогнозы листов (массивы)."""
    s = {k: np.asarray(ser[k], float) for k in CTRL}
    s.update({k: np.asarray(v, float) for k, v in x.items()})
    n = {len(v) for v in s.values()}
    if len(n) != 1 or any(not (np.isfinite(v).all() and (v > 0).all()) for v in s.values()):
        raise SystemExit("FIN: ряды разной длины / не положительны / не конечны — одинаковых дат нет")
    return {k: s[k] for k in ALL}


def _go_all(rv, s, refs, tp):
    return pd.concat([EV.go_table(rv, s[x], refs, tp).assign(config=x) for x in X], ignore_index=True)


def main_tables(rv, s: dict, tp) -> dict:
    rv, tp = np.asarray(rv, float), np.asarray(tp)
    refs = {k: s[k] for k in EV.REFS}
    tests = pd.concat([E1V._holm(EV.tests_table(rv, s[x], refs, tp), EV.TEST_WIN).assign(config=x) for x in X], ignore_index=True)
    return {"METRICS": EV.metrics_table(rv, s, tp), "TESTS": tests, "PAIRS": E1V._holm(E2.pairs_table(rv, s, tp, PAIRS), E2.PAIR_WIN), "GO": _go_all(rv, s, refs, tp),
            "YEARS": pd.concat([EV.years_table(rv, s[x], refs, tp).assign(config=x) for x in X], ignore_index=True), "STRESS": EV.stress_table(rv, s, tp), "MCS": EV.mcs_table(rv, s, tp)}


def state_table(rv, s: dict, tp, state) -> pd.DataFrame:
    rv, tp, state, rows = np.asarray(rv, float), np.asarray(tp), np.asarray(state), []
    for name in STATE_SERIES:
        q = Q.ql(rv, s[name])
        for w in E1V.STATE_WIN:
            for k, lab in ((0, "норма"), (1, "экстрем")):
                m = ST.wmask(tp, w) & (state == k)
                rows.append(dict(series=name, window=w, state=lab, n=int(m.sum()), qlike=float(q[m].mean()) if m.any() else float("nan")))
    return pd.DataFrame(rows)


def _stab_row(name: str, w: str, rv, s: dict, tp) -> dict:
    """Одна строка Ж-239 п. 10: ряд `name`, период `w` (год после 2021 или окно 2022–2025)."""
    m = ST.wmask(tp, w)
    f, h, r = s[name][m], s["K1"][m], rv[m]
    q, u, c = Q.ql(r, f), np.log(r / f), np.log(f / h)
    qb = {b: float(Q.ql(r, s[b][m]).mean()) for b in BOOST}
    best = min(BOOST, key=lambda b: (qb[b], b))
    d = q - Q.ql(r, s[best][m])
    k = max(1, int(round(EV.TRIM * int(m.sum()))))
    top = np.argsort(-np.abs(d), kind="stable")[:k]
    tot = float(d.sum())
    return dict(series=name, period=w, n=int(m.sum()), qlike=float(q.mean()), best_boost=best, gap_best_boost=float(q.mean() - qb[best]), mean_corr=float(c.mean()), mean_abs_corr=float(np.abs(c).mean()),
                share_corr_pos=float((c > 0).mean()), share_corr_neg=float((c < 0).mean()), share_under=float((u > 0).mean()), share_over=float((u <= 0).mean()), mean_u_under=float(u[u > 0].mean()) if (u > 0).any() else float("nan"),
                mean_u_over=float(u[u <= 0].mean()) if (u <= 0).any() else float("nan"), qlike_part_under=float((q * (u > 0)).sum() / len(q)), qlike_part_over=float((q * (u <= 0)).sum() / len(q)),
                top1_k=k, delta_total=tot / len(q), delta_top1=float(d[top].sum()) / len(q), share_top1=float(d[top].sum() / tot) if abs(tot) > 1e-12 else float("nan"))


def stab_table(rv, s: dict, tp) -> pd.DataFrame:
    rv, tp = np.asarray(rv, float), np.asarray(tp)
    return pd.DataFrame([_stab_row(n, w, rv, s, tp) for n in ALL if n != "K1" for w in PERIODS])


def adv_table(met: pd.DataFrame) -> pd.DataFrame:
    """Преимущество над K2 (QLIKE(K2) − QLIKE(ряд)) на 2018–2021 и 2022–2025 и его смена — «что устранено» из причины проигрыша (DE0)."""
    m = met.set_index(["series", "window"])["qlike"]
    rows = [dict(series=n, adv_2018_2021=float(m[("K2", "2018_2021")] - m[(n, "2018_2021")]), adv_2022_2025=float(m[("K2", "2022_2025")] - m[(n, "2022_2025")])) for n in ALL if n not in ("K2", "K1")]
    t = pd.DataFrame(rows)
    t["d_adv"] = t["adv_2022_2025"] - t["adv_2018_2021"]
    return t


def all_tables(rv, s: dict, tp, state) -> dict:
    t = main_tables(rv, s, tp)
    t.update(STATE=state_table(rv, s, tp, state), STAB=stab_table(rv, s, tp), ADV=adv_table(t["METRICS"]))
    return t
