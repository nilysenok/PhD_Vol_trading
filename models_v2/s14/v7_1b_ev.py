"""V7-1b-C, G25: расчёт оценки конфигураций 1–3 (чистые функции от массивов; файлы не читаются, ничего не пишется).
Ряды — на ОДИНАКОВЫХ датах (`same_dates`): K1, K2, K3, M = √(K2·K3), K8 = √(K1·M), V4:NTP2, V6:A2, V6:A0, V6:A0g и конфигурации 1–3. Окна — по году цели t⁺ (2018…2026, 2018–2021, 2019–2021, 2022–2025, 2026, all).
Метрики — определение проекта (`v6_0_q.metr_row`). Δ = mean QLIKE(конфигурация) − mean QLIKE(база) в % среднего QLIKE K1 окна, 90 %-й интервал и DM лаг 5, Холм N = 733 / N_wide = 3166, MCS 90 % — реализации V6 (`v6_1_eval`, `stats_d`).
«Go» 6.1 — шесть условий плана; «новая информация» 6.2 — `v7_1m_stat.new_information`. Условия 4–5 — по годовым средним лучшего из K2/K3 (как V6 `years_better`), условие 3 — отбрасывание 1 % дат окна 2018–2021 с наибольшей |q − min(q_K2, q_K3)|,
одинаковых для всех сравниваемых рядов."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q, v6_1_eval as EV
from models_v2.s14 import v7_1m_stat as ST

REFS = ST.BASES + ("V6:A0", "V6:A0g")                       # базы сравнения Δ: K1, K2, K3, M, K8, NTP2, A2, A0, A0g
GO_BASES = ("K2", "K3", "M", "K8", "V4:NTP2", "V6:A2")      # условие 1 «go»: ≤ min этих шести − δ
CFGS = ("cfg1", "cfg2", "cfg3")
YEARS = tuple(range(2018, 2027))
TEST_WIN = ("2018_2021", "2019_2021", "2022_2025", "2026", "all")
METR_WIN = tuple(str(y) for y in range(2018, 2026)) + TEST_WIN          # годы 2018…2025 + окна (2026 — в TEST_WIN)
MCS_WIN = ("2018_2021", "2022_2025", "all")
DELTA, TRIM, N_CP = Q.MARGIN, Q.TRIM, {"N": ST.N_HOLM, "N_wide": ST.N_WIDE}


def same_dates(frames: dict) -> bool:
    """Все ряды заданы на одном и том же индексе дат (сдвиг одного ряда на +1 день нарушает равенство)."""
    idx = [pd.DatetimeIndex(f.index) for f in frames.values()]
    return all(len(i) == len(idx[0]) and bool((i == idx[0]).all()) for i in idx)


def metrics_table(rv, series: dict, tp) -> pd.DataFrame:
    rows = []
    for name, f in series.items():
        for w in METR_WIN:
            m = ST.wmask(tp, w)
            if m.any():
                rows.append(dict(series=name, window=w, **Q.metr_row(np.asarray(rv, float)[m], np.asarray(f, float)[m])))
    return pd.DataFrame(rows)


def tests_table(rv, f, rf: dict, tp, windows=TEST_WIN) -> pd.DataFrame:
    """Строка на (окно, база): Δ % уровня K1 с 90 % интервалом, DM (НУ, лаг 5; p односторонний «конфигурация лучше»), p Холма N и N_wide по окну (семейство — все базы окна)."""
    rv, f = np.asarray(rv, float), np.asarray(f, float)
    rows = []
    for w in windows:
        m = ST.wmask(tp, w)
        qf, k1 = Q.ql(rv[m], f[m]), float(Q.ql(rv[m], rf["K1"][m]).mean())
        for name in REFS:
            fb = rf[name][m]
            rows.append(dict(window=w, base=name, **EV.delta_ci(qf, Q.ql(rv[m], fb), k1, b=ST.B, block=ST.BLOCK, seed=ST.SEED), **EV.dm_one(rv[m], fb, f[m])))
    t = pd.DataFrame(rows)
    t["p_holm_N"], t["p_holm_Nwide"] = np.nan, np.nan
    for w in windows:
        s = (t["window"] == w) & t["p_one"].notna()
        t.loc[s, "p_holm_N"], t.loc[s, "p_holm_Nwide"] = EV.holm_both(t.loc[s, "p_one"].to_numpy(), N_CP)
    return t


def _gap(q, qb, m) -> float:
    return float(q[m].mean() - min(b[m].mean() for b in qb))


def go_table(rv, f, rf: dict, tp) -> pd.DataFrame:
    """Шесть условий «go» 6.1: значение, порог, выполнено (+ строка «go» — все шесть)."""
    rv, tp = np.asarray(rv, float), np.asarray(tp)
    q, qb = Q.ql(rv, f), [Q.ql(rv, rf[b]) for b in GO_BASES]
    q2, q3 = Q.ql(rv, rf["K2"]), Q.ql(rv, rf["K3"])
    m1, m2, m3 = (ST.wmask(tp, w) for w in ("2018_2021", "2019_2021", "2022_2025"))
    dq = np.abs(q - np.minimum(q2, q3))
    keep = m1.copy()
    idx = np.flatnonzero(m1)
    keep[idx[np.argsort(-dq[idx], kind="stable")[: int(round(TRIM * len(idx)))]]] = False
    an = {y: (float(q[ST.wmask(tp, str(y))].mean()), min(float(q2[ST.wmask(tp, str(y))].mean()), float(q3[ST.wmask(tp, str(y))].mean())), int(ST.wmask(tp, str(y)).sum())) for y in range(2018, 2022)}
    better, gains = sum(a < b for a, b, _ in an.values()), [n * (b - a) for a, b, n in an.values()]
    total = float(sum(gains))
    share = max(gains) / total if total > 0 else float("inf")
    rows = [(1, "QLIKE 2018–2021 − min(K2, K3, M, K8, NTP2, A2)", _gap(q, qb, m1), -DELTA, _gap(q, qb, m1) <= -DELTA),
            (2, "то же 2019–2021 (без 2018)", _gap(q, qb, m2), -DELTA, _gap(q, qb, m2) <= -DELTA),
            (3, "то же без 1 % дат окна с наибольшим модулем разности q − min(q_K2, q_K3)", _gap(q, qb, keep), -DELTA, _gap(q, qb, keep) <= -DELTA),
            (4, "годов из 4 лучше лучшего из K2/K3 (годовые средние)", float(better), 3.0, better >= 3),
            (5, "доля максимального года в суммарном выигрыше над лучшим из K2/K3", share, 0.5, bool(total > 0 and share <= 0.5)),
            (6, "Δ к лучшему из K2/K3 на 2022–2025 (точка)", float(q[m3].mean() - min(q2[m3].mean(), q3[m3].mean())), 0.0, bool(q[m3].mean() - min(q2[m3].mean(), q3[m3].mean()) <= 0))]
    t = pd.DataFrame(rows, columns=["cond", "name", "value", "threshold", "passed"])
    return pd.concat([t, pd.DataFrame([dict(cond=0, name="go (все шесть)", value=float(t["passed"].sum()), threshold=6.0, passed=bool(t["passed"].all()))])], ignore_index=True)


def years_table(rv, f, rf: dict, tp) -> pd.DataFrame:
    """По годам 2018…2025: QLIKE ряда, лучшее годовое из K2/K3, выигрыш·n (вклад года в условие 5)."""
    rv, tp, rows = np.asarray(rv, float), np.asarray(tp), []
    for y in range(2018, 2026):
        m = ST.wmask(tp, str(y))
        a, b = float(Q.ql(rv[m], f[m]).mean()), min(float(Q.ql(rv[m], rf[k][m]).mean()) for k in ("K2", "K3"))
        rows.append(dict(year=y, n=int(m.sum()), qlike=a, qlike_best_k2k3=b, gain_times_n=(b - a) * int(m.sum())))
    return pd.DataFrame(rows)


def new_info_table(rv, f, rf: dict, tp) -> pd.DataFrame:
    return pd.DataFrame([dict(base=b, **ST.new_information(rv, f, rf[b], tp)) for b in ("V6:A0", "V6:A0g")])


def stress_table(rv, series: dict, tp) -> pd.DataFrame:
    """Стресс: QLIKE и разность к лучшему из K2/K3 — основное окно 2018–2021, без 1 % крайних дат (условие 3), без 2018 (2019–2021)."""
    rv, tp = np.asarray(rv, float), np.asarray(tp)
    q = {k: Q.ql(rv, v) for k, v in series.items()}
    m1 = ST.wmask(tp, "2018_2021")
    idx, rows = np.flatnonzero(m1), []
    for name in series:
        dq = np.abs(q[name] - np.minimum(q["K2"], q["K3"]))
        keep = m1.copy()
        keep[idx[np.argsort(-dq[idx], kind="stable")[: int(round(TRIM * len(idx)))]]] = False
        for label, m in (("2018_2021", m1), ("без 1 % крайних", keep), ("2019_2021", ST.wmask(tp, "2019_2021"))):
            rows.append(dict(series=name, stress=label, n=int(m.sum()), qlike=float(q[name][m].mean()), gap_best_k2k3=float(q[name][m].mean() - min(q["K2"][m].mean(), q["K3"][m].mean()))))
    return pd.DataFrame(rows)


def mcs_table(rv, series: dict, tp) -> pd.DataFrame:
    return pd.concat([ST.mcs90(rv, series, tp, w).assign(window=w) for w in MCS_WIN], ignore_index=True)
