"""Шаг 0 разработки главы 4 (v3): A без прогноза, 2018–2021. Чистые функции над движком `s9.Engine`.

Движок нужен только как объект с `port(pos, comm, wgt)`, `A_row`, `pairs`, `tickers`, `dates`,
`r["pair"|"tick"|"date"]`, `lr_row` — поэтому `build` проверяется на синтетическом движке (tests/test_s9_shag0.py).
Легенда столбцов — docs/otchety/2026-10-27_gl4_shag_0.md (шапка).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt.fast_kontrol import k3_sids
from fastbt.msp import sharpe
from fastbt.s9_stats import fko_fee, lw_sharpe_diff

SIDS = ("S1", "S2", "S3", "S4", "S5", "S6")
YEARS = (2018, 2019, 2020, 2021)
POMETKA = "разработка 2018–2021, после аудита"
NAN = float("nan")


def ce_ann(r, g: float) -> float:
    """CE годовой, %: степенное среднее (1+r)^(1−γ); γ = 1 — геометрическое (лог-полезность)."""
    w = np.maximum(1.0 + np.asarray(r, float), 1e-10)
    m = float(np.exp(np.mean(np.log(w)))) if g == 1 else float(np.mean(w ** (1 - g)) ** (1 / (1 - g)))
    return (m ** 252 - 1.0) * 100


def stats(p: pd.DataFrame, mask) -> dict:
    """Метрики §0.3 на срезе дней `mask`: Шарп избыточной, средняя доходность и волатильность (% год),
    MDD по избыточной (%), доля дней в рынке, экспозиция, оборот (сторон/год), FKO-доля времени
    не глубже −10 % от максимума капитала, CE (γ = 1, 3, 5)."""
    q = p[np.asarray(mask, bool)]
    ex, tot = q["excess"].values, q["total"].values
    if len(ex) < 2:
        return {k: NAN for k in ("sharpe", "mean_ret_pct_y", "vol_pct_y", "mdd_pct", "dni_v_rynke_pct",
                                 "expo_pct", "turn_y", "fko_nad_poroga_pct", "ce1_pct_y", "ce3_pct_y", "ce5_pct_y")}
    cum = np.cumsum(ex)
    eq = np.cumprod(1.0 + tot)
    return dict(sharpe=sharpe(ex), mean_ret_pct_y=float(np.mean(tot) * 252 * 100),
                vol_pct_y=float(np.std(ex, ddof=1) * np.sqrt(252) * 100),
                mdd_pct=float(np.max(np.maximum.accumulate(cum) - cum) * 100),
                dni_v_rynke_pct=float(np.mean(q["exposure"].values > 0) * 100),
                expo_pct=float(q["exposure"].mean() * 100), turn_y=float(q["turnover"].mean() * 252),
                fko_nad_poroga_pct=float(np.mean(eq >= 0.9 * np.maximum.accumulate(eq)) * 100),
                ce1_pct_y=ce_ann(tot, 1), ce3_pct_y=ce_ann(tot, 3), ce5_pct_y=ce_ann(tot, 5))


def year_mask(p: pd.DataFrame, y: int):
    return p.index.year == y


def d_sharpe(p1: pd.DataFrame, p0: pd.DataFrame) -> float:
    """Единственное определение Δ в таблицах гл. 4 (владелец 27.10): Шарп(ddof=1) p1 − Шарп(ddof=1) p0."""
    return sharpe(p1["excess"].values) - sharpe(p0["excess"].values)


def delta_p(p1: pd.DataFrame, p0: pd.DataFrame, reps: int) -> tuple:
    """(Δ = `d_sharpe`, p Ледуа–Вольфа, блок 5). ЛВ даёт только p; его оценка Δ (ddof=0) в таблицы не идёт."""
    _, _, pv = lw_sharpe_diff(p1["excess"].values, p0["excess"].values, reps, 5)
    return d_sharpe(p1, p0), pv


def delta(p1: pd.DataFrame, p0: pd.DataFrame, reps: int) -> dict:
    """Δ Шарпа к p0 (d_vs_A, разность табличных Шарпов), p ЛВ (блок 5, p_vs_A), знак Δ по годам
    (по годовым Шарпам ddof=1), ФКО-плата γ = 3 (% год)."""
    d, pv = delta_p(p1, p0, reps)
    out = dict(d_vs_A=d, p_vs_A=pv)
    n = 0
    for y in YEARS:
        m = year_mask(p1, y)
        dy = sharpe(p1["excess"].values[m]) - sharpe(p0["excess"].values[m])
        out[f"znak_{y}"] = "+" if dy > 0 else "−"
        n += dy > 0
    out["n_let_luchshe_A"] = n
    out["fko_fee_g3_pct_y"] = fko_fee(p1["total"].values, p0["total"].values, 3.0)
    return out


def sid_row(eng) -> np.ndarray:
    sid = np.array([s for _, s in eng.pairs])
    return sid[np.asarray(eng.r["pair"], np.int64)]


def zero_sids(pos, sidrow, sids) -> np.ndarray:
    """A без семей `sids`."""
    return np.where(np.isin(sidrow, list(sids)), 0.0, pos)


def keep_sids(pos, sidrow, sids) -> np.ndarray:
    """A только на семьях `sids`."""
    return np.where(np.isin(sidrow, list(sids)), pos, 0.0)


def semyi(eng) -> list:
    """Доля дней с позицией по каждой семье и пересечения: число семей, открытых одновременно
    на одном тикере в один день (тикер-дни). Валовый вклад семьи — Σ pos·lr / n пар, % год."""
    A, sidr = np.asarray(eng.A_row), sid_row(eng)
    tick, date = np.asarray(eng.r["tick"], np.int64), np.asarray(eng.r["date"], np.int64)
    nd, nt = len(eng.dates), len(eng.tickers)
    key = tick * nd + date
    rows = []
    for s in SIDS:
        m = sidr == s
        g = np.bincount(date, np.where(m, A * eng.lr_row, 0.0), nd) / eng.cnt
        rows.append(dict(pometka=POMETKA, rezhim="семья", semya=s, dnei_s_poziciey_pct=float(np.mean(A[m] != 0) * 100),
                         valovoy_vklad_pct_y=float(np.mean(g) * 252 * 100)))
    n_open = np.bincount(key, (A != 0).astype(float), nt * nd)
    has = np.bincount(key, minlength=nt * nd) > 0
    for k in range(len(SIDS) + 1):
        rows.append(dict(pometka=POMETKA, rezhim="пересечение", semya=f"открыто семей: {k}",
                         tikerdni_pct=float(np.mean(n_open[has] == k) * 100)))
    rows.append(dict(pometka=POMETKA, rezhim="пересечение", semya="открыто больше одной семьи",
                     tikerdni_pct=float(np.mean(n_open[has] > 1) * 100),
                     dni_s_poziciey_pct=float(np.mean(n_open[has] > 0) * 100)))
    return rows


def build(eng, comm: float, reps: int = 10000) -> dict:
    """Все таблицы шага 0 на одной комиссии: списки строк по ключам metriki, godam, tikeram, loo, k3, spravochno."""
    A, sidr = np.asarray(eng.A_row, float), sid_row(eng)
    port = lambda pos, **kw: eng.port(pos, comm=comm, **kw)      # noqa: E731
    pA = port(A)
    allday = lambda p: np.ones(len(p), bool)                      # noqa: E731
    k3pos = zero_sids(A, sidr, k3_sids())
    pK = port(k3pos)
    out = {k: [] for k in ("metriki", "godam", "tikeram", "loo", "k3", "spravochno")}

    def row(label, p, base=False, **ex):
        r = dict(pometka=POMETKA, row=label, comm_pct=comm * 100, **ex)
        r.update(stats(p, allday(p)))
        if not base:
            r.update(delta(p, pA, reps))
        return r
    out["metriki"].append(row("A", pA, base=True))
    for lab, p in (("A", pA), ("к3: A без " + "/".join(k3_sids()), pK)):
        for y in YEARS:
            out["godam"].append(dict(pometka=POMETKA, row=lab, comm_pct=comm * 100, god=y, **stats(p, year_mask(p, y))))
    out["k3"] += [row("A", pA, base=True), row("к3: A без " + "/".join(k3_sids()), pK, isklyucheno=",".join(k3_sids()))]
    sh_a = sharpe(pA["excess"].values)
    for s in SIDS:
        wo = row(f"A без {s}", port(zero_sids(A, sidr, [s])), semya=s, rezhim="без")
        wo["vklad_sharpe"] = sh_a - wo["sharpe"]
        out["loo"] += [wo, row(f"A только {s}", port(keep_sids(A, sidr, [s])), semya=s, rezhim="только")]
    tick, date = np.asarray(eng.r["tick"], np.int64), np.asarray(eng.r["date"], np.int64)
    nd = len(eng.dates)
    key = tick * nd + date
    cnt = np.bincount(key, minlength=len(eng.tickers) * nd)
    for ti, t in enumerate(eng.tickers):
        on, days = tick == ti, cnt[ti * nd:(ti + 1) * nd] > 0
        w = np.where(on, 1.0 / np.maximum(cnt[key], 1), 0.0)       # портфель одного тикера (6 семей поровну)
        pa, pk = port(np.where(on, A, 0.0), wgt=w), port(np.where(on, k3pos, 0.0), wgt=w)
        r = dict(pometka=POMETKA, row=f"A {t}", ticker=t, comm_pct=comm * 100, dnei=int(days.sum()))
        r.update(stats(pa, days))
        r["sharpe_k3"] = stats(pk, days)["sharpe"]
        r["d_k3_minus_A"] = r["sharpe_k3"] - r["sharpe"]
        out["tikeram"].append(r)
    bh = np.where(sidr == SIDS[0], 1.0, 0.0)                         # по одной позиции на тикер
    wb = bh / np.maximum(np.bincount(date, bh, nd)[date], 1.0)
    out["spravochno"] = [row("A", pA, base=True),
                         row("купи и держи: 17 тикеров поровну", port(bh, wgt=wb)),
                         row("A × 0,5 (масштаб позиций)", port(0.5 * A))]
    return out
