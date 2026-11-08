"""V4-RQC: RQC3 — ограниченная адаптация множителя после скачков (Ж-225 п. 6). F = K8·a_i,d, log a_i = иерархическая EWMA winsorized ln(RV/K8):
- y = ln(RV/K8) пары обрезается по 5-му и 95-му процентилям прошлых пар бумаги (расширяющееся окно, только пары до этой; до MIN_PAIRS пар — без обрезки);
- own: EWMA по парам бумаги (полураспад HL пар), панель: EWMA по дням (полураспад HL дней) по парам панели; цель = (n·own + κ·panel)/(n + κ), n = эффективное число пар, κ = KAPPA; цель обрезана в ln[0,6; 1,6];
- день-скачок d (данные ≤ d, процентили по прошлым дням бумаги): доля скачка J/RV > p90 или |Δ ln RV| > p95 — выход замораживается на FREEZE торговых дней (d, d+1, …, d+FREEZE−1; EWMA продолжает принимать пары);
- вне заморозки шаг |Δ ln a| ≤ STEP в день; до MIN_PAIRS закрытых пар бумаги a = 1. Гиперпараметры фиксированы объявлением."""
from __future__ import annotations

import numpy as np

HL, KAPPA, STEP, FREEZE, MIN_PAIRS = 60, 250.0, 0.02, 5, 60
Q_LO, Q_HI, Q_JS, Q_DL = 0.05, 0.95, 0.90, 0.95
LA_LO, LA_HI = float(np.log(0.6)), float(np.log(1.6))


def window(hist, full):
    """Прошлое окно бумаги для процентилей обрезки: только пары до текущей (тест подменяет на `full` — будущее)."""
    return hist


def _exceeds(v, past, q) -> bool:
    p = past[np.isfinite(past)]
    return bool(len(p) >= MIN_PAIRS and np.isfinite(v) and v > np.quantile(p, q))


def prep_ticker(y, js, dl) -> tuple:
    """-> (winsorized y, флаг дня-скачка) по строкам бумаги в порядке дат; js — доля скачка, dl — Δ ln RV."""
    n, yw, flag, adl = len(y), np.array(y, float), np.zeros(len(y), bool), np.abs(dl)
    for r in range(n):
        if r >= MIN_PAIRS:
            lo, hi = np.quantile(window(y[:r], y), [Q_LO, Q_HI])
            yw[r] = min(max(y[r], lo), hi)
        flag[r] = _exceeds(js[r], js[:r], Q_JS) or _exceeds(adl[r], adl[:r], Q_DL)
    return yw, flag


def prep(rv, F0, tix, js, dl, nT) -> tuple:
    """Строки отсортированы по (date, ticker). -> (yw, flag) по строкам; не зависит от состава панели и от моментов закрытия пар."""
    y = np.log(np.asarray(rv, float) / np.asarray(F0, float))
    yw, flag = np.empty(len(y)), np.zeros(len(y), bool)
    for i in range(nT):
        idx = np.flatnonzero(tix == i)
        yw[idx], flag[idx] = prep_ticker(y[idx], np.asarray(js, float)[idx], np.asarray(dl, float)[idx])
    return yw, flag


def run(yw, flag, avail, tix, date, nT, pm, step=None, freeze=None) -> tuple:
    """Дневной цикл по датам; -> (ln a, заморожено ли) по строкам. pm[i] — бумага в панели (в LOTO без исключённой)."""
    step, freeze = STEP if step is None else step, FREEZE if freeze is None else freeze
    n, order = len(yw), np.argsort(avail, kind="stable")
    lo, lp = 0.5 ** (1.0 / HL), 0.5 ** (1.0 / HL)
    num, den, den2, cnt = (np.zeros(nT) for _ in range(4))
    la, frz, pnum, pden, ptr = np.zeros(nT), np.zeros(nT, int), 0.0, 0.0, 0
    out, fz = np.zeros(n), np.zeros(n, bool)
    days, starts = np.unique(date, return_index=True)
    for d, a, e in zip(days, starts, list(starts[1:]) + [n]):
        pnum, pden = pnum * lp, pden * lp
        while ptr < n and avail[order[ptr]] <= d:
            j = order[ptr]
            ptr += 1
            i = tix[j]
            num[i], den[i], den2[i], cnt[i] = lo * num[i] + yw[j], lo * den[i] + 1.0, lo * lo * den2[i] + 1.0, cnt[i] + 1
            if pm[i]:
                pnum, pden = pnum + yw[j], pden + 1.0
        for r in range(a, e):
            i = tix[r]
            frz[i] = freeze if flag[r] else frz[i]
            fz[r] = frz[i] > 0
            if cnt[i] >= MIN_PAIRS and frz[i] == 0:
                ne = den[i] ** 2 / den2[i]
                tgt = float(np.clip((ne * num[i] / den[i] + KAPPA * (pnum / pden if pden > 0 else 0.0)) / (ne + KAPPA), LA_LO, LA_HI))
                la[i] += float(np.clip(tgt - la[i], -step, step))
            frz[i] = max(frz[i] - 1, 0)
            out[r] = la[i]
    return out, fz
