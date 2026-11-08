"""V4-E, E5: прогноз F = F0·exp(b), b = m + s²/2 из фильтра Калмана по закрытым парам бумаги (расписание оценки параметров — раз в год, см. v4_e_kalman)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_e_calib as CAL, v4_e_kalman as KF


def refit_dates(date, t_plus, tix, nT) -> list:
    """Пуск — первый день, когда у каждой бумаги ≥ MIN_PAIRS закрытых пар; далее — первый торговый день каждого следующего года (с первого года после пуска)."""
    days = np.unique(date)
    avail = np.asarray(CAL.closed_date(np.asarray(t_plus), np.asarray(date)))
    per = [np.sort(avail[tix == i]) for i in range(nT)]
    start = next((d for d in days if all(np.searchsorted(p, d, "right") >= CAL.MIN_PAIRS for p in per)), None)
    if start is None:
        return []
    yrs = pd.DatetimeIndex(days).year
    first = {int(y): days[np.argmax(yrs == y)] for y in np.unique(yrs)}
    y0 = int(pd.Timestamp(start).year)
    return [start] + [first[y] for y in sorted(first) if y > y0]


def forecast_b(rv, F0, tix, date, t_plus, nT, panel_mask):
    """-> b по строкам (отсортированы по (date, ticker)); 0 до пуска и при < MIN_PAIRS закрытых парах бумаги. Метаданные: ρ_i, r̂_i по датам оценки."""
    avail = np.asarray(CAL.closed_date(np.asarray(t_plus), np.asarray(date)))
    u = np.log(np.asarray(rv, float) / np.asarray(F0, float))
    idx = [np.where(tix == i)[0] for i in range(nT)]               # строки бумаги по возрастанию даты
    av = [avail[ix] for ix in idx]
    R = refit_dates(date, t_plus, tix, nT)
    b, meta = np.zeros(len(rv)), []
    for k, r in enumerate(R):
        nc = [int(np.searchsorted(a, r, "right")) for a in av]
        rho, rhat, rp = KF.params([u[ix[:n]] for ix, n in zip(idx, nc)], panel_mask)
        meta.append({"refit": str(r)[:10], "rho_panel": rp, **{"rho_%d" % i: v for i, v in enumerate(rho)}})
        hi = R[k + 1] if k + 1 < len(R) else None
        for i, ix in enumerate(idx):
            m, P, _, _ = KF.filt(u[ix], rho[i])
            sel = (date[ix] >= r) & ((date[ix] < hi) if hi is not None else True)
            kk = np.searchsorted(av[i], date[ix][sel], "right")      # число закрытых пар на день прогноза
            ok = kk >= CAL.MIN_PAIRS
            bb = np.zeros(int(sel.sum()))
            bb[ok] = KF.b_from_state(m[kk[ok] - 1], P[kk[ok] - 1], rho[i], rhat[i])
            b[ix[sel]] = bb
    return b, meta
