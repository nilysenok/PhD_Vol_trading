"""W9 ПОСЛЕ ПРОСМОТРА (записано 11.11 после вердикта блоков A–C; в вердикт не входит):
PIT-проход разделённых весов HAR-части и части энкодера C.
ln F_d = ln M_d + π_d·(κ_H,m D_H,d + κ_G,m D_G,d); (κ_H, κ_G) — argmin QLIKE по закрытым парам t⁺ ≤ D_m (D_m — первый d месяца), решётка 0…1,5;
пул PH-E — расширяющийся с 2018-01-04, PH-R — последние 252 закрытые пары; включение при ≥ 126 закрытых парах, иначе (1, 1) = X1 побитно.
Контроль PH-*0 — общий κ (κ_H = κ_G) на том же пуле: отделяет пользу разделения от общего сжатия доверия.
Перенос из scratchpad `w9/posthoc.py` без изменения логики (лист кодера 11.11.2026)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w9_diag_core as K

MIN_POOL = 126


def walk(x: pd.DataFrame, pool: str, split: bool, avail_shift: int = 0) -> tuple[np.ndarray, list]:
    """avail_shift = 1 — мутация «закрыто + 1» (пара доступна с t⁺ − 1 торговой строки), только для теста PIT."""
    rv, lnM = x["rv"].to_numpy(float), np.log(x["M"].to_numpy(float))
    p1, p2 = (x["pi"] * x["DH"]).to_numpy(float), (x["pi"] * x["DG"]).to_numpy(float)
    d, tp = x["d"].to_numpy(), x["t_plus"].to_numpy()
    avail = tp if not avail_shift else d          # мутация: пара известна уже на d
    mon = x["d"].dt.to_period("M").to_numpy()
    lnF = lnM + p1 + p2
    path = []
    starts = [i for i in range(len(x)) if i == 0 or mon[i] != mon[i - 1]]
    for k, i0 in enumerate(starts):
        i1 = starts[k + 1] if k + 1 < len(starts) else len(x)
        D = d[i0]
        closed = np.where(avail <= D)[0]
        if pool == "R":
            closed = closed[-252:]
        if len(closed) < MIN_POOL:
            path.append((str(pd.Timestamp(D).date()), 1.0, 1.0, len(closed)))
            continue
        m = np.zeros(len(x), bool)
        m[closed] = True
        if split:
            kh, kg = K.fit_kappa2(rv, lnM, p1, p2, m)
        else:
            kh = kg = K.fit_kappa(rv, lnM, p1 + p2, m)
        lnF[i0:i1] = lnM[i0:i1] + kh * p1[i0:i1] + kg * p2[i0:i1]
        path.append((str(pd.Timestamp(D).date()), kh, kg, len(closed)))
    return np.exp(lnF), path


def periods(x):
    yr = x["yr"].to_numpy()
    out = {"M": np.ones(len(x), bool), "h1": x["h1"].to_numpy(bool), "h2": x["h2"].to_numpy(bool), "h2b22": x["h2b22"].to_numpy(bool),
           "M_bez_2022": yr != 2022}
    for y in sorted(set(yr)):
        out[str(y)] = yr == y
    return out


def run(x) -> tuple[list, dict]:
    rv = x["rv"].to_numpy(float)
    lx = x["lX1"].to_numpy(float)
    lk2 = K.qlike(rv, x["K2"].to_numpy(float))
    rows, paths = [], {}
    for name, pool, split in (("PH-E", "E", True), ("PH-E0", "E", False), ("PH-R", "R", True), ("PH-R0", "R", False)):
        F, path = walk(x, pool, split)
        paths[name] = path
        l = K.qlike(rv, F)
        for per, m in periods(x).items():
            rows += [dict(block="PH", cand=name, window=per, group="", metric="Q", value=float(l[m].mean())),
                     dict(block="PH", cand=name, window=per, group="", metric="d_X1_pct", value=float((l[m].mean() / lx[m].mean() - 1) * 100)),
                     dict(block="PH", cand=name, window=per, group="", metric="d_K2_pct", value=float((l[m].mean() / lk2[m].mean() - 1) * 100))]
        kk = np.array([[p[1], p[2]] for p in path if p[3] >= MIN_POOL])
        rows += [dict(block="PH", cand=name, window="path", group="", metric="kH_mean", value=float(kk[:, 0].mean())),
                 dict(block="PH", cand=name, window="path", group="", metric="kG_mean", value=float(kk[:, 1].mean())),
                 dict(block="PH", cand=name, window="path", group="", metric="first_active", value=float(next(i for i, p in enumerate(path) if p[3] >= MIN_POOL)))]
        for y in range(2018, 2027):
            ky = np.array([[p[1], p[2]] for p in path if p[3] >= MIN_POOL and p[0].startswith(str(y))])
            if len(ky):
                rows += [dict(block="PH", cand=name, window="path_%d" % y, group="", metric="kH_mean", value=float(ky[:, 0].mean())),
                         dict(block="PH", cand=name, window="path_%d" % y, group="", metric="kG_mean", value=float(ky[:, 1].mean()))]
    return rows, paths


def check_pit(x) -> tuple[bool, bool]:
    """Порча RV пар с t⁺ > D не меняет прогноз месяца D; мутация «закрыто + 1» меняет."""
    F0, _ = walk(x, "R", True)
    i = int(np.searchsorted(x["d"].to_numpy(), np.datetime64("2023-06-01")))
    D = x["d"].to_numpy()[i]
    y = x.copy()
    bad = y["t_plus"].to_numpy() > D
    y.loc[bad, "rv"] = y.loc[bad, "rv"] * 7.0
    F1, _ = walk(y, "R", True)
    mon = x["d"].dt.to_period("M").to_numpy()
    same = mon == mon[i]
    honest = bool(np.array_equal(F0[same], F1[same]))
    Fm, _ = walk(x, "R", True, avail_shift=1)
    caught = not np.array_equal(F0, Fm)
    return honest, caught
