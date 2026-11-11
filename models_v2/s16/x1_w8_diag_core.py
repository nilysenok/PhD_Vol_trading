"""W8-A (Ж-253 п. 4; `docs/otchety/harj_x1/W8_SPEC.md` разд. 1.0, 9): перенос `w8/core.py` (scratchpad) БЕЗ изменения логики. Только чтение сохранённых рядов окна M; ничего не пишет.
Публичное API:
- `ROOT`, `MINPAST` = 60, `BMAX` = 0,5, `MIN_GROUP` = 15, `CUT_BAR` = 18:30 (последний бар с началом ≤ 18:30), `YEARS` = '2018'…'2026'.
- `ql(rv, f)`: QLIKE дня l = RV/F − ln(RV/F) − 1.
- `terc_past(v, minpast=60)`: PIT-терцили T1/T2/T3 — пороги по значениям строго до i (не меньше `minpast` конечных прошлых), иначе `na` (и при нефинитном v[i]).
- `planned(d_index, cal, canon) -> (L, first_wd_month)`: плановая длина перерыва L_d по календарю, известному на d 18:40 (`x1_w5_gap_cal.known`); месяц первого буднего неторгового дня.
- `gap_type(d_index, L)`: none / weekend / hol1 / long / worksat / other.
- `load()`: окно M (`x1_w3_core.load`) + w5g.parquet (W5G, gap, beta, refit, c_app) + L, type, delta_real, yr, half, u = ln(RV/X1), l_X1/l_W5G/l_K1/l_K2/l_K3, season. Расхождение входов — SystemExit.
- `beta_se(u, g)`, `beta_est(u1, u0)`: оценщик W5-G clip(Δū + ½Δs², 0, 0,5) (ddof = 1) и β̂/SE (дельта-метод).
Признаки W8D1…W8D10 — `x1_w8_diag_feat`."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401  (torch первым)
from models_v2.s16 import x1_w3_core as C
from models_v2.s16 import x1_w5_gap_cal as GC

ROOT = Path(__file__).resolve().parents[2]
MINPAST = 60
BMAX = 0.5
MIN_GROUP = 15
CUT_BAR = pd.Timedelta(hours=18, minutes=30)   # последний бар с началом ≤ 18:30 (кончается ≤ 18:40)
YEARS = [str(y) for y in range(2018, 2027)]


def ql(rv, f):
    r = np.asarray(rv, float) / np.asarray(f, float)
    return r - np.log(r) - 1.0


def terc_past(v, minpast=MINPAST):
    """Как w5/diag.terc_past: пороги по значениям строго до i (все дни), ≥ minpast прошлых; NaN → na."""
    v = np.asarray(v, float)
    g = np.full(len(v), "na", object)
    for i in range(minpast, len(v)):
        past = v[:i][np.isfinite(v[:i])]
        if len(past) < minpast or not np.isfinite(v[i]):
            continue
        q1, q2 = np.quantile(past, [1 / 3, 2 / 3])
        g[i] = "T1" if v[i] <= q1 else ("T2" if v[i] <= q2 else "T3")
    return g


# ---------- плановая длина перерыва на d 18:40 ----------

def _nontrading(n, d, rows, canon) -> bool:
    r = rows.get(n)
    if n.dayofweek >= 5:
        if r is None or r["kind"] != "weekend_trading" or n not in canon:
            return True
        return not GC.known(r, d)          # торговая суббота, не объявленная к d, — неторговая (11.3)
    if r is None or r["kind"] != "weekday_closed":
        return False
    return GC.known(r, d)                  # будний неторговый — только если объявлен к d


def planned(d_index, cal, canon):
    rows = {r["date"]: r for _, r in cal.iterrows()}
    L, first_wd_month = [], []
    for d in pd.DatetimeIndex(d_index):
        k, n, fm = 0, d + pd.Timedelta(days=1), 0
        while _nontrading(n, d, rows, canon):
            if n.dayofweek < 5 and fm == 0:
                fm = n.month
            k += 1
            n += pd.Timedelta(days=1)
            if k > 30:
                raise SystemExit("перерыв > 30 дней: %s" % d.date())
        L.append(k)
        first_wd_month.append(fm)
    return np.array(L, int), np.array(first_wd_month, int)


def gap_type(d_index, L):
    d = pd.DatetimeIndex(d_index)
    dow = d.dayofweek.to_numpy()
    t = np.full(len(d), "none", object)
    for i in range(len(d)):
        if L[i] == 0:
            continue
        if dow[i] == 5:
            t[i] = "worksat"
        elif L[i] == 1:
            t[i] = "hol1"
        elif L[i] >= 3:
            t[i] = "long"
        elif L[i] == 2 and dow[i] == 4:
            t[i] = "weekend"
        else:
            t[i] = "other"
    return t


# ---------- загрузка ----------

def load():  # xtask: длинная — перенос w8/core.py без изменения логики
    x = C.load()
    w = pd.read_parquet(ROOT / "data/regen/harj_x1/W5/w5g.parquet")
    w["d"], w["t_plus"] = pd.to_datetime(w["d"]), pd.to_datetime(w["t_plus"])
    if len(w) != len(x) or not (w["d"].to_numpy() == x["d"].to_numpy()).all() or not np.array_equal(w["X1"].to_numpy(), x["F"].to_numpy()) \
            or not np.array_equal(w["rv"].to_numpy(), x["rv"].to_numpy()) or not (w["t_plus"].to_numpy() == x["t_plus"].to_numpy()).all():
        raise SystemExit("w5g ≠ окну M (d, t⁺, X1, rv)")
    x["W5G"] = w["F"].to_numpy(float)
    x["gap"] = w["gap"].to_numpy(int)
    x["beta"] = w["beta"].to_numpy(float)
    x["refit"] = pd.to_datetime(w["refit_date"])
    x["c_app"] = np.log(x["W5G"] / x["F"])
    if not np.array_equal(x.loc[x["gap"] == 0, "W5G"].to_numpy(), x.loc[x["gap"] == 0, "F"].to_numpy()):
        raise SystemExit("W5-G ≠ X1 в днях gap = 0")
    cal = GC.load_calendar()
    canon = GC.canonical_saturdays()
    L, fm = planned(x["d"], cal, canon)
    if not np.array_equal((L >= 1).astype(int), x["gap"].to_numpy()):
        raise SystemExit("L ≥ 1 ≠ gap^PIT в %d парах" % int(((L >= 1).astype(int) != x["gap"].to_numpy()).sum()))
    x["L"] = L
    x["type"] = gap_type(x["d"], L)
    x["delta_real"] = ((x["t_plus"] - x["d"]).dt.days - 1).to_numpy(int)
    x["yr"] = x["t_plus"].dt.year.astype(str)
    x["half"] = np.where(x["t_plus"].dt.year <= 2021, "h1", "h2")
    x["u"] = np.log(x["rv"] / x["F"])
    for k, col in (("X1", "F"), ("W5G", "W5G"), ("K1", "K1"), ("K2", "K2"), ("K3", "K3")):
        x["l_" + k] = ql(x["rv"], x[col])
    mon_d = x["d"].dt.month.to_numpy()
    season = np.full(len(x), "none", object)
    for i in range(len(x)):
        if L[i] == 0:
            continue
        if x["type"].iat[i] == "weekend":
            season[i] = "weekend"
            continue
        m = fm[i] if fm[i] else mon_d[i]
        season[i] = {12: "dec_jan", 1: "dec_jan", 2: "feb_mar", 3: "feb_mar", 5: "may", 6: "jun", 11: "nov"}.get(m, "m%02d" % m)
    x["season"] = season
    return x


def beta_se(u, g):
    """Как w6/core.beta_se: β̂ = Δū + ½Δs², SE дельта-методом с эмпирическими 3-м и 4-м моментами."""
    out = []
    for k in (1, 0):
        vv = u[g == k]
        n, s2 = len(vv), vv.var(ddof=1)
        c = vv - vv.mean()
        m3, m4 = float(np.mean(c ** 3)), float(np.mean(c ** 4))
        out.append((vv.mean(), s2, n, max(s2 + m3 + 0.25 * (m4 - s2 ** 2), 1e-18) / n))
    (m1, s21, _, v1), (m0, s20, _, v0) = out
    return float((m1 - m0) + 0.5 * (s21 - s20)), float(np.sqrt(v1 + v0))


def beta_est(u1, u0):
    """Оценщик W5-G по двум группам (без PIT, для переноса): clip(Δū + ½Δs², 0, BMAX)."""
    return float(np.clip((u1.mean() - u0.mean()) + 0.5 * (u1.var(ddof=1) - u0.var(ddof=1)), 0.0, BMAX))
