"""Z2, HD — описание по сохранённым рядам ZB (`data/regen/harj_x1/Z/zb.parquet`), без конфигураций и прогнозов (план Z2, L2(б)). D_ZB = l(ZB) − l(X1) по дням E (e_d = 1) после включения ZB (active = 1).
Группы: терцили τ^last_d (минуты от последнего события дня до закрытия; пороги — по ПРОШЛЫМ дням E, d' < d, не меньше 30 дней, иначе день без группы) и N_d ∈ {1; ≥ 2}. «Поздний» = τ^last в нижнем терциле (событие у закрытия), «ранний» — в верхнем.
Контраст: поздний − ранний; (≥ 2) − 1. Интервал: скользящие блоки 20 по дням окна, 2000 повторов, seed 0, 90 % (5–95 %). Окна — по году t⁺: весь период и 2018–2021."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s14 import v7_1m_stat as ST

MIN_PAST, BLOCK, NBOOT, SEED, WINDOWS = 30, 20, 2000, 0, ("all", "2018_2021")


def tercile_labels(tau: np.ndarray, d) -> np.ndarray:
    """0 — поздний, 1 — средний, 2 — ранний, −1 — нет группы; пороги 1/3 и 2/3 по τ дней E со СТРОГО меньшей датой (tau — по дням E в порядке дат, NaN недопустим)."""
    d, lab = np.asarray(d), np.full(len(tau), -1)
    for i in range(len(tau)):
        past = tau[d < d[i]]
        if len(past) >= MIN_PAST:
            q1, q2 = np.quantile(past, [1 / 3, 2 / 3])
            lab[i] = 0 if tau[i] <= q1 else (2 if tau[i] > q2 else 1)
    return lab


def boot_diff(x: np.ndarray, g: np.ndarray, a: int, b: int) -> tuple:
    """(разность средних групп a − b, lo90, hi90, число валидных повторов): блоки 20, 2000, seed 0."""
    n = len(x)
    pt = float(x[g == a].mean() - x[g == b].mean()) if (g == a).any() and (g == b).any() else float("nan")
    rng, nb = np.random.default_rng(SEED), int(np.ceil(n / BLOCK))
    st = rng.integers(0, max(n - BLOCK, 0) + 1, size=(NBOOT, nb))
    ix = (st[:, :, None] + np.arange(BLOCK)).reshape(NBOOT, -1)[:, :n]
    xs, gs = x[ix], g[ix]
    ma, mb = (gs == a), (gs == b)
    ok = ma.any(1) & mb.any(1)
    d = (np.where(ma, xs, 0).sum(1)[ok] / ma.sum(1)[ok]) - (np.where(mb, xs, 0).sum(1)[ok] / mb.sum(1)[ok])
    lo, hi = (np.quantile(d, [0.05, 0.95]) if ok.any() else (float("nan"), float("nan")))
    return pt, float(lo), float(hi), int(ok.sum())


def hd_table(zb: pd.DataFrame, e: np.ndarray, tau_all: np.ndarray, nd_all: np.ndarray) -> pd.DataFrame:
    """zb: d, t_plus, rv, X1, F, active; e, tau_all, nd_all — по строкам zb (e_d, τ^last_d, N_d)."""
    d, tp = pd.DatetimeIndex(zb["d"]), pd.DatetimeIndex(zb["t_plus"])
    D = Q.ql(zb["rv"].to_numpy(), zb["F"].to_numpy()) - Q.ql(zb["rv"].to_numpy(), zb["X1"].to_numpy())
    E = np.asarray(e) == 1
    lab_E = tercile_labels(np.asarray(tau_all)[E], d.to_numpy()[E])
    lab = np.full(len(zb), -1)
    lab[E] = lab_E
    nlab = np.where(E, np.where(np.asarray(nd_all) >= 2, 1, 0), -1)
    act = zb["active"].to_numpy() == 1
    rows = []
    for w in WINDOWS:
        base = E & act & ST.wmask(tp, w)
        for name, g, a, b, la, lb in (("поздний − ранний (терцили τ^last)", lab, 0, 2, "поздний", "ранний"), ("N_d ≥ 2 − N_d = 1", nlab, 1, 0, "N≥2", "N=1")):
            k = base & (g >= 0)
            pt, lo, hi, nv = boot_diff(D[k], g[k], a, b) if k.sum() > 1 else (float("nan"),) * 3 + (0,)
            rows.append(dict(contrast=name, window=w, n=int(k.sum()), n_a=int((g[k] == a).sum()), n_b=int((g[k] == b).sum()), mean_a=float(D[k][g[k] == a].mean()) if (g[k] == a).any() else float("nan"),
                             mean_b=float(D[k][g[k] == b].mean()) if (g[k] == b).any() else float("nan"), diff=pt, lo90=lo, hi90=hi, n_boot=nv, group_a=la, group_b=lb))
    return pd.DataFrame(rows)


def hd_negative(t: pd.DataFrame) -> bool:
    """L2(б): хотя бы один контраст: разность < 0 за весь период, hi90 < 0 и разность < 0 на 2018–2021."""
    for c, g in t.groupby("contrast"):
        a, r = g[g["window"] == "all"].iloc[0], g[g["window"] == "2018_2021"].iloc[0]
        if a["diff"] < 0 and a["hi90"] < 0 and r["diff"] < 0:
            return True
    return False
