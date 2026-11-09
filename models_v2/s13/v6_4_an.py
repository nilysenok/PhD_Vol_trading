"""V6-4, G36/G37: историческая аналогия (план V6, п. 6) без обучателей. Состояние Z_D дня d — s1, s2, s4, s5, s7, s9, z-нормировка по среднему и СКО ПУЛА дня d (закрытые пары s: t⁺_s ≤ d, с 2015-07-07).
R_s = RV_{s⁺} / H_s. D1: M = среднее R по k = 50 ближайшим (евклидово расстояние в Z_D); D2: k = 100, w = exp(−dist²/(2h²)), h = медиана по пулу расстояний до 50-го соседа (сосед — другая точка пула); D3: M = (n_eff·M_D2 + 50·M_glob)/(n_eff + 50);
K9 (контроль): M_glob = среднее R пула, F = H·M_glob. Прогноз D: F = H·exp(clip(ln M, −1, 1)) = H·clip(M, e⁻¹, e) (движок `v6_1_fit.forecast`). Пересчёт ежедневно. Соседние точки (не конфигурации): k D1 25/100, k D2 50/200, κ D3 25/100."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from models_v2.s13 import v6_1_fit as FT, v6_1_wf as WF

ZN = ["s1", "s2", "s4", "s5", "s7", "s9"]
K_D1, K_D2, K_H, KAPPA = 50, 100, 50, 50
VAR_D1, VAR_D2, VAR_KAPPA = (25, 50, 100), (50, 100, 200), (25, 50, 100)
POOL_FIRST = "2015-07-07"


def closed_rows(t_plus, d) -> np.ndarray:
    """Пул дня d: пары с t⁺ ≤ d. МУТАНТ «закрыто + 1» подменяет эту функцию в тесте."""
    return WF.closed_rows(t_plus, d)


def norm_stats(Zp: np.ndarray) -> tuple:
    """Среднее и СКО (ddof = 0) по пулу дня d. МУТАНТ «по всей выборке» подменяет эту функцию в тесте."""
    return FT.norm_stats(Zp)


def h_scale(Zs: np.ndarray, nh: int = K_H) -> float:
    """h = медиана по точкам пула расстояний до nh-го другого соседа (в partition индекс nh: нулевое расстояние до себя стоит первым)."""
    if len(Zs) <= nh:
        raise ValueError("пул меньше %d точек" % (nh + 1))
    return float(np.median(np.partition(cdist(Zs, Zs), nh, axis=1)[:, nh]))


def clip_m(m):
    return np.exp(np.clip(np.log(m), -1.0, 1.0))


def day_stats(Zp, Rp, zq, k1=K_D1, k2=K_D2, nh=K_H, kappa=KAPPA, var=None) -> dict:
    """Все M дня d по пулу (Zp — состояния пула, Rp — R_s, zq — состояние дня d); var — соседние точки: {"d1": ks, "d2": ks, "kappa": ks}."""
    mu, sd = norm_stats(Zp)
    zp, q = (Zp - mu) / sd, (zq - mu) / sd
    dist = cdist(q[None, :], zp)[0]
    order = np.argsort(dist, kind="stable")
    hh = h_scale(zp, nh)
    mg = float(Rp.mean())
    out = dict(m_glob=mg, h=hh, n_pool=len(Rp))
    v = var or dict(d1=(k1,), d2=(k2,), kappa=(kappa,))
    for k in v["d1"]:
        out["m1_%d" % k] = float(Rp[order[:k]].mean())
    base = {}
    for k in v["d2"]:
        idx = order[:k]
        w = np.exp(-dist[idx] ** 2 / (2 * hh * hh))
        base[k] = (float((w * Rp[idx]).sum() / w.sum()), float(w.sum() ** 2 / (w ** 2).sum()))
        out["m2_%d" % k], out["neff_%d" % k] = base[k]
    m2, neff = base[k2]
    for kp in v["kappa"]:
        out["m3_%d" % kp] = (neff * m2 + kp * mg) / (neff + kp)
    return out


def tercile(hp, hq) -> int:
    """Терциль H_t: пороги — по H закрытых пар пула дня d (1 — нижний, 3 — верхний)."""
    q1, q2 = np.quantile(hp, [1 / 3, 2 / 3])
    return 1 + int(hq >= q1) + int(hq >= q2)


def run(tab: pd.DataFrame, S: pd.DataFrame, dates, **kw) -> pd.DataFrame:
    """Ежедневный прогноз M на датах `dates` (index = дни таблицы H): пул — закрытые пары, R = rv_next / h. -> по дате: m_glob, h, n_pool, tercile, M-колонки."""
    Z = S.reindex(tab.index)[ZN].to_numpy(float)
    R = tab["rv_next"].to_numpy(float) / tab["h"].to_numpy(float)
    tp, hh = tab["t_plus"].to_numpy(), tab["h"].to_numpy(float)
    pos = tab.index.get_indexer(pd.DatetimeIndex(dates))
    rows = []
    for d, i in zip(pd.DatetimeIndex(dates), pos):
        p = closed_rows(tp, d)
        r = day_stats(Z[p], R[p], Z[i], **kw)
        r["tercile"] = tercile(hh[p], hh[i])
        rows.append(r)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
