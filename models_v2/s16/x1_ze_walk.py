"""Лист ZE, помесячная переоценка (план ZE, разд. 3.3; протокол ZB из плана Z разд. 4 без изменений): в первый торговый день D каждого месяца с 2018-07-02 по ВСЕМ закрытым к D парам (t⁺ ≤ D) только e_d = 1;
включение при ≥ 30 закрытых экстремальных парах (до этого F = X1, active = 0); V_e = 60 (иначе ⌊n_e/2⌋) последних экстремальных пар — блок валидации; λ ∈ {0; 0,001; 0,01; 0,1}, ничья — больший λ; затем переобучение на всём пуле.
ZE: g = e_d·(b + a·s_d + c·G_d); ZEp: то же с G_{d−252} (для обучения и применения). Пара j: (d_j, t⁺_j, X1_j, RV(t⁺_j), s, e, G дня d_j). Копирует структуру `x1_z_walk.walk` (ZB), `x1_z_*` не правятся (мутации «закрыто + 1», валидация по всем парам — в `x1_z_walk.closed_date`, `x1_z_fit.val_block_b`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_z_fit as FT, x1_z_walk as ZW, x1_ze_feat as FE, x1_ze_fit as FZ

KINDS = ("ZE", "ZEp")
OUT_COLS = ["d", "t_plus", "rv", "X1", "F", "g", "active", "lam", "b", "a", "c", "G", "n_ext"]
ACTIVATION, MIN_PAIRS, month_starts, closed_pool = ZW.ACTIVATION, ZW.MIN_PAIRS, ZW.month_starts, ZW.closed_pool


def refit(pool, D) -> dict | None:
    """Параметры по экстремальным парам пула; None — меньше 30 закрытых экстремальных пар."""
    u, s, e, G = D["u"], D["s"], D["e"], D["G"]
    ext = FT.ext_pool(pool, e)
    if len(ext) < FT.MIN_EXT:
        return None
    v = FT.n_valid(len(ext), FT.V_EXT)
    fit = lambda i, lam: FZ.fit_ze(u[i], s[i], G[i], lam)
    lam, _ = FT.select(ext[: len(ext) - v], FT.val_block_b(pool, ext, v), fit, lambda th, i, o: FT.qlike_g(u[i], FZ.g_of(th, e[i], s[i], G[i])))
    th = fit(ext, lam)
    return dict(lam=lam, b=th[0], a=th[1], c=th[2])


def walk(x1: pd.DataFrame, D: dict, kind: str) -> pd.DataFrame:
    """x1: d, t_plus, rv, F (= X1); D: u, s, e, G (n,) — G сырой (для ZEp сдвиг внутри). -> прогнозы (OUT_COLS)."""
    d, tp, X1, n = pd.DatetimeIndex(x1["d"]), pd.DatetimeIndex(x1["t_plus"]), x1["F"].to_numpy(float), len(x1)
    ZW.check_activation(d, tp)
    D = dict(D, G=FE.shift_g(D["G"]) if kind == "ZEp" else np.asarray(D["G"], float))
    g, act, nx, par = np.zeros(n), np.zeros(n, int), np.full(n, np.nan), np.full((n, 4), np.nan)
    starts = [p for p in month_starts(d) if d[p] >= ACTIVATION] + [n]
    for a, b in zip(starts[:-1], starts[1:]):
        pool = closed_pool(tp, d, d[a])
        nx[a:b] = FT.ext_pool(pool, D["e"]).size
        p = refit(pool, D)
        if p is None:
            continue
        g[a:b] = FZ.g_of((p["b"], p["a"], p["c"]), D["e"][a:b], D["s"][a:b], D["G"][a:b])
        act[a:b], par[a:b] = 1, [p["lam"], p["b"], p["a"], p["c"]]
    return pd.DataFrame(dict(d=d, t_plus=tp, rv=x1["rv"].to_numpy(float), X1=X1, F=X1 * np.exp(g), g=g, active=act, lam=par[:, 0], b=par[:, 1], a=par[:, 2], c=par[:, 3], G=D["G"], n_ext=nx))[OUT_COLS]
