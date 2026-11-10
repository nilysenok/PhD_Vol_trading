"""Лист ZE (Ж-247 п. 4), признак разрыва G_d и предусловие дубля (план ZE, разд. 3.2, 3.5; PIT). Файлов не пишет.
G_d = ln RV_d − ln X1_d: RV_d — Σ r² присутствующих баров дня d склада M10 V7 (`x1_z_feat.store_stats`, не таблица пар: в ней нет строки t⁺ = 2022-04-29, а у d = 2022-04-29 e_d = 1); X1_d — колонка F строки d.
RV_d ≤ 0 или X1_d ≤ 0: G = 0, счётчик в отчёте. ZEp: G_{d−252} циклическим сдвигом по ряду общих дат (np.roll, как ZCp).
Предусловие: на парах e_d = 1 с t⁺ ≤ 2021-12-31 |corr(G, v_d)| и |corr(G, s_d)| ≤ 0,8, иначе ZE — дубль (не регистрируется и не считается)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_z_feat as ZF

SHIFT, LIMIT, DEV_END = ZF.SHIFT, 0.8, pd.Timestamp("2021-12-31")


def rv_row(p):
    """Строка склада RV дня d (мутация: p + 1 — RV дня t⁺ в признаке)."""
    return p


def x1_row(i):
    """Строка ряда X1 прогноза дня d (мутация: i + 1 — X1 со строки t⁺)."""
    return i


def gap(st: dict, d, X1) -> tuple:
    """-> (G, число дней с вырожденными RV_d / X1_d, где G = 0)."""
    pos = st["days"].get_indexer(pd.DatetimeIndex(d))
    if (pos < 0).any():
        raise SystemExit("день d вне склада M10")
    rv, X1 = ZF.store_stats(st)[0], np.asarray(X1, float)
    r = rv[np.minimum(rv_row(pos), len(rv) - 1)]
    x = X1[np.minimum(x1_row(np.arange(len(pos))), len(X1) - 1)]
    ok = (r > 0) & (x > 0)
    G = np.zeros(len(pos))
    G[ok] = np.log(r[ok]) - np.log(x[ok])
    return G, int((~ok).sum())


def shift_g(G) -> np.ndarray:
    """ZEp: G_{d−252} циклическим сдвигом (мутация: без сдвига)."""
    return np.roll(np.asarray(G, float), SHIFT)


def corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def precondition(G, v, s, x, e, tp) -> dict:
    """Корреляции G с v_d, s_d (и справочно x_d) на парах e_d = 1 с t⁺ ≤ 2021-12-31. ok = max |corr(G, v), corr(G, s)| ≤ 0,8."""
    m = (np.asarray(e) == 1) & (pd.DatetimeIndex(tp) <= DEV_END)
    cv, cs, cx = corr(G[m], np.asarray(v)[m]), corr(G[m], np.asarray(s, float)[m]), corr(G[m], np.asarray(x)[m])
    mx = max(abs(cv), abs(cs))
    return dict(n=int(m.sum()), corr_G_v=cv, corr_G_s=cs, corr_G_x_info=cx, max_abs=mx, limit=LIMIT, ok=bool(mx <= LIMIT), g_mean=float(G[m].mean()), g_std=float(G[m].std()))
