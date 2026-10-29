"""Оси К25–К26 (v6, итерация 24): условия по внешнему фактору f ∈ {brent, usdrub, rvi} (контракт #376 §2Б.6).

z_f,d = изменение f за день d / σ_f,d−1 (σ — rv22 по 10-минуткам f, на d−1; `change` из `vneshnie_1850`).
β_t,f = МНК r_t на изменение f за d−252…d−1 (окно без дня d).
Условия (vn_uslovie): net (выкл.: все True — точка v6 = точка v5), spokojno_1 |z| ≤ 1, spokojno_2 |z| ≤ 2,
po_bete sign(β·z) = сторона, protiv_bety sign(β·z) = −сторона.
Нет значения f или β (дыра, прогрев) — вход разрешён (Д4, без ffill), как в контракте §2.8.
Константы SIGMA_LAG, BETA_LAG — для тестов-мутаций (подмена в памяти): 0 = утечка дня d.
Данные не читает: на входе готовые ряды.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

USLOVIYA = ("net", "spokojno_1", "spokojno_2", "po_bete", "protiv_bety")
FAKTORY = ("brent", "usdrub", "rvi")
WINDOW, MIN_N = 252, 126
SIGMA_LAG = 1
BETA_LAG = 1


def z_f(dx: pd.Series, sigma: pd.Series) -> pd.Series:
    """z = dx_d / σ_{d−1}; sigma — rv22 фактора по дням (выровнен на d), берётся вчерашняя."""
    return dx / sigma.shift(SIGMA_LAG).reindex(dx.index)


def beta_f(r: pd.Series, dx: pd.Series, window: int = WINDOW, min_n: int = MIN_N) -> pd.Series:
    """Наклон МНК r на dx по последним `window` парам, заканчивающимся на d−1; меньше min_n пар — NaN."""
    x, y = dx.reindex(r.index), r
    ok = x.notna() & y.notna()
    x, y = x.where(ok), y.where(ok)
    cov = x.rolling(window, min_periods=min_n).cov(y)
    var = x.rolling(window, min_periods=min_n).var()
    return (cov / var.where(var > 0)).shift(BETA_LAG)


def uslovie(name: str, z: pd.Series, beta: pd.Series, side) -> pd.Series:
    """Разрешён ли вход в день d (bool). side — +1 лонг / −1 шорт (скаляр или ряд). Неизвестное — True."""
    if name not in USLOVIYA:
        raise ValueError(f"vn_uslovie: {name!r} не из {USLOVIYA}")
    if name == "net":
        return pd.Series(True, index=z.index)
    unknown = z.isna() | (beta.reindex(z.index).isna() if not name.startswith("spokojno") else False)
    if name.startswith("spokojno"):
        ok = z.abs() <= (1.0 if name == "spokojno_1" else 2.0)
    else:
        sgn = pd.Series(np.sign(beta.reindex(z.index) * z), index=z.index)
        ok = sgn == (side if name == "po_bete" else -side)
    return (ok | unknown).astype(bool)
