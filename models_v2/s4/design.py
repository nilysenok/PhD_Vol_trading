"""Дизайны Ж-200: форма цели (уровень / Б-2 / прирост / остаток к HAR-J(б)) × признаки (база / + рынок)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models_v2.s2 import features as ft
from models_v2.s3 import resid, scale
from models_v2.s4 import mkt

FORMS = ("level", "m", "delta", "resid")
LN_MKT = ("proxy_d", "proxy_w", "proxy_m", "avg_d", "avg_w", "avg_m", "imoex_d", "imoex_w", "imoex_m")  # минус ln S в Б-2
NEED = (0, 1, 2, 3)


@dataclass
class Dz:
    X: np.ndarray
    y: np.ndarray     # y[t] — цель пары (x_t, y_t)
    off: np.ndarray   # ln σ̂² = off[t] + f (+ s²/2)
    mono: tuple | None = None


def make(F: ft.Fp, form: str, M=None, lb: np.ndarray | None = None) -> Dz:
    """M — DataFrame рыночных признаков (или None); lb — PIT ln σ̂²_HAR-J(б) (для формы resid)."""
    zero = np.zeros(len(F.rv))
    if form == "m":
        S = scale.scale(F, "m")
        X, ln_s = scale.normalize(F.X, S), np.log(S)
        y, off = resid.target_minus(F, ln_s), ln_s
    else:
        X = F.X[:, :ft.N_BASE].copy()
        off = {"level": zero, "delta": F.lnrv, "resid": lb}[form]
        if form == "resid" and lb is None:
            raise ValueError("форме resid нужен lb")
        y = resid.target_minus(F, off)
    if M is not None:
        cols = []
        for c in M.columns:
            v = M[c].to_numpy(float)
            cols.append(v - off_ln(F) if (form == "m" and c in LN_MKT) else v)
        X = np.column_stack([X] + cols)
    return Dz(X, y, off)


def off_ln(F: ft.Fp) -> np.ndarray:
    return np.log(scale.scale(F, "m"))


def mono_vec(n_cols: int) -> tuple:
    return tuple(1 if j in ft.MONO else 0 for j in range(n_cols))


def columns(M, which: str):
    """Подбор столбцов рынка: 'proxy' — mkt.BASE, 'imoex' — BASE + IDX."""
    if M is None or which == "none":
        return None
    return M[list(mkt.BASE) + (list(mkt.IDX) if which == "imoex" else [])]
