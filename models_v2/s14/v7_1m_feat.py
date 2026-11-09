"""V7-1a-M, признаки режима Z (9, план п. 3), PIT. Таблица `tab`: индекс — дата d; H (K1), K2, K3, rv_next (цель, RV дня t⁺), t_plus, rv_d (RV дня d, известна на закрытии d), jshare (J/RV дня d).
Закрытая пара на дату d — пара с t⁺ ≤ d; «последняя закрытая» — последняя такая строка (`last_closed`). Признаки по прошлым ошибкам берутся ТОЛЬКО из закрытых пар:
 0 ln H; 1 ℓ = ln(M/H), M = √(K2·K3); 2 ln(K2/K3); 3 EWMA₂₀ (полураспад 20 пар) разности QLIKE(H) − QLIKE(M); 4 ln(RV/H) последней закрытой пары; 5 ln(RV/M) последней закрытой пары;
 6 перцентиль ln RV_d среди ln RV строго ДО дня d (не меньше 20 наблюдений); 7 J/RV дня d; 8 CUSUM прошлых лог-ошибок H: `v4_cld_brk.cusum` (определение V4 H6, Ж-225: порог 5, дрейф 0,5, окно 252 строго до, ≥ 60) от ряда ln(RV/H) закрытых пар по датам t⁺.
Отличие от V4 H6: там CUSUM строился по медиане ln RV 17 бумаг; здесь, по плану V7, — по лог-ошибкам H одной бумаги (SBER). Стандартизация Z — не здесь, а в walk-forward по закрытым парам."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_cld_brk as BK

ZCOLS = ["ln_H", "ell", "ln_K2_K3", "ewma_dq", "last_err_H", "last_err_M", "pct_lnRV", "jshare", "cusum_errH"]
HALFLIFE, MIN_PCT = 20, 20


def last_closed(t_plus: np.ndarray, d) -> np.ndarray:
    """Позиция последней закрытой пары (t⁺ ≤ d) для каждой даты d; −1, если закрытых нет. МУТАНТ «текущая пара» (позиция строки d) подменяет эту функцию в тесте."""
    return np.searchsorted(np.asarray(t_plus, "datetime64[ns]"), np.asarray(d, "datetime64[ns]"), side="right") - 1


def pct_past(x: np.ndarray, min_obs: int = MIN_PCT) -> np.ndarray:
    """Доля прошлых значений (строго до строки i) меньше x[i]; NaN при < min_obs конечных."""
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        p = x[:i][np.isfinite(x[:i])]
        if len(p) >= min_obs and np.isfinite(x[i]):
            out[i] = float(np.mean(p < x[i]))
    return out


def build_Z(tab: pd.DataFrame) -> pd.DataFrame:
    H, K2, K3 = (tab[c].to_numpy(float) for c in ("H", "K2", "K3"))
    rv, tp = tab["rv_next"].to_numpy(float), tab["t_plus"].to_numpy("datetime64[ns]")
    M = np.sqrt(K2 * K3)
    eH, eM = np.log(rv / H), np.log(rv / M)
    dq = (rv / H - eH - 1.0) - (rv / M - eM - 1.0)                       # QLIKE(H) − QLIKE(M) пары j
    ew = pd.Series(dq).ewm(halflife=HALFLIFE, adjust=False).mean().to_numpy()
    cu = BK.cusum(pd.Series(eH, index=pd.DatetimeIndex(tp)))[0].to_numpy()
    j = last_closed(tp, tab.index.to_numpy("datetime64[ns]"))
    pick = lambda a: np.where(j >= 0, a[np.clip(j, 0, None)], np.nan)
    out = pd.DataFrame({"ln_H": np.log(H), "ell": np.log(M / H), "ln_K2_K3": np.log(K2 / K3), "ewma_dq": pick(ew), "last_err_H": pick(eH), "last_err_M": pick(eM),
                        "pct_lnRV": pct_past(np.log(tab["rv_d"].to_numpy(float))), "jshare": tab["jshare"].to_numpy(float), "cusum_errH": pick(cu)}, index=tab.index)
    return out[ZCOLS]
