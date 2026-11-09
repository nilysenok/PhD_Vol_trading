"""E2 (X2 = D2): признаки и метка стресса. Чистые функции от массивов, файлов нет.
Вход логистики p (11 колонок, день d, известны на закрытии d): Z V7 (9) + e_d (экстремальный M10-бар, определение E1) + доля отрицательной семивариации дня d (Σr² баров с r < 0 / Σr² баров дня).
«Доля скачка» плана — колонка jshare уже внутри Z (дубля нет). Вход c_n, c_s — Z (первые 9 колонок). Метка стресса — ТОЛЬКО в потере: RV(t⁺) > PIT-порог (90-й перцентиль RV по закрытым парам t⁺ ≤ d0 переоценки);
метка не входит в признаки (будущее событие разрешено в обучении, Ж-232 п. 8)."""
from __future__ import annotations

import numpy as np

ZNAMES = ["ln_H", "ell", "ln_K2_K3", "ewma_dq", "last_err_H", "last_err_M", "pct_lnRV", "jshare", "cusum_errH"]
FEATURES = ZNAMES + ["e_d", "rs_neg_share"]
NZ, NP, Q_STRESS = len(ZNAMES), len(FEATURES), 0.9


def rs_neg_share(store: dict, dates) -> np.ndarray:
    """Доля Σr² баров M10 дня d с r < 0 в Σr² дня d; день без баров — NaN (в обучение не входит)."""
    pos = store["days"].get_indexer(dates)
    if (pos < 0).any():
        raise SystemExit("в складе M10 нет дней таблицы")
    r = store["r"][pos]
    fin = np.isfinite(r)
    r0 = np.where(fin, r, 0.0)
    tot, neg = (r0 ** 2).sum(1), np.where(r0 < 0, r0 ** 2, 0.0).sum(1)
    return np.where(fin.any(1) & (tot > 0), neg / np.where(tot > 0, tot, 1.0), np.nan)


def day_inputs(Z: np.ndarray, e: np.ndarray, rsn: np.ndarray) -> np.ndarray:
    """(n, 11): Z, e_d, rs_neg_share. МУТАНТ «метка во входе» подменяет эту функцию в тесте."""
    return np.column_stack([np.asarray(Z, float), np.asarray(e, float), np.asarray(rsn, float)])


def stress_threshold(rv: np.ndarray, closed: np.ndarray) -> float:
    """90-й перцентиль RV(t⁺) по ЗАКРЫТЫМ парам (индексы `closed`, t⁺ ≤ d0). МУТАНТ «с будущими парами» берёт все пары."""
    return float(np.quantile(np.asarray(rv, float)[closed], Q_STRESS))


def stress_label(rv: np.ndarray, thr: float) -> np.ndarray:
    return (np.asarray(rv, float) > thr).astype(float)
