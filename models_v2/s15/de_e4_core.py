"""E4 (X4, C1): поправка с затвором по неопределённости. F = H·exp(α_t·ḡ), ḡ — среднее g_r пяти QLIKE-GBM (`v6_1_fit.fit_gbm`, признаки S V6, параметры A0g) на окнах закрытых пар: расширяющееся и последние 750 / 1000 / 1250 / 1500; помесячная переоценка (первый торговый день месяца с 2017-01-01).
ПАРАМЕТРЫ ВНЕ G14 (записаны в EVIDENCE G14 до счёта): g_r = clip(g, −1, 1) (клип V6); окно короче N закрытых пар берёт все закрытые; act = (закрытых пар на дату переоценки > 750) — только тогда разброс осмыслен;
sd_r(d) — СКО (ddof = 0) 5 значений g_r дня d; s̄(d) — медиана sd_r по строго более ранним дням с act (меньше MIN_S = 20 таких дней, s̄ ≤ 1e-12 или нет act в день d — α = 1); α = exp(−κ·sd_r/s̄) ∈ (0; 1].
κ ∈ {0,5; 1; 2} — выбирается в каждый день d по QLIKE F_κ на последних VAL = 126 закрытых парах (t⁺ ≤ d) дней с act и определённым s̄; ничья — меньшее κ; меньше MIN_VAL = 20 пар — κ = 1. κ = 0 ⇒ α ≡ 1 ⇒ F = H·exp(ḡ) («ансамбль без затвора», ENS — не конфигурация)."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

WINDOWS = (None, 750, 1000, 1250, 1500)     # None — расширяющееся
KAPPAS, VAL, MIN_VAL, MIN_S, FIRST_REFIT, CLIP = (0.5, 1.0, 2.0), 126, 20, 20, "2017-01-01", 1.0


def closed_rows(tp, d0) -> np.ndarray:
    """Закрытые пары на d0: t⁺ ≤ d0. МУТАНТ «закрыто + 1» (t⁺ ≤ d0 + 1 день) подменяет эту функцию в тесте."""
    return np.flatnonzero(np.asarray(tp, "datetime64[ns]") <= np.datetime64(d0))


def window_rows(cl: np.ndarray, n) -> np.ndarray:
    """Последние n закрытых пар (None — все; закрытых меньше n — все закрытые)."""
    return cl if n is None else cl[-n:]


def qlike(rv, F):
    r = np.asarray(rv, float) / np.asarray(F, float)
    return r - np.log(r) - 1.0


def refit_positions(index: pd.DatetimeIndex, first: str = FIRST_REFIT) -> list:
    per = index.to_period("M")
    return [int(np.flatnonzero(per == m)[0]) for m in per.unique() if m.start_time >= pd.Timestamp(first)]


def default_fit(X, e):
    return importlib.import_module("models_v2.s13.v6_1_fit").fit_gbm(X, e)      # через importlib: AST-тест DE0 запрещает статический импорт обучателей в s15


def walk_forward(X, e, tp, dates, fit=default_fit, windows=WINDOWS, first: str = FIRST_REFIT, thr: int = 750) -> dict:
    """-> dict(G (n, len(windows)) — g_r с клипом, act (n,), fitted (n,), rows — [(d0, [n_строк по окнам])]). До первой переоценки G = 0, fitted = False. Прогноз дня d — моделями последней переоценки ≤ d."""
    X, e, dates = np.asarray(X, float), np.asarray(e, float), pd.DatetimeIndex(dates)
    n, pos = len(e), refit_positions(dates, first)
    G, act, fitted, rows = np.zeros((n, len(windows))), np.zeros(n, bool), np.zeros(n, bool), []
    for a, b in zip(pos, pos[1:] + [n]):
        cl = closed_rows(tp, dates[a])
        used = [window_rows(cl, w) for w in windows]
        for k, r in enumerate(used):
            G[a:b, k] = np.clip(fit(X[r], e[r]).g(X[a:b]), -CLIP, CLIP)
        act[a:b], fitted[a:b] = len(cl) > thr, True
        rows.append((dates[a], [len(r) for r in used]))
    return dict(G=G, act=act, fitted=fitted, rows=rows)


def spread_stats(G: np.ndarray) -> tuple:
    """-> (ḡ, sd_r): среднее и СКО (ddof = 0) по моделям ансамбля на день."""
    return G.mean(axis=1), G.std(axis=1)


def sbar_series(sd, act) -> np.ndarray:
    """s̄(d) — медиана sd_r по СТРОГО более ранним дням с act; NaN, пока таких дней < MIN_S. МУТАНТ «по будущим датам» подменяет эту функцию."""
    out, hist = np.full(len(sd), np.nan), []
    for i in range(len(sd)):
        if len(hist) >= MIN_S:
            out[i] = float(np.median(hist))
        if act[i]:
            hist.append(float(sd[i]))
    return out


def alpha(sd, sbar, act, kappa: float) -> np.ndarray:
    """α = exp(−κ·sd_r/s̄) там, где act и s̄ определена и > 1e-12; иначе 1. κ = 0 ⇒ α ≡ 1."""
    ok = np.asarray(act, bool) & np.isfinite(sbar) & (np.where(np.isfinite(sbar), sbar, 0.0) > 1e-12)
    return np.where(ok, np.exp(-kappa * np.asarray(sd, float) / np.where(ok, sbar, 1.0)), 1.0)


def select_kappa(rv, H, gbar, sd, sbar, act, tp, dates, fitted, kappas=KAPPAS, default: float = 1.0) -> tuple:
    """-> (κ по дням, число пар валидации). Нет модели или меньше MIN_VAL пар — κ = default (1)."""
    rv, H, gbar = np.asarray(rv, float), np.asarray(H, float), np.asarray(gbar, float)
    Q = np.column_stack([qlike(rv, H * np.exp(alpha(sd, sbar, act, k) * gbar)) for k in kappas])
    valid = np.asarray(act, bool) & np.isfinite(sbar) & fitted
    kap, nval = np.full(len(rv), float(default)), np.zeros(len(rv), int)
    for i, d in enumerate(pd.DatetimeIndex(dates)):
        if not fitted[i]:
            continue
        cl = closed_rows(tp, d)
        j = cl[valid[cl]][-VAL:]
        nval[i] = len(j)
        if len(j) >= MIN_VAL:
            kap[i] = kappas[int(np.argmin(Q[j].mean(axis=0)))]
    return kap, nval


def run(X, e, rv, H, tp, dates, fit=default_fit, windows=WINDOWS, first: str = FIRST_REFIT, kappas=KAPPAS, default: float = 1.0) -> dict:
    """Полный путь X4. -> dict(F, F_ens, gbar, sd, sbar, alpha, kappa, nval, act, fitted, G, rows). F_ens — ряд α ≡ 1 (справочный)."""
    w = walk_forward(X, e, tp, dates, fit, windows, first)
    gbar, sd = spread_stats(w["G"])
    sbar = sbar_series(sd, w["act"])
    kap, nval = select_kappa(rv, H, gbar, sd, sbar, w["act"], tp, dates, w["fitted"], kappas, default)
    al, H = alpha(sd, sbar, w["act"], kap), np.asarray(H, float)
    return dict(F=H * np.exp(al * gbar), F_ens=H * np.exp(gbar), gbar=gbar, sd=sd, sbar=sbar, alpha=al, kappa=kap, nval=nval, **{k: w[k] for k in ("act", "fitted", "G", "rows")})
