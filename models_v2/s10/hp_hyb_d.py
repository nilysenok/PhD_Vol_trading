"""P2, G9/G10, архитектура D (У-11): Confidence-Gated поверх A1.
Ḡ_{i,t} — среднее по закрытым 63 парам бумаги [QLIKE(K1) − QLIKE(A1 при λ = 1)] (A1 при выбранных на год глубине и c).
D1: λ_{i,t} = clip(a + b·Ḡ, 0, 1), a ∈ {0; 0,25; 0,5}, b ∈ {0; 5; 20; 50} — раз в год по закрытым прогнозам Y−1; f = K1·exp(λ·clip(g_A1)); нет Ḡ — Ḡ = 0; первый год — λ = 1 (a = 1, b = 0).
D2: λ = σ(w·z), z = (Ḡ, разброс g по 5 бэггинг-копиям A1 (seed 42…46, тот же модуль), z ln RV_d/RV_m IMOEX, ln RV_d/RV_w бумаги, доля одного знака g за 22 строки); w — ридж-логистическая регрессия (C = 1)
цели «A1 при λ = 1 лучше K1» на закрытых парах (t⁺ < 01.01.Y), признаки — винзоризация по 1/99 процентилям обучающих строк (У-1) и стандартизация; нет z в строке — 0. λ = 0 ≡ K1 побитно."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from models_v2.s10 import hp_engine as E, hp_hyb_core as C

GAIN_WIN, SIGN_WIN, MIN_TRAIN = 63, 22, 500
A_GRID, B_GRID, DEFAULT_AB = (0.0, 0.25, 0.5), (0.0, 5.0, 20.0, 50.0), (1.0, 0.0)
ZCOLS = ["gbar", "disp", "z_m_imo_d_m", "d_w", "sign_share"]


def gain(rv, k1, f1) -> np.ndarray:
    """QLIKE(K1) − QLIKE(A1 при λ = 1) на пару (выигрыш поправки); нефинитно — NaN."""
    ok = np.isfinite(rv) & (rv > 0) & np.isfinite(k1) & (k1 > 0) & np.isfinite(f1) & (f1 > 0)
    return np.where(ok, E.qlike_rows(np.where(ok, rv, 1.0), np.where(ok, k1, 1.0)) - E.qlike_rows(np.where(ok, rv, 1.0), np.where(ok, f1, 1.0)), np.nan)


def gain_a1(panel: pd.DataFrame, k1, g, c) -> np.ndarray:
    """Выигрыш A1 при λ = 1 (глубина и c выбраны на год строки) на пару; нет g (модель не обучена) — NaN, а не 0."""
    f1 = np.where(np.isfinite(g), C.apply_lam(k1, g, c, 1.0), np.nan)
    return gain(panel["rv_tplus"].to_numpy(float), np.asarray(k1, float), f1)


def gbar(panel: pd.DataFrame, gn: np.ndarray, win: int = GAIN_WIN) -> np.ndarray:
    """Ḡ на строку — среднее выигрыша по последним win ЗАКРЫТЫМ парам бумаги (t⁺ <= d); нет пар — NaN. Мутант «с незакрытым днём» подменяет `closed_counts` ядра."""
    return C.trail_agg(np.where(np.isfinite(gn), gn, 0.0), C.trail_index(panel, np.isfinite(gn), win))[0]


def lam_d1(a: float, b: float, gb: np.ndarray) -> np.ndarray:
    return np.clip(a + b * np.nan_to_num(gb, nan=0.0), 0.0, 1.0)


def select_ab(panel, k1, g, c, gb, Y: int) -> dict:
    """(a, b) на год Y по закрытым прогнозам Y−1; критерий — средний по бумагам QLIKE."""
    m = E.closed_mask(panel["date"], panel["t_plus"], Y)
    tk, rv = panel["ticker"].to_numpy(), panel["rv_tplus"].to_numpy(float)
    cands = [(C.mean_q(rv, C.apply_lam(k1, g, c, lam_d1(a, b, gb)), tk, m), dict(a=a, b=b)) for a, b in itertools.product(A_GRID, B_GRID)]
    return C.pick_min(cands, dict(a=DEFAULT_AB[0], b=DEFAULT_AB[1]))


def finalize_d1(panel, k1, g, c, gb, years=None) -> tuple:
    """-> (σ̂² D1 по строкам панели — NaN до 2016, таблица выбора (a, b))."""
    yr = pd.DatetimeIndex(panel["date"]).year.to_numpy()
    out, ch = np.full(len(panel), np.nan), []
    for Y in sorted(set(yr[yr >= E.FIRST_YEAR])) if years is None else years:
        s = select_ab(panel, k1, g, c, gb, Y)
        ch.append({"year": int(Y), **s})
        m = yr == Y
        out[m] = C.apply_lam(k1, g, c, lam_d1(s["a"], s["b"], gb))[m]
    return out, pd.DataFrame(ch)


# --- D2 ---
def bag_module(module: str) -> str:
    """Модуль бэггинг-копий = модуль конфигурации (XGB и LGBM не смешиваются). Мутант «другой модуль» подменяет эту функцию."""
    return module


def bag_runs(module: str, panel, feats, base, panel_sha: str, depths, nthread: int = 4, log=None, hp=None) -> list:
    """Копии A1 seed 43…46 (подвыборка 0,8 — HP движка; hp — только тесты): [{глубина: walk_forward}] — только нужные глубины."""
    return [C.cached_runs(bag_module(module), "bag%d" % s, panel, feats, base, panel_sha, "qlike", {**(hp or {}), "seed": s}, tuple(depths), nthread, log) for s in C.SEEDS_BAG]


def dispersion(g_a1: np.ndarray, g_bags: list) -> np.ndarray:
    """Стандартное отклонение (ddof = 1) g по 5 копиям; хотя бы одна нет g — NaN."""
    G = np.column_stack([g_a1] + list(g_bags))
    ok = np.isfinite(G).all(axis=1)
    out = np.full(len(G), np.nan)
    out[ok] = G[ok].std(axis=1, ddof=1)
    return out


def sign_share(panel: pd.DataFrame, g: np.ndarray, win: int = SIGN_WIN) -> np.ndarray:
    """Доля одного знака g за последние win строк бумаги, включая d (≥ 5 значений); g на дату d известна к d."""
    pos = pd.Series(np.where(np.isfinite(g), (g > 0).astype(float), np.nan))
    p = pos.groupby(panel["ticker"].to_numpy()).transform(lambda s: s.rolling(win, min_periods=5).mean()).to_numpy()
    return np.maximum(p, 1.0 - p)


def zmatrix(panel, gb, disp, share) -> np.ndarray:
    return np.column_stack([gb, disp, panel["z_m_imo_d_m"].to_numpy(float), (panel["x_d_m"] - panel["x_w_m"]).to_numpy(float), share])


def train_mask(panel: pd.DataFrame, Y: int) -> np.ndarray:
    """Пары, закрытые к 01.01.Y (t⁺ < 01.01.Y). Мутант «закрытые + 1» подменяет эту функцию."""
    return (pd.DatetimeIndex(panel["t_plus"]) < pd.Timestamp(Y, 1, 1))


def fit_w(Z: np.ndarray, y: np.ndarray) -> dict:
    """Ридж-логистика (C = 1) по винзоризованным 1/99 и стандартизованным признакам; вырожденный случай (один класс, мало строк) — ValueError с причиной."""
    from sklearn.linear_model import LogisticRegression
    if len(y) < MIN_TRAIN or len(np.unique(y)) < 2:
        raise ValueError("D2: вырожденное обучение логистики: строк %d, классов %d" % (len(y), len(np.unique(y))))
    lo, hi = np.percentile(Z, 1, axis=0), np.percentile(Z, 99, axis=0)
    W = np.clip(Z, lo, hi)
    mu, sd = W.mean(axis=0), W.std(axis=0)
    if not (sd > 0).all():
        raise ValueError("D2: постоянный признак в обучении (sd = 0): %s" % [c for c, s in zip(ZCOLS, sd) if not s > 0])
    m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000).fit((W - mu) / sd, y)
    return dict(lo=lo, hi=hi, mu=mu, sd=sd, coef=m.coef_[0], intercept=float(m.intercept_[0]))


def lam_d2(Z: np.ndarray, w: dict) -> np.ndarray:
    """λ = σ(w·z) (стандартизованные винзоризованные признаки; NaN — 0)."""
    S = np.nan_to_num((np.clip(Z, w["lo"], w["hi"]) - w["mu"]) / w["sd"], nan=0.0)
    return 1.0 / (1.0 + np.exp(-(S @ w["coef"] + w["intercept"])))


def finalize_d2(panel, k1, g, c, Z, gn, years=None) -> tuple:
    """-> (σ̂² D2 по строкам панели, таблица w по годам). Год без обучающих пар (первый) — λ = 1 (R8)."""
    yr = pd.DatetimeIndex(panel["date"]).year.to_numpy()
    out, ch = np.full(len(panel), np.nan), []
    for Y in sorted(set(yr[yr >= E.FIRST_YEAR])) if years is None else years:
        tr = train_mask(panel, Y) & np.isfinite(Z).all(axis=1) & np.isfinite(gn)
        m = yr == Y
        if tr.sum() == 0:
            lam, row = np.ones(len(panel)), dict(source="default", n_train=0)
        else:
            w = fit_w(Z[tr], (gn[tr] > 0).astype(int))
            lam = lam_d2(Z, w)
            row = dict(source="closed", n_train=int(tr.sum()), intercept=w["intercept"], **{"w_" + k: float(v) for k, v in zip(ZCOLS, w["coef"])})
        ch.append({"year": int(Y), **row})
        out[m] = C.apply_lam(k1, g, c, lam)[m]
    return out, pd.DataFrame(ch)
