"""P1b, G8: движок панельной поправки (R4, R6, R8; У-6, У-7; Ж-217). Им пользуются K4/K5 (hp_ctrl) и гибриды P2.
Вход: таблица строк (бумага i, дата d) с t_plus, rv_tplus; `base` — уровень базы Ĥ (>0; K1 σ̂² или RV_m), признаки, функция потерь, модуль (xgb|lgbm), исключаемые бумаги (LOTO), seed.
offset = ln(base) (`base_margin` XGB / `init_score` LGBM); итог σ̂² = base·exp(λ·(clip(g, ±c) [+ s²_g/2 при l2log])) — множитель, а не exp(ln base + …): при λ = 0 σ̂² ≡ base побитно.
QLIKE: L = ln f + RV/f, f = exp(offset + g), grad = 1 − RV/f, hess = RV/f. l2log (только A2): цель ln RV, σ̂² = Ĥ·exp(g + s²_g/2), s²_g — дисперсия остатка на блоке ранней остановки.
Месячное переобучение (первый торговый день месяца панели) на расширяющемся окне: пары (j, s) всех бумаг с s⁺ <= d (R4); последние 126 дат окна (по s⁺) — блок ранней остановки, ВНЕ обучения (№ 99).
Глубина {2; 4} — два прохода walk-forward (`walk_forward`); c, λ — при прогнозе (`apply_corr`); выбор на год Y — по закрытым прогнозам года Y−1 (t⁺ < 01.01.Y), средний по бумагам QLIKE (`choose`, №102)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

DEPTHS, CS, LAMS = (2, 4), (0.5, 1.0, 2.0), (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT = (2, 1.0, 1.0)      # R8: глубина 2, c = 1, λ = 1 (первый год без закрытой валидации)
FIRST_YEAR = 2016
HP = dict(lr=0.03, rounds=800, patience=50, min_leaf=500, l2=10.0, subsample=0.8, colsample=0.8, seed=42, es_dates=126, min_train=1000)


def qlike_gh(margin, rv):
    """grad, hess QLIKE по margin = offset + g."""
    r = np.asarray(rv, float) / np.exp(margin)
    return 1.0 - r, r


def qlike_value(margin, rv) -> float:
    return float(np.mean(margin + np.asarray(rv, float) / np.exp(margin)))


# --- крючки правил (мутации подменяют их в тестах; боевой путь вызывает только их) ---
def pair_closed(t_plus: np.ndarray, d) -> np.ndarray:
    """R4: пара (j, s) годна к обучению на дату d, если s⁺ <= d."""
    return t_plus <= d


def split_es(t_plus: np.ndarray, n_dates: int) -> tuple:
    """(обучение, ранняя остановка): ES — строки с s⁺ в последних n_dates датах окна; в обучение не входят (№ 99)."""
    last = np.unique(t_plus)[-n_dates:]
    es = np.isin(t_plus, last)
    return ~es, es


def kept(tk: np.ndarray, exclude) -> np.ndarray:
    """LOTO (R11): бумаги из exclude в обучение не входят."""
    return ~np.isin(tk, list(exclude)) if len(exclude) else np.ones(len(tk), bool)


def closed_mask(date, t_plus, Y: int) -> np.ndarray:
    """Закрытые прогнозы года Y−1: d в Y−1 и t⁺ < 01.01.Y (№ 102)."""
    return (pd.DatetimeIndex(date).year == Y - 1) & (pd.DatetimeIndex(t_plus) < pd.Timestamp(Y, 1, 1))


def apply_corr(base, g, s2g, c: float, lam: float, loss: str) -> np.ndarray:
    """σ̂² = base·exp(λ·(clip(g,±c) [+ s²_g/2])); нет g (модель не обучена) — поправка 0."""
    gc = np.clip(np.nan_to_num(np.asarray(g, float), nan=0.0), -c, c)
    if loss == "l2log":
        gc = gc + 0.5 * np.nan_to_num(np.asarray(s2g, float), nan=0.0)
    return np.asarray(base, float) * np.exp(lam * gc)


# --- модели ---
def _xgb(depth, loss, tr, es, hp, nth):
    import xgboost as xgb
    p = dict(max_depth=depth, eta=hp["lr"], min_child_weight=hp["min_leaf"], reg_lambda=hp["l2"], subsample=hp["subsample"], colsample_bytree=hp["colsample"],
             seed=hp["seed"], tree_method="hist", nthread=nth)
    kw = {}
    if loss == "qlike":
        p["disable_default_eval_metric"] = 1
        kw = dict(obj=lambda m, dm: qlike_gh(m, dm.get_label()), custom_metric=lambda m, dm: ("qlike", qlike_value(m, dm.get_label())))
    else:
        p.update(objective="reg:squarederror", eval_metric="rmse")
    lab = (lambda y: y) if loss == "qlike" else np.log
    dtr = xgb.DMatrix(tr[0], label=lab(tr[1]), base_margin=tr[2])
    des = xgb.DMatrix(es[0], label=lab(es[1]), base_margin=es[2])
    b = xgb.train(p, dtr, hp["rounds"], evals=[(des, "es")], early_stopping_rounds=hp["patience"], verbose_eval=False, **kw)
    n = b.best_iteration + 1
    return lambda X: b.predict(xgb.DMatrix(X, base_margin=np.zeros(len(X))), iteration_range=(0, n), output_margin=True).astype(float), n


def _lgbm(depth, loss, tr, es, hp, nth):
    import lightgbm as lgb
    p = dict(max_depth=depth, num_leaves=2 ** depth, learning_rate=hp["lr"], min_data_in_leaf=hp["min_leaf"], lambda_l2=hp["l2"], bagging_fraction=hp["subsample"], bagging_freq=1,
             feature_fraction=hp["colsample"], seed=hp["seed"], deterministic=True, force_row_wise=True, num_threads=nth, verbosity=-1)
    fe = None
    if loss == "qlike":
        p.update(objective=lambda m, d: qlike_gh(m, d.get_label()), metric="None")
        fe = lambda m, d: ("qlike", qlike_value(m, d.get_label()), False)
    else:
        p.update(objective="regression", metric="l2")
    lab = (lambda y: y) if loss == "qlike" else np.log
    dtr = lgb.Dataset(tr[0], label=lab(tr[1]), init_score=tr[2], free_raw_data=False)
    des = lgb.Dataset(es[0], label=lab(es[1]), init_score=es[2], reference=dtr)
    b = lgb.train(p, dtr, hp["rounds"], valid_sets=[des], feval=fe, callbacks=[lgb.early_stopping(hp["patience"], verbose=False)])
    n = max(b.best_iteration, 1)
    return lambda X: b.predict(X, num_iteration=n).astype(float), n


def fit_model(module, depth, loss, tr, es, hp, nthread):
    """tr, es = (X, RV_{t⁺}, offset). -> (g(X), s²_g, число деревьев). Ранняя остановка по es; финальная модель — лучшая итерация (без дообучения на es)."""
    pred, n = {"xgb": _xgb, "lgbm": _lgbm}[module](depth, loss, tr, es, hp, nthread)
    s2g = float(np.var(np.log(es[1]) - es[2] - pred(es[0]), ddof=1)) if loss == "l2log" else 0.0
    return pred, s2g, n


def train_sets(tp, tk, y, base, d, exclude, hp) -> tuple:
    """Индексы (обучение, ES) на дату d: s⁺ <= d, бумаги не из exclude, RV > 0, base > 0."""
    ok = pair_closed(tp, d) & kept(tk, exclude) & np.isfinite(y) & (y > 0) & np.isfinite(base) & (base > 0)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return idx, idx
    a, b = split_es(tp[idx], hp["es_dates"])
    return idx[a], idx[b]


def month_starts(dates: np.ndarray, first_year: int) -> list:
    """Первый торговый день каждого месяца панели (по объединению дат бумаг) с first_year."""
    s = pd.Series(pd.DatetimeIndex(np.unique(dates)))
    s = s[s.dt.year >= first_year]
    return list(s.groupby([s.dt.year, s.dt.month]).min().to_numpy())


def walk_forward(panel: pd.DataFrame, feats: list, base, loss="qlike", module="xgb", depth=2, exclude=(), hp=None, nthread=6, first_year=FIRST_YEAR, log=None) -> pd.DataFrame:
    """Один проход walk-forward для глубины `depth`. panel: ticker, date, t_plus, rv_tplus (+ feats). -> g, s2g, n_train, n_trees, fit_sec, retrain на строку (NaN — модель не обучена)."""
    hp = {**HP, **(hp or {})}
    n = len(panel)
    date, tp, tk = (panel[c].to_numpy("datetime64[ns]") if c != "ticker" else panel[c].to_numpy() for c in ("date", "t_plus", "ticker"))
    y, base = panel["rv_tplus"].to_numpy(float), np.asarray(base, float)
    X = panel[feats].to_numpy(float)
    off = np.log(np.where(base > 0, base, np.nan))
    out = {k: np.full(n, np.nan) for k in ("g", "s2g", "n_train", "n_trees", "fit_sec")}
    out["retrain"] = np.full(n, np.datetime64("NaT"), "datetime64[ns]")
    starts = month_starts(date, first_year)
    for k, d in enumerate(starts):
        span = (date >= d) & ((date < starts[k + 1]) if k + 1 < len(starts) else True)
        itr, ies = train_sets(tp, tk, y, base, d, exclude, hp)
        if len(itr) < hp["min_train"] or len(ies) == 0:
            continue
        t0 = time.time()
        pred, s2g, nt = fit_model(module, depth, loss, (X[itr], y[itr], off[itr]), (X[ies], y[ies], off[ies]), hp, nthread)
        sec = time.time() - t0
        out["g"][span] = pred(X[span])
        out["s2g"][span], out["n_train"][span], out["n_trees"][span], out["fit_sec"][span], out["retrain"][span] = s2g, len(itr), nt, sec, d
        if log:
            log("%s %s d%d %s: train %d es %d trees %d %.1fs" % (module, loss, depth, str(d)[:10], len(itr), len(ies), nt, sec))
    return pd.DataFrame(out, index=panel.index)


def mean_by_ticker(q: np.ndarray, tk: np.ndarray) -> float:
    """Средний по бумагам QLIKE (среднее по датам внутри бумаги, затем по бумагам); нефинитные строки пропускаются."""
    ok = np.isfinite(q)
    if not ok.any():
        return float("nan")
    u, inv = np.unique(tk[ok], return_inverse=True)
    return float(np.mean(np.bincount(inv, weights=q[ok]) / np.bincount(inv)))


def qlike_rows(rv, s2) -> np.ndarray:
    r = np.asarray(rv, float) / np.asarray(s2, float)
    return r - np.log(r) - 1.0


def choose(panel: pd.DataFrame, base, runs: dict, Y: int, loss: str, cs=CS, lams=LAMS, default=DEFAULT) -> dict:
    """Выбор (глубина, c, λ) на год Y по закрытым прогнозам года Y−1 (t⁺ < 01.01.Y). Нет закрытых прогнозов — default. runs: {глубина: walk_forward}."""
    m = closed_mask(panel["date"], panel["t_plus"], Y)
    best = dict(depth=default[0], c=default[1], lam=default[2], crit=float("nan"), n=int(m.sum()), source="default")
    tk, rv, bs = panel["ticker"].to_numpy()[m], panel["rv_tplus"].to_numpy(float)[m], np.asarray(base, float)[m]
    for depth in sorted(runs):
        g, s2 = runs[depth]["g"].to_numpy()[m], runs[depth]["s2g"].to_numpy()[m]
        fin = np.isfinite(g)
        for c in cs:
            for lam in lams:
                q = mean_by_ticker(qlike_rows(rv[fin], apply_corr(bs[fin], g[fin], s2[fin], c, lam, loss)), tk[fin])
                if np.isfinite(q) and (best["source"] == "default" or q < best["crit"]):
                    best = dict(depth=depth, c=c, lam=lam, crit=q, n=int(fin.sum()), source="closed %d" % (Y - 1))
    return best


def finalize(panel: pd.DataFrame, base, runs: dict, loss="qlike", years=None, cs=CS, lams=LAMS, default=DEFAULT) -> tuple:
    """-> (таблица прогнозов: ticker, date, t_plus, depth, c, lam, g, s2g, sigma2, n_train, n_trees, fit_sec; таблица выбора по годам). Строки до first_year не выдаются."""
    base = np.asarray(base, float)
    yr = pd.DatetimeIndex(panel["date"]).year.to_numpy()
    sig, dep, cc, ll, g, s2, ntr, ntt, sec = (np.full(len(panel), np.nan) for _ in range(9))
    ch = []
    for Y in sorted(set(yr[yr >= FIRST_YEAR]) if years is None else years):
        sel = choose(panel, base, runs, Y, loss, cs, lams, default)
        ch.append({"year": int(Y), **sel})
        m, r = yr == Y, runs[sel["depth"]]
        g[m], s2[m] = r["g"].to_numpy()[m], r["s2g"].to_numpy()[m]
        ntr[m], ntt[m], sec[m] = r["n_train"].to_numpy()[m], r["n_trees"].to_numpy()[m], r["fit_sec"].to_numpy()[m]
        dep[m], cc[m], ll[m] = sel["depth"], sel["c"], sel["lam"]
        sig[m] = apply_corr(base[m], g[m], s2[m], sel["c"], sel["lam"], loss)
    out = panel[["ticker", "date", "t_plus"]].assign(depth=dep, c=cc, lam=ll, g=g, s2g=s2, sigma2=sig, n_train=ntr, n_trees=ntt, fit_sec=sec)
    return out[np.isfinite(sig)].reset_index(drop=True), pd.DataFrame(ch)
