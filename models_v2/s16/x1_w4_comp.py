"""W4-B2 (Ж-250; `docs/otchety/harj_x1/W4_SPEC.md`, разделы 2, 4): ступень 1 — компонент M⁺ = M·exp(clip(ρ·ĝ(x), ±1)). Ядро: таблица, признаки, обучатели, помесячный проход.
Выходы, статистика, решение, `--check`, `--check-decision` — `x1_w4_comp_run.py`, `x1_w4_comp_stat.py` (запуск: `python3 -W ignore -m models_v2.s16.x1_w4_comp [--check|--check-decision]`).

ПУБЛИЧНОЕ API (мутации подкладываются через monkeypatch ИМЕННО этих имён модуля `x1_w4_comp`; внутри ядра они вызываются по глобальному имени):
- `load_table() -> DataFrame` (index d; t_plus, rv, M = √(K2·K3)): все пары K2/K3 «архив + продолжение», с t⁺ 2017-01-03; `in_window(tab) -> bool[]` — окно M (t⁺ ≥ 2018-01-04).
- `feature_source() -> (S, Sigma)` — ИСТОЧНИК признаков (`x1_w4_feat.read_ext`): S s1…s9, Σ 12 площадей, index d. `shifted(X, k)` — СДВИГ для K_b: x_d → x_{d−k} по кругу (`np.roll`) на ПОЛНОЙ таблице признаков.
- `closed(t_plus, d0) -> позиции` — ОТБОР закрытых пар t⁺ ≤ d0 (делегирует `v6_1_wf.closed_rows`; мутация «закрыто + 1» подменяет `closed` или `WF.closed_rows`).
- `select_rho_trees(X, e) -> (n_trees, rho)` — ВЫБОР ρ и деревьев по последним 252 закрытым парам (`FT.split_es`: блок ранней остановки, мутация № 99); `pick_rho(e_es, g_es) -> rho` — argmin QLIKE по RHOS, ничья |Δ| ≤ 1e-12 — меньшее ρ.
- `fit_final(X, e, n_trees) -> booster` — финальная модель на `FT.final_idx` (мутация № 100).
- `gbm_learner(X, e) -> (n_trees, rho, predict)`; `const_learner(X, e)` — K_a: g = ln mean(RV/M) по закрытым парам, ρ — то же правило; `const_g(e)`.
- `mplus(M, g, rho) -> F` = M·exp(clip(ρ·g, ±1)), при ρ = 0 побитно M.
- `walk(tab, X, learner=gbm_learner) -> DataFrame` (d, t_plus, rv, M, g, rho, n_trees, refit_date, F) по ВСЕМ строкам таблицы; до первой переоценки с ≥ 312 закрытыми парами g = 0, ρ = 0.
- `NoFit` — обучатель не строится (обучающих строк < 100): переоценка пропущена, g = 0 (отступление: так на 2018-04-02 и 2018-05-02).
- `first_refit(tab) -> Timestamp` — первый d₀ с ≥ 312 закрытыми парами; `job(spec) -> (spec, DataFrame окна)`, spec = (имя ∈ {S1, S2, Ka}, сдвиг)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401  (torch первым)
from models_v2.s13 import v6_1_fit as FT
from models_v2.s13 import v6_1_wf as WF
from models_v2.s13 import v6_1b_sig as SG
from models_v2.s16 import x1_w3_core as W3
from models_v2.s16 import x1_w4_feat as FEAT

RHOS, TIE, SHIFT = (0.0, 0.25, 0.5, 0.75, 1.0), 1e-12, 252
PERM_SHIFTS = tuple(SHIFT + 100 * (j - 1) for j in range(1, 20))
WINDOW_START, N_WINDOW = pd.Timestamp("2018-01-04"), 2198
S_COLS = ["s%d" % i for i in range(1, 10)]
OUT_COLS = ["d", "t_plus", "rv", "M", "g", "rho", "n_trees", "refit_date", "F"]


def load_table() -> pd.DataFrame:
    """Пары (d, t⁺): K2/K3 архива (t⁺ ≤ 2026-02-27) + продолжения; M = √(K2·K3); дубль d — остановка."""
    ka, kn = W3._rd(W3.SRC["k23_a"], {"date": "d"}), W3._rd(W3.SRC["k23_n"], {"date": "d"})
    k = pd.concat([ka[ka["t_plus"] <= W3.ARCH_END], kn], ignore_index=True).sort_values("d").reset_index(drop=True)
    if k["d"].duplicated().any():
        raise SystemExit("k23: дубль d")
    k["M"] = np.sqrt(k["sigma2_k2"] * k["sigma2_k3"])
    k = k[np.isfinite(k["rv"]) & np.isfinite(k["M"]) & (k["M"] > 0) & (k["rv"] > 0)]
    return k.set_index("d")[["t_plus", "rv", "M"]]


def in_window(tab: pd.DataFrame) -> np.ndarray:
    return (tab["t_plus"] >= WINDOW_START).to_numpy()


def feature_source() -> tuple:
    S, G = FEAT.read_ext()
    return S[S_COLS], G[list(SG.SIGMA_NAMES)]


def shifted(X: pd.DataFrame, k: int) -> pd.DataFrame:
    """x_d → x_{d−k} по кругу на полной таблице признаков (строка i получает строку i − k). Цель e не сдвигается."""
    return pd.DataFrame(np.roll(X.to_numpy(float), k, axis=0), index=X.index, columns=X.columns)


def closed(t_plus, d0) -> np.ndarray:
    return WF.closed_rows(t_plus, d0)


def pick_rho(e_es, g_es) -> float:
    """ρ = argmin среднего QLIKE(clip(ρ·g)) на проверочных парах; ничья (|Δ| ≤ 1e-12) — меньшее ρ."""
    L = [float(FT.qlike_g(e_es, FT.clip_g(r * np.asarray(g_es, float))).mean()) for r in RHOS]
    return min(r for r, v in zip(RHOS, L) if v <= min(L) + TIE)


class NoFit(Exception):
    """Обучатель LightGBM не может строить дерево: на обучающей части (закрытые пары без последних 252) меньше 2·min_data_in_leaf строк (`Check failed: num_features > 0`)."""


def select_rho_trees(X: np.ndarray, e: np.ndarray) -> tuple:
    """Модель на закрытых парах без последних 252 (`FT.split_es`), ≤ 300 деревьев; лучшая итерация по QLIKE(clip g) на этих 252; ρ — на той же модели при лучшей итерации."""
    tr, es = FT.split_es(len(e))
    if len(tr) < 2 * FT.GBM_PARAMS["min_data_in_leaf"]:
        raise NoFit("обучающих строк %d < %d" % (len(tr), 2 * FT.GBM_PARAMS["min_data_in_leaf"]))
    b = FT._train(X[tr], e[tr], FT.MAX_TREES)
    sc = [float(FT.qlike_g(e[es], FT.clip_g(b.predict(X[es], num_iteration=k, raw_score=True))).mean()) for k in range(1, FT.MAX_TREES + 1)]
    best = int(np.argmin(sc)) + 1
    return best, pick_rho(e[es], b.predict(X[es], num_iteration=best, raw_score=True))


def fit_final(X: np.ndarray, e: np.ndarray, n_trees: int):
    fi = FT.final_idx(len(e))
    return FT._train(X[fi], e[fi], n_trees)


def gbm_learner(X: np.ndarray, e: np.ndarray) -> tuple:
    k, rho = select_rho_trees(X, e)
    b = fit_final(X, e, k)
    return k, rho, lambda Z: b.predict(np.asarray(Z, float), raw_score=True)


def const_g(e) -> float:
    """QLIKE-оптимальная константа: ln mean(RV/M) = ln mean(exp(e))."""
    return float(np.log(np.mean(np.exp(np.asarray(e, float)))))


def const_learner(X: np.ndarray, e: np.ndarray) -> tuple:
    tr, es = FT.split_es(len(e))
    rho = pick_rho(e[es], np.full(len(es), const_g(e[tr])))
    c = const_g(e[FT.final_idx(len(e))])
    return 0, rho, lambda Z: np.full(len(Z), c)


def mplus(M, g, rho) -> np.ndarray:
    """M⁺ = M·exp(clip(ρ·g, ±1)); при ρ = 0 exp(0) = 1 и M⁺ = M побитно."""
    return np.asarray(M, float) * np.exp(FT.clip_g(np.asarray(rho, float) * np.asarray(g, float)))


def first_refit(tab: pd.DataFrame):
    tp = tab["t_plus"].to_numpy()
    for a in WF.refit_positions(tab.index):
        if len(closed(tp, tab.index[a])) >= FT.MIN_TRAIN:
            return tab.index[a]
    return None


def walk(tab: pd.DataFrame, X: np.ndarray, learner=gbm_learner) -> pd.DataFrame:
    """Переоценка в первый торговый день месяца d₀ по парам t⁺ ≤ d₀; прогноз g месяца — моделью d₀. Нет ≥ 312 пар — g = 0, ρ = 0 (M⁺ = M)."""
    e = np.log(tab["rv"].to_numpy(float) / tab["M"].to_numpy(float))
    tp, n = tab["t_plus"].to_numpy(), len(tab)
    g, rho, nt, rd = np.zeros(n), np.zeros(n), np.zeros(n, int), np.full(n, np.datetime64("NaT"), "datetime64[ns]")
    pos = WF.refit_positions(tab.index)
    for a, b in zip(pos, pos[1:] + [n]):
        d0 = tab.index[a]
        tr = closed(tp, d0)
        if len(tr) < FT.MIN_TRAIN:
            continue
        try:
            k, r, pred = learner(X[tr], e[tr])
        except NoFit:     # отступление W4-B2 №1: переоценки 2018-04-02, 2018-05-02 (313 и 335 пар) — g = 0
            continue
        g[a:b], rho[a:b], nt[a:b], rd[a:b] = pred(X[a:b]), r, k, np.datetime64(d0)
    out = pd.DataFrame(dict(d=tab.index, t_plus=tab["t_plus"].to_numpy(), rv=tab["rv"].to_numpy(float), M=tab["M"].to_numpy(float), g=g, rho=rho, n_trees=nt, refit_date=rd))
    out["F"] = mplus(out["M"], out["g"], out["rho"])
    return out[OUT_COLS]


def job(spec: tuple) -> tuple:
    """spec = (S1 | S2 | Ka, сдвиг; 0 — без сдвига) -> (spec, строки окна M). Читает реестр не здесь, а в `x1_w4_comp_run.require_all`."""
    name, shift = spec
    tab = load_table()
    if name == "Ka":
        df = walk(tab, np.zeros((len(tab), 1)), const_learner)
    else:
        S, G = feature_source()
        X = S if name == "S1" else S.join(G)
        X = (shifted(X, shift) if shift else X).reindex(tab.index)
        if X.isna().any().any():
            raise SystemExit("признаки: нет строки на %d датах пар" % int(X.isna().any(axis=1).sum()))
        df = walk(tab, X.to_numpy(float))
    return spec, df[in_window(tab)].reset_index(drop=True)


if __name__ == "__main__":
    import sys

    from models_v2.s16 import x1_w4_comp_run as RUN
    sys.exit(RUN.main(sys.argv[1:]))
