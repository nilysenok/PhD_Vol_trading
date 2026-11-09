"""V7-DX, G40 (г): дублирование информации HAR-J и бустингов.
1) Частные корреляции (по годам 2018–2025, внутри года одна система координат z: энкодер года, seeds 0–4) компонент z и поправки G (сохранённые прогнозы 4 и 6, у 1 — g) с ln H, ℓ = ln(M/H), последними ЗАКРЫТЫМИ ln(RV/K2) и ln(RV/H):
   частная корреляция с каждой из четырёх переменных при остальных трёх зафиксированных (линейно, со свободным членом); для z — среднее |частной корреляции| по компонентам и seeds (знак компоненты произволен) и R² компоненты на всех четырёх.
2) PIT-регрессия: ln(RV/K2) на z — расширяющееся окно, помесячная переоценка (позиции `refit_positions`), обучение только на закрытых парах t⁺ ≤ d0 (`closed`), гребень (штраф из {1; 10; 100; 1000} по последним 126 закрытым парам), z — энкодера года d0.
   R² вне выборки по годам t⁺ 2018–2025 = 1 − SSE/SST, SST — вокруг среднего ln(RV/K2) закрытых пар (прогноз «без z»); по seeds — среднее (z разных seeds не выровнены). Ж-237 п. 20.
3) Разрыв конфигураций 4 и 6 к K2 по годам: Δ_y = QLIKE(cfg) − QLIKE(K2) на датах с t⁺ в году y; вклад года = n_y·Δ_y / Σn; доля 2022 в суммарном разрыве 2018–2025 и разрыв без 2022."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s13 import v6_1_wf as W6
from models_v2.s14 import v7_1b_cfg as CFG, v7_1m_feat as FE, v7_1m_wf as WF, v7_dx_core as K

ALPHAS, HOLD, MIN_TRAIN, MIN_ROWS = (1.0, 10.0, 100.0, 1000.0), 126, 60, 20
YEARS = tuple(range(2018, 2026))


def closed(tp, d0) -> np.ndarray:
    """Закрытые пары на d0: t⁺ ≤ d0. МУТАНТ «обучение на будущих датах» (все пары) подменяет эту функцию в тесте."""
    return W6.closed_rows(tp, d0)


def _resid(a: np.ndarray, C: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(C)), C])
    return a - A @ np.linalg.lstsq(A, a, rcond=None)[0]


def partial_corrs(v: np.ndarray, C: np.ndarray) -> list:
    """Частная корреляция v с каждым столбцом C при остальных; плюс R² v на всех столбцах C."""
    out = []
    for j in range(C.shape[1]):
        o = np.delete(C, j, axis=1)
        out.append(float(np.corrcoef(_resid(v, o), _resid(C[:, j], o))[0, 1]))
    r = _resid(v, C)
    return out + [float(1.0 - r.var() / v.var())]


def controls(tab: pd.DataFrame) -> tuple:
    """-> (C (n, 4) = ln H, ℓ, ln(RV/K2) и ln(RV/H) последней закрытой пары, валидные строки)."""
    H, K2, K3, rv = (tab[c].to_numpy(float) for c in ("H", "K2", "K3", "rv_next"))
    j = FE.last_closed(tab["t_plus"].to_numpy("datetime64[ns]"), tab.index.to_numpy("datetime64[ns]"))
    pk = lambda a: np.where(j >= 0, a[np.clip(j, 0, None)], np.nan)
    C = np.column_stack([np.log(H), np.log(np.sqrt(K2 * K3) / H), pk(np.log(rv / K2)), pk(np.log(rv / H))])
    return C, np.isfinite(C).all(1)


CNAMES = ("ln_H", "ell", "last_ln_RV_K2", "last_ln_RV_H")


def pcorr_table(inp: dict) -> pd.DataFrame:
    tab = inp["tab"]
    C, okC = controls(tab)
    yr = tab.index.year.to_numpy()
    zs = {s: {y: CFG.z_of_year(inp["kind"], s, tab.index, y) for y in YEARS} for s in range(5)}
    rows = []
    for y in YEARS:
        m = okC & (yr == y)
        for s in range(5):
            Z = zs[s][y][m]
            for k in range(4):
                pc = partial_corrs(Z[:, k], C[m])
                rows.append(dict(var="z%d" % (k + 1), year=y, seed=s, **{"pc_" + n: abs(p) for n, p in zip(CNAMES, pc)}, r2_on_controls=pc[-1]))
        for cfg in K.CFGS:
            sv = pd.read_parquet(K.SAVED[cfg]).set_index("d")[K.GCOL[cfg]]
            g = sv.reindex(tab.index).to_numpy(float)
            mg = m & np.isfinite(g)
            pc = partial_corrs(g[mg], C[mg])
            rows.append(dict(var="G_cfg" + cfg, year=y, seed=-1, **{"pc_" + n: p for n, p in zip(CNAMES, pc)}, r2_on_controls=pc[-1]))
    return pd.DataFrame(rows)


def pit_ridge(y: np.ndarray, z: np.ndarray, tp: np.ndarray, index: pd.DatetimeIndex, pos: list, first_eval: str = WF.FIRST_EVAL) -> pd.DataFrame:
    """Одна матрица z на все переоценки (синтетика тестов); боевой путь — `pit_blocks` с z энкодера года переоценки."""
    return pit_blocks(y, lambda d0: z, tp, index, pos, first_eval)


def pit_blocks(y: np.ndarray, zfun, tp: np.ndarray, index: pd.DatetimeIndex, pos: list, first_eval: str = WF.FIRST_EVAL) -> pd.DataFrame:
    """-> по каждой дате d ≥ first_eval: y, прогноз ŷ с z, прогноз «без z» (среднее закрытых), строка переоценки. zfun(d0) -> (n, 4) z для переоценки на d0."""
    n, i0, out = len(y), int(np.searchsorted(index.to_numpy("datetime64[ns]"), np.datetime64(first_eval))), []
    for a, b in zip(pos, pos[1:] + [n]):
        if b <= i0:
            continue
        lo, d0, z = max(a, i0), index[a], zfun(index[a])
        cl = closed(tp, d0)
        cl = cl[np.isfinite(z[cl]).all(1) & np.isfinite(y[cl])]
        if len(cl) < MIN_TRAIN + HOLD:
            continue
        va, tr = cl[-HOLD:], cl[:-HOLD]
        mu, sd = z[cl].mean(0), z[cl].std(0)
        A = (z - mu) / np.where(sd > 1e-12, sd, 1.0)

        def fit(rows, al):
            m = y[rows].mean()
            return m, np.linalg.solve(A[rows].T @ A[rows] + al * np.eye(A.shape[1]), A[rows].T @ (y[rows] - m))
        best = min(ALPHAS, key=lambda al: float(np.mean((y[va] - (lambda mb: mb[0] + A[va] @ mb[1])(fit(tr, al))) ** 2)))
        m, beta = fit(cl, best)
        for i in range(lo, b):
            if np.isfinite(A[i]).all() and np.isfinite(y[i]):
                out.append(dict(i=i, y=y[i], pred=m + A[i] @ beta, naive=float(y[cl].mean()), alpha=best))
    return pd.DataFrame(out)


def r2_by_year(df: pd.DataFrame, tp_years: np.ndarray) -> dict:
    out = {}
    for yv in sorted(set(tp_years[df["i"].to_numpy()])):
        m = tp_years[df["i"].to_numpy()] == yv
        sse, sst = float(np.sum((df["y"][m] - df["pred"][m]) ** 2)), float(np.sum((df["y"][m] - df["naive"][m]) ** 2))
        out[int(yv)] = 1.0 - sse / sst if sst > 0 else float("nan")
    return out


def pit_table(inp: dict) -> pd.DataFrame:
    tab = inp["tab"]
    y = np.log(tab["rv_next"].to_numpy(float) / tab["K2"].to_numpy(float))
    tp = tab["t_plus"].to_numpy("datetime64[ns]")
    ty, pos, rows = pd.DatetimeIndex(tab["t_plus"]).year.to_numpy(), W6.refit_positions(tab.index, WF.FIRST_REFIT), []
    for s in range(5):
        cache = {}
        zf = lambda d0, s=s: cache.setdefault(d0.year, CFG.z_of_year(inp["kind"], s, tab.index, d0.year))
        r2 = r2_by_year(pit_blocks(y, zf, tp, tab.index, pos), ty)
        rows += [dict(seed=s, year_t_plus=yy, r2_oos=v) for yy, v in r2.items() if yy in YEARS]
    return pd.DataFrame(rows)


def gap_table(inp: dict) -> pd.DataFrame:
    """Разрыв конфигураций 4 и 6 к K2 (и 1 справочно) по годам t⁺; вклад года и доля 2022."""
    tab, rows = inp["tab"], []
    for cfg in K.CFGS:
        sv = pd.read_parquet(K.SAVED[cfg]).set_index("d")
        t = tab.reindex(sv.index)
        ty = pd.DatetimeIndex(sv["t_plus"]).year.to_numpy()
        ds = {y: float(np.mean(metrics.qlike(sv["rv"][ty == y], sv["F"][ty == y])) - np.mean(metrics.qlike(sv["rv"][ty == y], t["K2"][ty == y]))) for y in YEARS}
        n = {y: int((ty == y).sum()) for y in YEARS}
        tot = sum(n[y] * ds[y] for y in YEARS) / sum(n.values())
        for y in YEARS:
            c = n[y] * ds[y] / sum(n.values())
            rows.append(dict(cfg=cfg, year=y, n=n[y], delta_vs_K2=ds[y], contribution=c, share_of_total=c / tot if tot != 0 else float("nan"), total_gap=tot,
                             gap_without_2022=(sum(n[v] * ds[v] for v in YEARS if v != 2022) / sum(n[v] for v in YEARS if v != 2022))))
    return pd.DataFrame(rows)


def tables(inp: dict) -> dict:
    return dict(PCORR=pcorr_table(inp), PIT=pit_table(inp), GAP=gap_table(inp))
