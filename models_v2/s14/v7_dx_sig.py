"""V7-DX, G38 (б): есть ли сигнал вообще.
1) В выборке против вне выборки (конфигурации 1, 4, 6): на ПОСЛЕДНЕЙ переоценке каждого года Y — снижение QLIKE к H (в % QLIKE(H)) ансамбля 5 seeds на обучающих парах (tr ∪ va, закрытые t⁺ ≤ d0);
   вне выборки — то же на датах с t⁺ в году Y + 1 из СОХРАНЁННЫХ прогнозов (новых прогнозов нет). Выводы — по годам разработки 2018–2021 (Y = 2017…2020); остальные годы — справочно.
2) Верхняя оценка ЗАДНИМ ЧИСЛОМ («ex post, не для отбора»): гребень ln(RV/H) на z энкодера 2021 (TCN-w5, одна система координат, seeds 0–4) по датам с t⁺ 2018–2021; 5 блоков подряд по датам (в разработке 4 года —
   блок «по годам» 5-кратно невозможен, отступление: 5 равных последовательных блоков), R² вне блока относительно среднего обучающих блоков; частный R² = R²(z, Z) − R²(Z). Сетка гребня {1; 10; 100; 1000} объявлена, сообщается каждая точка."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_1b_cfg as CFG, v7_dx_core as K

ALPHAS = (1.0, 10.0, 100.0, 1000.0)
N_BLOCKS, DEV = 5, (2018, 2021)


def const_control(full: pd.DataFrame) -> dict:
    """Контроль без модели (otbor_statistika п. 2а; добавлен после первого просмотра, правила вывода не менялись): F = H·e^c, e^c = среднее rv/H закрытых пар (QLIKE-оптимальный множитель), переоценка помесячно (`refit_positions`).
    -> {год Y: (снижение в выборке на закрытых парах последней переоценки Y, % ; снижение вне выборки на t⁺ в году Y + 1, %)}. МУТАНТ «множитель по всем парам» — `closed` из `v7_dx_dup`."""
    from models_v2.s13 import v6_1_wf as W6
    from models_v2.s14 import v7_1m_wf as WF, v7_dx_dup as DUP
    H, rv, tp = full["H"].to_numpy(float), full["rv_next"].to_numpy(float), full["t_plus"].to_numpy("datetime64[ns]")
    pos, F, last = W6.refit_positions(full.index, WF.FIRST_REFIT), np.full(len(H), np.nan), {}
    for a, b in zip(pos, pos[1:] + [len(H)]):
        cl = DUP.closed(tp, full.index[a])
        if len(cl) < WF.MIN_TRAIN:
            continue
        c = float(np.mean(rv[cl] / H[cl]))
        F[a:b] = H[a:b] * c
        last[full.index[a].year] = K.q_red(rv[cl], H[cl], H[cl] * c)[2]
    ty, ok = pd.DatetimeIndex(full["t_plus"]).year.to_numpy(), np.isfinite(F) & (full.index >= pd.Timestamp(WF.FIRST_EVAL))
    return {y: (red, K.q_red(rv[ok & (ty == y + 1)], H[ok & (ty == y + 1)], F[ok & (ty == y + 1)])[2] if (ok & (ty == y + 1)).sum() > 20 else float("nan")) for y, red in last.items()}


def in_vs_out(cfg: str, rt: pd.DataFrame, const: dict) -> pd.DataFrame:
    """rt — `K.refit_table`; сохранённый прогноз читается из `K.SAVED`; const — `const_control` на тех же строках, что у конфигурации (для 1 — с прогревом H)."""
    sv = pd.read_parquet(K.SAVED[cfg])
    ty = pd.DatetimeIndex(sv["t_plus"]).year.to_numpy()
    out = []
    for y, g in rt.groupby("year"):
        last = g.sort_values("d0").iloc[-1]
        m = ty == y + 1
        row = dict(cfg=cfg, year=int(y), d0=str(last["d0"].date()), n_in=int(last["n_pairs"]), red_in_pct=float(last["red_in_pct"]), dev_window=bool(DEV[0] <= y + 1 <= DEV[1]))
        red = K.q_red(sv["rv"].to_numpy(float)[m], sv["H"].to_numpy(float)[m], sv["F"].to_numpy(float)[m])[2] if m.sum() > 20 else float("nan")
        ci, co = const.get(int(y), (float("nan"), float("nan")))
        row.update(year_out=int(y + 1), n_out=int(m.sum()), red_out_pct=red, const_red_in_pct=ci, const_red_out_pct=co)
        out.append(row)
    return pd.DataFrame(out)


def _std(A: np.ndarray, tr: np.ndarray) -> np.ndarray:
    mu, sd = A[tr].mean(0), A[tr].std(0)
    return (A - mu) / np.where(sd > 1e-12, sd, 1.0)


def ridge_cv(X: np.ndarray, y: np.ndarray, alpha: float, blocks: int = N_BLOCKS) -> float:
    """R² вне блока: 1 − ΣSSE/ΣSST, где SST — вокруг среднего обучающих блоков; признаки стандартизуются по обучающим блокам. Блоки — последовательные равные части."""
    n, edges = len(y), np.linspace(0, len(y), blocks + 1).astype(int)
    sse = sst = 0.0
    for b in range(blocks):
        te = np.arange(edges[b], edges[b + 1])
        tr = np.setdiff1d(np.arange(n), te)
        A = _std(X, tr)
        mu = y[tr].mean()
        beta = np.linalg.solve(A[tr].T @ A[tr] + alpha * np.eye(A.shape[1]), A[tr].T @ (y[tr] - mu))
        sse += float(np.sum((y[te] - mu - A[te] @ beta) ** 2))
        sst += float(np.sum((y[te] - mu) ** 2))
    return 1.0 - sse / sst


def ex_post_rows(y: np.ndarray, z: dict, Z: np.ndarray) -> pd.DataFrame:
    """z — {seed: (n, 4)}; Z (n, 9) — признаки режима. Все массивы — на одних датах (без NaN). -> по seed и alpha: R²(Z), R²(z), R²(z, Z), частный R² = R²(z, Z) − R²(Z)."""
    rows = []
    for a in ALPHAS:
        rz = ridge_cv(Z, y, a)
        for s, zz in z.items():
            r_z, r_zZ = ridge_cv(zz, y, a), ridge_cv(np.hstack([zz, Z]), y, a)
            rows.append(dict(seed=s, alpha=a, r2_Z=rz, r2_z=r_z, r2_zZ=r_zZ, partial_r2=r_zZ - rz))
    return pd.DataFrame(rows)


def ex_post_table(inp: dict) -> pd.DataFrame:
    tab = inp["tab"]
    ty = pd.DatetimeIndex(tab["t_plus"]).year.to_numpy()
    z = {s: CFG.z_of_year(inp["kind"], s, tab.index, 2021) for s in range(5)}
    ok = (ty >= DEV[0]) & (ty <= DEV[1]) & np.isfinite(inp["Z"]).all(1) & np.all([np.isfinite(v).all(1) for v in z.values()], axis=0)
    y = np.log(tab["rv_next"].to_numpy(float) / tab["H"].to_numpy(float))[ok]
    df = ex_post_rows(y, {s: v[ok] for s, v in z.items()}, inp["Z"][ok])
    df.insert(0, "n_dates", int(ok.sum()))
    return df


def tables(inp: dict, rts: dict) -> dict:
    const = {"1": const_control(CFG.with_warm(inp["tab"], inp["warm"])), "4": const_control(inp["tab"])}
    const["6"] = const["4"]
    return dict(SIG=pd.concat([in_vs_out(c, rts[c], const[c]) for c in K.CFGS], ignore_index=True), SIG_EXPOST=ex_post_table(inp))
