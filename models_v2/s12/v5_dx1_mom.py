"""V5, лист 1b, G16: условные моментные ошибки m_k = mean(s·φ_k), k = 0…6 (φ₀ = 1), s = RV/F − 1 для баз K8 и M на M16; окно — по году цели t⁺.
Панель: среднее по бумагам за дату цели -> ряд по датам; t по Ньюи–Уэсту (лаг 5, Бартлетт, центрированный ряд — как `v5_0_dx_ols.nw_ols`); совместный тест Вальда по вектору (m₀…m₆): W = n·m'S⁻¹m ~ χ²(rank S).
Правило (п. 6.3, объявлено): направление 3 считается, если p Вальда < 0,01 на 2017–2021 для базы K8; иначе закрывается. 2022–2025 показывается, в решение не входит. Диагностика задним числом."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from models_v2.s12 import v5_dx1_phi as PH

LAG = 5
P_RULE = 0.01
BASES = {"K8": "K8", "M": "M"}
WINDOWS = {"2017_2021": (2017, 2021), "2022_2025": (2022, 2025)}
YEARS = list(range(2017, 2026))
NAMES = ["const"] + ["phi%d" % i for i in range(1, 7)]


def date_series(c: pd.DataFrame, phi: pd.DataFrame, col: str) -> pd.DataFrame:
    """Ряд по датам цели: U[d, k] = среднее по бумагам s·φ_k (k = 0…6). Строки с любой нерасчётной φ отбрасываются (один набор строк для всех k и баз)."""
    d = c.merge(phi, on=["ticker", "date"], how="inner").dropna(subset=PH.PHI)
    s = (d["rv_tplus"] / d[col] - 1.0).to_numpy()
    x = np.column_stack([np.ones(len(d))] + [d[p].to_numpy() for p in PH.PHI]) * s[:, None]
    return pd.DataFrame(x, columns=NAMES, index=pd.DatetimeIndex(d["t_plus"])).groupby(level=0).mean().sort_index()


def nw_cov(G: np.ndarray, lag: int = LAG) -> np.ndarray:
    """Долгосрочная ковариация (Бартлетт) центрированных рядов G (n×k): S = Γ0 + Σ w_l(Γl + Γl'), Γ = G'G/n."""
    G = G - G.mean(axis=0)
    n = len(G)
    S = G.T @ G / n
    for l in range(1, min(lag, n - 1) + 1):
        A = G[l:].T @ G[:-l] / n
        S += (1.0 - l / (lag + 1.0)) * (A + A.T)
    return S


def wald(U: pd.DataFrame, lag: int = LAG) -> dict:
    """-> m (k), t (k), W, df, p, n. Нужно n ≥ 20, иначе всё NaN."""
    k = U.shape[1]
    n = len(U)
    if n < 20:
        return dict(m=np.full(k, np.nan), t=np.full(k, np.nan), W=np.nan, df=0, p=np.nan, n=n)
    x = U.to_numpy(float)
    m = x.mean(axis=0)
    S = nw_cov(x, lag)
    t = m / np.sqrt(np.maximum(np.diag(S) / n, 1e-300))
    df = int(np.linalg.matrix_rank(S))
    W = float(n * m @ np.linalg.pinv(S) @ m)
    return dict(m=m, t=t, W=W, df=df, p=float(stats.chi2.sf(W, df)), n=n)


def table(c: pd.DataFrame, phi: pd.DataFrame) -> pd.DataFrame:
    """DX1_MOMENTS: по базе (K8, M) × окну (годы 2017…2025, 2017_2021, 2022_2025) — строки m_k, t_nw, и строка Вальда (item = wald)."""
    rows = []
    for base, col in BASES.items():
        U = date_series(c, phi, col)
        wins = {str(y): (y, y) for y in YEARS} | WINDOWS
        for win, (lo, hi) in wins.items():
            r = wald(U[(U.index.year >= lo) & (U.index.year <= hi)])
            rows += [dict(base=base, window=win, item=nm, m=r["m"][i], t_nw=r["t"][i], wald=np.nan, df=np.nan, p_wald=np.nan, n_dates=r["n"]) for i, nm in enumerate(NAMES)]
            rows.append(dict(base=base, window=win, item="wald", m=np.nan, t_nw=np.nan, wald=r["W"], df=r["df"], p_wald=r["p"], n_dates=r["n"]))
    return pd.DataFrame(rows)


def decision(t: pd.DataFrame) -> dict:
    """Правило п. 6.3: база K8, окно 2017_2021, p < 0,01 -> «считать», иначе «закрыть»; крупнейший |t| среди m₁…m₆ (и всех m₀…m₆) называется."""
    g = t[(t["base"] == "K8") & (t["window"] == "2017_2021")]
    p = float(g[g["item"] == "wald"]["p_wald"].iloc[0])
    mm = g[g["item"] != "wald"].set_index("item")
    big = mm["t_nw"].abs().idxmax()
    big_phi = mm.loc[mm.index != "const", "t_nw"].abs().idxmax()
    return dict(verdict="считать" if p < P_RULE else "закрыть", p=p, big=big, big_t=float(mm.loc[big, "t_nw"]), big_phi=big_phi, big_phi_t=float(mm.loc[big_phi, "t_nw"]))
