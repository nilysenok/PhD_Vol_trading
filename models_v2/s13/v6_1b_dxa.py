"""V6-1b, G19: диагностика DX-A «площади сверх спектра» — ЗАДНИМ ЧИСЛОМ, НЕ ВЫБОР: никакой её результат не используется для выбора признаков или параметров A1–A3. Модуль не импортируется путём прогноза (AST-тест `test_v2_v6_1b_iso`);
его вызывает только оркестратор `run_v6.py` после расчёта прогнозов.
Выборка — закрытые пары 2015-07-07 … 2021-12-30 (t⁺ ≤ 2021-12-30). Остаток e = ln RV_{t⁺} − ln H; QLIKE-регрессия g = θ₀ + θ'z (без штрафа) на S + доли энергий полос DCT-II V5 по барам SBER (low, mid; доля high = 1 − low − mid линейно зависима — не входит) + Σ.
Тест Вальда для блока (θ_B' V_BB⁻¹ θ_B ~ χ²(df)), V = H⁻¹ Ω H⁻¹ / n, Ω — Ньюи–Уэст лаг 5 по скорам; «без спектра» — те же признаки без долей полос. Единицы θ — z (нормировка по выборке)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from models_v2.s12 import v5_0_bars as B5
from models_v2.s13 import v6_1_feat as FE, v6_1_fit as FT, v6_1b_sig as SG

LAST_TP = "2021-12-30"
LAG = 5
SPEC = ["q_low", "q_mid"]


def spectral_shares(index) -> pd.DataFrame:
    """Доли энергий полос DCT-II по барам SBER (код V5 `v5_0_bars.day_table`), на днях index."""
    bars, _ = FE.load_inputs()
    t = B5.day_table(bars).reindex(index)
    tot = t[["e_low", "e_mid", "e_high"]].sum(axis=1)
    return pd.DataFrame({"q_low": t["e_low"] / tot, "q_mid": t["e_mid"] / tot}, index=index)


def hac_cov(Z: np.ndarray, e: np.ndarray, th: np.ndarray, lag: int = LAG) -> np.ndarray:
    """Робастная ковариация θ QLIKE-M-оценки: H⁻¹ Ω H⁻¹ / n; Ω — Ньюи–Уэст (веса Бартлетта) по скорам (1 − exp(e − g))·A_i."""
    A = np.column_stack([np.ones(len(Z)), Z])
    u = np.exp(e - A @ th)
    s, n = (1.0 - u)[:, None] * A, len(e)
    Om = s.T @ s / n
    for k in range(1, lag + 1):
        G = s[k:].T @ s[:-k] / n
        Om += (1 - k / (lag + 1)) * (G + G.T)
    Hs = (A * u[:, None]).T @ A / n
    Hi = np.linalg.inv(Hs)
    return Hi @ Om @ Hi / n


def fit_wald(X: pd.DataFrame, e: np.ndarray, blocks: dict) -> list:
    """QLIKE-регрессия e на X (z-нормировка по выборке) и тест Вальда по блокам имён. -> строки (test, df, stat, p, theta, se) + строки коэффициентов блоков."""
    Z = (X.to_numpy(float) - X.to_numpy(float).mean(0)) / X.to_numpy(float).std(0)
    th = FT.fit_lin_fixed(Z, e, 0.0)
    V = hac_cov(Z, e, th)
    names = ["const"] + list(X.columns)
    rows = []
    for b, cols in blocks.items():
        ix = [names.index(c) for c in cols]
        st = float(th[ix] @ np.linalg.solve(V[np.ix_(ix, ix)], th[ix]))
        rows.append(dict(test="Wald:" + b, df=len(ix), stat=st, p=float(stats.chi2.sf(st, len(ix))), theta=np.nan, se=np.nan))
        rows += [dict(test="coef:%s" % names[i], df=1, stat=float(th[i] ** 2 / V[i, i]), p=float(stats.chi2.sf(th[i] ** 2 / V[i, i], 1)), theta=float(th[i]), se=float(np.sqrt(V[i, i]))) for i in ix]
    return rows


def compute(tab: pd.DataFrame, S: pd.DataFrame, sigma: pd.DataFrame) -> pd.DataFrame:
    m = (tab["t_plus"] <= pd.Timestamp(LAST_TP)).to_numpy()
    idx = tab.index[m]
    e = np.log(tab.loc[idx, "rv_next"].to_numpy(float) / tab.loc[idx, "h"].to_numpy(float))
    spec, sg = spectral_shares(idx), sigma[SG.SIGMA_NAMES].reindex(idx)
    base = S.reindex(idx)
    out = []
    for model, X, blocks in (("S + спектр + Σ", pd.concat([base, spec, sg], axis=1), {"Σ (12)": SG.SIGMA_NAMES, "спектр (2)": SPEC}),
                             ("S + Σ (без спектра)", pd.concat([base, sg], axis=1), {"Σ (12)": SG.SIGMA_NAMES}),
                             ("S + спектр (без Σ)", pd.concat([base, spec], axis=1), {"спектр (2)": SPEC})):
        out += [dict(model=model, n=len(e), first_d=str(idx[0].date()), last_t_plus=str(tab.loc[idx, "t_plus"].max().date()), **r) for r in fit_wald(X, e, blocks)]
    return pd.DataFrame(out)
