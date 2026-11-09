"""V7-2b, G30: специализация и веса экспертов (план 6.3, 6.5) для MoE 5–7 и условие конфигурации 9 (план, раздел 5). Чистые функции от таблиц прогнозов (d, t_plus, rv, H, G, F, pi0, pi1, pi2, g1, g2; среднее по 5 seeds).
Эксперт k = 1, 2 (π₀ — нулевой эксперт, g₀ ≡ 0). «Без эксперта k»: F₋k = H·exp(G − π_k·g_k) — π и g по seeds усреднены, поэтому вычитание приближённое: `approx_err` = max |G − Σπ_k g_k| (в таблице).
Разность «без» − «с» (QLIKE) > 0 — эксперт помогает. Интервалы — блоковый бутстрап (блок 20, 2000, seed 0). Всё — описательно, без выбора; связь π с преимуществом — ЗАДНИМ ЧИСЛОМ (по фактическим RV).
Условие 9: окно разработки 2018–2021 (2022+ в условие не входит)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s5.d import stats_d as SD
from models_v2.s14 import v7_1m_stat as ST

THR, SWITCH, SHARE_MIN = 0.3, 0.5, 0.10
MOE = ("cfg5", "cfg6", "cfg7")
SPEC_WIN = ("2018_2021", "2022_2025", "all")
PERIODS = tuple(str(y) for y in range(2018, 2027)) + ("до 2021 (≤ 2021)", "после 2021 (≥ 2022)")


def _pi(df, k):
    return df["pi%d" % k].to_numpy(float)


def gain(df, k) -> np.ndarray:
    """Дневная разность QLIKE: без эксперта k минус с ним (плюс = эксперт помогает)."""
    rv, H, G = (df[c].to_numpy(float) for c in ("rv", "H", "G"))
    return Q.ql(rv, H * np.exp(G - _pi(df, k) * df["g%d" % k].to_numpy(float))) - Q.ql(rv, df["F"].to_numpy(float))


def approx_err(df) -> float:
    return float(np.abs(df["G"].to_numpy(float) - sum(_pi(df, k) * df["g%d" % k].to_numpy(float) for k in (1, 2))).max())


def _on_rest_ci(g, on) -> tuple:
    """90 % интервал разности средних (дни π_k > 0,3 минус остальные) — блоковый бутстрап по дням окна."""
    idx = SD.block_idx(len(g), B_=ST.B, block=ST.BLOCK, seed=ST.SEED)
    gi, oi = g[idx], on[idx].astype(float)
    n1, n0 = oi.sum(1), (1 - oi).sum(1)
    d = (gi * oi).sum(1) / np.maximum(n1, 1) - (gi * (1 - oi)).sum(1) / np.maximum(n0, 1)
    d = d[(n1 > 0) & (n0 > 0)]
    if len(d) == 0:                                                       # все дни окна по одну сторону порога — разности нет
        return float("nan"), float("nan")
    return float(np.percentile(d, 5)), float(np.percentile(d, 95))


def _mean_ci(g) -> tuple:
    if len(g) == 0:
        return float("nan"), float("nan"), float("nan")
    bm = SD.boot_means(g, SD.block_idx(len(g), B_=ST.B, block=ST.BLOCK, seed=ST.SEED))
    return float(g.mean()), float(np.percentile(bm, 5)), float(np.percentile(bm, 95))


def spec_table(frames: dict) -> pd.DataFrame:
    """Строка на (конфигурация, k, окно): доля дней π_k > 0,3; частота |Δπ_k| > 0,5 между соседними датами; средняя |Δπ|; связь π_k с преимуществом эксперта (Спирмен, ЗАДНИМ ЧИСЛОМ);
    разность «без» − «с» на днях π_k > 0,3 и на остальных (среднее, 90 %), разность «на днях − остальные» с интервалом; вывод 6.3 — по окну 2018–2021."""
    rows = []
    for c, df in frames.items():
        tp, err = df["t_plus"].to_numpy(), approx_err(df)
        for k in (1, 2):
            p, g = _pi(df, k), gain(df, k)
            dp = np.r_[np.nan, np.abs(np.diff(p))]
            for w in SPEC_WIN:
                m = ST.wmask(tp, w)
                on = p[m] > THR
                a, b, ci = _mean_ci(g[m][on]), _mean_ci(g[m][~on]), _on_rest_ci(g[m], on)
                spearman = float(pd.Series(p[m]).corr(pd.Series(g[m]), method="spearman"))
                share, specialized = float(on.mean()), bool(on.mean() >= SHARE_MIN and (ci[0] > 0 or ci[1] < 0))
                rows.append(dict(config=c, k=k, window=w, n=int(m.sum()), share_pi_gt03=share, switch_freq=float(np.mean(dp[m][1:] > SWITCH)) if m.sum() > 1 else float("nan"), mean_abs_dpi=float(np.nanmean(dp[m])),
                                 gain_on=a[0], gain_on_lo90=a[1], gain_on_hi90=a[2], gain_rest=b[0], gain_rest_lo90=b[1], gain_rest_hi90=b[2], diff_on_rest_lo90=ci[0], diff_on_rest_hi90=ci[1],
                                 spearman_pi_gain_hindsight=spearman, specialized=("да" if specialized else "нет") if w == "2018_2021" else "—", approx_err=err))
    return pd.DataFrame(rows)


def weights_table(frames: dict) -> pd.DataFrame:
    """Средние π_0, π_1, π_2 по годам цели и до / после 2021 (6.5)."""
    rows = []
    for c, df in frames.items():
        y = pd.DatetimeIndex(df["t_plus"]).year.to_numpy()
        for k in (0, 1, 2):
            p = _pi(df, k)
            sel = [(str(yy), y == yy) for yy in range(2018, 2027)] + [(PERIODS[-2], y <= 2021), (PERIODS[-1], y >= 2022)]
            rows += [dict(config=c, k=k, period=n, n=int(m.sum()), mean_pi=float(p[m].mean())) for n, m in sel]
    return pd.DataFrame(rows)


def cond9(df6: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Условие 9 по конфигурации 6 (версия 2): у обоих экспертов π > 0,3 не менее чем на 10 % дней 2018–2021 И 6 несёт новую информацию против 4 (6.2, строка H2 окна 2018–2021)."""
    m = ST.wmask(df6["t_plus"].to_numpy(), "2018_2021")
    sh = [float((_pi(df6, k)[m] > THR).mean()) for k in (1, 2)]
    r = pairs[(pairs["hyp"] == "H2") & (pairs["x"] == "cfg6") & (pairs["y"] == "cfg4") & (pairs["window"] == "2018_2021")].iloc[0]
    ok_sh, ok_ni = bool(min(sh) >= SHARE_MIN), r["rule62"] == "да"
    return dict(window="2018_2021", n_dates=int(m.sum()), share_pi1_gt03=sh[0], share_pi2_gt03=sh[1], shares_ok=ok_sh, delta_q_6_vs_4=float(r["delta_q"]), p_dm_one=float(r["p_one"]), years_neg=int(r["years_neg"]),
                new_information_vs_4=ok_ni, cond9=bool(ok_sh and ok_ni), note="окно только 2018–2021; 2022+ в условие не входит; конфигурация 9 не считается")
