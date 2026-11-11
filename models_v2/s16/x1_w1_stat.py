"""W1: статистика пар (бутстрап 90 %: блок 20, B 2000, seed 0; DM Ньюи–Уэст лаг 5, односторонний; Холм с N программы — не число строк, № 103) и вклад лет.
Вклад года c_y = Σ_{t ∈ y}(l_s,t − l_b,t) / (n · Q_b), в % базы; Σ_y c_y = полному Δ (тождество). Периоды: 2018…2025 по году t⁺, 2026 — две части: до 27.02 и окно F."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s16 import x1_t0 as T0, x1_w1_join as J
from models_v2.s16.x1_ze_crit import N_HOLM, N_WIDE

ST, EVX = T0.ST, T0.EVX
PAIRS = (("X1", "K2"), ("X1", "K3"), ("X1", "K1"), ("ZB", "K2"), ("ZB", "K3"), ("ZB", "K1"), ("ZB", "X1"))
YPAIRS = (("X1", "K2"), ("X1", "K3"), ("ZB", "K2"), ("ZB", "K3"), ("ZB", "X1"))
SWINS = ("M", "H", "F")
ALPHA = 0.05
PERIODS = tuple(str(y) for y in range(2018, 2026)) + ("2026_до_27.02", "2026_F")


def verdict(delta_pct: float, p_holm: float) -> str:
    """Численное и статистическое — раздельно; «подтверждено» только при Δ < 0 и p Холма < 0,05 (№ 158); знак Δ: минус — лучше (№ 125/140)."""
    num = "численно лучше базы" if delta_pct < 0 else "численно не лучше базы"
    return num + "; " + ("подтверждено (p Холма < 0,05)" if delta_pct < 0 and p_holm < ALPHA else "не подтверждено (p Холма ≥ 0,05)" if delta_pct < 0 else "не подтверждено")


def stats_table(wins: dict) -> pd.DataFrame:
    rows = []
    for w in SWINS:
        al = wins[w]
        rv, tp = al["K1"]["rv"].to_numpy(float), pd.DatetimeIndex(al["K1"]["t_plus"])
        F = {k: v["F"].to_numpy(float) for k, v in al.items()}
        for h, b in PAIRS:
            qh, qb = Q.ql(rv, F[h]), Q.ql(rv, F[b])
            ci = EVX.delta_ci(qh, qb, float(qb.mean()), b=ST.B, block=ST.BLOCK, seed=ST.SEED)
            dm = EVX.dm_one(rv, F[b], F[h])
            rows.append(dict(window=w, hybrid=h, base=b, n=len(tp), first=str(tp.min().date()), last=str(tp.max().date()), qlike_h=float(qh.mean()), qlike_b=float(qb.mean()),
                             delta_pct=ci["delta_pct"], lo90=ci["lo90"], hi90=ci["hi90"], dm_stat=dm["stat"], p_one=dm["p_one"]))
    t = pd.DataFrame(rows)
    t["p_holm_N"], t["p_holm_Nwide"] = np.nan, np.nan
    for w in SWINS:
        k = t["window"] == w
        t.loc[k, "p_holm_N"], t.loc[k, "p_holm_Nwide"] = EVX.holm_both(t.loc[k, "p_one"].to_numpy(), {"N": N_HOLM, "N_wide": N_WIDE})
    t["verdict"] = [verdict(d, p) for d, p in zip(t["delta_pct"], t["p_holm_N"])]
    t["n_holm"], t["n_holm_wide"] = N_HOLM, N_WIDE
    return t


def period_masks(tp) -> dict:
    tp = pd.DatetimeIndex(tp)
    y = tp.year.to_numpy()
    m = {str(k): y == k for k in range(2018, 2026)}
    m["2026_до_27.02"] = (y == 2026) & (tp <= J.ARCH_END)
    m["2026_F"] = tp > J.ARCH_END
    return m


def contrib(ls: np.ndarray, lb: np.ndarray, masks: dict) -> dict:
    """c_y = 100 · Σ_{t ∈ y}(l_s − l_b) / (n · Q_b)."""
    n, qb = len(ls), float(lb.mean())
    return {p: 100.0 * float((ls[m] - lb[m]).sum()) / (n * qb) for p, m in masks.items()}


def years_conc(al: dict) -> tuple:
    rv, tp = al["K1"]["rv"].to_numpy(float), pd.DatetimeIndex(al["K1"]["t_plus"])
    ls, mk = {k: Q.ql(rv, v["F"].to_numpy(float)) for k, v in al.items()}, period_masks(tp)
    rows, conc = [], []
    for k in al:
        rows += [dict(series=k, base="-", period=p, n=int(m.sum()), q_series=float(ls[k][m].mean()), q_base=np.nan, delta_pct=np.nan, contrib_pct=np.nan, delta_total_pct=np.nan) for p, m in mk.items()]
    for s, b in YPAIRS:
        c, tot = contrib(ls[s], ls[b], mk), 100.0 * float(ls[s].mean() - ls[b].mean()) / float(ls[b].mean())
        rows += [dict(series=s, base=b, period=p, n=int(m.sum()), q_series=float(ls[s][m].mean()), q_base=float(ls[b][m].mean()), delta_pct=100.0 * float(ls[s][m].mean() / ls[b][m].mean() - 1.0),
                      contrib_pct=c[p], delta_total_pct=tot) for p, m in mk.items()]
        conc += conc_rows(s, b, ls, mk, c, tot)
    return pd.DataFrame(rows), pd.DataFrame(conc)


def conc_rows(s: str, b: str, ls: dict, mk: dict, c: dict, tot: float) -> list:
    """Концентрация выигрыша (выигрыш — вклад года < 0, т. е. ряд лучше базы), сумма вкладов ухудшающих лет, Δ без каждого года, число лет с выигрышем."""
    v = np.array(list(c.values()))
    gain = float(-v[v < 0].sum())
    best = np.sort(v)
    out = [("summary", "delta_total_pct", tot), ("summary", "years_with_gain", int((v < 0).sum())), ("summary", "periods", len(v)), ("summary", "gain_total_pct", gain),
           ("summary", "share_best_year", float(-best[0] / gain) if gain > 0 and best[0] < 0 else np.nan), ("summary", "share_two_best_years", float(-best[:2][best[:2] < 0].sum() / gain) if gain > 0 else np.nan),
           ("summary", "worsening_sum_pct", float(v[v > 0].sum()))]
    for p, m in mk.items():
        out.append(("loyo", p, 100.0 * float((ls[s][~m] - ls[b][~m]).sum()) / float(ls[b][~m].sum())))
    return [dict(series=s, base=b, item=i, key=k, value=x) for i, k, x in out]
