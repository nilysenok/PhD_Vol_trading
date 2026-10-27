"""МСП ALL6 × EW по логике stage5_msp.py (ST5:179-235, 421-442) и Шарп печати (FB:20-22).

Шаги: среднее позиции и валовой доходности по имеющимся стратегиям на (дата, тикер) →
комиссия от |Δ средней позиции| на тикере → EW по тикерам → свободный капитал под ставку:
total = r + max(0, 1−exp)·rf·cal_days − max(0, exp−1)·(key+0,02)/365·cal_days,
excess = total − rf·cal_days.
cal_days — дни [d, d_next) периода позиции d→d+1 (аудит FA, Н2); у восстановленной ставки
печати (ворота) — прежние (d_prev, d].
Шарп = mean(excess)/std(excess, ddof=1)·√252 (печать; fix_sharpe — то же «везде»).

Ставка: gate — восстановлена из MSP_A_inst.csv (спецификация п. 9); fix_rate — решения ЦБ
(moex_discovery/.../key_rate_cbr.parquet, в процентах).
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt.data import MAIN, STRAT

MSP_CSV = STRAT / "msp" / "perstrat_daily_returns"
KEY_RATE = MAIN / "moex_discovery" / "data" / "external" / "macro" / "raw" / "key_rate_cbr.parquet"
RATE_OWNER = Path(__file__).resolve().parent / "rates" / "key_rate_cbr_vladelec.csv"
RATE_SPLICE = pd.Timestamp("2022-01-01")
PRINTS ={"A": 1.32, "B": 1.47, "C": 2.51, "D": 2.54}
# комиссия, с которой собраны печатные ряды (спецификация п. 8: A, D — без вычета; B — 0,05%)
PRINT_COMM = {"A": 0.0, "B": 0.0005, "C": 0.0, "D": 0.0}


def rate_restored() -> pd.Series:
    d = pd.read_csv(MSP_CSV / "MSP_A_inst.csv", parse_dates=["date"]).set_index("date")
    idx = d.index
    cal = np.ones(len(idx))
    cal[1:] = np.diff(idx.values).astype("timedelta64[D]").astype(float)
    s = pd.Series((d["total_ret"] - d["excess_ret"]).values / cal * 365.0, index=idx)
    s.attrs["cal"] = "back"          # печатные ряды собраны с днями (d_prev, d] — ворота
    return s


def cal_days(index, forward: bool = True) -> np.ndarray:
    """Календарные дни начисления ставки на дату портфеля.
    forward=True (аудит FA, Н2): дни [d, d_next) — период позиции d→d+1, последняя дата — 1;
    forward=False (ST5:216-219, печати): дни (d_prev, d], первая дата — 1."""
    v = pd.DatetimeIndex(index).values
    cal = np.ones(len(v))
    if len(v) > 1:
        dd = np.diff(v).astype("timedelta64[D]").astype(float)
        if forward:
            cal[:-1] = dd
        else:
            cal[1:] = dd
    return cal


def rate_vladelec() -> pd.Series:
    """Ключевая ставка по датам вступления в силу (Ъ владельца 27.10, 12:00), доли; 40 решений."""
    d = pd.read_csv(RATE_OWNER, comment="#", parse_dates=["data_vstupleniya"])
    return d.set_index("data_vstupleniya")["stavka_pct"].sort_index() / 100.0


def rate_cbr() -> pd.Series:
    """Ставка разработки и теста: до 2022-01-01 — ряд владельца (parquet ЦБ ставил 09.02.2018 вместо
    12.02 и 29.10.2019 вместо 28.10); с 2022-01-01 — parquet ЦБ как был (≥ 2022 не пересчитывается)."""
    k = pd.read_parquet(KEY_RATE)
    p = k.set_index(pd.to_datetime(k["date"]))["value"].sort_index() / 100.0
    o = rate_vladelec()
    return pd.concat([o[o.index < RATE_SPLICE], p[p.index >= RATE_SPLICE]])


def portfolio(pos: pd.DataFrame, rate: pd.Series, comm: float, agg: str = "pairs",
              cap_share: float = 1.0) -> pd.DataFrame:
    """pos: date, ticker, strategy, position, daily_gross_return (одного подхода).
    agg="pairs" — EW по имеющимся парам (стратегия, тикер): так собраны печатные ряды
    MSP_{A,C,D}_inst.csv (trading_ret и exposure совпадают до 1e-16);
    agg="ticker" — как stage5_msp.py: среднее по стратегиям внутри тикера, затем EW.
    cap_share — доля номинала, занятая капиталом (1 — спот; 0,2 — сценарий с маржой 20%):
    под ставку идёт 1 − cap_share·экспозиция.
    Последняя строка каждой пары (конец окна) платит комиссию выхода |pos| (Н5), кроме печатей v1."""
    if agg == "pairs":
        g = pos.rename(columns={"daily_gross_return": "gross"})[
            ["date", "ticker", "strategy", "position", "gross"]]
        keys = ["ticker", "strategy"]
    else:
        g = pos.groupby(["date", "ticker"]).agg(position=("position", "mean"),
                                                gross=("daily_gross_return", "mean")).reset_index()
        keys = ["ticker"]
    g = g.sort_values(keys + ["date"])
    dp = g.groupby(keys)["position"].diff().fillna(g["position"])
    sides = dp.abs()
    if rate.attrs.get("cal") != "back":
        # Ъ 26.10, 23:50 (аудит FA2a, Н5): позиция, открытая на последнем баре окна, закрыта по
        # close первого бара после окна — платит комиссию выхода. Печати v1 (ставка «back») —
        # прежнее соглашение спецификации п. 8 (без выхода на конце)
        last = ~g.duplicated(keys, keep="last")
        sides = sides + g["position"].abs().where(last, 0.0)
    g["net"] = g["gross"] - sides * comm
    g["abs_pos"] = g["position"].abs()
    g["turn"] = sides
    port = g.groupby("date").agg(r=("net", "mean"), exposure=("abs_pos", "mean"),
                                 turnover=("turn", "mean")).sort_index()
    kr = rate.reindex(rate.index.union(port.index)).sort_index().ffill().bfill()
    kr = kr.reindex(port.index)
    rf = kr.values / 365.0
    cal = cal_days(port.index, forward=rate.attrs.get("cal") != "back")
    exp_ = port["exposure"].values
    used = cap_share * exp_
    port["fdr"] = np.maximum(0.0, 1.0 - used) * rf * cal
    # плата за плечо — за те же календарные дни, что и ставка (аудит FA2a, Н4; в воротах не
    # срабатывает: экспозиция ≤ 1)
    port["margin"] = np.maximum(0.0, used - 1.0) * (kr.values + 0.02) / 365.0 * cal
    port["total"] = port["r"] + port["fdr"] - port["margin"]
    port["excess"] = port["total"] - rf * cal
    port["cal_days"] = cal
    return port


def sharpe(x) -> float:
    x = np.asarray(x, float)
    s = np.std(x, ddof=1)
    return float(np.mean(x) * sqrt(252) / s) if s > 1e-12 else 0.0


def metrics(port: pd.DataFrame) -> dict:
    tot, exc = port["total"].values, port["excess"].values
    neg = tot[tot < 0]
    ds = np.std(neg, ddof=1) if len(neg) > 1 else 1e-12
    cum = np.cumsum(tot)
    return {"sharpe": sharpe(exc),
            "sortino": float(np.mean(exc) * sqrt(252) / ds) if ds > 1e-12 else 0.0,
            "ann_ret_pct": float(np.mean(tot) * 252 * 100),
            "ann_vol_pct": float(np.std(tot, ddof=1) * sqrt(252) * 100),
            "mdd_pct": float(np.max(np.maximum.accumulate(cum) - cum) * 100),
            "exposure_pct": float(port["exposure"].mean() * 100),
            "n_days": int(len(port))}


def msp_table(pos: pd.DataFrame, rate: pd.Series, comm, years=(2022, 2025),
              agg: str = "pairs") -> pd.DataFrame:
    """comm: число или {подход: комиссия}. Возвращает метрики по подходам."""
    pos = pos[(pos["test_year"] >= years[0]) & (pos["test_year"] <= years[1])]
    rows = []
    for ap in sorted(pos["approach"].unique()):
        c = comm[ap] if isinstance(comm, dict) else comm
        port = portfolio(pos[pos["approach"] == ap], rate, c, agg)
        m = metrics(port)
        m.update({"approach": ap, "comm": c})
        for y in range(years[0], years[1] + 1):
            ex = port.loc[port.index.year == y, "excess"]
            m[f"sharpe_{y}"] = sharpe(ex) if len(ex) > 2 else np.nan
        rows.append(m)
    return pd.DataFrame(rows)
