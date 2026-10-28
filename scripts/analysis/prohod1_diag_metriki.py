# -*- coding: utf-8 -*-
"""Метрики «редко и метко» и терцили σ по журналу сделок EvaluatorV2 (Ж-091 п. 2–3). Справочно, train4, потолок.

Журнал сделок — ev._zctx[2] после evaluate_v2: [(тикер, Result, a, b)]; сделка закрытая, вошла в окне (entry_day >= a).
Доходность сделки на единицу стоимости входа: side·(exit/entry − 1) − комиссия входа − комиссия выхода (веса размера не учтены).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

OKNO_RANG, MIN_RANG, GODA = 252, 126, 4


def sdelki(ev) -> pd.DataFrame:
    """Закрытые сделки последнего evaluate_v2: t, god, side, ret, dni (дней, затронутых позицией), rang_sigma (терциль 0/1/2 или nan)."""
    out, dni_vsego, dni_okna = [], 0, 0
    for ti, r, a, b in ev._zctx[2]:
        t, dates = ev.sk.tickers[ti], ev.inp.meta[ev.sk.tickers[ti]]["dates"]
        sg = np.asarray(ev.inp.sigma["rv22"][t], float)
        dni_okna += b - a
        for tr in r.trades:
            if tr.entry_day < a:
                continue
            dni_vsego += (b if tr.exit_day is None else tr.exit_day) - tr.entry_day + 1
            if tr.exit_day is None:
                continue
            ret = tr.side * (tr.exit_px / tr.entry_px - 1.0) - tr.fee_in - tr.fee_out
            out.append(dict(t=t, god=int(str(dates[tr.entry_day])[:4]), side=tr.side, ret=ret,
                            dni=tr.exit_day - tr.entry_day + 1, rang=rang_sigma(sg, tr.entry_day)))
    df = pd.DataFrame(out, columns=["t", "god", "side", "ret", "dni", "rang"])
    df.attrs["ekspoziciya"] = dni_vsego / dni_okna if dni_okna else float("nan")
    return df


def rang_sigma(sg: np.ndarray, e: int) -> float:
    """Ранг rv22 тикера на день входа e в его же истории: значение на e−1 против окна из 252 значений, оканчивающегося e−1.
    Только данные ≤ e−1; меньше 126 известных значений — nan. Доля значений окна строго ниже (0…1)."""
    w = sg[max(e - OKNO_RANG, 0):e]
    w = w[np.isfinite(w)]
    x = sg[e - 1] if e >= 1 else float("nan")
    if len(w) < MIN_RANG or not math.isfinite(x):
        return float("nan")
    return float((w < x).sum()) / len(w)


def tercil(r: float) -> int:
    return -1 if not math.isfinite(r) else min(int(r * 3), 2)


def metriki(row: dict, df: pd.DataFrame) -> dict:
    """Одна точка/контроль: сделок в год, экспозиция, средняя доходность сделки, σ, IR; Шарп расчётный и фактический; skew/kurt, mSR."""
    n = len(df)
    mu = float(df.ret.mean()) if n else float("nan")
    sd = float(df.ret.std(ddof=1)) if n > 1 else float("nan")
    ir = mu / sd if n > 1 and sd > 0 else float("nan")
    return dict(sdelok=n, sdelok_v_god=n / GODA, ekspoziciya=df.attrs.get("ekspoziciya", float("nan")), sr_sdelki=mu, sigma_sdelki=sd,
                ir_sdelki=ir, sharpe_rasch=ir * math.sqrt(n / GODA) if math.isfinite(ir) else float("nan"),
                sharpe_fakt=row.get("sharpe", float("nan")), skew=row.get("skew", float("nan")), kurt=row.get("kurt", float("nan")),
                msr=row.get("msr", float("nan")))


def terciles(df: pd.DataFrame) -> pd.DataFrame:
    """По терцилям ранга σ (0 — низкая … 2 — высокая; -1 — ранг не определён): сделок, средняя доходность, доля прибыльных."""
    g = df.assign(terc=[tercil(r) for r in df.rang]).groupby("terc")
    return pd.DataFrame(dict(sdelok=g.size(), sr_sdelki=g.ret.mean(), dolya_pribyl=g.ret.apply(lambda s: float((s > 0).mean())))).reset_index()
