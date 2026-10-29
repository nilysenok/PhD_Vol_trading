"""V6-12b: оси движка и тень паузы для EvaluatorV6 (docs/otchety/2026-10-29_setka_v6_inventarizaciya.md §1.2: И1, И2, И3, И10, И14, И15).

Оси: time_decay (V6-9: после tau//2 дней тейк net — половина дистанции; нужны tau и tejk = net, иначе NeSchitana), sigma_ist (ряд σ̂
уровней и размера: rv22 | rv5 | ewma094 | rv66 из inp.sigma), ed_lim (единица отступа лимитки; действует при k_e > 0), zhizn (жизнь
лимитки, баров по 5 мин), пауза: pauza_porog / pauza_okno.
Тень паузы: сначала считается та же точка БЕЗ паузы; её дневной excess тикера (g − комиссия·стороны − занятость·ставка·cal, как
`EvaluatorV2._series`) подаётся в e1_pauza.allow, и вторым прогоном новые входы запрещены в дни паузы. Окно тени — с первого дня периода:
раньше полного окна (pauza_okno строк) пауза не включается (вход разрешён, e1_pauza). Открытые позиции паузой не закрываются.
"""
from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fastbt import e1_pauza
from fastbt.etap1_eval import NET
from fastbt.etap1_eval_v2 import NeSchitana
from fastbt.indicators import compute_base
from fastbt.signals.interface import TickerData

DVIG = ("time_decay", "sigma_ist", "ed_lim", "zhizn")
PAUZA = ("pauza_porog", "pauza_okno")
PAUZA_OKNO = 63                      # «выкл.» дочерней оси pauza_okno (kub_S*.yaml)


def primenit(r, v6: dict, sigmy) -> None:
    """Оси движка → разбор точки v5 (`r` — eval_v5_osi.Razbor): ext (Params), pt4 (σ), nov (жизнь лимитки)."""
    if v6.get("time_decay") == "da":
        r.ext["time_decay"] = True
    if "sigma_ist" in v6:
        if v6["sigma_ist"] not in sigmy:
            raise ValueError(f"sigma_ist={v6['sigma_ist']!r}: в inp.sigma нет такого ряда ({sorted(sigmy)})")
        r.pt4["sigma"] = v6["sigma_ist"]
    if "zhizn" in v6:
        r.nov["zhizn"] = v6["zhizn"]


def rel_td_ind(df: pd.DataFrame, td, progrev: bool = False) -> tuple:
    """Дневной td_rel → (TickerData, ind) на сетке td.dates; нет дня — ValueError (без ffill, Д4).
    progrev=True: дни сетки ДО первого дня td_rel (10-минуток ещё нет: 2014-06…08) остаются NaN, не заполняются; дыра внутри — отказ."""
    cols = ["open", "high", "low", "close", "volume"]
    d = df.set_index(pd.to_datetime(df["date"]))[cols].reindex(pd.DatetimeIndex(td.dates))
    bad = d.isna().any(axis=1).to_numpy()
    lead = (np.cumsum(~bad) == 0) if progrev else np.zeros(len(bad), bool)
    if bad[~lead].any():
        raise ValueError(f"{td.ticker}: в td_rel нет {int(bad[~lead].sum())} дней сетки td")
    o, h, l, c, v = (d[k].to_numpy(float) for k in cols)
    lr = np.zeros(len(c))
    lr[:-1] = np.log(c[1:] / np.maximum(c[:-1], 1e-12))
    return TickerData(td.ticker, td.dates, o, h, l, c, v, lr, td.sigma), compute_base(c, h, l, o, v)


def proverit(r, v6: dict) -> None:
    """time_decay без tau или при tejk ≠ net ничего не делает (движок отказал бы ValueError) — строка ne_schitana."""
    if v6.get("time_decay") == "da" and (r.pt4["tau"] == NET or r.nov["tejk_signal"] != NET):
        raise NeSchitana("time_decay: нужен tau и tejk = net (иначе ось ничего не делает)")


def ten(evr, rows, period: str, anchor: bool) -> dict:
    """{тикер: pd.Series(дата → excess тени за день)} по строкам `_rows` без паузы (как EvaluatorV2._series, полный период)."""
    eng, inv = evr._engine(period)
    rs = [x for _, x, _, _ in rows]
    cat = lambda f: np.concatenate([f(x) for x in rs])                           # noqa: E731
    u = evr._u(rs, eng, inv, anchor) if evr.fdr is not None else cat(lambda x: x.expo)
    ex = cat(lambda x: x.g) - cat(lambda x: x.sides) * evr.comm - u * (eng.rf[inv] * eng.cal[inv])
    dts, out, lo = pd.DatetimeIndex(eng.dates[inv]), {}, 0
    for (ti, x, _, _) in rows:
        out[evr.sk.tickers[ti]] = pd.Series(ex[lo:lo + len(x.g)], index=dts[lo:lo + len(x.g)])
        lo += len(x.g)
    return out


def razreshit(v6: dict, t: str, dates, tenyu: dict) -> np.ndarray:
    """bool[d]: вход разрешён паузой Sk (e1_pauza.allow); оба направления запрещены вместе."""
    pt = {"pauza_okno": int(v6.get("pauza_okno", PAUZA_OKNO)), "pauza_porog": float(v6["pauza_porog"])}
    with warnings.catch_warnings():                  # FutureWarning pandas из e1_pauza (fillna на object), результат тот же
        warnings.simplefilter("ignore", FutureWarning)
        return e1_pauza.allow(pt, t, dates, SimpleNamespace(ten=tenyu))[0]
