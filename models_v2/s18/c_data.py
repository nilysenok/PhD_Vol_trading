"""S-C-DATA: синхронизированные M10 SBER, 16 бумаг пула и IMOEX на барах основной сессии SBER (Ж-254 п. 8).

Источник баров — тот же, что у K2/K3: `fwd_src.Src` (архив + продолжение 2026, обрезка begin <= cut 18:40). Новых данных нет.
Сетка — канонические сессии SBER (`daily.build_daily`, правило IMOEX), бары — основная сессия 10:00…18:40 (53 слота по 10 минут).
Доходность бара SBER — как в целевой RV (`daily.intraday_returns`): первый бар ln(c/o), далее ln(c_i/c_prev) по соседним ПРИСУТСТВУЮЩИМ барам.
Доходность бумаги — на том же интервале бара SBER: ln c_k − ln c_prev (первый бар: ln c_k − ln o_k); нет бара на концах интервала — NaN
(константой не заполняется, Д4); интервал из m слотов (дыра SBER) нормируется на √m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s4.prov4 import PAPERS
from models_v2.s17.fwd_src import LAST_SESSION, Src

POOL = tuple(PAPERS)          # 16 разрешённых бумаг, SBER не входит (Ж-248, K2/K3)
NSLOT = 53                    # слоты 10:00 … 18:40


@dataclass
class Inputs:
    days: pd.DatetimeIndex    # канонические сессии SBER
    rv: np.ndarray            # RV целевого определения (%², `daily_SBER`), по сессиям
    nbars: np.ndarray         # баров основной сессии по build_daily
    off: np.ndarray           # off[i]:off[i+1] — бары сессии i
    day: np.ndarray           # (n,) номер сессии бара
    slot: np.ndarray          # (n,) слот 0…52
    r: np.ndarray             # (n,) доходность бара SBER (log)
    R: np.ndarray             # (n, 16) доходности бумаг пула на интервалах баров SBER, / √m
    ri: np.ndarray            # (n,) то же для IMOEX (только покрытие; в фактор не входит)
    stats: dict               # дубли, не-10-минутные бары, исключённые дни


def slot_arrays(bars: pd.DataFrame, days: pd.DatetimeIndex) -> tuple:
    """-> (LC, LO, n_dup, n_off_grid): ln close/open по (сессия, слот), NaN — бара нет."""
    b = D.main_session(bars)
    t = b["begin"]
    di = days.get_indexer(t.dt.normalize())
    mins = (t.dt.hour * 60 + t.dt.minute - 600).to_numpy()
    ok = (di >= 0) & (mins >= 0) & (mins <= 520)
    di, mins, b = di[ok], mins[ok], b[ok]
    slot = mins // 10
    n_dup = int(len(di) - len(np.unique(di * NSLOT + slot)))
    LC = np.full((len(days), NSLOT), np.nan)
    LO = np.full((len(days), NSLOT), np.nan)
    LC[di, slot] = np.log(b["close"].to_numpy(float))
    LO[di, slot] = np.log(b["open"].to_numpy(float))
    return LC, LO, n_dup, int((mins % 10 != 0).sum())


def sber_bars(bars: pd.DataFrame, days: pd.DatetimeIndex) -> tuple:
    """Бары SBER канонических сессий: (day, slot, prev_slot (−1 — первый бар), r) в порядке времени."""
    m = D.main_session(bars).sort_values("begin").reset_index(drop=True)
    rr = D.intraday_returns(m)
    t = m["begin"]
    day = days.get_indexer(rr["date"])
    slot = ((t.dt.hour * 60 + t.dt.minute - 600) // 10).to_numpy()
    ok = day >= 0
    day, slot, r = day[ok], slot[ok], rr["r"].to_numpy(float)[ok]
    prev = np.full(len(day), -1)
    same = day[1:] == day[:-1]
    prev[1:][same] = slot[:-1][same]
    return day, slot, prev, r


def interval_returns(LC, LO, day, slot, prev) -> np.ndarray:
    """Доходность инструмента на интервале бара SBER, / √m; NaN, если нет бара на любом конце."""
    first = prev < 0
    pp = np.where(first, 0, prev)
    ret = np.where(first, LC[day, slot] - LO[day, slot], LC[day, slot] - LC[day, pp])
    return ret / np.sqrt(np.where(first, 1, slot - prev))


def build_inputs(cut=LAST_SESSION, src: Src | None = None, pool=POOL) -> Inputs:
    src = src or Src()
    sb = src.bars("SBER", cut)
    dd, ex = D.build_daily(sb, index_days=src.index_days(cut))
    days = dd.index
    day, slot, prev, r = sber_bars(sb, days)
    cols, st = [], {"excluded": ex}
    for t in list(pool) + ["IMOEX"]:
        LC, LO, nd, no = slot_arrays(src.bars(t, cut), days)
        cols.append(interval_returns(LC, LO, day, slot, prev))
        st[t] = (nd, no)
    off = np.searchsorted(day, np.arange(len(days) + 1))
    return Inputs(days, dd["rv"].to_numpy(float), dd["n_bars"].to_numpy(int), off, day, slot, r,
                  np.column_stack(cols[:-1]), cols[-1], st)
