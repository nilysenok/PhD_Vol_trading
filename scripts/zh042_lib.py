"""Конвейер Ж-042, общие функции. Только train 2018–2019: validation 2020–2021 и данные ≥ 2022 не читаются.

Движок `fastbt/*` не меняется: оценщик — `Sweep`/`SkEvaluator`, боевой путь — `ekviv_s9` (`s9.Engine.port`).
Запуск из корня репозитория: FASTBT_MAIN=<корень> /usr/bin/python3 scripts/zh042_konveyer.py S3 <этап>.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import ekviv_s9 as ek
from fastbt import s9, shag0
from fastbt.data import year_bounds
from fastbt.sweep import Sweep, make_evaluator
from fastbt.sweep_run import load
from fastbt.sweep_spec import Spec

E1 = Path("docs/tables/sweep_e1v2")
OUT = Path("docs/tables/zh042")
COMMS = (0.0, 0.0004, 0.00035)
SHAPKA = "# статус: разработка, train 2018–2019; validation не открыта; Ж-042\n"
COMM_TRADE = 2 * 0.04                      # комиссия на сделку (две стороны) при 0,04 % на сторону, в %


def open_sweep(sid: str):
    """Оценщик стратегии и развёртка сетки; ledger в памяти (проба = точка на 2018 при 0,04 %)."""
    spec_d, exp, _ = load(f"fastbt/designs/sweep/{sid}_setka_e1b.yaml")
    spec = Spec(**spec_d, out=None)
    return Sweep(spec, make_evaluator(spec)), exp


def coerce(exp: dict, key: str, raw):
    """Значение из CSV -> значение сетки (тип узла: int/float/str/None)."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    for v in list(exp["grid"][key]) + [exp["base"][key]]:
        if v is None:
            continue
        try:
            if float(v) == float(raw):
                return v
        except (TypeError, ValueError):
            if str(v) == str(raw):
                return v
    raise ValueError(f"{key}={raw!r}: нет в сетке")


def point_from_row(exp: dict, row) -> dict:
    return {k: coerce(exp, k, row.get(f"p.{k}")) for k in exp["base"]}


def sides_gross(pos, lr, first, m) -> tuple:
    """(сделок = стороны/2, суммарный доход на единицу позиции) по строкам маски m. Стороны — как в `Engine.port` (Н5)."""
    dp = np.diff(pos, prepend=0.0)
    dp[first] = pos[first]
    last = np.r_[first[1:], True]
    sides = np.abs(dp) + np.where(last, np.abs(pos), 0.0)
    return float(sides[m].sum() / 2), float((pos * lr)[m].sum())


def _stat(sharpes: dict, trades: float, gross: float) -> dict:
    """Строка диагностики: доход и комиссия на сделку в %, отношение (доход валовый, до комиссии)."""
    nm = {0.0: "sharpe_do_izderzhek", 0.0004: "sharpe_posle_004", 0.00035: "sharpe_posle_0035"}
    d = {nm[c]: v for c, v in sharpes.items()}
    gp = gross / trades * 100 if trades else float("nan")
    return {**d, "sdelok_god": trades, "dohod_na_sdelku_pct": gp, "komissiya_na_sdelku_pct": COMM_TRADE,
            "otnoshenie": gp / COMM_TRADE, "chistyy_na_sdelku_pct": gp - COMM_TRADE}


def main_rows(sid: str) -> list:
    """Главная строка: «A только Sk» через `s9.Engine.port`, окна 2018 и 2018–2019 (как `ekviv_s9.WINDOWS`)."""
    cache, out = ek.a_cache(2019), []
    for god, ys in ek.WINDOWS.items():
        eng = ek.s9_engine(cache, ys)
        sidr = shag0.sid_row(eng)
        pos = shag0.keep_sids(np.asarray(eng.A_row, float), sidr, [sid])
        sh = {c: s9.sharpe((x := eng.port(pos, comm=c)["excess"])[x.index.year == god]) for c in COMMS}
        m = (sidr == sid) & (np.asarray(eng.r["year"]) == god)
        tr, gr = sides_gross(pos, eng.lr_row, eng.first, m)
        out.append(dict(stroka="A_tolko_Sk_s9", god=god, **_stat(sh, tr, gr)))
    return out


def ref_stats(sid: str, ev, p: dict) -> dict:
    """{god: (сделок, доход)} по позициям боевого пути (`ek.boevoy_pos`) точки p на всех тикерах."""
    res = {}
    for god in (2018, 2019):
        end, P, L, F, Y = pd.Timestamp(f"{god + 1}-01-01"), [], [], [], []
        for t in ev.tickers:
            td, ind = ev._td[t]
            i0, i1 = year_bounds(td.dates, 2018)[0], int(np.searchsorted(td.dates, np.datetime64(end)))
            P.append(ek.boevoy_pos(sid, p, t, td, ind)[i0:i1].astype(float))
            L.append(np.asarray(td.log_ret)[i0:i1])
            Y.append(pd.DatetimeIndex(td.dates[i0:i1]).year.values)
            F.append(np.r_[True, np.zeros(i1 - i0 - 1, bool)])
        y = np.concatenate(Y)
        res[god] = sides_gross(np.concatenate(P), np.concatenate(L), np.concatenate(F), y == god)
    return res


def ref_rows(sid: str, ev, p: dict, name: str, klass: str = "zh042") -> list:
    """Справочная строка: точка p через `ekviv_s9.run_sk` (Шарп боевого пути, статус сверки) + сделки/доход."""
    res = ek.run_sk(dict(id=name, klass=klass, sid=sid, p=p), ev, COMMS)
    st, out = ref_stats(sid, ev, p), []
    for god in (2018, 2019):
        r = [x for x in res if x["god"] == god]
        sh = {x["comm_pct"] / 100: x["sharpe_s9"] for x in r}
        sh = {c: next(v for k, v in sh.items() if abs(k - c) < 1e-12) for c in COMMS}
        ok = all(x["status"] == "совпало" for x in r)
        out.append(dict(stroka=name, god=god, **_stat(sh, *st[god]), sverka_s9="sovpalo" if ok else "RASHOZhDENIE",
                        max_dpnl=max(x["max_dpnl"] for x in r), bar_poz_razl=sum(x["bar_poz_razl"] for x in r)))
    return out


def write_csv(df: pd.DataFrame, name: str, extra: str = ""):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(SHAPKA + extra + df.to_csv(index=False), "utf-8")
