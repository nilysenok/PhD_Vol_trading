"""Диагностика параметров S1–S6 (Ж-055) по готовым матрицам: ledger_it*.pkl (только чтение), zh047-таблицы, сетки.
Новых прогонов оценщика и движка нет. Запуск: FASTBT_MAIN=$PWD python3 scripts/diag_zh055.py [основная_копия] [выход_каталог]"""
from __future__ import annotations

import csv
import importlib.util
import math
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs" / "otchety"
DELTA, INFL, NA = 0.1, 0.05, "нет в матрице"
for _n in ("fastbt", "fastbt.s9"):                       # metriki_e тянет s9 -> data; нам нужен только msr
    sys.modules[_n] = types.ModuleType(_n)
sys.modules["fastbt.s9"].sharpe = None
_sp = importlib.util.spec_from_file_location("metriki_e", ROOT / "fastbt" / "metriki_e.py")
ME = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(ME)

A_GRID = {   # паспорт A (2026-10-27_pasport_A.md): единой точки нет, в каждом (тикер, год) топ-10 из сетки
    "ma_window": "{15,20,25}", "z_entry": "{1.5,2,2.5}", "z_exit": "{0.3,0.5,0.7}", "bb_window": "{15,20,25}",
    "bb_std": "{1.5,2,2.5}", "dc_window": "{10,15,20,25}", "atr_period": "{10,14,20}", "multiplier": "{2,2.5,3,3.5}",
    "pivot_period": "{weekly,monthly}", "pivot_type": "{classic,fibonacci}", "filter_set": "S5 {2,3,4}; S6 {1,2,3,4}",
    "vwap_window": "{5,10,20,60}", "dev_mult": "{0.5,1,1.5,2}", "dir_bias": "{OFF,ON}"}


def num(s):
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except ValueError:
        return s


def pkey(p):
    q = dict(p)
    if q.get("k_s") == "net":
        q["R"] = "net"
    return tuple(sorted(q.items()))


def fin(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def f3(x):
    return f"{x:.3f}" if fin(x) else "nan"


def calendar():
    import pyarrow.parquet as pq
    t = pq.read_table(MAIN / "data/fastbt/daily/ohlcv_daily_1850.parquet", columns=["date"],
                      filters=[("date", "<=", pd.Timestamp("2019-12-31"))])
    d = pd.to_datetime(t.column("date").to_pandas()).drop_duplicates().sort_values()
    d = d[d >= "2018-01-01"]
    return {y: list(d[d.dt.year == y]) for y in (2018, 2019)}


def load_cfg(sid):
    y = yaml.safe_load((ROOT / f"fastbt/designs/sweep/zh047_{sid}_run1.yaml").read_text())
    axes = {a: v["setka"] for sec in ("signal", "obshchie") for a, v in y[sec].items()}
    rd = lambda n: [r for r in csv.DictReader(l for l in open(ROOT / f"docs/tables/zh047/{sid}_{n}.csv") if not l.startswith("#"))]  # noqa: E731
    row = rd("itog_v2")[-1]
    pt = {a: num(row["p_" + a]) for a in axes}
    mor = {r["os"]: (float(r["mu_star"]), float(r["sigma"])) for r in rd("morris")}
    return axes, pt, row, mor


def fetch(sid, keys):
    got = {}
    for f in sorted((MAIN / f"docs/tables/zh051/_ledger/{sid}").glob("ledger_it*.pkl"), key=lambda p: len(p.name) * 100 + ord(p.name[-5])):
        miss = keys - set(got)
        if not miss:
            break
        d = pickle.load(open(f, "rb"))
        for k in miss:
            if (k, "train") in d:
                got[k] = d[(k, "train")][1]
        del d
    return got


def cell(r):
    if r is None:
        return NA
    if r.get("ne_schitana"):
        return "не считана (div/cbr_div)"
    if r.get("vyrozhdena"):
        return "вырожден (нет сделок)"
    return None


def shape(vals, i):
    """Правило формы (задано до счёта): vals — целевая min(mSR2018, mSR2019) по значениям оси в порядке сетки, i — выбранное.
    Соседи — ближайшие определённые значения слева и справа. Есть сосед лучше выбранного более чем на 0.1 — «склон»;
    все соседи не хуже выбранного более чем на 0.1 — «плато»; иначе (соседи хуже, none лучше) — «пик»."""
    if not fin(vals[i]):
        return NA
    nb = [next((v for v in (vals[:i][::-1] if s < 0 else vals[i + 1:]) if fin(v)), None) for s in (-1, 1)]
    nb = [v for v in nb if v is not None]
    if not nb:
        return NA
    if any(v - vals[i] > DELTA for v in nb):
        return "склон"
    res = "плато" if all(vals[i] - v <= DELTA for v in nb) else "пик"
    return res + (" (один сосед)" if len(nb) == 1 else "")


def slices(sid, axes, pt, got):
    out = {}
    for a, grid in axes.items():
        rows = []
        for v in grid:
            r = got.get(pkey({**pt, a: v}))
            m = (r or {}).get("msr_by_year") or {}
            sd = (r or {}).get("sdelok_y") or {}
            rows.append((v, r, cell(r), m.get(2018, float("nan")), m.get(2019, float("nan")), sd.get(2018), sd.get(2019)))
        out[a] = rows
    return out


def tsel(r):
    return min(r[3], r[4]) if fin(r[3]) and fin(r[4]) else float("nan")


def influence(rows):
    ok = [r for r in rows if r[2] is None]
    sp = [max(x) - min(x) for x in ([r[3] for r in ok if fin(r[3])], [r[4] for r in ok if fin(r[4])]) if len(x) >= 2]
    return max(sp) if sp else float("nan")


def monthly(rec, cal):
    ex = rec["excess_d"]
    out = {}
    for y, xs in ex.items():
        if len(xs) != len(cal[y]):
            return None
        for d, x in zip(cal[y], xs):
            out[f"{y}-{d.month:02d}"] = out.get(f"{y}-{d.month:02d}", 0.0) + float(x)
    return out
