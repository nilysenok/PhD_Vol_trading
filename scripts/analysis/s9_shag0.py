"""Шаг 0 разработки главы 4 (владелец 27.10, 11:15): стратегии A без прогноза, 2018–2021.

Строки: A; к3 (A без S5/S6); для каждого S1…S6 — «A только Sk» и «A без Sk».
Вклад Sk = Шарп A − Шарп «A без Sk». Δ к A — Ледуа–Вольф, блок 5, 10 000 повторов.
Столбцы по годам 2018…2021 и за период: Шарп, Сортино, просадка (%, по избыточной), экспозиция (%), оборот (в год).

    python3 scripts/analysis/s9_shag0.py [--comm 0.0004] [--reps 10000]
Кэш — s9 --sigma-set v2 (cache_latest_v2.txt); σ не нужны (A от прогноза не зависит). Только dev.
Самопроверка без аргументов: python3 scripts/analysis/s9_shag0.py --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastbt import s9  # noqa: E402
from fastbt.msp import sharpe  # noqa: E402

POMETKA = "после аудита и ОВ/К3; шаг 0, без прогноза, разработка 2018–2021"
PERIODS = ("2018", "2019", "2020", "2021", "period")


def block_stats(p: pd.DataFrame, mask: np.ndarray) -> dict:
    """Шарп, Сортино, просадка, экспозиция, оборот на срезе; формулы — s9.metrics."""
    q = p[mask]
    ex, tot = q["excess"].values, q["total"].values
    if len(ex) < 2:
        return dict(sharpe=np.nan, sortino=np.nan, mdd=np.nan, expo=np.nan, turn=np.nan)
    neg = tot[tot < 0]
    ds = np.std(neg, ddof=1) if len(neg) > 1 else 1e-12
    cum = np.cumsum(ex)
    return dict(sharpe=sharpe(ex), sortino=float(np.mean(ex) * np.sqrt(252) / ds),
                mdd=float(np.max(np.maximum.accumulate(cum) - cum) * 100),
                expo=float(q["exposure"].mean() * 100), turn=float(q["turnover"].mean() * 252))


def wide(p: pd.DataFrame) -> dict:
    yr = p.index.year
    out = {}
    for per in PERIODS:
        mask = np.ones(len(p), bool) if per == "period" else (yr == int(per))
        for k, v in block_stats(p, mask).items():
            out[f"{k}_{per}"] = v
    out["worst_year"] = min(out[f"sharpe_{y}"] for y in PERIODS[:4])
    return out


def masked(A_row: np.ndarray, drop: np.ndarray, pair: np.ndarray) -> np.ndarray:
    return np.where(drop[pair], 0.0, A_row)


def build(comm: float, reps: int) -> pd.DataFrame:
    s9.use_v2(comm)
    eng = s9.Engine(s9.load_cache())
    x = s9.S9(eng, reps)
    pair = np.asarray(eng.r["pair"], np.int64)
    sid = np.array([s for _, s in eng.pairs])
    pA = x.A
    rows = []

    def add(label, pos, base=False):
        p = eng.port(pos)
        r = dict(pometka=POMETKA, row=label, comm_pct=comm * 100)
        r.update(wide(p))
        if not base:
            x.delta(r, "vs_A", p, pA)
        rows.append(r)
        return r
    ra = add("A", eng.A_row, base=True)
    add("к3: A без S5/S6", masked(eng.A_row, np.isin(sid, ["S5", "S6"]), pair))
    for s in s9.SIDS:
        add(f"A только {s}", masked(eng.A_row, sid != s, pair))
        r = add(f"A без {s}", masked(eng.A_row, sid == s, pair))
        r["vklad_sharpe"] = ra["sharpe_period"] - r["sharpe_period"]
    return pd.DataFrame(rows)


def write(df: pd.DataFrame, name: str) -> Path:
    p = Path(__file__).resolve().parents[2] / "docs" / "tables" / "strat_dev_v2" / name
    df.to_csv(p, sep=";", decimal=",", index=False, float_format="%.4f", encoding="utf-8-sig")
    return p


def selftest() -> None:
    idx = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(1)
    ex = rng.normal(0, 0.01, 600)
    p = pd.DataFrame(dict(excess=ex, total=ex, exposure=0.5, turnover=0.1), index=idx)
    w = wide(p)
    assert abs(w["sharpe_period"] - sharpe(ex)) < 1e-12 and abs(w["expo_period"] - 50) < 1e-9
    assert np.isnan(block_stats(p, np.zeros(600, bool))["sharpe"])        # пустой срез — NaN, не ноль
    pos = np.array([1.0, 2.0, 3.0, 4.0])
    out = masked(pos, np.array([False, True]), np.array([0, 1, 0, 1]))
    assert out.tolist() == [1.0, 0.0, 3.0, 0.0]
    print("OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--comm", type=float, default=0.0004)
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args()
    if a.selftest or len(sys.argv) == 1:
        selftest()
    else:
        name = "shag0_bez_prognoza.csv" if abs(a.comm - 0.0004) < 1e-12 else "shag0_bez_prognoza_k0050.csv"
        print(write(build(a.comm, a.reps), name))
