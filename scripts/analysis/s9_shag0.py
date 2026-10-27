"""Шаг 0 разработки главы 4 (v3): A без прогноза, 2018–2021 → docs/tables/strat_dev_v3/shag0_*.csv.

    FASTBT_MAIN=<своя копия> python3 scripts/analysis/s9_shag0.py [--reps 10000]
Кэш — s9 v2 (cache_latest_v2.txt); σ не нужны (A от прогноза не зависит), проверка сдвига σ не применяется.
Комиссия 0,04 % (основная) и 0,05 % (рядом). Расчёт — fastbt.shag0.build; сверка быстрого слоя (Engine.port)
с медленным (msp.portfolio) на 5 строках — shag0_sverka_sloy.csv. Самопроверка без аргументов: --selftest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import shag0  # noqa: E402
from fastbt.fast_kontrol import k3_sids  # noqa: E402

OUT = ROOT / "docs" / "tables" / "strat_dev_v3"
COMMS = (0.00035, 0.0004)          # институциональный инвестор; «Премиум» Т-Банка (Ъ 27.10, 12:05)
COMMS_PRIL = (0.0005, 0.0006)      # чувствительность — приложение


def write(rows: list, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    pd.DataFrame(rows).to_csv(p, sep=";", decimal=",", index=False, float_format="%.4f", encoding="utf-8-sig")
    return p


def slow_port(eng, pos, comm):
    """Медленный слой: msp.portfolio на длинной таблице позиций (так считает пайплайн)."""
    from fastbt import msp
    df = pd.DataFrame(dict(date=eng.dates[np.asarray(eng.r["date"])],
                           ticker=np.array(eng.tickers)[np.asarray(eng.r["tick"])],
                           strategy=shag0.sid_row(eng), position=pos, daily_gross_return=pos * eng.lr_row))
    return msp.portfolio(df, msp.rate_cbr(), comm)


def sverka(eng, comm: float) -> list:
    """5 строк: A, к3, A без S3, A только S4, A × 0,5 — строки метрик быстрого и медленного слоя (формат CSV) равны."""
    A, sidr = np.asarray(eng.A_row, float), shag0.sid_row(eng)
    cases = [("A", A), ("к3", shag0.zero_sids(A, sidr, k3_sids())), ("A без S3", shag0.zero_sids(A, sidr, ["S3"])),
             ("A только S4", shag0.keep_sids(A, sidr, ["S4"])), ("A × 0,5", 0.5 * A)]
    rows = []
    for lab, pos in cases:
        a, b = eng.port(pos, comm=comm), slow_port(eng, pos, comm)
        b = b.reindex(a.index)
        sa = pd.DataFrame([shag0.stats(a, np.ones(len(a), bool))]).to_csv(sep=";", float_format="%.4f", index=False)
        sb = pd.DataFrame([shag0.stats(b, np.ones(len(b), bool))]).to_csv(sep=";", float_format="%.4f", index=False)
        rows.append(dict(pometka=shag0.POMETKA, row=lab, comm_pct=comm * 100, metriki_do_4_znaka=sa == sb,
                         max_abs_excess="%.1e" % float(np.abs(a["excess"].values - b["excess"].values).max())))
    return rows


def write_pair(eng, pair: tuple, reps: int, suffix: str, res: dict) -> None:
    """Таблицы для пары комиссий; suffix — "" (основные) или "_prilozhenie"."""
    c0, c1 = pair
    cat = lambda k: res[c0][k] + res[c1][k]          # noqa: E731
    write(cat("metriki"), f"shag0_A_metriki{suffix}.csv")
    write(cat("godam"), f"shag0_A_po_godam{suffix}.csv")
    t0, t1 = res[c0]["tikeram"], res[c1]["tikeram"]
    for a, b in zip(t0, t1):
        a[f"sharpe_k{c1 * 1e5:.0f}"], a[f"sharpe_k3_k{c1 * 1e5:.0f}"] = b["sharpe"], b["sharpe_k3"]
    write(t0, f"shag0_A_po_tikeram{suffix}.csv")
    write(cat("loo"), f"shag0_signaly_LOO{suffix}.csv")
    write(cat("k3"), f"shag0_k3{suffix}.csv")
    write(cat("spravochno"), f"shag0_spravochno{suffix}.csv")


def run(reps: int) -> None:
    """Основные колонки — 0,035 % и 0,04 % (Ъ 27.10, 12:05); 0,05 и 0,06 % — файлы *_prilozhenie."""
    from fastbt import s9
    s9.use_v2(0.0004)
    eng = s9.Engine(s9.load_cache())
    res = {c: shag0.build(eng, c, reps) for c in COMMS + COMMS_PRIL}
    write_pair(eng, COMMS, reps, "", res)
    write_pair(eng, COMMS_PRIL, reps, "_prilozhenie", res)
    write(shag0.semyi(eng), "shag0_semyi_dni.csv")
    sv = sverka(eng, COMMS[1])
    write(sv, "shag0_sverka_sloy.csv")
    print("сверка слоёв:", [(r["row"], r["metriki_do_4_znaka"], r["max_abs_excess"]) for r in sv])


def selftest() -> None:
    assert k3_sids() == ("S5", "S6")
    idx = pd.bdate_range("2018-01-01", periods=300)
    ex = np.random.default_rng(1).normal(0, 0.01, 300)
    p = pd.DataFrame(dict(excess=ex, total=ex, exposure=0.5, turnover=0.1), index=idx)
    s = shag0.stats(p, np.ones(300, bool))
    assert abs(s["expo_pct"] - 50) < 1e-9 and abs(s["dni_v_rynke_pct"] - 100) < 1e-9
    assert np.isnan(shag0.stats(p, np.zeros(300, bool))["sharpe"])
    print("OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sverka", action="store_true", help="только сверка слоёв")
    ap.add_argument("--reps", type=int, default=10000)
    a = ap.parse_args()
    if a.sverka:
        from fastbt import s9
        s9.use_v2(0.0004)
        sv = sverka(s9.Engine(s9.load_cache()), COMMS[1])
        write(sv, "shag0_sverka_sloy.csv")
        print(sv)
    else:
        selftest() if (a.selftest or len(sys.argv) == 1) else run(a.reps)
