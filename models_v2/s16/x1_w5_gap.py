"""W5-B, ступень 1 (Ж-250, Ж-251, Ж-252; `W5_SPEC.md` разделы 1–3, 11): W5-G = X1·exp(gap^PIT·β_m) и контроли. Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS.
Метка — только `x1_w5_gap_cal.gap_pit` (календарь, известный на d 18:40; Δ_d в модель не входит). Ряды X1, K2, rv — окно M (`x1_w3_core.load`, n = 2198, t⁺ 2018-01-04…2026-11-10).
Публичное API:
- `beta_gap(u, g)`: β = clip((ū₁ − ū₀) + ½(s²₁ − s²₀), 0; 0,5), ddof = 1, u = ln(RV/база); None, если пар gap = 1 или gap = 0 меньше 30 (`MIN_G`).
- `beta_all(u, g=None)` (g не используется; единая сигнатура beta_fn(u, g)):β^a = clip(ū + ½s², −0,5; 0,5), ddof = 1; None, если пар меньше 60 (`MIN_ALL`). Контроль W5-Ga.
- `shift_labels(gap, shift=SHIFT)`: gap^b_i = gap_{(i − shift) mod n} (сдвиг по кругу по строкам окна M; 252). Контроль W5-Gb и перестановки.
- `walk(d, tp, rv, base, gap, beta_fn=beta_gap, all_days=False, pool_fn=None, base_name="X1") -> DataFrame`: первый торговый день D каждого месяца ≥ 2018-07-02 (`x1_ya_walk.ACTIVATION`),
  пул — закрытые пары t⁺ ≤ D (`x1_z_walk.closed_pool`; параметр `pool_fn` для мутации «закрыто + 1»); β_m действует на дни D … конец месяца; до включения и при нехватке пар β = 0, F = база побитно,
  refit_date = NaT. F = база·exp(β) в днях gap = 1 (при `all_days` — во всех днях), в остальных F = база (побитно). Колонки: d, t_plus, rv, <base_name>, gap, beta, F, refit_date.
- `OUTS` — имена parquet в `data/regen/harj_x1/W5/`: w5g, w5ga, w5gb, w5gk2 (у w5gk2 колонка базы — `K2`).
- `compute(cal=None, canon=None) -> dict`: `require` для 4 id (первым действием), окно M, gap^PIT, четыре ряда, таблица GAP_PIT.
- `write(out=OUT)`: пишет parquet и `W5_GAP_PIT.csv`; существующий выход с иным содержимым — отказ (SystemExit).
- `check(out=OUT) -> list`, `check_pit(...) -> list`: только чтение. Запуск: `python3 -W ignore -m models_v2.s16.x1_w5_gap [--write | --check | --check-pit]`."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C, x1_w5_gap_cal as CAL, x1_w5_reg as REG, x1_z_walk as ZW
from models_v2.s16.x1_ya_walk import ACTIVATION, check_activation, month_starts
from models_v2.s16.x1_z_fit import MIN_EXT as MIN_G

OUT = C.ROOT / "data/regen/harj_x1/W5"
GAP_PIT = CAL.GAP_PIT
SHIFT, MIN_ALL, BMAX = 252, 60, 0.5
OUTS = {"W5-G": "w5g.parquet", "W5-Ga": "w5ga.parquet", "W5-Gb": "w5gb.parquet", "W5-GK2": "w5gk2.parquet"}
COLS = ["d", "t_plus", "rv", "X1", "gap", "beta", "F", "refit_date"]


def beta_gap(u, g):
    u, g = np.asarray(u, float), np.asarray(g, int)
    u1, u0 = u[g == 1], u[g == 0]
    if len(u1) < MIN_G or len(u0) < MIN_G:
        return None
    return float(np.clip((u1.mean() - u0.mean()) + 0.5 * (u1.var(ddof=1) - u0.var(ddof=1)), 0.0, BMAX))


def beta_all(u, g=None):
    u = np.asarray(u, float)
    return None if len(u) < MIN_ALL else float(np.clip(u.mean() + 0.5 * u.var(ddof=1), -BMAX, BMAX))


def shift_labels(gap, shift: int = SHIFT) -> np.ndarray:
    return np.roll(np.asarray(gap), shift)


def walk(d, tp, rv, base, gap, beta_fn=beta_gap, all_days: bool = False, pool_fn=None, base_name: str = "X1") -> pd.DataFrame:
    d, tp, rv, base, gap = pd.DatetimeIndex(d), pd.DatetimeIndex(tp), np.asarray(rv, float), np.asarray(base, float), np.asarray(gap, int)
    pool_fn = pool_fn or ZW.closed_pool
    check_activation(d, tp)
    n, u = len(d), np.log(rv / base)
    beta, refit = np.zeros(n), np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    starts = [p for p in month_starts(d) if d[p] >= ACTIVATION] + [n]
    for a, b in zip(starts[:-1], starts[1:]):
        pool = pool_fn(tp, d, d[a])
        beta_m = beta_fn(u[pool], gap[pool])
        if beta_m is not None:
            beta[a:b], refit[a:b] = beta_m, np.datetime64(d[a])
    on = np.ones(n, bool) if all_days else gap == 1
    F = np.where(on & ~np.isnat(refit), base * np.exp(beta), base)
    return pd.DataFrame({"d": d, "t_plus": tp, "rv": rv, base_name: base, "gap": gap, "beta": beta, "F": F, "refit_date": refit})


def compute(cal: pd.DataFrame | None = None, canon: set | None = None) -> dict:
    for k in OUTS:
        REG.require(k)
    x = C.load()
    cal = CAL.load_calendar() if cal is None else cal
    canon = CAL.canonical_saturdays() if canon is None else canon
    tbl = CAL.pit_table(x, cal, canon)
    g = tbl["gap_pit"].to_numpy(int)
    a = (x["d"], x["t_plus"], x["rv"], x["F"])
    s = {"W5-G": walk(*a, g), "W5-Ga": walk(*a, g, beta_all, True), "W5-Gb": walk(*a, shift_labels(g)), "W5-GK2": walk(x["d"], x["t_plus"], x["rv"], x["K2"], g, base_name="K2")}
    return dict(x=x, cal=cal, canon=canon, tbl=tbl, series=s)


def write(out: Path = OUT, gap_path: Path = GAP_PIT) -> None:
    r = compute()
    out.mkdir(parents=True, exist_ok=True)
    for k, f in OUTS.items():
        p = out / f
        if p.exists() and not _equal(pd.read_parquet(p), r["series"][k]):
            raise SystemExit("%s: существует и отличается от пересчёта — перезапись отказана" % p)
        r["series"][k].to_parquet(p, index=False)
    gap_path.write_text(CAL.csv_text(r["tbl"], r["cal"], r["canon"]), encoding="utf-8")


def _equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True)
        return True
    except AssertionError:
        return False


def tree_hash(out: Path = OUT, gap_path: Path = GAP_PIT) -> dict:
    ps = sorted(out.glob("*.parquet")) if out.is_dir() else []
    return {str(p): REG._sha(p) for p in ps + ([gap_path] if gap_path.exists() else [])}


def check_series(s: dict, n: int) -> list:
    bad = []
    for k, f in s.items():
        base = "K2" if k == "W5-GK2" else "X1"
        off = (f["gap"] == 0) if k != "W5-Ga" else np.zeros(len(f), bool)
        off = off | f["refit_date"].isna()
        if len(f) != n or not (f.loc[off, "F"].to_numpy() == f.loc[off, base].to_numpy()).all():
            bad.append("%s: F ≠ базе побитно в днях gap = 0 или до включения (или n ≠ %d)" % (k, n))
        lo, hi = (-BMAX, BMAX) if k == "W5-Ga" else (0.0, BMAX)
        if not f["beta"].between(lo, hi).all():
            bad.append("%s: β вне [%s; %s]" % (k, lo, hi))
    return bad


def check(out: Path = OUT, gap_path: Path = GAP_PIT) -> list:
    """B.G1: только чтение; хэши выходов до и после обязаны совпасть."""
    h0, r = tree_hash(out, gap_path), compute()
    bad = check_series(r["series"], 2198)
    for k, f in OUTS.items():
        p = out / f
        bad += [] if p.exists() and _equal(pd.read_parquet(p), r["series"][k]) else ["%s: нет файла или ≠ пересчёту" % f]
    want = CAL.csv_text(r["tbl"], r["cal"], r["canon"])
    bad += [] if gap_path.exists() and gap_path.read_text(encoding="utf-8") == want else ["W5_GAP_PIT.csv ≠ пересчёту"]
    g = r["series"]["W5-G"]
    print("W5-G: месяц включения %s; доля месяцев β = 0 после включения: %.3f; дат %d" % (
        g["refit_date"].min().date(), float((g.dropna(subset=["refit_date"]).drop_duplicates("refit_date")["beta"] == 0).mean()), len(g)))
    return bad + ([] if tree_hash(out, gap_path) == h0 else ["--check изменил выходы"])


def check_pit(step: int = 1) -> list:
    """B.G2: порча RV пар с t⁺ > D не меняет β и F строк до D включительно (все месяцы с шагом `step`)."""
    r = compute()
    x, g = r["x"], r["tbl"]["gap_pit"].to_numpy(int)
    ref = walk(x["d"], x["t_plus"], x["rv"], x["F"], g)
    bad = []
    for p in month_starts(x["d"])[::step]:
        D = x["d"][p]
        rv = x["rv"].to_numpy(float).copy()
        rv[(x["t_plus"] > D).to_numpy()] *= 7.0
        w = walk(x["d"], x["t_plus"], rv, x["F"], g)
        m = (x["d"] <= D).to_numpy()
        if not (w.loc[m, ["beta", "F"]].to_numpy() == ref.loc[m, ["beta", "F"]].to_numpy()).all():
            bad.append("порча RV пар с t⁺ > %s изменила β или F до D" % D.date())
    return bad


if __name__ == "__main__":
    if "--write" in sys.argv:
        write()
        print("W5-B: записаны parquet и W5_GAP_PIT.csv")
    elif "--check" in sys.argv or "--check-pit" in sys.argv:
        b = check() if "--check" in sys.argv else check_pit()
        print("W5 gap %s: %s" % ("--check" if "--check" in sys.argv else "--check-pit", "ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write, --check или --check-pit")
