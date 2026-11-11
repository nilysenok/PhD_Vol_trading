"""W8-B, ступень 1 (Ж-253 п. 4; `docs/otchety/harj_x1/W8_SPEC.md` разд. 2–4, 9): W8-G = X1·exp(gap^PIT·β_{m,G(d)}), G(d) — PIT-терциль ln RV^IMOEX дня d. Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS.
Константы (из спецификации, не из данных): `MIN_G` = 30 (= `x1_z_fit.MIN_EXT`), `BMAX` = 0,5, `MINPAST` = 60, `CUT_BAR` = 18:30, `SHIFT` = 252, ddof = 1. Кандидат один, сетки нет.
Публичное API:
- `load_bars`, `day_rv`, `z_series`, `labels`, `shift_z`, `corrupt_bars` — из `x1_w8_gap_z` (см. его docstring): бары IMOEX 10:00…18:30, z = ln RV^IMOEX дня d, PIT-терцили (пороги по дням строго до d, не меньше 60).
- `walk(d, tp, rv, base, gap, grp, w5_beta, w5_refit, pool_fn=None, min_g=MIN_G) -> DataFrame`: первый торговый день D каждого месяца ≥ 2018-07-02 (`ACTIVATION`); пул — закрытые пары t⁺ ≤ D
  (`x1_z_walk.closed_pool`; параметр `pool_fn` — для мутации «закрыто + 1»); метка пары — её собственная `grp[j]` (посчитана один раз, не пересчитывается). В месяце, где W5-G включена (`w5_refit` не NaT):
  β_{m,G} = clip((ū₁,G − ū₀) + ½(s²₁,G − s²₀), 0; 0,5) по парам пула (u = ln(RV/base), ddof = 1; ū₁,G, s²₁,G — по парам gap = 1 группы G, ū₀, s²₀ — по gap = 0), если в пуле не меньше `min_g` пар gap = 1 группы G;
  иначе (группа не включена, группа `na`) — β W5-G месяца `w5_beta`. Месяц без W5-G: β = 0, F = base. F = base·exp(β_{m,G(d)}) в днях gap = 1, в остальных F = base (побитно).
  Колонки: d, t_plus, rv, X1, gap, group, beta (β_{m,G(d)}), F, refit_date, b_T1, b_T2, b_T3 (β_{m,G} месяца), n_T1, n_T2, n_T3 (пар gap = 1 группы в пуле).
  Группа включена в месяце, если W5-G включена и n_G ≥ min_g. Группа не включена ⇒ F = W5-G побитно (тот же `base*np.exp(β W5-G)`).
- `OUTS` — `w8g.parquet`, `w8gb.parquet`; сдвиги — `shift_01.parquet`…`shift_19.parquet`, сдвиг 252 + 100·(j − 1). `shift_z(z, s)` = `np.roll(z, s)`; W8-Gb — сдвиг 252; метки терцилей пересчитываются заново.
- `compute() -> dict`: `require` для W8-G и W8-Gb (первым действием), окно M, gap^PIT и W5-G из `w5g.parquet` (сверка d, rv, X1), W8-G, W8-Gb, 19 сдвигов.
- `write(out=OUT)`: пишет parquet; существующий выход с иным содержимым — отказ (SystemExit). `check(out=OUT) -> list`, `check_pit(step=1) -> list`: только чтение (хэши выходов до и после).
Запуск: `python3 -W ignore -m models_v2.s16.x1_w8_gap [--write | --check | --check-pit]`."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C, x1_w8_diag_feat as FE, x1_w8_reg as REG, x1_z_walk as ZW
from models_v2.s16.x1_w8_diag_core import ROOT
from models_v2.s16.x1_w8_gap_z import SHIFT, corrupt_bars, day_rv, labels, load_bars, shift_z, z_series
from models_v2.s16.x1_ya_walk import ACTIVATION, check_activation, month_starts
from models_v2.s16.x1_z_fit import MIN_EXT as MIN_G

OUT = ROOT / "data/regen/harj_x1/W8"
W5G = ROOT / "data/regen/harj_x1/W5/w5g.parquet"
BMAX, GROUPS = 0.5, ("T1", "T2", "T3")
OUTS = {"W8-G": "w8g.parquet", "W8-Gb": "w8gb.parquet"}
SHIFTS = {j: "shift_%02d.parquet" % j for j in range(1, 20)}


def _beta(u, g, sel) -> float:
    u1, u0 = u[sel], u[g == 0]
    return float(np.clip((u1.mean() - u0.mean()) + 0.5 * (u1.var(ddof=1) - u0.var(ddof=1)), 0.0, BMAX))


def walk(d, tp, rv, base, gap, grp, w5_beta, w5_refit, pool_fn=None, min_g: int = MIN_G) -> pd.DataFrame:  # xtask: длинная — один проход месяцев, как x1_w5_gap.walk
    d, tp, rv, base, gap = pd.DatetimeIndex(d), pd.DatetimeIndex(tp), np.asarray(rv, float), np.asarray(base, float), np.asarray(gap, int)
    grp, w5_beta, w5_refit = np.asarray(grp, object), np.asarray(w5_beta, float), np.asarray(w5_refit, dtype="datetime64[ns]")
    pool_fn = pool_fn or ZW.closed_pool
    check_activation(d, tp)
    n, u = len(d), np.log(rv / base)
    beta, refit = np.zeros(n), np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    bg, ng = {G: np.full(n, np.nan) for G in GROUPS}, {G: np.zeros(n, int) for G in GROUPS}
    starts = [p for p in month_starts(d) if d[p] >= ACTIVATION] + [n]
    for a, b in zip(starts[:-1], starts[1:]):
        if np.isnat(w5_refit[a]):
            continue
        pool = pool_fn(tp, d, d[a])
        refit[a:b], beta[a:b] = np.datetime64(d[a]), w5_beta[a]
        for G in GROUPS:
            sel = (gap[pool] == 1) & (grp[pool] == G)
            ng[G][a:b] = int(sel.sum())
            bG = _beta(u[pool], gap[pool], sel) if sel.sum() >= min_g else float(w5_beta[a])
            bg[G][a:b] = bG
            beta[a:b] = np.where(grp[a:b] == G, bG, beta[a:b])
    F = np.where((gap == 1) & ~np.isnat(refit), base * np.exp(beta), base)
    return pd.DataFrame({"d": d, "t_plus": tp, "rv": rv, "X1": base, "gap": gap, "group": grp.astype(str), "beta": beta, "F": F, "refit_date": refit,
                         **{"b_" + G: bg[G] for G in GROUPS}, **{"n_" + G: ng[G] for G in GROUPS}})


def _inputs() -> tuple:
    x, w = C.load(), pd.read_parquet(W5G)
    if len(w) != len(x) or not (pd.to_datetime(w["d"]).to_numpy() == x["d"].to_numpy()).all() or not np.array_equal(w["rv"].to_numpy(), x["rv"].to_numpy()) or not np.array_equal(w["X1"].to_numpy(), x["F"].to_numpy()):
        raise SystemExit("w5g.parquet ≠ окну M (d, rv, X1)")
    return x, w


def _run(x, w, z, **kw) -> pd.DataFrame:
    return walk(x["d"], x["t_plus"], x["rv"], x["F"], w["gap"], labels(z), w["beta"], pd.to_datetime(w["refit_date"]), **kw)


def compute() -> dict:
    for k in OUTS:
        REG.require(k)
    x, w = _inputs()
    z = z_series(x["d"])
    s = {"W8-G": _run(x, w, z), "W8-Gb": _run(x, w, shift_z(z))}
    sh = {j: _run(x, w, shift_z(z, 252 + 100 * (j - 1))) for j in SHIFTS}
    return dict(x=x, w5=w, z=z, series=s, shifts=sh)


def _files(r: dict) -> dict:
    return {OUTS[k]: v for k, v in r["series"].items()} | {SHIFTS[j]: v for j, v in r["shifts"].items()}


def _equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True)
        return True
    except AssertionError:
        return False


def write(out: Path = OUT) -> None:
    r = compute()
    out.mkdir(parents=True, exist_ok=True)
    for f, df in _files(r).items():
        p = out / f
        if p.exists() and not _equal(pd.read_parquet(p), df):
            raise SystemExit("%s: существует и отличается от пересчёта — перезапись отказана" % p)
        df.to_parquet(p, index=False)


def tree_hash(out: Path = OUT) -> dict:
    ps = sorted(out.glob("*.parquet")) if out.is_dir() else []
    return {str(p): REG._sha(p) for p in ps}


def check_series(f: pd.DataFrame, w5: pd.DataFrame, n: int, name: str) -> list:
    bad, off = [], (f["gap"] == 0) | f["refit_date"].isna()
    if len(f) != n or not (f.loc[off, "F"].to_numpy() == f.loc[off, "X1"].to_numpy()).all():
        bad.append("%s: F ≠ X1 побитно в днях gap = 0 или до включения (или n ≠ %d)" % (name, n))
    inc = np.zeros(len(f), bool)
    for G in GROUPS:
        inc |= ((f["group"] == G) & (f["n_" + G] >= MIN_G)).to_numpy()
    nz = ((f["gap"] == 1) & f["refit_date"].notna()).to_numpy() & ~inc
    if not (f.loc[nz, "F"].to_numpy() == w5.loc[nz, "F"].to_numpy()).all():
        bad.append("%s: группа не включена, а F ≠ W5-G побитно" % name)
    if not f["beta"].between(0.0, BMAX).all():
        bad.append("%s: β вне [0; %s]" % (name, BMAX))
    return bad


def inclusion(f: pd.DataFrame) -> dict:
    m = f.dropna(subset=["refit_date"]).drop_duplicates("refit_date")
    return {G: (str(m.loc[m["n_" + G] >= MIN_G, "refit_date"].min().date()) if (m["n_" + G] >= MIN_G).any() else "нет") for G in GROUPS}


def check(out: Path = OUT) -> list:
    """B.G1: только чтение; хэши выходов до и после обязаны совпасть."""
    h0, r = tree_hash(out), compute()
    bad = []
    for f, df in _files(r).items():
        p = out / f
        bad += [] if p.exists() and _equal(pd.read_parquet(p), df) else ["%s: нет файла или ≠ пересчёту" % f]
    for k, f in r["series"].items():
        bad += check_series(f, r["w5"], 2198, k)
    for j, f in r["shifts"].items():
        bad += check_series(f, r["w5"], 2198, "shift_%02d" % j)
    d = pd.DatetimeIndex(r["x"]["d"])
    if not np.array_equal(np.nan_to_num(FE.imoex_day()["rv"].reindex(d).to_numpy()), np.nan_to_num(day_rv(load_bars()).reindex(d).to_numpy())):
        bad.append("RV IMOEX ≠ источнику диагностики (x1_w8_diag_feat.imoex_day)")
    print("W8-G: месяцы включения групп %s; дат %d; групп na в днях gap: %d" % (inclusion(r["series"]["W8-G"]), len(r["series"]["W8-G"]), int(((r["series"]["W8-G"]["group"] == "na") & (r["series"]["W8-G"]["gap"] == 1)).sum())))
    return bad + ([] if tree_hash(out) == h0 else ["--check изменил выходы"])


def check_pit(step: int = 1) -> list:
    """B.G2: порча RV пар с t⁺ > D и баров IMOEX после d 18:30 не меняет β_{m,G} и F строк d ≤ D (все месяцы с шагом `step`)."""
    x, w = _inputs()
    bars = load_bars()
    ref, bad = _run(x, w, z_series(x["d"], bars)), []
    for p in month_starts(x["d"])[::step]:
        D = x["d"][p]
        rv = x["rv"].to_numpy(float).copy()
        rv[(x["t_plus"] > D).to_numpy()] *= 7.0
        xx = x.copy()
        xx["rv"] = rv
        f = _run(xx, w, z_series(x["d"], corrupt_bars(bars, D)))
        m = (x["d"] <= D).to_numpy()
        cols = ["group", "beta", "F"] + ["b_" + G for G in GROUPS]
        if not f.loc[m, cols].equals(ref.loc[m, cols]):
            bad.append("порча RV пар с t⁺ > %s или баров после d 18:30 изменила β_{m,G} или F до D" % D.date())
    return bad


if __name__ == "__main__":
    if "--write" in sys.argv:
        write()
        print("W8-B: записаны parquet")
    elif "--check" in sys.argv or "--check-pit" in sys.argv:
        b = check() if "--check" in sys.argv else check_pit()
        print("W8 gap %s: %s" % ("--check" if "--check" in sys.argv else "--check-pit", "ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write, --check или --check-pit")
