"""Л5: мощность семейства из 7 односторонних гипотез нового периода для ФАКТИЧЕСКОГО n_new (протокол, раздел 8) — ТОЛЬКО по архиву (t⁺ <= 2026-02-27).
Перенос `power.py` архитектора (sha256 9966efb32cf7a681…) без изменения расчёта; новое: n_new — длина календаря Л1 (`data/fwd/models/calendar.parquet`, правило IMOEX, по begin),
запись — `docs/otchety/harj_fwd/POWER.csv`, прежний файл (n = 170/179/185) сохранён как `POWER_pre.csv`. Новых данных не читает.

Модель: дневные разности потерь e_t = L_B − L_A (плюс = «A лучше B») семи пар берутся СОВМЕСТНО кольцевым блочным бутстрапом исторических дней окна (блок 10, чувствительность 20);
статистика — DM (Ньюи–Уэст, лаг 5, как `models_v2/s3/dm.py`), p односторонний; Холм по 7; мощность = доля из R = 2000 повторов (seed 0), где гипотеза отвергнута при α = 0,05.
Эффект: (а) исторический — среднее окна; (б) границы 90 %-го блочного интервала — шум окна центрируется и сдвигается на границу; (в) кривая — центрированные e окна плюс сдвиг δ (% базы B)
только на днях-носителях пары (у ZB против X1 — дни e_d = 1). Запуск: `python3 -m models_v2.s17.fwd_power [--n N]` (по умолчанию n_new из календаря).
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s3 import dm as DM
from models_v2.s5.d import stats_d as SD
from models_v2.s17.fwd_src import ARCHIVE_END, LAST_SESSION, ROOT

OUT = ROOT / "docs/otchety/harj_fwd/POWER.csv"
CAL = ROOT / "data/fwd/models/calendar.parquet"
PAIRS = [("X1", "K2"), ("X1", "K3"), ("ZB", "K2"), ("ZB", "K3"), ("ZB", "X1"), ("X1", "K1"), ("ZB", "K1")]
ALPHA, R, SEED, LAG = 0.05, 2000, 0, 5
BLOCK_MAIN, BLOCK_ALT = 10, 20
GRID = (0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -5.0, -7.5, -10.0, -15.0)
SRC = ("data/regen/harj_deep/E1/x1.parquet", "data/regen/harj_x1/Z/zb.parquet", "data/regen/harj_panel_hybrid/har_SBER.parquet", "data/regen/harj_panel_hybrid/k23_SBER.parquet")


def sha16(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def n_new_of_calendar(path: Path = CAL) -> int:
    """Число пар нового периода: t⁺ ∈ (2026-02-27; 2026-11-10] по календарю Л1 (только begin)."""
    c = pd.read_parquet(path)
    return int((c["t_plus"].notna() & (c["t_plus"] > ARCHIVE_END) & (c["t_plus"] <= LAST_SESSION)).sum())


def load() -> tuple:
    x1, zb, h, k = (pd.read_parquet(ROOT / p) for p in SRC)
    d = pd.DatetimeIndex(x1["d"])
    h, k = h.set_index(pd.DatetimeIndex(h["d"])).reindex(d), k.set_index(pd.DatetimeIndex(k["date"])).reindex(d)
    rv = x1["rv"].to_numpy(float)
    assert np.array_equal(pd.DatetimeIndex(zb["d"]), d) and np.array_equal(h["rv"].to_numpy(float), rv) and np.array_equal(k["rv_tplus"].to_numpy(float), rv)
    assert np.array_equal(pd.DatetimeIndex(h["target_date"]), pd.DatetimeIndex(x1["t_plus"]))
    f = {"X1": x1["F"].to_numpy(float), "ZB": zb["F"].to_numpy(float), "K1": h["sigma2"].to_numpy(float), "K2": k["sigma2_k2"].to_numpy(float), "K3": k["sigma2_k3"].to_numpy(float)}
    assert np.allclose(x1["M"].to_numpy(float), np.sqrt(f["K2"] * f["K3"]), rtol=1e-12, atol=0)
    return {n: metrics.qlike(rv, v) for n, v in f.items()}, pd.DatetimeIndex(x1["t_plus"]).year.to_numpy(), {p: sha16(ROOT / p) for p in SRC}


def E_of(L: dict, mask) -> tuple:
    """(n, 7): e = L_B − L_A на днях mask; base (7,) — средний QLIKE базы B окна."""
    return np.column_stack([L[b][mask] - L[a][mask] for a, b in PAIRS]), np.array([L[b][mask].mean() for a, b in PAIRS])


def nw_var_batch(X: np.ndarray, lag: int = LAG) -> np.ndarray:
    """X (R, n) -> (R,) дисперсия среднего по Ньюи–Уэсту (как `dm.nw_variance`)."""
    n = X.shape[1]
    U = X - X.mean(1, keepdims=True)
    v = (U * U).sum(1) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (U[:, k:] * U[:, :-k]).sum(1) / n
    return v / n


def p_one(X: np.ndarray) -> np.ndarray:
    z = X.mean(1) / np.sqrt(np.where(nw_var_batch(X) > 0, nw_var_batch(X), np.nan))
    return np.array([0.5 * math.erfc(t / math.sqrt(2)) if np.isfinite(t) else 1.0 for t in z])


def power(E: np.ndarray, n: int, block: int, seed: int = SEED) -> tuple:
    """E (N, 7) — исторические дни (совместно) -> ((7,) мощность с Холмом по 7, (7,) без поправки)."""
    if n > E.shape[0]:
        raise SystemExit("n > N")
    idx = SD.block_idx(E.shape[0], B_=R, block=block, seed=seed)[:, :n]
    P = np.column_stack([p_one(E[:, j][idx]) for j in range(E.shape[1])])
    return np.array([DM.holm(P[r]) < ALPHA for r in range(R)]).mean(0), (P < ALPHA).mean(0)


def shifted(E: np.ndarray, base: np.ndarray, delta_pct) -> np.ndarray:
    """Центрировать e и сдвинуть на δ (% базы) только на днях-носителях (e ≠ 0): среднее сдвига по всем дням = δ."""
    out = E - E.mean(0, keepdims=True)
    for j in range(E.shape[1]):
        sup = E[:, j] != 0
        out[:, j] = np.where(sup, E[:, j] - E[sup, j].mean(), 0.0) if (~sup).any() else out[:, j]
        out[sup, j] += -delta_pct[j] / 100 * base[j] * len(E) / sup.sum()
    return out


def acf(x: np.ndarray, k: int) -> float:
    u = x - x.mean()
    return float((u[k:] * u[:-k]).sum() / (u * u).sum())


def historical(w: str, E, base, n: int) -> list:
    """Строки «исторический эффект» окна w для n_new = n (оба блока)."""
    eff = -100 * E.mean(0) / base
    bm = SD.boot_means(E, SD.block_idx(len(E), B_=R, block=BLOCK_MAIN, seed=SEED))
    lo, hi = -100 * np.percentile(bm, 95, axis=0) / base, -100 * np.percentile(bm, 5, axis=0) / base
    rows = []
    for blk in (BLOCK_MAIN, BLOCK_ALT):
        ph, p1 = power(E, n, blk)
        ph_lo, ph_hi = power(shifted(E, base, lo), n, blk)[0], power(shifted(E, base, hi), n, blk)[0]
        for j, (a, b) in enumerate(PAIRS):
            rows.append(dict(kind="istoricheskij", window=w, hyp=f"{a}<{b}", n_new=n, block=blk, n_hist=len(E), n_support=int((E[:, j] != 0).sum()), base_qlike=base[j], effect_pct=eff[j],
                             effect_lo90=lo[j], effect_hi90=hi[j], power_holm7=ph[j], power_holm7_at_lo=ph_lo[j], power_holm7_at_hi=ph_hi[j], power_bez_popravki=p1[j], delta_grid_pct=np.nan))
    return rows


def curve(wn: str, E, base, n: int) -> list:
    rows = []
    for g in GRID:
        ph, p1 = power(shifted(E, base, np.full(7, g)), n, BLOCK_MAIN)
        for j, (a, b) in enumerate(PAIRS):
            rows.append(dict(kind="krivaya", window=wn + "_shum", hyp=f"{a}<{b}", n_new=n, block=BLOCK_MAIN, n_hist=len(E), n_support=int((E[:, j] != 0).sum()), base_qlike=base[j], effect_pct=np.nan,
                             effect_lo90=np.nan, effect_hi90=np.nan, power_holm7=ph[j], power_holm7_at_lo=np.nan, power_holm7_at_hi=np.nan, power_bez_popravki=p1[j], delta_grid_pct=g))
    return rows


def compute(n: int, L: dict, y: np.ndarray) -> pd.DataFrame:
    wins = {"2018_2025": (y >= 2018) & (y <= 2025), "2023_2025": (y >= 2023) & (y <= 2025), "2018_2025_bez_2022": (y >= 2018) & (y <= 2025) & (y != 2022)}
    rows = []
    for w, m in wins.items():
        E, base = E_of(L, m)
        rows += historical(w, E, base, n)
        if w != "2018_2025_bez_2022":
            rows += curve(w, E, base, n)
    return pd.DataFrame(rows)


def header(n: int, shas: dict) -> str:
    return (f"# мощность нового периода — ОЦЕНКА ПО АРХИВУ (t⁺ ≤ 2026-02-27), не результат проверки; Ж-248 п. 4; n_new = {n} (календарь Л1, правило IMOEX); DM лаг 5 односторонний, Холм по 7, "
            f"α = 0,05; R = {R}, seed {SEED}; блоки {BLOCK_MAIN} (объявленный) и {BLOCK_ALT} (чувствительность); входы sha16 {shas}; код {sha16(__file__)} (перенос power.py 9966efb32cf7a681)\n")


def main(argv: list) -> int:
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else n_new_of_calendar()
    L, y, shas = load()
    df = compute(n, L, y)
    if "--write" in argv:
        with open(OUT, "w", encoding="utf-8") as f:
            f.write(header(n, shas))
            df.to_csv(f, index=False, lineterminator="\n", float_format="%.6g")
    h = df[(df.kind == "istoricheskij") & (df.block == BLOCK_MAIN)]
    print(f"n_new={n}; строк {len(df)}\n", h[["window", "hyp", "effect_pct", "effect_lo90", "effect_hi90", "power_holm7"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
