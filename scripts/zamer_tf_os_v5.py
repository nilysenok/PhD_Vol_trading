"""Замер оси ТФ (лист П5): с/точку на 20 точках каждой семьи S1–S6 v5, tf = 1h, train4 2016–2019 (прогрев 2015), 17 тикеров.

Бары — 10-минутки основной копии (candles_10m, чтение строго < 2020-01-01, fastbt.bars_agg.load_10m). Точки — случайная
выборка (seed 0) из осей семьи; оси S1/S2 здесь — для замера, не сетка v5 (её задаёт zh126_*.yaml). Время — сигналы +
исполнение внутри дня + строка издержек, без чтения баров; выводится на экран, в репозиторий не пишется.
Статус: разработка train4, до аудита; доходы — потолок, не результат.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastbt import bars_agg as ba  # noqa: E402
from fastbt import tf_os_v5 as T  # noqa: E402
from fastbt.data import TICKERS  # noqa: E402
from fastbt.signals import v5_s12, v5_s34, v5_s56  # noqa: E402

MAIN = Path(os.environ.get("TF_MAIN", "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading"))
TF, N_TOCHEK, A, B, WARM = "1h", 20, "2016-01-01", "2020-01-01", "2015-01-01"
OSI = {
    "S1": dict(ma_tip=list(v5_s12.TIPY), ma_window=[5, 10, 20, 30, 50, 100], dev_tip=list(v5_s12.DEV_TIPY),
               z_entry=[1.0, 1.5, 2.0, 2.5, 3.0], z_exit=[0.0, 0.5, 1.0, "net"]),
    "S2": dict(ma_tip=list(v5_s12.TIPY), bb_window=[10, 20, 30, 50], bb_std=[1.5, 2.0, 2.5, 3.0],
               podtv=list(v5_s12.PODTV)),
    "S3": v5_s34.S3_GRID_V5, "S4": v5_s34.S4_GRID_V5, "S5": v5_s56.G5, "S6": v5_s56.G6,
}


def tochki(osi: dict, n: int, rng) -> list:
    return [{k: v[rng.integers(len(v))] for k, v in sorted(osi.items())} for _ in range(n)]


def main() -> None:
    t0 = time.perf_counter()
    tds = {}
    for t in TICKERS:
        b10 = ba.load_10m(t, B, MAIN)
        tds[t] = T.ticker_data(T.bary(b10[b10["date"] >= pd.Timestamp(WARM)], TF), t)
    print(f"чтение и бары {TF}: {time.perf_counter() - t0:.1f} с, {len(tds)} тикеров, "
          f"баров на тикер ~{int(np.mean([td.n for td in tds.values()]))}")
    rng, vse = np.random.default_rng(0), 0.0
    for sid, osi in OSI.items():
        grid = [{k: (x.item() if hasattr(x, "item") else x) for k, x in p.items()} for p in tochki(osi, N_TOCHEK, rng)]
        t1 = time.perf_counter()
        rows = T.progon(sid, TF, tds, grid, A, B)
        dt = time.perf_counter() - t1
        vse += dt
        r = pd.DataFrame(rows)
        print(f"{sid} {TF}: {N_TOCHEK} точек, {dt / N_TOCHEK:.3f} с/точку; сделок в день (медиана) "
              f"{r['sdelok_v_den'].median():.3f}; издержки (медиана) {r['izderzhki_pct_god'].median():.2f} %/год")
    print(f"{TF}: {N_TOCHEK} точек x 6 семей, {vse / (6 * N_TOCHEK):.3f} с/точку (17 тикеров, train4); {T.STATUS}")


if __name__ == "__main__":
    main()
