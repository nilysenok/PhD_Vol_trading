"""Замер исполнения Ж-111 на оси ТФ (лист П5b): с/точку на 20 точках каждой семьи S1–S6 v5, tf = 1h, train4 2016–2019 (прогрев 2015).

Те же 17 тикеров, бары и точки (seed 0), что у scripts/zamer_tf_os_v5.py (П5); исполнение — fastbt.tf_ispolnenie_v5.PARAMS
(лимит-вход 0,25σ, стоп 2σ, тейк 2σ, Chandelier 3 ATR, выход по сигналу лимитом на t+1, закрытие в конце сессии).
Чтение 10-минуток строго < 2020-01-01. Выводится на экран, в репозиторий не пишется. Статус: разработка train4, до аудита; потолок.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import zamer_tf_os_v5 as Z  # noqa: E402

from fastbt import bars_agg as ba  # noqa: E402
from fastbt import tf_ispolnenie_v5 as X  # noqa: E402
from fastbt import tf_os_v5 as T  # noqa: E402
from fastbt.data import TICKERS  # noqa: E402


def main() -> None:
    t0 = time.perf_counter()
    tds = {}
    for t in TICKERS:
        b10 = ba.load_10m(t, Z.B, Z.MAIN)
        tds[t] = T.ticker_data(T.bary(b10[b10["date"] >= pd.Timestamp(Z.WARM)], Z.TF), t)
    print(f"чтение и бары {Z.TF}: {time.perf_counter() - t0:.1f} с, {len(tds)} тикеров")
    rng, vse = np.random.default_rng(0), 0.0
    for sid, osi in Z.OSI.items():
        grid = [{k: (x.item() if hasattr(x, "item") else x) for k, x in p.items()} for p in Z.tochki(osi, Z.N_TOCHEK, rng)]
        t1 = time.perf_counter()
        r = pd.DataFrame(X.progon(sid, Z.TF, tds, grid, Z.A, Z.B))
        dt = time.perf_counter() - t1
        vse += dt
        print(f"{sid} {Z.TF}: {Z.N_TOCHEK} точек, {dt / Z.N_TOCHEK:.2f} с/точку; сделок в день (медиана) "
              f"{r['sdelok_v_den'].median():.3f}; заполнено лимитов {r['zapolneno_pct'].median():.1f} %; "
              f"брутто {r['brutto_pct_god'].median():.2f}, издержки {r['izderzhki_pct_god'].median():.2f}, "
              f"нетто {r['netto_pct_god'].median():.2f} %/год (медианы)", flush=True)
    print(f"{Z.TF}: {Z.N_TOCHEK} точек x 6 семей, {vse / (6 * Z.N_TOCHEK):.2f} с/точку (17 тикеров, train4); {T.STATUS}")


if __name__ == "__main__":
    main()
