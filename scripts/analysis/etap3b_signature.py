#!/usr/bin/env python3
"""etap3b п.5: signature plot и автокорреляция 10-минутных доходностей.

Использование: etap3b_signature.py [ТИКЕР ...]  (без аргументов - все 17).
Дневная RV = сумма квадратов лог-доходностей close-to-close внутри дня на
сетке частоты f минут (10/20/30/60); ночные интервалы не входят. Сетка:
каждый (f/10)-й 10-минутный бар от первого бара дня. Дни 28.02-24.03.2022
исключены. Автокорреляция 1-го порядка - Пирсон между r_t и r_{t-1}
внутри дня (пары через ночь не берутся). Пишет data/out/etap3b_signature.json
и docs/otchety/img/signature_plot.png (если есть matplotlib).
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
CANDLES = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/moex_discovery/data/clean/candles_10m")
TICKERS = ['AFLT', 'ALRS', 'HYDR', 'IRAO', 'LKOH', 'LSRG', 'MGNT', 'MOEX', 'MTLR',
           'MTSS', 'NVTK', 'OGKB', 'PHOR', 'RTKM', 'SBER', 'TATN', 'VTBR']
FREQS = (10, 20, 30, 60)
HALT = (pd.Timestamp("2022-02-28"), pd.Timestamp("2022-03-24"))


def one(tk):
    d = pd.read_parquet(CANDLES / f"{tk}.parquet", columns=["timestamp", "close"])
    d = d[d.close > 0].sort_values("timestamp")
    d["day"] = d.timestamp.dt.normalize()
    d = d[(d.day < HALT[0]) | (d.day > HALT[1])]
    d["lc"] = np.log(d.close)
    d["k"] = d.groupby("day").cumcount()
    res = {}
    for f in FREQS:
        s = d[d.k % (f // 10) == 0]
        r = s.groupby("day").lc.diff()
        res[f] = (r ** 2).groupby(s.day).sum(min_count=1).dropna()
    r = d.groupby("day").lc.diff().dropna()  # 10-мин доходности внутри дня
    prev = r.groupby(d.day.loc[r.index]).shift(1)
    m = prev.notna()
    ac = float(np.corrcoef(r[m], prev[m])[0, 1])
    return {f: (float(v.mean()), len(v)) for f, v in res.items()}, ac, len(r)


def plot(table, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tk, v in table.items():
        ax.plot(FREQS, [v[str(f)] / v["10"] for f in FREQS], color="0.8", lw=0.8)
    ax.plot(FREQS, [np.mean([v[str(f)] / v["10"] for v in table.values()]) for f in FREQS],
            color="C0", lw=2.2, marker="o", label="среднее по 17 тикерам")
    ax.axhline(1, color="k", lw=0.6, ls="--")
    ax.set_xticks(FREQS)
    ax.set_xlabel("частота, минут")
    ax.set_ylabel("средняя дневная RV / RV(10 мин)")
    ax.set_title("Signature plot, MOEX, 17 тикеров, 2014-2026")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    return True


def main(argv):
    tks = argv or TICKERS
    out, t0 = {}, time.time()
    for tk in tks:
        rv, ac, n = one(tk)
        out[tk] = {**{str(f): rv[f][0] for f in FREQS}, "ac1": ac, "n_ret": n, "n_days": rv[10][1]}
        print(tk, f"{time.time() - t0:.1f}с", {k: round(v, 6) for k, v in out[tk].items()}, flush=True)
    if argv:
        return
    (REPO / "data/out").mkdir(parents=True, exist_ok=True)
    (REPO / "data/out/etap3b_signature.json").write_text(json.dumps(out, indent=1))
    print("png:", plot(out, REPO / "docs/otchety/img/signature_plot.png"))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
