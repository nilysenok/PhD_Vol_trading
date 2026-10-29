"""Δ Шарпа ансамбля s9 (путь A, train4 2016–2019) между старым и новым кэшем: по Sk и по тикерам.
Запуск: FASTBT_MAIN=<копия> python3 scripts/delta_kesh_s9_train4.py <старый.pkl> <новый.pkl> <выход.csv> [--comm 0.0004].
Шарп — s9.metrics (ddof=1, 252) от портфеля Engine.port: по Sk — все тикеры, по паре (тикер, Sk) — одна пара.
Комиссия 0,04 % за сторону (рабочая; в кэше позиции от неё не зависят)."""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastbt import s9  # noqa: E402


def sharpes(cache: dict, comm: float) -> pd.DataFrame:
    """Строки: (тикер|ALL, Sk|ALL) → Шарп; ALL по Sk и по тикеру — срезы портфеля."""
    eng = s9.Engine(cache)
    pair = eng.r["pair"]
    out = {}

    def one(mask, key):
        p = eng.port(np.where(mask, eng.A_row, 0.0), comm=comm)
        out[key] = s9.metrics(p)["sharpe"]

    one(np.ones(len(pair), bool), ("ALL", "ALL"))
    for sid in s9.SIDS:
        one(np.isin(pair, [i for i, (_, s) in enumerate(eng.pairs) if s == sid]), ("ALL", sid))
    for t in eng.tickers:
        one(np.isin(pair, [i for i, (tt, _) in enumerate(eng.pairs) if tt == t]), (t, "ALL"))
        for sid in s9.SIDS:
            one(np.isin(pair, [i for i, pr in enumerate(eng.pairs) if pr == (t, sid)]), (t, sid))
    return pd.Series(out, name="sharpe").rename_axis(["ticker", "sk"]).reset_index()


def main(old: str, new: str, dest: str, comm: float) -> pd.DataFrame:
    res = []
    with s9.period_ctx("train4"):
        for lab, p in (("old", old), ("new", new)):
            df = sharpes(pickle.loads(Path(p).read_bytes()), comm).rename(columns={"sharpe": lab})
            res.append(df.set_index(["ticker", "sk"]))
    d = res[0].join(res[1])
    d["delta"] = d["new"] - d["old"]
    d.round(4).to_csv(dest)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("dest")
    ap.add_argument("--comm", type=float, default=0.0004)
    a = ap.parse_args()
    d = main(a.old, a.new, a.dest, a.comm)
    print(d.loc[[("ALL", s) for s in ["ALL"] + list(s9.SIDS)]].round(4).to_string())
    t = d.xs("ALL", level="sk").drop("ALL")
    print(t.round(4).to_string())
    print("max|Δ| по парам:", float(d["delta"].abs().max().round(4)), "; пар с Δ≠0:", int((d["delta"].abs() > 1e-9).sum()), "из", len(d))
