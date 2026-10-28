"""«Было / стало» по дивидендам (Ж-057 п. 1, Ж-059/60): Шарп и mSR «A только Sk» (S1–S6), 2018, 2019, train (2018–2019).
Три колонки: без дивидендов (прежнее поведение: lr и сигналы на сырых ценах), D×0,87 (основной), полный D.
Запуск: FASTBT_MAIN=<своя копия> python3 -m fastbt.dividendy_bylo_stalo <out.csv>. Только train (< 2020), validation не читается."""
from __future__ import annotations

import sys
from contextlib import ExitStack
from unittest import mock

import numpy as np
import pandas as pd

from fastbt import dividendy, etap1, msp
from fastbt.metriki_e import msr

END, COMM = pd.Timestamp("2020-01-01"), 0.0004
VARIANTS = ("bez_dividendov", "D_x087", "D_polnyy")


def _patches(st: ExitStack, variant: str) -> None:
    if variant == "bez_dividendov":              # прежний код: событий нет, ряд для сигналов не корректируется
        st.enter_context(mock.patch.object(dividendy, "gap_divs", lambda t, d: np.zeros(len(d))))
        st.enter_context(mock.patch.object(dividendy, "adjust_ohlc", lambda t, d, close, *b: tuple(np.array(x, float) for x in b)))
    elif variant == "D_polnyy":
        st.enter_context(mock.patch.object(dividendy, "DIV_NALOG", 1.0))


def one(sid: str, variant: str) -> dict:
    with ExitStack() as st:
        _patches(st, variant)
        ev = etap1.sk_evaluator(sid)
        p = etap1.base_point(sid)
        sp = {k: v for k, v in p.items() if k in ev.sig_names}
        ex = ev.pnl(ev.signals(sp), p, COMM, END)["excess"]
    out = {}
    for lab, m in (("2018", ex.index.year == 2018), ("2019", ex.index.year == 2019), ("train", ex.index.year >= 2018)):
        x = ex[m].to_numpy()
        out[f"sharpe_{lab}"], out[f"msr_{lab}"] = float(msp.sharpe(x)), float(msr(x))
    return out


def main(path: str) -> pd.DataFrame:
    rows = []
    for sid in ("S1", "S2", "S3", "S4", "S5", "S6"):
        for v in VARIANTS:
            rows.append({"sid": sid, "variant": v, **one(sid, v)})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, float_format="%.4f")
    return df


if __name__ == "__main__":
    main(sys.argv[1])
