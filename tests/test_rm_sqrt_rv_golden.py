"""E1в: старые пути ATR/pivot/vwap не меняются ни на байт. Golden снят ДО правки (sweep_post без sqrt_rv).

S1–S6 × {база, узел RM}, 3 тикера, позиции до 2019-01-01 (2018 в окне) + PnL pnl(): sha256 байтов.
Пересъём 29.10 (Д6): после #370 (дивиденды в PnL) старый golden устарел — он в tests/staryj_golden_2026-10-29_div/ (sha256, причина).
Запуск: FASTBT_MAIN=$PWD pytest tests/test_rm_sqrt_rv_golden.py   (пересъём golden: python3 -m tests.test_rm_sqrt_rv_golden --snyat)
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import sweep_setka as sk

GOLD = Path(__file__).with_name("test_rm_sqrt_rv_golden.json")
DESIGN = Path(__file__).resolve().parent.parent / "fastbt" / "designs" / "sweep"
TICKERS = ["SBER", "LKOH", "MGNT"]
END = pd.Timestamp("2019-01-01")
NODES = {"S1": {"sl_mult": 1.0, "max_hold": 15}, "S2": {"sl_mult": 1.0, "cooldown_bars": 2},
         "S3": {"trail_atr_mult": 4.0, "trail_n": 20}, "S4": {"trail_atr_mult": 4.0, "cooldown_bars": 2},
         "S5": {"rm_tip": "atr", "rm_atr.sl_mult": 0.75, "rm_atr.max_hold": 5},
         "S6": {"rm_tip": "vwap", "rm_vwap.sl_dev": 2.5, "rm_vwap.max_hold": 5}}


def points(sid: str) -> dict:
    base = sk.expand(DESIGN / f"{sid}_setka_e1b.yaml")["base"]
    return {"base": dict(base), "uzel": {**base, **NODES[sid]}}


def snimok(sid: str) -> dict:
    from fastbt.sweep_post import build_view, positions
    from fastbt.sweep_sk import SkEvaluator
    ev = SkEvaluator(sid, TICKERS)
    out = {}
    for name, p in points(sid).items():
        sigs = ev.signals({k: v for k, v in p.items() if k in ev.sig_names})
        h = hashlib.sha256()
        for t, u in sigs.items():
            td, ind = ev._td[t]
            v, ex = build_view(sid, u, td, ind, p, ev.sig_names)
            h.update(np.ascontiguousarray(positions(v, ex, int(np.searchsorted(u.dates, np.datetime64(END))))).tobytes())
        df = ev.pnl(sigs, p, 0.0004, END)
        out[name] = dict(pos=h.hexdigest(), pnl=hashlib.sha256(np.ascontiguousarray(df.values.astype(np.float64)).tobytes()).hexdigest(),
                         n=int(len(df)))
    return out


@pytest.mark.parametrize("sid", ["S1", "S2", "S3", "S4", "S5", "S6"])
def test_starye_puti_pobaytno(sid):
    gold = json.loads(GOLD.read_text("utf-8"))
    assert snimok(sid) == gold[sid], f"{sid}: пути ATR/pivot/vwap изменились относительно golden до правки"


def test_golden_ne_pustoy():
    gold = json.loads(GOLD.read_text("utf-8"))
    assert set(gold) == {"S1", "S2", "S3", "S4", "S5", "S6"}
    assert all(g[k]["n"] > 200 and len(g[k]["pos"]) == 64 for g in gold.values() for k in ("base", "uzel"))
    assert all(g["base"]["pos"] != g["uzel"]["pos"] for g in gold.values()), "узел RM не меняет позиции — golden не чувствителен"


if __name__ == "__main__":
    if "--snyat" not in sys.argv:
        sys.exit("пересъём golden — только до правки кода (--snyat)")
    GOLD.write_text(json.dumps({s: snimok(s) for s in NODES}, indent=1), "utf-8")
    print("golden записан:", GOLD)
