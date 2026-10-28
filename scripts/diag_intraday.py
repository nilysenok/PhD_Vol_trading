"""Ж-097: внутридневные признаки — информация без стратегий (терцили, контроль) и на сделках A_strogaya.
Запуск: python3 scripts/diag_intraday.py [--no-sdelki]   (10-минутки — основная копия, только чтение; train4 2016–2019).
Выход: docs/tables/progon_e1/intraday_golaya.csv; кэш признаков — data/fastbt/intraday/features_train4.parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastbt.daily_1850 import default_main  # noqa: E402
from fastbt.intraday_priznaki import (END, FEATURES, cell_stats, load_features, panel, pit_f1_f3,  # noqa: E402
                                      tercile_cell, with_pit)

OUT = ROOT / "docs" / "tables" / "progon_e1" / "intraday_golaya.csv"
CACHE = ROOT / "data" / "fastbt" / "intraday" / "features_train4.parquet"
KS, NREP = (1, 5), 100
TREND, KONTR = ("S3", "S4"), ("S1", "S2", "S5", "S6")
TICK = ["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX", "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM",
        "SBER", "TATN", "VTBR"]


def get_features() -> dict:
    if CACHE.exists():
        d = pd.read_parquet(CACHE)
        return {t: g.drop(columns="ticker").set_index("date") for t, g in d.reset_index().groupby("ticker")}
    src = default_main() / "moex_discovery" / "data" / "raw" / "candles_10m"
    f = with_pit(load_features(TICK, src), pit_f1_f3(default_main() / "data" / "regen" / "features_pit_v3.parquet"))
    late = {t: int((pd.read_parquet(src / f"{t}.parquet", columns=["timestamp"], filters=[("timestamp", "<", END.to_pydatetime())])
                    ["timestamp"].dt.hour * 60 + 0 >= 19 * 60).sum()) for t in TICK}
    print("бары с началом ≥ 19:00 до 2020 (вечерняя сессия):", {t: n for t, n in late.items() if n}, flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([v.assign(ticker=t) for t, v in f.items()]).reset_index().to_parquet(CACHE, index=False)
    return f


def row(razdel, gruppa, f, gor, s, ctrl=None) -> dict:
    r = dict(razdel=razdel, gruppa=gruppa, priznak=f, gorizont=gor, n_nizh=int(s["n"][0]), n_sred=int(s["n"][1]), n_verh=int(s["n"][2]),
             m_nizh_pct=100 * s["m"][0], m_sred_pct=100 * s["m"][1], m_verh_pct=100 * s["m"][2], spred_pct=100 * s["spread"],
             t_spred=s["t"], t_nizh=s["t_cell"][0], t_sred=s["t_cell"][1], t_verh=s["t_cell"][2])
    c = ctrl if ctrl is not None else np.array([])
    r.update(kontrol_n=len(c), kontrol_dolya_abs_ge=float((np.abs(c) >= abs(s["spread"])).mean()) if len(c) else np.nan,
             kontrol_q95_abs_pct=100 * float(np.quantile(np.abs(c), 0.95)) if len(c) else np.nan,
             kontrol_sred_pct=100 * float(c.mean()) if len(c) else np.nan)
    return r


def informatsiya(feats: dict, rf: pd.Series) -> list:
    rows, rng = [], np.random.default_rng(97)
    for f in FEATURES:
        real = panel(feats, rf, f, KS)
        ctrl = {k: [] for k in KS}
        for _ in range(NREP):
            p = panel(feats, rf, f, KS, shuffle=rng)
            for k in KS:
                y, c, d = p[k]
                ctrl[k].append(cell_stats(y, c, d)["spread"])
        for k in KS:
            rows.append(row("informatsiya", "vse_tikery", f, f"d+1" if k == 1 else "d+1..d+5", cell_stats(*real[k]), np.array(ctrl[k])))
        print(f, "готово", flush=True)
    return rows


def sdelki_A() -> pd.DataFrame:
    """Сделки A_strogaya (якорь, S1–S6): тикер, день сигнала, sid, pnl сделки = side·(exit/entry − 1) − комиссии."""
    from fastbt import etap1
    from fastbt.a_strogaya import COMM, PER, SIDS, AStrogaya
    out = []
    for sid in SIDS:
        inp = etap1.build_inputs(sid, ev=None, period=PER)
        ev = AStrogaya(sid, inp, etap1.base_point(sid), COMM)
        ev.evaluate_v2({"t_vyhod": "open2"}, PER)
        for run in ev._last[4]:
            for tr in run.trades:
                if not tr.reason:
                    continue                                     # открытые на конце окна не входят
                dts = inp.meta[tr.ticker]["dates"]
                out.append(dict(tiker=tr.ticker, d_sig=pd.Timestamp(dts[tr.d_sig]), sid=sid,
                                pnl=tr.side * (tr.exit_px / tr.entry_px - 1.0) - tr.fee_in - tr.fee_out))
        print(sid, "сделок", sum(o["sid"] == sid for o in out), flush=True)
    d = pd.DataFrame(out)
    return d[(d["d_sig"] >= "2016-01-01") & (d["d_sig"] <= "2019-12-31")]


def na_signalah(feats: dict, tr: pd.DataFrame) -> list:
    rows = []
    for f in FEATURES:
        cells = {t: tercile_cell(d[f]) for t, d in feats.items()}
        tr = tr.assign(cell=[cells[t].get(d, np.nan) for t, d in zip(tr["tiker"], tr["d_sig"])])
        ok = tr[tr["cell"].notna()]
        for name, sids in (("kontrtrend", KONTR), ("trend", TREND), ("vse", KONTR + TREND)):
            g = ok[ok["sid"].isin(sids)]
            rows.append(row("sdelki_A_strogaya", name, f, "sdelka", cell_stats(g["pnl"].to_numpy(), g["cell"].to_numpy(),
                                                                                 g["d_sig"].to_numpy().astype("datetime64[D]"))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sdelki", action="store_true")
    a = ap.parse_args()
    from fastbt import msp
    feats = get_features()
    days = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in feats.values()])))
    rf = msp.rate_on(days)
    rows = informatsiya(feats, rf)
    if not a.no_sdelki:
        rows += na_signalah(feats, sdelki_A())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.5g")
    print(pd.DataFrame(rows)[["razdel", "gruppa", "priznak", "gorizont", "spred_pct", "t_spred", "kontrol_dolya_abs_ge"]].to_string())


if __name__ == "__main__":
    main()
