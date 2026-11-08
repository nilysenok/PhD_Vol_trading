"""V5 лист 2: счёт SP1–SP4 с кэшем (ключ = хэш входов и кода, Д5) и LOTO. Кэш `data/regen/harj_hybrid_v5/SP{n}.parquet` (прогнозы 17 бумаг), `SP{n}_params.parquet`, `SP{n}_loto.parquet` (прогноз каждой бумаги моделью без неё).
Основа — K8 листа 0; входы — энергии полос (кэш листа 0) и форма дня H3 V4 (firsth — `v4_h_feat.firsth_table`)."""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import pandas as pd

from models_v2.s11 import v4_h_feat as HF
from models_v2.s12 import v5_0_bars as BR, v5_0_io as IO, v5_gs_panel as PA, v5_sp_feat as FE, v5_sp_fit as FI, v5_sp_har as HAR

CODE_FILES = ["v5_sp_feat.py", "v5_sp_har.py", "v5_sp_fit.py", "v5_sp_run.py", "v5_gs_panel.py", "v5_gs_fit.py", "v5_0_bars.py"]
CFGS = ["SP1", "SP2", "SP3", "SP4"]


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(f.encode() + (Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:16]


def key(cfg: str) -> str:
    return hashlib.sha256(("%s|%s|%s" % (IO.inputs_hash(), code_hash(), cfg)).encode()).hexdigest()[:16]


def paths(cfg: str) -> dict:
    return {"main": IO.OUT / ("%s.parquet" % cfg), "params": IO.OUT / ("%s_params.parquet" % cfg), "loto": IO.OUT / ("%s_loto.parquet" % cfg)}


def inputs(write: bool = False) -> FE.Feat:
    return FE.build(BR.cached_bands(write), HF.firsth_table(IO.T17), IO.T17)


def _long(P, res: dict) -> pd.DataFrame:
    return PA.long(P, F=res["F"], mult=res["M"])


def compute(P, F: FE.Feat, cfg: str) -> tuple:
    """-> (прогнозы, параметры по месяцам, LOTO). SP1 — модели по бумаге независимы: LOTO = основной прогноз."""
    if cfg == "SP1":
        res = HAR.run(P, F)
        main = _long(P, res)
        return main, res["params"], pd.concat([main[main["ticker"] == t].assign(excl=t) for t in P.tick], ignore_index=True)
    res = FI.run_config(P, F, cfg)
    main, lot = _long(P, res), []
    for j, t in enumerate(P.tick):
        lot.append(_long(P, FI.run_config(P, F, cfg, excl=j)).query("ticker == @t").assign(excl=t))
    return main, res["params"], pd.concat(lot, ignore_index=True)


def get(P, cfg: str, write: bool = False, F: FE.Feat | None = None) -> dict:
    """Кэш с ключом: совпал — читаем; иначе считаем (в `--check` — только в памяти)."""
    p, k = paths(cfg), key(cfg)
    if all(f.exists() for f in p.values()):
        m = pd.read_parquet(p["main"])
        if len(m) and str(m["key"].iloc[0]) == k:
            return {"main": m.drop(columns="key"), "params": pd.read_parquet(p["params"]), "loto": pd.read_parquet(p["loto"]).drop(columns="key"), "cached": True, "key": k}
    main, par, lot = compute(P, inputs(write) if F is None else F, cfg)
    if write:
        IO.OUT.mkdir(parents=True, exist_ok=True)
        main.assign(key=k).to_parquet(p["main"], index=False)
        par.to_parquet(p["params"], index=False)
        lot.assign(key=k).to_parquet(p["loto"], index=False)
    return {"main": main, "params": par, "loto": lot, "cached": False, "key": k}


def frame_with(k: pd.DataFrame, runs: dict) -> pd.DataFrame:
    """Общий кадр листа 0 + колонки SP (прогноз F на строках (бумага, d))."""
    k = k.copy()
    for g, r in runs.items():
        k = k.merge(r["main"][["ticker", "date", "F"]].rename(columns={"F": g}), on=["ticker", "date"], how="left")
    return k
