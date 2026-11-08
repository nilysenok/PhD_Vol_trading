"""V5 лист 1a: счёт пяти конфигураций GS1–GS5 с кэшем (ключ кэша = хэш входов и кода, Д5) и LOTO. Кэш: `data/regen/harj_hybrid_v5/GS{n}.parquet` (прогнозы 17 бумаг), `GS{n}_params.parquet` (θ по месяцам), `GS{n}_loto.parquet` (прогноз каждой бумаги моделью без неё).
Форма score для GS3–GS5 (GS1 — сырой, GS2 — min(s, 3)) читается из CHECKPOINT.json["GS"]["form"]; её записывает `decide_form` по меньшему `crit_panel` ДО счёта GS3–GS5."""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s12 import v5_0_io as IO, v5_gs_fit as FI, v5_gs_panel as PA

CODE_FILES = ["v5_gs_core.py", "v5_gs_panel.py", "v5_gs_graph.py", "v5_gs_fit.py", "v5_gs_run.py"]
CP = IO.REP / "CHECKPOINT.json"
CFGS = list(FI.VAR)


@functools.lru_cache(maxsize=1)
def code_hash() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(f.encode() + (Path(__file__).parent / f).read_bytes())
    return h.hexdigest()[:16]


def key(cfg: str, clip: bool) -> str:
    return hashlib.sha256(("%s|%s|%s|%s" % (IO.inputs_hash(), code_hash(), cfg, clip)).encode()).hexdigest()[:16]


def clip_of(cfg: str) -> bool:
    """GS2 — ограниченный; GS1 — сырой; GS3–GS5 — по форме, записанной в CHECKPOINT до их счёта."""
    if cfg in ("GS1", "GS2"):
        return cfg == "GS2"
    form = json.loads(CP.read_text()).get("GS", {}).get("form")
    if form not in ("GS1", "GS2"):
        raise RuntimeError("форма score не записана в CHECKPOINT.json (сначала GS1, GS2 и decide_form)")
    return form == "GS2"


def _long(P, res: dict, base: np.ndarray, sel_cols=None) -> pd.DataFrame:
    F = base * np.exp(res["U"])
    return PA.long(P, F=F, u=res["U"], s=res["S"], x=res["X"])


def compute(P, cfg: str, clip: bool) -> tuple:
    """-> (прогнозы, параметры по месяцам, LOTO). LOTO — для каждой бумаги модель без неё, оставляем её собственные строки."""
    var, res = FI.VAR[cfg], FI.run_config(P, P.k8, FI.VAR[cfg], clip)
    main = _long(P, res, P.k8)
    lot = []
    for j, t in enumerate(P.tick):
        r = FI.run_config(P, P.k8, var, clip, excl=j)
        lot.append(_long(P, r, P.k8).query("ticker == @t").assign(excl=t))
    return main, res["params"], pd.concat(lot, ignore_index=True)


def paths(cfg: str) -> dict:
    return {"main": IO.OUT / ("%s.parquet" % cfg), "params": IO.OUT / ("%s_params.parquet" % cfg), "loto": IO.OUT / ("%s_loto.parquet" % cfg)}


def get(P, cfg: str, write: bool = False) -> dict:
    """Кэш с ключом: совпал — читаем; иначе считаем (в `--check` — только в памяти)."""
    clip, p = clip_of(cfg), paths(cfg)
    k = key(cfg, clip)
    if all(f.exists() for f in p.values()):
        m = pd.read_parquet(p["main"])
        if len(m) and str(m["key"].iloc[0]) == k:
            return {"main": m.drop(columns="key"), "params": pd.read_parquet(p["params"]), "loto": pd.read_parquet(p["loto"]).drop(columns="key"), "cached": True, "key": k, "clip": clip}
    main, par, lot = compute(P, cfg, clip)
    if write:
        IO.OUT.mkdir(parents=True, exist_ok=True)
        main.assign(key=k).to_parquet(p["main"], index=False)
        par.to_parquet(p["params"], index=False)
        lot.assign(key=k).to_parquet(p["loto"], index=False)
    return {"main": main, "params": par, "loto": lot, "cached": False, "key": k, "clip": clip}


def frame_with(k: pd.DataFrame, runs: dict) -> pd.DataFrame:
    """Общий кадр листа 0 + колонки GS (прогноз F на строках (бумага, d))."""
    k = k.copy()
    for g, r in runs.items():
        k = k.merge(r["main"][["ticker", "date", "F"]].rename(columns={"F": g}), on=["ticker", "date"], how="left")
    return k


def decide_form(k: pd.DataFrame, g1: pd.DataFrame, g2: pd.DataFrame) -> tuple:
    """Правило формы: меньший `crit_panel` из GS1, GS2 (равные — GS1). -> (форма, {GS1:…, GS2:…})."""
    kk = frame_with(k, {"GS1": {"main": g1}, "GS2": {"main": g2}})
    cp = S.crit_panel(kk, ["GS1", "GS2"]).set_index("series")["crit_panel"]
    return ("GS2" if cp["GS2"] < cp["GS1"] else "GS1"), {s: float(cp[s]) for s in ("GS1", "GS2")}
