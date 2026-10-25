"""Варианты дизайна B/C/D (С3): ручки поверх сеток кода. Применяются в каждом процессе
(run_unit) из Config.design — кортежа пар (ключ, значение); пустой = как в коде.

Ключи yaml (в fastbt.dev всё, что не поле Config, кроме tickers/sids/vpred):
  horizons            — горизонты σ_pred в B/C/D, напр. [h1] (без h22 term_filter выключен);
  sigma_smooth        — ewm(span) σ_pred до B/C/D (None — нет);
  c_thresholds        — пороги процентильного ранга C (трендовые/контртрендовые, v4 и v41);
  c_lookbacks         — окна ранга C;
  c_trend_direction   — направления σ для трендовых в C (OFF/rise/fall);
  c_trend_hysteresis  — гистерезис трендовых в C;
  d_gate_thresholds, d_target_vols, d_smooth — сетка затвора D;
  d_mode              — all | standard | inverse;
  ens_k, vote_frac    — размер ансамбля A (топ-k) и порог голосования (доля ненулевых).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from fastbt import grids

_NAMES = ["C_HORIZONS", "D_HORIZONS", "B_GRIDS", "C_TERM_FILTER", "C_THRESHOLDS",
          "C_THRESHOLDS_V41", "C_LOOKBACKS", "C_DIRECTION_TREND", "C_HYSTERESIS_TREND",
          "D_GATE_THRESHOLDS", "D_TARGET_VOLS", "D_SMOOTH", "D_STANDARD", "D_INVERSE",
          "ENS_K", "VOTE_FRAC"]
_DEFAULT = {k: copy.deepcopy(getattr(grids, k)) for k in _NAMES}
_APPLIED = None
KEYS = {"horizons", "sigma_smooth", "c_thresholds", "c_lookbacks", "c_trend_direction",
        "c_trend_hysteresis", "d_gate_thresholds", "d_target_vols", "d_smooth", "d_mode",
        "ens_k", "vote_frac"}


def freeze(d: dict) -> tuple:
    def f(v):
        return tuple(f(x) for x in v) if isinstance(v, (list, tuple)) else v
    bad = set(d) - KEYS
    if bad:
        raise KeyError(f"неизвестные ключи дизайна: {sorted(bad)}")
    return tuple(sorted((k, f(v)) for k, v in d.items()))


def apply(design: tuple) -> None:
    global _APPLIED
    if design == _APPLIED:
        return
    for k, v in _DEFAULT.items():
        setattr(grids, k, copy.deepcopy(v))
    d = dict(design)
    if "horizons" in d:
        hz = list(d["horizons"])
        grids.C_HORIZONS, grids.D_HORIZONS = hz, hz
        for cat in grids.B_GRIDS:
            grids.B_GRIDS[cat]["horizon"] = hz
        if "h22" not in hz:
            grids.C_TERM_FILTER = [False]
    simple = {"c_lookbacks": "C_LOOKBACKS", "c_trend_direction": "C_DIRECTION_TREND",
              "c_trend_hysteresis": "C_HYSTERESIS_TREND", "d_gate_thresholds": "D_GATE_THRESHOLDS",
              "d_target_vols": "D_TARGET_VOLS", "d_smooth": "D_SMOOTH"}
    for k, name in simple.items():
        if k in d:
            setattr(grids, name, list(d[k]))
    if "c_thresholds" in d:
        grids.C_THRESHOLDS = grids.C_THRESHOLDS_V41 = list(d["c_thresholds"])
    if "d_mode" in d:
        grids.D_STANDARD = d["d_mode"] in ("all", "standard")
        grids.D_INVERSE = d["d_mode"] in ("all", "inverse")
    if "ens_k" in d:
        grids.ENS_K = int(d["ens_k"])
    if "vote_frac" in d:
        grids.VOTE_FRAC = float(d["vote_frac"])
    from fastbt import masks
    masks._CACHE.clear()
    _APPLIED = design


def prepare_td(td, design: tuple):
    """σ_pred со сглаживанием (ewm, только прошлое) — копия TickerData."""
    span = dict(design).get("sigma_smooth")
    if not span:
        return td
    import dataclasses
    sg = {h: pd.Series(v).ewm(span=int(span), min_periods=1, ignore_na=True).mean()
          .where(~np.isnan(v)).values for h, v in td.sigma.items()}
    return dataclasses.replace(td, sigma=sg)
