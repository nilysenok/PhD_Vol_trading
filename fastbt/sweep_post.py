"""Пост-сигнальный слой оценщика Sk (Ж-031 Э1-а, Ж-036): всё, что не входит в ключ кэша сигналов.

Работает на КОПИИ закешированного Unit (copy.copy): кэш сигналов не трогается, ядро fastbt не правится.
- RM: любые значения (не только узлы grids.rm_grid) — Unit._build_rm на одну строку, под замком подменены
  unit.PIV_COMBOS / VWAP_DEVS / TREND_TN;
- маски входа после кэша (объём, ADX, vol_regime, свечи, vf_porog) — только УЖЕСТОЧЕНИЕ зашитых порогов;
  ослабление — «вне v1» (ValueError), нужна правка fastbt;
- квантиль силы q_sily по скользящей истории сигналов (окно 252 бара, ТЕКУЩИЙ БАР НЕ ВХОДИТ);
- порог выхода S6 vyhod_dev; слой исполнения (execlayer.exec_layer); фильтр оборота (turnover.tfilter).
Ключи: `имя` или `<блок>.<имя>` (условные блоки S5/S6 — max_hold и cooldown_bars есть в нескольких блоках).
"""
from __future__ import annotations

import copy
import threading

import numpy as np

from fastbt import unit as unit_mod
from fastbt.batch import trades_one
from fastbt.engine import next_signal
from fastbt.execlayer import exec_layer
from fastbt.stats import pos_from_trades
from fastbt.turnover import tfilter

LOCK = threading.Lock()
QWIN, QMIN = 252, 10                      # окно квантиля силы; меньше QMIN прошлых сигналов — фильтра нет
MASKS = {"vf_porog", "adx_min", "obyom_mnozh_sma20", "vol_regime_max", "svechi_min_iz5", "obyom_nizhe_sma5", "vyhod_dev"}
EXEC = {"consec_entry": 1, "min_strength": 0.0, "q_sily": 0.0, "exec_cooldown": 0, "exec_min_hold": 0,
        "tf_min_hold": 0, "tf_reentry": 0}
SELECT = {"rm_tip", "rm_edinitsa"}
RM_DEF = {
    "atr": dict(sl_mult=1.5, tp_mult=3.0, max_hold=20, breakeven_trigger=None, cooldown_bars=0, time_decay=False,
                partial_exit=False),
    "pivot": dict(sl_target="next_level", tp_target="P", buffer=0.1, max_hold=8, breakeven=False, cooldown_bars=0),
    "vwap": dict(sl_dev=2.0, max_hold=8, breakeven=False, cooldown_bars=0),
    "trend": dict(initial_sl_mult=2.5, trail_type="fixed_atr", trail_n=15, trail_atr_mult=3.0, breakeven_thresh=None,
                  cooldown_bars=0, parabolic_step=0.02, parabolic_max=0.15)}


class VneV1(ValueError):
    """Значение требует правки fastbt (ослабление зашитого порога, sqrt_rv и пр.)."""


def split(sid: str, p: dict, sig_names) -> tuple:
    """Плоские параметры -> (rm_type, rm dict, mask dict, exec dict). Сигнальные ключи пропускаются."""
    trend = sid in ("S3", "S4")
    rtype = "trend" if trend else p.get("rm_tip", "pivot" if sid == "S5" else "atr")
    if rtype == "sqrt_rv" or p.get("rm_edinitsa", "atr14") == "sqrt_rv":
        raise VneV1("RM в единицах √RV (ядро B kind=2) — вне v1")
    rm, mask, ex = dict(RM_DEF[rtype]), {}, dict(EXEC)
    for k, v in p.items():
        blk, _, name = k.rpartition(".")
        if k in sig_names or k in SELECT:
            continue
        if blk.startswith("rm_") and blk not in (f"rm_{rtype}", "rm_parabolic"):
            continue                                   # неактивный условный блок
        if name in MASKS:
            mask[name] = v
        elif name in EXEC:
            ex[name] = v
        elif name in rm:
            rm[name] = v
        elif blk or name not in sig_names:
            raise KeyError(f"неизвестный параметр {k!r} для {sid}/{rtype}")
    if rm.get("trail_type") == "off":
        raise VneV1("trail_type=off — вне v1")
    return rtype, rm, mask, ex


def entry_masks(td, ind, m: dict) -> tuple:
    """(long_ok, short_ok) bool[T]. Только ужесточение зашитых порогов (иначе VneV1)."""
    n = td.n
    lo, sh = np.ones(n, bool), np.ones(n, bool)
    vs20 = ind["vol_sma20"]
    ok20 = ~np.isnan(vs20) & (vs20 > 0)

    def weaker(name, v, base, more):
        if v == "off" or (v < base if more else v > base):
            raise VneV1(f"{name}={v}: ослабление зашитого порога {base} — вне v1")
    if "vf_porog" in m:
        weaker("vf_porog", m["vf_porog"], 0.5, True)
        f = ~(ok20 & (td.volume < m["vf_porog"] * np.where(ok20, vs20, 0.0)))
        lo, sh = lo & f, sh & f
    if "adx_min" in m:
        weaker("adx_min", m["adx_min"], 20, True)
        f = ~np.isnan(ind["adx14"]) & (ind["adx14"] > m["adx_min"])
        lo, sh = lo & f, sh & f
    if "obyom_mnozh_sma20" in m:
        weaker("obyom_mnozh_sma20", m["obyom_mnozh_sma20"], 1.0, True)
        f = ok20 & (td.volume > m["obyom_mnozh_sma20"] * np.where(ok20, vs20, 0.0))
        lo, sh = lo & f, sh & f
    if "vol_regime_max" in m:
        weaker("vol_regime_max", m["vol_regime_max"], 1.2, False)
        f = ~np.isnan(ind["vol_regime"]) & (ind["vol_regime"] < m["vol_regime_max"])
        lo, sh = lo & f, sh & f
    if "svechi_min_iz5" in m:
        weaker("svechi_min_iz5", m["svechi_min_iz5"], 3, True)
        lo = lo & ~np.isnan(ind["red_count5"]) & (ind["red_count5"] >= m["svechi_min_iz5"])
        sh = sh & ~np.isnan(ind["green_count5"]) & (ind["green_count5"] >= m["svechi_min_iz5"])
    if m.get("obyom_nizhe_sma5", "on") != "on":
        raise VneV1("obyom_nizhe_sma5=off — вне v1")
    return lo, sh


def strength_quantile_ok(sig, strength, q: float) -> np.ndarray:
    """bool[T]: сила сигнала на баре t >= q-квантиль силы ПРОШЛЫХ сигналов в окне [t-QWIN, t-1] (бар t не входит).
    Меньше QMIN прошлых сигналов — фильтра нет (True). q=0 — фильтра нет."""
    ok = np.ones(len(sig), bool)
    if q <= 0:
        return ok
    idx = np.flatnonzero(sig != 0)
    for t in idx:
        past = idx[(idx >= t - QWIN) & (idx < t)]
        if len(past) >= QMIN:
            ok[t] = strength[t] >= np.quantile(strength[past], q)
    return ok


def build_view(sid: str, u, td, ind, p: dict, sig_names):
    """Копия Unit с масками, выходом S6, RM-строкой p; возвращает (view, exec dict)."""
    rtype, rm, mask, ex = split(sid, p, sig_names)
    v = copy.copy(u)
    lo, sh = entry_masks(td, ind, mask)
    s = u.sigs[0].copy()
    s = np.where(((s > 0) & ~lo) | ((s < 0) & ~sh), 0, s).astype(np.int8)
    s = np.where(strength_quantile_ok(s, u.strength[0], ex["q_sily"]), s, 0).astype(np.int8)
    v.sigs = np.ascontiguousarray(s[None, :])
    v.nxt = np.stack([next_signal(x) for x in v.sigs])
    if "vyhod_dev" in mask:
        x = u.ss.extra
        e = ~np.isnan(x["vwap"][0]) & (np.abs(td.close - x["vwap"][0]) < mask["vyhod_dev"] * x["dev"][0])
        v.exl = v.exs = np.ascontiguousarray(e[None, :])
    row = dict(rm, _rm_type=rtype if rtype in ("pivot", "vwap") else "atr")
    v.rm = [row]
    with LOCK:
        saved = unit_mod.PIV_COMBOS, unit_mod.VWAP_DEVS, unit_mod.TREND_TN
        try:
            pv, vw = RM_DEF["pivot"], RM_DEF["vwap"]       # расстояния _build_dist считаются всегда; для чужого типа RM — заглушка
            unit_mod.PIV_COMBOS = [(rm.get("sl_target", pv["sl_target"]), rm.get("tp_target", pv["tp_target"]),
                                    rm.get("buffer", pv["buffer"]))]
            unit_mod.VWAP_DEVS = [rm.get("sl_dev", vw["sl_dev"])]
            unit_mod.TREND_TN = [int(rm.get("trail_n", 10))]
            v._build_rm()
        finally:
            unit_mod.PIV_COMBOS, unit_mod.VWAP_DEVS, unit_mod.TREND_TN = saved
    return v, ex


def positions(v, ex: dict, end: int) -> np.ndarray:
    """Позиции [n]: сделки ядра -> слой исполнения -> фильтр оборота."""
    te, tx, td = trades_one(v, 0, v.rmP[0], v.trend, end=end)
    pos = pos_from_trades(te, tx, td, len(te), v.n)
    if ex["consec_entry"] > 1 or ex["exec_min_hold"] > 0 or ex["exec_cooldown"] > 0 or ex["min_strength"] > 0:
        pos = exec_layer(pos, v.strength[0], int(ex["consec_entry"]), int(ex["exec_min_hold"]),
                         int(ex["exec_cooldown"]), float(ex["min_strength"]), end) * np.where(pos < 0, -1.0, 1.0)
    if ex["tf_min_hold"] > 0 or ex["tf_reentry"] > 0:
        pos = tfilter(pos.astype(np.float64), int(ex["tf_min_hold"]), int(ex["tf_reentry"]))
    return pos
