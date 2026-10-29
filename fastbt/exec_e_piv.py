"""V6-10 (И11, И12; v4 `s5_rerun.precompute_pivot_sl_tp`, `s5s6_rerun.precompute_vwap_sl_tp`): стоп и тейк по уровням пивота S5, стоп по VWAP S6.

Расстояния — как в v4 (от цены входа, фиксируются на входе, как tT в Ж-084):
  стоп piv_sl "next": лонг |S1 − S2|, шорт |R2 − R1|; "half": половина; ×(1 + piv_buf). NaN S2/R2 — S1 − |P − S1| / R1 + |R1 − P|;
  стоп piv_sl "vwap": лонг |close_d − (VWAP − m·dev)|, шорт |(VWAP + m·dev) − close_d|, m = vwap_m;
  тейк piv_tp "P": лонг |P − S1|, шорт |R1 − P|; "R": |R1 − S1|; "half": ½|R1 − S1|.
Нижняя граница расстояния 0,001·close_d; нечисловое — тоже она (как в v4 engine.py: sd != sd → 0,001·c).
Данные: пивот — как `signals/v5_s56.pivots` (бар периода видит уровни прошлого периода, ≤ d−1); VWAP/dev и close дня сигнала d — на закрытии d
(сам сигнал считается по ним же; вход не раньше close d / d+1, Д2). Уровни дня d_sig берутся строкой d_sig, не d_sig+1.
Уровни передаются в run через `tejk_M` (канал уровней по дням, Ж-084) объектом PivUrovni при Params.tejk = "net".
Стоп один: piv_sl несовместим с b, atr_m, chand, k_tr, be, b0…, стоп-заявкой. Выключено (None) — путь прежний побайтно.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fastbt.exec_e_rules import ceil_tick, floor_tick

PIV_SL, PIV_TP, MIN_D = ("next", "half", "vwap"), ("P", "R", "half"), 0.001


@dataclass(frozen=True)
class PivUrovni:
    """Ряды по дням (длина days), известные на close дня: P, S1, S2, R1, R2 (пивот) и/или vwap, dev (S6). Нужные для оси — не None."""
    P: np.ndarray | None = None
    S1: np.ndarray | None = None
    S2: np.ndarray | None = None
    R1: np.ndarray | None = None
    R2: np.ndarray | None = None
    vwap: np.ndarray | None = None
    dev: np.ndarray | None = None


def piv_on(p) -> bool:
    return p.piv_sl is not None or p.piv_tp is not None


def piv_check(p, lv, n_days: int) -> None:
    if not piv_on(p):
        return
    if p.piv_sl not in (None, *PIV_SL) or p.piv_tp not in (None, *PIV_TP):
        raise ValueError("piv_sl: next | half | vwap; piv_tp: P | R | half")
    if p.piv_sl is not None and (p.b is not None or p.atr_m is not None or p.chand is not None or p.k_tr is not None or p.be
                                 or p.vhod_tip != "podtverzhdenie" or p.parab is not None):
        raise ValueError("piv_sl — единственный стоп: без b, atr_m, chand, k_tr, be и стоп-заявки")
    if p.piv_tp is not None and (p.a is not None or p.R is not None or p.tejk != "net"):
        raise ValueError("piv_tp — единственный тейк: без a, R, tejk")
    if (p.piv_sl == "vwap") != (p.vwap_m is not None) or (p.vwap_m is not None and not p.vwap_m > 0):
        raise ValueError("vwap_m > 0 — ровно при piv_sl = vwap")
    if not isinstance(lv, PivUrovni):
        raise ValueError("piv_sl / piv_tp требуют PivUrovni в tejk_M")
    need = {"vwap": ("vwap", "dev"), "next": ("P", "S1", "S2", "R1", "R2"), "half": ("P", "S1", "S2", "R1", "R2"), None: ()}[p.piv_sl]
    need += {"P": ("P", "S1", "R1"), "R": ("S1", "R1"), "half": ("S1", "R1"), None: ()}[p.piv_tp]
    for k in dict.fromkeys(need):
        a = getattr(lv, k)
        if a is None or len(a) != n_days:
            raise ValueError(f"PivUrovni.{k}: нужен ряд длины days")


def _fin(x) -> float:
    return float(x) if x is not None and math.isfinite(x) else float("nan")


def _floor(d: float, close: float) -> float:
    return max(d, MIN_D * close) if math.isfinite(d) else MIN_D * close


def stop_dist(p, lv, sd: int, d: int, close: float) -> float | None:
    """Расстояние стопа от цены входа по уровням дня сигнала d; None — ось выключена."""
    if p.piv_sl is None:
        return None
    if p.piv_sl == "vwap":
        w, v = _fin(lv.vwap[d]), _fin(lv.dev[d])
        return _floor(abs(close - (w - sd * p.vwap_m * v)), close)
    P, S1, R1 = _fin(lv.P[d]), _fin(lv.S1[d]), _fin(lv.R1[d])
    S2, R2 = _fin(lv.S2[d]), _fin(lv.R2[d])
    S2 = S2 if math.isfinite(S2) else S1 - abs(P - S1)
    R2 = R2 if math.isfinite(R2) else R1 + abs(R1 - P)
    base = abs(S1 - S2) if sd > 0 else abs(R2 - R1)
    return _floor(base * (0.5 if p.piv_sl == "half" else 1.0) * (1.0 + p.piv_buf), close)


def take_dist(p, lv, sd: int, d: int, close: float) -> float | None:
    if p.piv_tp is None:
        return None
    P, S1, R1 = _fin(lv.P[d]) if lv.P is not None else float("nan"), _fin(lv.S1[d]), _fin(lv.R1[d])
    if p.piv_tp == "P":
        base = abs(P - S1) if sd > 0 else abs(R1 - P)
    else:
        base = abs(R1 - S1) * (0.5 if p.piv_tp == "half" else 1.0)
    return _floor(base, close)


def enter_levels(p, lv, sd: int, px: float, d: int, close: float, tick) -> tuple:
    """(стоп, тейк) сделки, фиксируются на входе по цене входа px и уровням дня сигнала d; выключенная ось — None."""
    ds, dt = stop_dist(p, lv, sd, d, close), take_dist(p, lv, sd, d, close)
    S = px - sd * ds if ds is not None else None
    T = px + sd * dt if dt is not None else None
    if T is not None and tick is not None:
        T = ceil_tick(T, tick) if sd > 0 else floor_tick(T, tick)
    return S, T
