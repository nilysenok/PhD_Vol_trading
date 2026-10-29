"""Единица счёта (тикер × стратегия): данные, сигналы, RM-матрицы, сегменты лет."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastbt import grids
from fastbt.batch import (L_A1, L_A2, L_ADX, L_BE, L_CD, L_DIST, L_KIND, L_MH, L_TD,
                          T_BTH, T_CD, T_ISL, T_KIND, T_PM, T_PS, T_RH, T_TM, T_TT)
from fastbt.data import load_ticker, year_bounds
from fastbt.engine import next_signal
from fastbt.indicators import compute_base, rolling_hl
from fastbt.signals.interface import WARMUP
from fastbt.sigprov import get_signals

SEG_YEARS = list(range(2016, 2027))   # границы сегментов: WARMUP, 1.01.2016 … 1.01.2026, n
TREND_TN = [10, 15, 20, 30, 50, 100, 200]
PIV_COMBOS = [(s, t, b) for s in ("next_level", "next_level_half")
              for t in ("P", "next_R", "half_to_next") for b in (0, 0.1, 0.2)]
VWAP_DEVS = [1.5, 2.0, 2.5, 3.0]


@dataclass
class Unit:
    ticker: str
    sid: str
    td: object
    ind: dict
    ss: object

    def __post_init__(self):
        td, ss = self.td, self.ss
        self.cat = grids.CATEGORY[self.sid]
        self.trend = self.cat == "Trend"
        self.n = td.n
        self.start = WARMUP
        self.close, self.high, self.low, self.lr = td.close, td.high, td.low, td.log_ret
        self.dates = td.dates
        self.atr = self.ind["atr14"]
        self.atr0 = np.where(np.isnan(self.atr), 0.0, self.atr)
        self.adx = self.ind["adx14"]
        self.sigs = np.ascontiguousarray(ss.sig)
        self.exl = np.ascontiguousarray(ss.exit_l)
        self.exs = np.ascontiguousarray(ss.exit_s)
        self.strength = np.ascontiguousarray(ss.strength)
        self.nxt = np.stack([next_signal(s) for s in self.sigs])
        b = [WARMUP] + [year_bounds(self.dates, y)[0] for y in SEG_YEARS] + [self.n]
        self.segb = np.array(b, np.int64)
        self.nbars = np.diff(self.segb).astype(np.float64)
        self.seg_of = np.full(self.n, -1, np.int64)
        for k in range(len(b) - 1):
            self.seg_of[b[k]:b[k + 1]] = k
        self.sig3 = np.full((1, self.n), np.nan)
        self.med3 = np.zeros(1)
        self.rm = grids.rm_grid(self.sid)
        self._build_rm()

    def seg_index(self, bound: int) -> int:
        """Номер последнего сегмента окна [WARMUP, bound)."""
        j = int(np.searchsorted(self.segb, bound)) - 1
        assert self.segb[j + 1] == bound, (bound, self.segb)
        return j

    def _build_rm(self):
        sid, n = self.sid, self.n
        dummy = np.full((len(self.sigs), 1, n), np.nan)
        self.sdl = self.sds = self.tdl = self.tds = dummy
        if self.trend:
            rh, rl = [], []
            for tn in TREND_TN:
                if f"rh_{tn}" in self.ind:
                    rh.append(self.ind[f"rh_{tn}"])
                    rl.append(self.ind[f"rl_{tn}"])
                else:
                    a, c = rolling_hl(self.high, self.low, tn)
                    rh.append(a)
                    rl.append(c)
            self.rh, self.rl = np.stack(rh), np.stack(rl)
            P = np.zeros((len(self.rm), 12))
            tt = {"fixed_atr": 0, "chandelier": 1, "parabolic_step": 2, "atr_trail": 3}
            for i, r in enumerate(self.rm):
                be = r["breakeven_thresh"]
                P[i, [T_KIND, T_ISL, T_TM, T_BTH, T_TT, T_PS, T_PM, T_CD, T_RH]] = [
                    0, r["initial_sl_mult"], r["trail_atr_mult"], 1e12 if be is None else be,
                    tt[r["trail_type"]], r["parabolic_step"], r["parabolic_max"],
                    r["cooldown_bars"], TREND_TN.index(int(r["trail_n"]))]
            self.rmP = P
            return
        self.rh = self.rl = np.zeros((1, n))
        P = np.zeros((len(self.rm), 11))
        for i, r in enumerate(self.rm):
            t = r.get("_rm_type", "atr")
            if t == "atr":
                assert not r.get("partial_exit", False)
                be = r["breakeven_trigger"]
                P[i, [L_KIND, L_A1, L_A2, L_BE, L_TD, L_ADX]] = [
                    0, r["sl_mult"], r["tp_mult"], -1.0 if be is None else be,
                    1 if r.get("time_decay") else 0, 0 if self.cat == "Contrarian" else 1]
            else:
                di = (PIV_COMBOS.index((r["sl_target"], r["tp_target"], r["buffer"]))
                      if t == "pivot" else VWAP_DEVS.index(r["sl_dev"]))
                P[i, [L_KIND, L_BE, L_ADX, L_DIST]] = [1, 0.5 if r["breakeven"] else -1.0, 1, di]
            P[i, L_MH] = r["max_hold"]
            P[i, L_CD] = r["cooldown_bars"]
        self.rmP = P
        if sid in ("S5", "S6"):
            self._build_dist()

    def _build_dist(self):
        c, x = self.close, self.ss.extra
        md = 0.001 * c
        K = len(self.sigs)
        combos = PIV_COMBOS if self.sid == "S5" else VWAP_DEVS
        out = [np.empty((K, len(combos), self.n)) for _ in range(4)]
        for k in range(K):
            if self.sid == "S5":
                P, S1, S2, R1, R2 = (x[n][k] for n in ("P", "S1", "S2", "R1", "R2"))
                s2 = np.where(np.isnan(S2), S1 - np.abs(P - S1), S2)
                r2 = np.where(np.isnan(R2), R1 + np.abs(R1 - P), R2)
                lsb, ssb = np.abs(S1 - s2), np.abs(r2 - R1)
                tpl, tps, tnr = np.abs(P - S1), np.abs(R1 - P), np.abs(R1 - S1)
                for j, (slt, tpt, buf) in enumerate(combos):
                    m = 0.5 if slt == "next_level_half" else 1.0
                    out[0][k, j] = np.maximum(lsb * m * (1.0 + buf), md)
                    out[1][k, j] = np.maximum(ssb * m * (1.0 + buf), md)
                    lt, st = {"P": (tpl, tps), "next_R": (tnr, tnr)}.get(
                        tpt, (0.5 * tnr, 0.5 * tnr))
                    out[2][k, j] = np.maximum(lt, md)
                    out[3][k, j] = np.maximum(st, md)
            else:
                vwap, dev = x["vwap"][k], x["dev"][k]
                for j, sd in enumerate(combos):
                    out[0][k, j] = np.maximum(np.abs(c - (vwap - sd * dev)), md)
                    out[1][k, j] = np.maximum(np.abs((vwap + sd * dev) - c), md)
                    out[2][k, j] = np.maximum(np.abs(vwap - c), md)
                    out[3][k, j] = np.maximum(np.abs(c - vwap), md)
        self.sdl, self.sds, self.tdl, self.tds = out


def make_unit(ticker: str, sid: str, td=None) -> Unit:
    td = load_ticker(ticker) if td is None else td
    ind = compute_base(td.close, td.high, td.low, td.open, td.volume)
    return Unit(ticker, sid, td, ind, get_signals(sid, td, ind))
