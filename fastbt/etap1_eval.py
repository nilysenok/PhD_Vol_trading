"""Оценщик этапа 1 (К4, Ж-047): точка сетки → портфель через exec_e и `Engine.port_rows` → строка v1.0.

Режимы (толкование [Ж-047, приёмка #345] п. 3):
 - «якорь» (все оси = yakor): родные выходы Sk на месте, σ-выходы выключены, вход рынком, равный размер — результат
   обязан совпасть с «A только Sk» через s9 (SkEvaluator.pnl) до 1e-12;
 - «сетка»: родные RM-выходы выключены (RM_OFF), σ-выходы включены, сигнальные выходы остаются, k_s = нет — стопа нет.
Режим, события и квантиль силы запрещают только вход (сделка Sk целиком пропускается). Повторный вход после
σ-выхода блокирован до новой сделки Sk (reentry=False). Вес фиксирован на сделку (w = цель/(σ_d·√252), L = 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import cache, metriki_e as me, rezhim_e1
from fastbt.data import year_bounds
from fastbt.etap1_data import _one_bar, align_b, gate_trades
from fastbt.etap1_run import run_e1
from fastbt.exec_e_types import Params
from fastbt.fdr import REZHIMY as REZHIMY_FDR
from fastbt.s9 import Engine
from fastbt.sweep_post import build_view, positions
from fastbt.sweep_sk import _Mini

YAKOR = dict(q_sily=0.0, sigma="rv22", vhod="rynok", k_s="net", R="net", k_tr="net", bezubytok="net", tau="net",
             proverka="zakrytie", razmer="ravnyy", rezhim="net", sobytiya="net")
OFF, FAR = 1e6, 10 ** 6
_ATR = {"rm_tip": "atr", "rm_atr.sl_mult": OFF, "rm_atr.tp_mult": OFF, "rm_atr.max_hold": FAR,
        "rm_atr.breakeven_trigger": None, "rm_atr.cooldown_bars": 0, "rm_atr.time_decay": False}
RM_OFF = {"S1": dict(sl_mult=OFF, tp_mult=OFF, max_hold=FAR, breakeven_trigger=None, cooldown_bars=0),
          "S3": dict(initial_sl_mult=OFF, trail_atr_mult=OFF, breakeven_thresh=None, cooldown_bars=0), "S5": _ATR}
RM_OFF.update(S2=RM_OFF["S1"], S4=RM_OFF["S3"], S6=_ATR)
PERIODS = {"train": (2018, 2019), "ctrl2019": (2019, 2019), "tune2018": (2018, 2018), "train4": (2016, 2019)}
HONEST = {"train": "net", "ctrl2019": "kontrol", "tune2018": "net", "train4": "net"}     # train4 — В1, 2015 — прогрев


def years_of(period: str) -> tuple:
    """Годы метрик периода (прогрев в них не входит)."""
    y0, y1 = PERIODS[period]
    return tuple(range(y0, y1 + 1))
NET = "net"
REZHIM = {"imoex_ma200": "ma200", "rang_sigma_rynka_08": "rank08"}     # имена оси в zh047_S*.yaml → rezhim_e1
NE_SCHITANO = ("div", "cbr_div")                                         # ❓5: даты дивидендов не сверены с эмитентом


def split_years(x: np.ndarray, yrs: np.ndarray, years: tuple) -> tuple:
    """(ряд только по годам периода, {год: ряд}); дни прогрева (годы вне периода) в метрики не входят."""
    keep = np.isin(yrs, years)
    return x[keep], {int(y): x[yrs == y] for y in sorted(set(yrs[keep]))}


def is_anchor(pt: dict) -> bool:
    return all(pt.get(k, v) == v for k, v in YAKOR.items()) and pt.get("vyhod", NET) == NET


def params_of(pt: dict, tick: float, h: float, anchor: bool, comm: float) -> Params:
    """Точка → Params exec_e. Якорь: все σ-правила выключены."""
    if anchor:
        return Params(check="zakrytie", comm=comm)
    num = lambda v: None if v == NET else v                                      # noqa: E731
    ks, tau = num(pt["k_s"]), num(pt["tau"])
    lim = None if pt["vhod"] == "rynok" else float(pt["vhod"])
    return Params(k=lim, b=ks, R=num(pt["R"]) if ks is not None else None, h=h if tau is None else None,
                  k_tr=num(pt["k_tr"]) if ks is not None else None, be=pt["bezubytok"] != NET and ks is not None,
                  tau=None if tau is None else int(tau), tick=tick, stop_touch=True, reentry=False,
                  check="zakrytie" if pt["proverka"] == "zakrytie" else "bary", comm=comm,
                  vyhod_lim=pt.get("vyhod", NET) == "lim")                # ось vyhod: net | lim (Ж-065 п. 3)


def v11(out: dict, fdr, pt: dict, rs: list, done: list) -> dict:
    """§1.3 Г: поля v1.1 — только при включённом fdr или vyhod = lim, иначе строка v1.0 как была."""
    vy = pt.get("vyhod", NET)
    if fdr is None and vy == NET:
        return out
    xo, xf = sum(r.x_orders for r in rs), sum(r.x_fills for r in rs)
    out["prichiny"]["lim"] = sum(tr.reason == "lim" for tr in done) / max(len(done), 1)
    dolya = xf / xo if xo else float("nan")
    return {**out, "v": "1.1", "fdr": fdr, "vyhod": vy, "dolya_ispolneniya_vyhoda": dolya}


class Evaluator:
    fdr = None                                      # подклассы без __init__ родителя: режим выключен

    def fdr_kw(self) -> dict:                       # fdr = None: вызов run_e1 прежний, без ключа
        return {} if self.fdr is None else {"fdr": self.fdr}

    def __init__(self, sid: str, inp, base_p: dict, commission: float = 0.0004, fdr=None):
        """fdr: None — ставка по доле баров; "t0" | "t1" — занятость на 18:50 (Ж-065 п. 3); у режима свой ledger."""
        if fdr is not None and fdr not in REZHIMY_FDR:
            raise ValueError(f"fdr: None или {REZHIMY_FDR}, получено {fdr!r}")
        self.sid, self.inp, self.base, self.comm, self.fdr = sid, inp, dict(base_p), commission, fdr
        self.sk, self._sigs, self._pos, self._eng = inp.sk, {}, {}, {}

    def full(self, pt: dict) -> dict:
        """Точка с умолчаниями: оси общие — якорь, сигнальные — центр сетки (ekviv_20.yaml#Sk_a)."""
        return {**YAKOR, **{k: self.base[k] for k in self.sk.sig_names}, **pt}

    def _b(self, t: str, period: str) -> int:
        """Локальный индекс b первого дня после окна (строка b−1 — последняя строка окна, Н5)."""
        y1 = PERIODS[period][1]
        return int(np.searchsorted(self.inp.meta[t]["dates"], np.datetime64(f"{y1 + 1}-01-01")))

    def _positions(self, t: str, sp: dict, q: float, anchor: bool, gi1: int) -> np.ndarray:
        k = (t, tuple(sorted(sp.items())), q, anchor, gi1)
        if k not in self._pos:
            sk = tuple(sorted(sp.items()))
            if sk not in self._sigs:
                self._sigs[sk] = self.sk.signals(sp)
            td, ind = self.sk._td[t]
            p = {**self.base, **sp, "q_sily": q, **({} if anchor else RM_OFF[self.sid])}
            v, ex = build_view(self.sid, self._sigs[sk][t], td, ind, p, self.sk.sig_names)
            self._pos[k] = positions(v, ex, gi1)
        return self._pos[k]

    def _h(self, sp: dict, q: float) -> float:
        """❓1: h = медианное удержание (дней) сделок базы Sk в 2017 (ex-ante, до окна); нет сделок — 5."""
        runs = []
        for t, m in self.inp.meta.items():
            pos = self._positions(t, sp, q, False, m["i1"])[m["s0"]:m["a18"]]
            ch = np.flatnonzero(np.diff(pos, prepend=0.0, append=0.0) != 0)
            runs += [e - s for s, e in zip(ch[:-1], ch[1:]) if pos[s] != 0]
        return float(np.median(runs)) if runs else 5.0

    def _rows(self, pt: dict, period: str, anchor: bool):
        y0, y1 = PERIODS[period]
        sp = {k: pt[k] for k in self.sk.sig_names}
        q = float(pt.get("q_sily", 0.0))
        h = 5.0 if (anchor or pt["tau"] != NET or pt["R"] == NET) else self._h(sp, q)
        out = []
        for ti, t in enumerate(self.sk.tickers):
            m, inp = self.inp.meta[t], self.inp
            s0, b = m["s0"], self._b(t, period)
            dates = m["dates"][:b + 1]
            a = int(np.searchsorted(dates, np.datetime64(f"{y0}-01-01")))
            sig = self._positions(t, sp, q, anchor, s0 + b)[s0:s0 + b + 1].astype(float)
            sig[-1] = 0.0                                   # день b — только close для строки b−1
            days = inp.bars[t][:b] + [_one_bar(float(inp.close_n6[t][b]))]
            sg = np.ones(len(sig)) if anchor else inp.sigma[pt["sigma"]][t][:b + 1]
            if not anchor:
                allow = ~inp.sobytiya[pt["sobytiya"]][t]
                if pt["rezhim"] != NET:
                    allow = allow & align_b(inp.rezhim[REZHIM.get(pt["rezhim"], pt["rezhim"])], dates)
                sig = gate_trades(sig, allow)
            w = np.ones(len(sig))
            if not anchor and pt["razmer"] != "ravnyy":
                w = rezhim_e1.razmer_ves(sg, "obratnyj", float(pt["razmer"]) / 100.0)
            p = params_of(pt, inp.tick[t], h, anchor, self.comm)
            out.append((ti, run_e1(t, days, sig, sg, p, close=inp.close_n6[t][:b + 1], w=w, win=(a, b),
                                         div=inp.div[t][:b + 1], **self.fdr_kw()), a, b))
        return out

    def _engine(self, period: str):
        if period not in self._eng:
            y0, _ = PERIODS[period]
            dates, lr, tick = [], [], []
            for ti, t in enumerate(self.sk.tickers):
                m, td = self.inp.meta[t], self.sk._td[t][0]
                i0 = year_bounds(td.dates, y0)[0]
                idx = np.arange(i0, m["s0"] + self._b(t, period))
                dates.append(td.dates[idx]), lr.append(td.log_ret[idx]), tick.append(np.full(len(idx), ti))
            dts, inv = np.unique(np.concatenate(dates), return_inverse=True)
            tick = np.concatenate(tick)
            first = np.r_[True, tick[1:] != tick[:-1]]
            self._eng[period] = (_Mini(dts, inv, first, np.concatenate(lr)), inv)
        return self._eng[period]

    def excess(self, pt: dict, period: str):
        """(df портфеля, стороны по строкам, лента сделок) для точки; точки без якоря — сетка."""
        pt = self.full(pt)
        anchor = is_anchor(pt)
        rows = self._rows(pt, period, anchor)
        eng, inv = self._engine(period)
        cat = lambda f: np.concatenate([f(r) for _, r, _, _ in rows])           # noqa: E731
        noch = cat(lambda r: r.noch) if self.fdr is not None else None
        df = eng.port_rows(cat(lambda r: r.g), cat(lambda r: r.sides), cat(lambda r: r.expo), self.comm, noch=noch)
        return df, inv, eng, cat(lambda r: r.cnt), [r for _, r, _, _ in rows]

    def evaluate(self, pt: dict, period: str) -> dict:
        if period not in PERIODS:
            raise PermissionError(f"период {period!r}: этап 1 считает только {sorted(PERIODS)}")
        if self.full(pt)["sobytiya"] in NE_SCHITANO:        # не считаем молча как net: строка «не считана», mSR = nan
            nan = {y: float("nan") for y in range(PERIODS[period][0], PERIODS[period][1] + 1)}
            return {"v": "1.0", "ne_schitana": "sobytiya div/cbr_div ждёт сверки дат (❓5)", "msr_by_year": nan,
                    "tselevaya": float("nan"), "sdelok_y": {}, "excess_d": {}, "vyrozhdena": True}
        df, inv, eng, cnt, rs = self.excess(pt, period)
        ex = df["excess"]
        yrs = ex.index.year
        x, yy = split_years(ex.to_numpy(), np.asarray(yrs), years_of(period))
        ryear = eng.dates.year.to_numpy()[inv]
        trades = [tr for r in rs for tr in r.trades]
        done = [tr for tr in trades if tr.reason]
        n_o, n_f = sum(r.n_orders for r in rs), sum(r.n_fills for r in rs)
        pt = self.full(pt)
        out = {"v": "1.0", "sharpe": me.sr_dnevnoy(x) * np.sqrt(252),
                "by_year": {y: me.sr_dnevnoy(v) * np.sqrt(252) for y, v in yy.items()},
                "msr": me.msr(x), "msr_by_year": {y: me.msr(v) for y, v in yy.items()},
                "tselevaya": min(me.msr(v) for v in yy.values()), "sortino": me.sortino_e(x), "kalmar": me.kalmar(x),
                "mdd": me.mdd(x), "T": len(x), "skew": me.asimmetriya(x), "kurt": me.ekscess(x) + 3.0,
                "storony_y": {y: float(cnt[ryear == y].sum()) for y in yy},
                "sdelok_y": {y: float(cnt[ryear == y].sum()) / 2 for y in yy},
                "dolya_ispolneniya_limita": n_f / n_o if n_o else float("nan"),
                "prichiny": {k: sum(tr.reason == k for tr in done) / max(len(done), 1)
                             for k in ("stop", "take", "trail", "bu", "vremya", "signal")},
                "okno_s": f"{PERIODS[period][0]}-01-01", "okno_po": f"{PERIODS[period][1]}-12-31",
                "chestnaya": HONEST[period], "comm_pct": self.comm * 100,
                "vyrozhdena": bool(pt["k_s"] == NET and pt["R"] != NET), "kod": cache.code_hash(),
                "vhody": self.inp.digest(), "excess_d": {y: list(v) for y, v in yy.items()}}
        return v11(out, self.fdr, pt, rs, done)
