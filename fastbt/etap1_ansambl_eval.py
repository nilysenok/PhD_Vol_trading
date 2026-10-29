"""Оценщик ансамбля (Ж-051, итерация 2): позиции каждой точки — путь оценщика этапа 1 (сверен с боевым s9),
сведение (`etap1_ansambl.combine`) — до движка, выходы/σ/размер/вход — центральной точки. Ансамбль из одной точки
побайтно равен этой точке (golden-тест на данных). Только 2017–2019.
"""
from __future__ import annotations

import numpy as np

from fastbt import ekviv_s9
from fastbt.etap1_ansambl import combine
from fastbt.etap1_data import _one_bar, align_b, gate_trades
from fastbt.etap1_eval import NET, PERIODS, REZHIM, Evaluator, params_of
from fastbt.etap1_run import run_e1
from fastbt import rezhim_e1


class AnsamblEvaluator(Evaluator):
    ens = None                                      # (список сигнальных точек, режим) или None — обычная точка

    def _ens_signal(self, t: str, pts: list, mode: str, dates, s0: int, b: int):
        """Позиции каждой точки (оценщик этапа 1), вход по режиму/событиям этой точки, затем сведение по дням."""
        P = []
        for q in pts:
            q = self.full(q)
            sp = {k: q[k] for k in self.sk.sig_names}
            pos = self._positions(t, sp, float(q.get("q_sily", 0.0)), False, s0 + b)[s0:s0 + b + 1].astype(float)
            pos[-1] = 0.0                           # день b — только close для строки b−1
            allow = ~self.inp.sobytiya[q["sobytiya"]][t]
            if q["rezhim"] != NET:
                allow = allow & align_b(self.inp.rezhim[REZHIM.get(q["rezhim"], q["rezhim"])], dates)
            P.append(gate_trades(pos, allow))
        return combine(np.array(P), mode)

    def _rows(self, pt: dict, period: str, anchor: bool):
        if self.ens is None or anchor:
            return super()._rows(pt, period, anchor)
        pts, mode = self.ens
        y0, _ = PERIODS[period]
        sp = {k: pt[k] for k in self.sk.sig_names}
        q = float(pt.get("q_sily", 0.0))
        h = 5.0 if (pt["tau"] != NET or pt["R"] == NET) else self._h(sp, q)
        out = []
        for ti, t in enumerate(self.sk.tickers):
            m, inp = self.inp.meta[t], self.inp
            s0, b = m["s0"], self._b(t, period)
            dates = m["dates"][:b + 1]
            a = int(np.searchsorted(dates, np.datetime64(f"{y0}-01-01")))
            d, mag = self._ens_signal(t, pts, mode, dates, s0, b)
            days = inp.bars[t][:b] + [_one_bar(float(inp.close_n6[t][b]))]
            sg = inp.sigma[pt["sigma"]][t][:b + 1]
            w = np.ones(len(d))
            if pt["razmer"] != "ravnyy":
                w = rezhim_e1.razmer_ves(sg, "obratnyj", float(pt["razmer"]) / 100.0)
            p = params_of(pt, inp.tick[t], h, False, self.comm)
            out.append((ti, run_e1(t, days, d, sg, p, close=inp.close_n6[t][:b + 1], w=w * mag, win=(a, b),
                                         div=inp.div[t][:b + 1], dates=dates, **self.fdr_kw()), a, b))
        return out

    def evaluate_ens(self, center: dict, pts: list, mode: str, period: str = "train") -> dict:
        self.ens = (pts, mode)
        try:
            return self.evaluate(center, period)
        finally:
            self.ens = None


def sverka_s9(evr: Evaluator, pts: list) -> dict:
    """Каждая сигнальная точка против боевого пути s9 (`ekviv_s9.boevoy_pos`: Unit + ApproachA.raw_pos + exec_layer +
    tfilter), родные выходы Sk, позиции 2018–2019 по всем тикерам. Нет боевого аналога (q_sily ≠ 0, маски, значения RM,
    которых Unit не принимает) — точка в `net_etalona`, не в `sovpalo`."""
    ok, no_ref, bad = 0, 0, []
    for i, q in enumerate(pts):
        q = evr.full(q)
        sp = {k: q[k] for k in evr.sk.sig_names}
        if float(q.get("q_sily", 0.0)) != 0.0:
            no_ref += 1
            continue
        diff = 0
        try:
            for t in evr.sk.tickers:
                td, ind = evr.sk._td[t]
                m = evr.inp.meta[t]
                bo = ekviv_s9.boevoy_pos(evr.sid, {**evr.base, **sp}, t, td, ind)
                mine = evr._positions(t, sp, 0.0, True, m["i1"])
                diff += int((mine[m["a18"]:m["i1"]] != bo[m["a18"]:m["i1"]]).sum())
        except ekviv_s9.NoRef:
            no_ref += 1
            continue
        ok, bad = (ok + 1, bad) if diff == 0 else (ok, bad + [(i, diff)])
    return dict(sovpalo=ok, net_etalona=no_ref, rashozhdeniy=bad)
