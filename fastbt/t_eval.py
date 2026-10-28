"""Оценщик T16–T20 (контракт #376 §2.1, К-T0): события → sig (`postroit_sig`) → `run_e1` → `Engine.port_rows`.

Вход рынком по close Н6 (k=None), выход по времени e + τ; в сетке стопа нет (k_s = нет), досчёт k_s — `stop_b`.
Дневной путь (`dnevnoy=True`) — тот же sig на однобарных днях с check="zakrytie": для ворот «воспроизводится s9».
Движок (`eng` с `port_rows`, `inv` — индекс даты строки) отдаёт вызывающий, как в `etap1_eval.Evaluator._engine`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fastbt.etap1_data import _one_bar
from fastbt.etap1_run import run_e1
from fastbt.exec_e_types import Params
from fastbt.t_obshchee import postroit_sig


@dataclass
class TInput:
    """Один тикер: days — бары дней 0..b (последний — однобарный close дня b, Н5); close — close Н6 дней 0..b."""
    days: list
    close: np.ndarray
    sigma: np.ndarray               # σ̂ (sigma_real), длина b+1
    tick: float
    a: int                          # окно строк [a; b)
    b: int


class TEvaluator:
    def __init__(self, inputs: dict, eng, commission: float = 0.0004):
        self.inp, self.eng, self.comm = dict(inputs), eng, commission

    def _one(self, t: str, events, tau: int, porog: float, stop_b, dnevnoy: bool, lag: int):
        x = self.inp[t]
        n = x.b + 1
        sig, w, acc = postroit_sig(n, events, tau, x.sigma[:n], porog, lag)
        sig[-1], w[-1] = 0, 0.0                         # день b — только close для строки b−1
        days = [_one_bar(float(c)) for c in x.close[:n]] if dnevnoy else list(x.days[:n])
        p = Params(k=None, b=stop_b, tau=tau, tick=x.tick, stop_touch=True, reentry=False,
                   check="zakrytie" if dnevnoy else "bary", comm=self.comm)
        return run_e1(t, days, sig, x.sigma[:n], p, close=x.close[:n], w=w, win=(x.a, x.b)), acc

    def rows(self, events_by_t: dict, tau: int, porog: float, stop_b=None, dnevnoy: bool = False, lag: int = 1):
        """(df портфеля, ленты E1Rows, принятые события по тикерам). Нет событий — нулевая лента."""
        res = {t: self._one(t, events_by_t.get(t, []), tau, porog, stop_b, dnevnoy, lag) for t in self.inp}
        cat = lambda f: np.concatenate([f(r) for r, _ in res.values()])        # noqa: E731
        df = self.eng.port_rows(cat(lambda r: r.g), cat(lambda r: r.sides), cat(lambda r: r.expo), self.comm)
        return df, [r for r, _ in res.values()], {t: a for t, (_, a) in res.items()}

    def evaluate(self, events_by_t: dict, tau: int, porog: float, **kw) -> dict:
        df, rs, acc = self.rows(events_by_t, tau, porog, **kw)
        return {"excess": df["excess"].to_numpy(), "sdelok": sum(len(r.trades) for r in rs),
                "pnl": float(df["gross"].sum()), "prinyato": {t: len(a) for t, a in acc.items()}}
