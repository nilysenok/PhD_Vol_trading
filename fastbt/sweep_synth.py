"""Синтетический Evaluator для тестов движка перебора: AR(1)-доходности, позиция = знак скользящего среднего.

`a` — параметр сигнала (окно), `w` — параметр PnL (плечо). `late` умножает доходности с 2020 года
(проверка, что validation не влияет на выбор). Реальных данных не читает.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


class SynthEvaluator:
    sig_names = {"a"}

    def __init__(self, seed: int = 0, late: float = 1.0, code: str = "v1"):
        self.dates = pd.bdate_range("2018-01-01", "2021-12-31")
        e = np.random.default_rng(seed).normal(0, 0.01, len(self.dates))
        r = np.zeros(len(e))
        for t in range(1, len(e)):
            r[t] = 0.15 * r[t - 1] + e[t] + 0.0002
        r[self.dates.year >= 2020] *= late
        self.ret, self.code, self.signal_calls = r, code, 0

    def series_digest(self) -> str:
        return hashlib.sha1(self.ret.tobytes()).hexdigest()

    def code_id(self) -> str:
        return self.code

    def signals(self, sp: dict):
        self.signal_calls += 1
        a = int(sp["a"])
        past = pd.Series(self.ret).shift(1).rolling(a).mean().fillna(0.0).values
        return np.sign(past)

    def pnl(self, sigs, p: dict, comm: float, end) -> pd.DataFrame:
        m = self.dates < end
        pos = sigs[m] * p.get("w", 1.0)
        d = np.abs(np.diff(pos, prepend=0.0))
        net = pos * self.ret[m] - d * comm - 0.0001 * pos**2    # плата за удержание: Шарп зависит от w
        return pd.DataFrame(dict(excess=net, exposure=np.abs(pos), turnover=d, sides=(d > 0).astype(float)),
                            index=self.dates[m])
