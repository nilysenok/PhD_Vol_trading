"""B2: синтетический мир для тестов (без сети и реальных данных): бары SBER / IMOEX / бумаги, F, признаки F1–F6, матрица Б-2."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from models_v2.s2 import features as ft
from models_v2.s4 import design, fit, mkt, synth
from models_v2.s5.b2 import engine, feat

HP = fit.HP(3, 10.0, 2.0, 0.1, 0.8)


def bars(n: int, seed: int, start: str = "2019-01-02") -> pd.DataFrame:
    dts = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(seed + 1)
    b = synth.from_returns(dts, synth.random_returns(n, seed), 0.002 * rng.standard_normal(n), evening=0.001 * rng.standard_normal((n, 2)))
    b["volume"] = rng.integers(1000, 5000, len(b))
    return b


def damage(b: pd.DataFrame, d: pd.Timestamp) -> pd.DataFrame:
    out = synth.damage_after(b, d)
    t = out["begin"].dt.hour * 60 + out["begin"].dt.minute
    bad = (out["begin"].dt.normalize() > d) | ((out["begin"].dt.normalize() == d) & (t >= 18 * 60 + 50))
    out.loc[bad, "volume"] = out.loc[bad, "volume"] * 7 + 3
    return out


class World:
    """F, feats, dz0 (Б-2: форма m + 17 рыночных), alarms; по барам SBER (a), IMOEX (b), бумаги (c)."""

    def __init__(self, a, b, c):
        self.F = ft.build(a)
        self.feats = feat.build(self.F, a, b)
        M, _ = mkt.block(None, self.F.X[:, [0, 1, 2]], self.F.seg, self.F.dates, [mkt.paper(a), mkt.paper(c)], b)
        self.dz0 = design.make(self.F, "m", design.columns(M, "imoex"))
        from models_v2.s5.b2 import cusum
        self.alarms = {h: cusum.run(cusum.har_arrivals(self.F), h)[2] for h in (5.0, 8.0)}
        self.i2 = defaultdict(lambda: HP)

    def forecast(self, cfg: engine.Cfg, T: int, d: int):
        dz = engine.make_dz(self.dz0, self.feats, cfg)
        return engine.fit_block(self.F, dz, cfg, HP, T, [d], self.alarms.get(cfg.cusum_h))["f"]


def pair(n: int = 700, seed: int = 11, start: str = "2019-01-02") -> tuple:
    return bars(n, seed, start), bars(n, seed + 1, start), bars(n, seed + 2, start)
